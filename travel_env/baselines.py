"""Baseline policies for the eval harness.

All three are pure functions: `(obs) -> action_dict`. They never call an
LLM, they're cheap, and (given the same obs + rng) deterministic.

The three baselines exist for three different jobs:

  random_policy     — floor. "What does no strategy look like?"
                      Useful as a sanity check that the env doesn't reward
                      noise.

  cheapest_policy   — "always pick cheapest" exploit probe. The reward
                      function should structurally push back; if cheapest
                      out-scores heuristic on overall reward, our reward
                      design is broken (we're rewarding the exploit).

  heuristic_policy  — informed ceiling for non-learned policies. Uses the
                      persona profile to pick options that match soft prefs
                      and respect hard constraints. A trained agent should
                      eventually beat this — but it shouldn't lose to it.

The eval harness passes `profile` (a real PersonaProfile, sampled with the
same seed as the env) so policies can make informed choices. Random uses
the profile only for origin/dest/dates (so its searches are valid); the
content of each action is random.

Signatures kept stable with the skeleton so eval.py can call them directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from travel_env.persona import PersonaProfile
from travel_env.world import CITIES


# Module-level: cheap two-way lookup so search results tagged with IATA
# codes match against city names carried by the profile.
_IATA_TO_NAME: dict[str, str] = {c.iata: name for name, c in CITIES.items()}


# --- Public API ----------------------------------------------------------

def random_policy(
    obs: dict, *, rng: np.random.Generator, profile: PersonaProfile | None = None,
) -> dict:
    """Pick uniformly from currently-valid tool calls (random args within each)."""
    state = obs.get("state", {}) or {}
    itinerary = state.get("itinerary", []) or []
    last_result = obs.get("last_result") or {}

    tentative = [s for s in itinerary if s.get("status") == "tentative"]
    last_results: list[dict] = last_result.get("results") or []

    candidates: list[tuple[str, dict]] = []

    # Always-available searches (with reasonable random args).
    candidates.append(("search_flights", _random_flight_args(obs, profile, rng)))
    candidates.append(("search_hotels", _random_hotel_args(obs, profile, rng)))
    candidates.append(("search_activities", _random_activity_args(obs, profile, rng)))

    # Conditionally available actions.
    if last_results:
        picked = last_results[int(rng.integers(0, len(last_results)))]
        candidates.append(("add_to_itinerary", {
            "item_id": picked["id"],
            "slot": f"slot_{int(rng.integers(0, 9999))}",
        }))
        candidates.append(("get_details", {"item_id": picked["id"]}))

    if tentative:
        # Book a random subset of tentative items.
        k = int(rng.integers(1, len(tentative) + 1))
        ids = [t["item_id"] for t in tentative[:k]]
        candidates.append(("book", {"item_ids": ids}))

    if itinerary:
        slot = itinerary[int(rng.integers(0, len(itinerary)))]["slot"]
        candidates.append(("remove_from_itinerary", {"slot": slot}))

    candidates.append(("propose_to_client", {}))
    candidates.append(("submit_final", {}))

    tool, args = candidates[int(rng.integers(0, len(candidates)))]
    return {"tool": tool, "args": args}


def cheapest_policy(
    obs: dict, *, profile: PersonaProfile | None = None,
) -> dict:
    """Follow the same trip-building flow as the heuristic, but always pick
    cheapest from each search.

    Does NOT add activities. Does NOT propose. The whole point is "minimize
    spend, ignore everything else" — that's the exploit shape we want to
    surface."""
    return _flow_action(obs, mode="cheapest", profile=profile, rng=None)


def heuristic_policy(
    obs: dict, *, rng: np.random.Generator, profile: PersonaProfile | None = None,
) -> dict:
    """Persona-aware. Uses soft_prefs to rank options; respects hard prefs in
    search filters; adds 2 activities targeting the persona's top categories;
    proposes once before submitting."""
    return _flow_action(obs, mode="heuristic", profile=profile, rng=rng)


# --- Shared trip-building state machine ----------------------------------
#
# Both heuristic and cheapest follow the same shape:
#   handle disruption  ->  add outbound  ->  add return  ->  add hotel
#     ->  book base    ->  add activities (heuristic only)
#     ->  propose (heuristic only)  ->  submit
#
# They differ only in selection (`_pick`) and a few choices around activities
# / proposing. Keeps the logic in one place; the mode flag toggles behavior.

def _flow_action(
    obs: dict,
    *,
    mode: str,                              # "cheapest" | "heuristic"
    profile: PersonaProfile | None,
    rng: np.random.Generator | None,
) -> dict:
    state = obs.get("state", {}) or {}
    itinerary = state.get("itinerary", []) or []
    clock = state.get("clock", {}) or {}
    last_result = obs.get("last_result") or {}
    history = obs.get("history", []) or []
    budget = state.get("budget", {}) or {}

    cancelled = [s for s in itinerary if s.get("status") == "cancelled"]
    tentative = [s for s in itinerary if s.get("status") == "tentative"]
    booked = [s for s in itinerary if s.get("status") == "booked"]
    booked_flights = [s for s in booked if _kind(s) == "flight"]
    booked_hotels = [s for s in booked if _kind(s) == "hotel"]
    booked_activities = [s for s in booked if _kind(s) == "activity"]

    # Budget context: shared with _need_slot so it can pre-filter searches
    # by remaining cap. Avoids the "always pick the cheapest of top-k where
    # top-k is ranked by stars/dollar, not absolute price" failure mode.
    budget_ctx = {
        "cap": float(budget.get("cap", 0) or 0),
        "spent": float(budget.get("spent", 0) or 0),
        "tentative": tentative,
    }

    # Bail-out: too many recent book failures or empty searches means the
    # current itinerary is structurally infeasible. Submit and let the gate
    # fail — better than burning turns to truncation.
    recent_book_fails = sum(
        1 for h in history[-5:]
        if (h.get("action") or {}).get("tool") == "book"
        and (h.get("result") or {}).get("ok") is False
    )
    recent_empty_searches = sum(
        1 for h in history[-5:]
        if str((h.get("action") or {}).get("tool", "")).startswith("search_")
        and (h.get("result") or {}).get("count") == 0
    )
    if recent_book_fails >= 2 or recent_empty_searches >= 2:
        return {"tool": "submit_final", "args": {}}

    # If the previous turn was a failed book, remove the most-expensive
    # tentative item before re-entering the build/book loop. The state
    # machine will re-search for that slot with a tighter max_price next turn.
    last_h = history[-1] if history else None
    last_book_failed = (
        last_h is not None
        and (last_h.get("action") or {}).get("tool") == "book"
        and (last_h.get("result") or {}).get("ok") is False
        and tentative
    )
    if last_book_failed:
        # Skip rebook-tentatives (they're emergency replacements; removing
        # them just re-triggers the disruption recovery flow).
        candidates = [t for t in tentative
                      if not str(t.get("slot", "")).endswith("_rebook")]
        if candidates:
            most_exp = max(candidates, key=lambda t: float(t.get("price", 0)))
            return {"tool": "remove_from_itinerary", "args": {"slot": most_exp["slot"]}}

    # Trip params: prefer profile if eval/caller passed one, otherwise read from
    # obs (env exposes origin/dest in clock so policies don't need profile).
    origin = (profile.origin_city if profile else None) or clock.get("origin")
    dest = (profile.dest_city if profile else None) or clock.get("dest")
    depart_date = clock.get("depart_date")
    return_date = clock.get("return_date")

    # Hard constraints from the client. Env exposes these in clock so baselines
    # can filter cleanly without parsing the natural-language request.
    hard_constraints = {
        "no_overnight_flights": (
            profile.hard.no_overnight_flights if profile
            else bool(clock.get("no_overnight_flights"))
        ),
        "max_stops": (
            profile.hard.max_stops if profile
            else clock.get("max_stops")
        ),
        "required_amenities": tuple(
            profile.hard.required_amenities if profile
            else clock.get("required_amenities") or ()
        ),
    }

    # 1) DISRUPTION RECOVERY
    if cancelled and not _has_replacement_for_any(cancelled, tentative + booked_flights):
        return _rebook_action(obs, cancelled, mode, profile, rng, last_result, hard_constraints)

    # If we have a tentative rebook flight from last turn, book it now.
    rebook_tentatives = [t for t in tentative if str(t.get("slot", "")).endswith("_rebook")]
    if rebook_tentatives:
        return {"tool": "book", "args": {"item_ids": [t["item_id"] for t in rebook_tentatives]}}

    # 2) BUILD OUTBOUND
    if not _has_route(booked + tentative, origin, dest):
        return _need_slot(
            obs, slot="outbound_flight", kind="flight",
            src=origin, dst=dest, date=depart_date,
            mode=mode, profile=profile, rng=rng, last_result=last_result,
            hard_constraints=hard_constraints, budget_ctx=budget_ctx,
        )

    # 3) BUILD RETURN
    if not _has_route(booked + tentative, dest, origin):
        return _need_slot(
            obs, slot="return_flight", kind="flight",
            src=dest, dst=origin, date=return_date,
            mode=mode, profile=profile, rng=rng, last_result=last_result,
            hard_constraints=hard_constraints, budget_ctx=budget_ctx,
        )

    # 4) BUILD HOTEL
    if not booked_hotels and not any(_kind(t) == "hotel" for t in tentative):
        return _need_slot(
            obs, slot="hotel", kind="hotel",
            src=dest, dst=dest, date=depart_date,
            mode=mode, profile=profile, rng=rng, last_result=last_result,
            hard_constraints=hard_constraints, budget_ctx=budget_ctx,
        )

    # 5) BOOK BASE TRIP — anything tentative gets committed in bulk
    base_tentatives = [t for t in tentative if not str(t.get("slot", "")).endswith("_rebook")]
    if base_tentatives:
        return {"tool": "book", "args": {"item_ids": [t["item_id"] for t in base_tentatives]}}

    # 6) ACTIVITIES (heuristic only — cheapest deliberately skips to minimize spend)
    if mode == "heuristic" and _wants_activities(profile) and len(booked_activities) < 2:
        slot_name = f"activity_{len(booked_activities) + 1}"
        return _need_slot(
            obs, slot=slot_name, kind="activity",
            src=dest, dst=dest, date=depart_date,
            mode=mode, profile=profile, rng=rng, last_result=last_result,
            hard_constraints=hard_constraints, budget_ctx=budget_ctx,
        )

    # 7) PROPOSE (heuristic only — once per episode)
    if mode == "heuristic":
        proposed = any(
            (h.get("action") or {}).get("tool") == "propose_to_client" for h in history
        )
        if not proposed:
            return {"tool": "propose_to_client", "args": {}}

    # 8) SUBMIT
    return {"tool": "submit_final", "args": {}}


def _need_slot(
    obs: dict, *, slot: str, kind: str,
    src: str | None, dst: str | None, date: str | None,
    mode: str, profile: PersonaProfile | None,
    rng: np.random.Generator | None, last_result: dict,
    hard_constraints: dict | None = None,
    budget_ctx: dict | None = None,
) -> dict:
    """If we just searched for this kind/route, pick from results; else search."""
    results = _results_matching(last_result, kind, src, dst)
    if results:
        chosen = _pick(
            results, mode=mode, profile=profile, rng=rng, kind=kind,
            hard_constraints=hard_constraints,
        )
        return {"tool": "add_to_itinerary", "args": {
            "item_id": chosen["id"], "slot": slot,
        }}

    # Emit a search. Push known hard constraints into the search args so we get
    # pre-filtered results (server-side filter is more efficient + smaller token
    # bill in the LLM rollout case).
    #
    # If the immediately-previous search of this same kind returned 0 results,
    # broaden the search by dropping max_price (and min_stars for hotels). The
    # filter was too aggressive; better to surface options that fail soft prefs
    # than to spin on zero-result retries until the bail-out fires.
    hc = hard_constraints or {}
    prev_empty_same_kind = (
        last_result.get("tool") == f"search_{kind}"
        and last_result.get("count") == 0
    )
    max_price = None if prev_empty_same_kind else _slot_max_price(kind, slot, budget_ctx, obs)

    if kind == "flight":
        args: dict[str, Any] = {"origin": src, "dest": dst, "depart_date": date}
        if hc.get("max_stops") is not None:
            args["max_stops"] = hc["max_stops"]
        if max_price is not None:
            args["max_price"] = max_price
        return {"tool": "search_flights", "args": args}

    if kind == "hotel":
        clock = obs["state"]["clock"]
        args = {
            "city": dst,
            "checkin": clock["depart_date"],
            "checkout": clock["return_date"],
        }
        if mode == "heuristic" and profile and not prev_empty_same_kind:
            target_stars = (profile.soft_prefs or {}).get("hotel_stars", 3)
            try:
                args["min_stars"] = max(1, int(target_stars) - 1)
            except (TypeError, ValueError):
                pass
        if max_price is not None:
            args["max_price"] = max_price
        return {"tool": "search_hotels", "args": args}

    if kind == "activity":
        args = {"city": dst}
        if mode == "heuristic" and profile:
            soft = profile.soft_prefs or {}
            top_cats = _top_activity_categories(soft, k=2)
            if top_cats:
                args["categories"] = top_cats
        if max_price is not None:
            args["max_price"] = max_price
        return {"tool": "search_activities", "args": args}

    # Shouldn't reach.
    return {"tool": "submit_final", "args": {}}


def _slot_max_price(
    kind: str, slot: str, budget_ctx: dict | None, obs: dict,
) -> float | None:
    """Per-slot price ceiling derived from remaining budget.

    Allocation (rough): flights ~40% of cap split across two legs, hotel
    ~50%, activities ~10%. We subtract already-tentative items from
    "remaining" so a return-flight search shrinks if the outbound has
    been added; same idea for the hotel after both flights are tentative.

    Returns None when budget is unknown (so callers omit the filter and
    fall back to the broader top-k).
    """
    if not budget_ctx:
        return None
    cap = float(budget_ctx.get("cap", 0) or 0)
    if cap <= 0:
        return None
    spent = float(budget_ctx.get("spent", 0) or 0)
    tentative = budget_ctx.get("tentative") or []
    other_tentative = sum(
        float(t.get("price", 0) or 0) for t in tentative
        if t.get("slot") != slot
    )
    remaining = max(0.0, cap - spent - other_tentative)
    if remaining <= 0:
        return None

    if kind == "flight":
        # Need room for the *other* flight too. Bias toward the lower half.
        return round(remaining * 0.55, 2)

    if kind == "hotel":
        nights = _nights_in_obs(obs)
        return round(remaining * 0.95 / max(1, nights), 2)

    if kind == "activity":
        # One activity at a time; cap at remaining * 0.25 so we leave room
        # for further activities.
        return round(min(remaining * 0.25, 250.0), 2)

    return None


def _nights_in_obs(obs: dict) -> int:
    """Best-effort nights count from the clock dates."""
    clock = (obs.get("state") or {}).get("clock") or {}
    try:
        from datetime import date as _date
        a = _date.fromisoformat(str(clock.get("depart_date", "")))
        b = _date.fromisoformat(str(clock.get("return_date", "")))
        return max(1, (b - a).days)
    except (ValueError, TypeError):
        return 1


def _rebook_action(
    obs: dict, cancelled: list[dict], mode: str,
    profile: PersonaProfile | None, rng: np.random.Generator | None,
    last_result: dict, hard_constraints: dict | None = None,
) -> dict:
    """Search for + add a replacement flight on the cancelled route."""
    c = cancelled[0]
    cm = _meta(c)
    src = cm.get("origin")
    dst = cm.get("dest")
    date = (cm.get("depart_iso") or "")[:10] or obs["state"]["clock"].get("depart_date")

    results = _results_matching(last_result, "flight", src, dst)
    if results:
        chosen = _pick(
            results, mode=mode, profile=profile, rng=rng, kind="flight",
            hard_constraints=hard_constraints,
        )
        return {"tool": "add_to_itinerary", "args": {
            "item_id": chosen["id"],
            "slot": c.get("slot", "rebook") + "_rebook",
        }}

    args: dict[str, Any] = {"origin": src, "dest": dst, "depart_date": date}
    hc = hard_constraints or {}
    if hc.get("max_stops") is not None:
        args["max_stops"] = hc["max_stops"]
    return {"tool": "search_flights", "args": args}


# --- Selection: cheapest vs heuristic vs random --------------------------

_CABIN_RANK = {"economy": 0, "premium": 1, "business": 2}


def _pick(
    items: list[dict], *, mode: str,
    profile: PersonaProfile | None, rng: np.random.Generator | None, kind: str,
    hard_constraints: dict | None = None,
) -> dict:
    """Choose one item from a search result list based on mode.

    Hard-constraint filtering runs first so we don't pick options that would
    fail the gate. If nothing survives the filter, fall back to the unfiltered
    set — gate will fail, but at least we return a valid id."""
    pool = _filter_hard(items, kind, hard_constraints) or items

    if mode == "random":
        return pool[int(rng.integers(0, len(pool)))]

    if mode == "cheapest":
        price_key = "price_per_night" if kind == "hotel" else "price"
        return min(pool, key=lambda x: (float(x.get(price_key, 0)), x.get("id", "")))

    # mode == "heuristic"
    if profile is None:
        # Without persona soft prefs, fall back to cheapest among the
        # hard-feasible pool. Better than first item — at least we know it
        # passes the gate.
        price_key = "price_per_night" if kind == "hotel" else "price"
        return min(pool, key=lambda x: (float(x.get(price_key, 0)), x.get("id", "")))
    return max(pool, key=lambda x: _heuristic_score(x, kind, profile))


def _filter_hard(
    items: list[dict], kind: str, hc: dict | None,
) -> list[dict]:
    """Drop options that would fail the hard_constraint_gate. Returns the
    full list unchanged if hc is None/empty."""
    if not hc:
        return list(items)
    out = list(items)
    if kind == "flight":
        if hc.get("no_overnight_flights"):
            out = [x for x in out if not x.get("overnight")]
        if hc.get("max_stops") is not None:
            out = [x for x in out if int(x.get("stops", 0)) <= int(hc["max_stops"])]
    elif kind == "hotel":
        required = set(hc.get("required_amenities") or ())
        if required:
            out = [x for x in out if required.issubset(set(x.get("amenities", [])))]
    return out


def _heuristic_score(item: dict, kind: str, profile: PersonaProfile) -> float:
    """Higher = better. Combines persona fit with mild cost preference."""
    soft = profile.soft_prefs or {}

    if kind == "flight":
        score = 0.0
        if profile.hard.no_overnight_flights and item.get("overnight"):
            score -= 100.0
        if profile.hard.max_stops is not None:
            stops = int(item.get("stops", 0))
            if stops > profile.hard.max_stops:
                score -= 50.0
        # cabin fit
        target_cabin = str(soft.get("flight_cabin", "economy"))
        actual_rank = _CABIN_RANK.get(item.get("cabin", "economy"), 0)
        target_rank = _CABIN_RANK.get(target_cabin, 0)
        score -= 10.0 * abs(actual_rank - target_rank)
        # soft stops preference (above hard.max_stops if any)
        try:
            soft_max = int(soft.get("flight_max_stops", 2))
            over = max(0, int(item.get("stops", 0)) - soft_max)
            score -= 5.0 * over
        except (TypeError, ValueError):
            pass
        # mild cost pressure
        score -= float(item.get("price", 0)) / 200.0
        return score

    if kind == "hotel":
        score = 0.0
        target_stars = soft.get("hotel_stars", 3)
        try:
            tgt = int(target_stars)
        except (TypeError, ValueError):
            tgt = 3
        diff = abs(int(item.get("stars", 0)) - tgt)
        score -= 8.0 * diff
        required = set(profile.hard.required_amenities or ())
        if required and not required.issubset(set(item.get("amenities", []))):
            score -= 50.0
        # mild cost pressure (per night)
        score -= float(item.get("price_per_night", 0)) / 100.0
        return score

    if kind == "activity":
        cat = item.get("category", "")
        weight = float(soft.get(f"act_{cat}", 0.05))
        score = 10.0 * weight
        score -= float(item.get("price", 0)) / 200.0
        return score

    return 0.0


# --- State queries -------------------------------------------------------

def _kind(slot: dict) -> str | None:
    return (slot.get("meta") or {}).get("kind")


def _meta(slot: dict) -> dict:
    return slot.get("meta") or {}


def _has_route(slots: list[dict], src: str | None, dst: str | None) -> bool:
    """True if any flight slot covers src -> dst (booked or tentative)."""
    if src is None or dst is None:
        return False
    for s in slots:
        if _kind(s) != "flight":
            continue
        m = _meta(s)
        if _city_eq(m.get("origin"), src) and _city_eq(m.get("dest"), dst):
            return True
    return False


def _has_replacement_for_any(
    cancelled: list[dict], candidates: list[dict],
) -> bool:
    """True if every cancelled flight has a tentative or booked replacement."""
    for c in cancelled:
        cm = _meta(c)
        has = any(
            _kind(x) == "flight"
            and _city_eq(_meta(x).get("origin"), cm.get("origin"))
            and _city_eq(_meta(x).get("dest"), cm.get("dest"))
            and x.get("item_id") != c.get("item_id")
            for x in candidates
        )
        if not has:
            return False
    return bool(cancelled)


def _results_matching(
    last_result: dict, kind: str, src: str | None, dst: str | None,
) -> list[dict]:
    """Return last_result.results if it's from the right search + correct route."""
    if not last_result.get("ok"):
        # Tool errors / disruption envelopes — drill into tool_result if present
        inner = last_result.get("tool_result")
        if isinstance(inner, dict):
            return _results_matching(inner, kind, src, dst)
        return []

    tool_map = {
        "flight": "search_flights",
        "hotel": "search_hotels",
        "activity": "search_activities",
    }
    if last_result.get("tool") != tool_map.get(kind):
        return []

    results = last_result.get("results") or []
    if not results:
        return []

    if kind == "flight":
        if not (src and dst):
            return []
        sample = results[0]
        if _city_eq(sample.get("origin"), src) and _city_eq(sample.get("dest"), dst):
            return results
        return []

    if kind in ("hotel", "activity"):
        if not dst:
            return []
        sample = results[0]
        if _city_eq(sample.get("city"), dst):
            return results
        return []

    return []


def _wants_activities(profile: PersonaProfile | None) -> bool:
    # No profile: assume the client wants some activities (the typical case).
    # With profile: only if at least one category has nontrivial weight.
    if profile is None:
        return True
    soft = profile.soft_prefs or {}
    return any(
        k.startswith("act_") and float(v) > 0.10
        for k, v in soft.items() if isinstance(v, (int, float))
    )


def _top_activity_categories(soft: dict, *, k: int = 2) -> list[str]:
    pairs = []
    for axis, weight in soft.items():
        if isinstance(axis, str) and axis.startswith("act_"):
            try:
                pairs.append((axis[4:], float(weight)))
            except (TypeError, ValueError):
                pass
    pairs.sort(key=lambda x: -x[1])
    return [c for c, _ in pairs[:k]]


def _city_eq(actual: Any, expected: Any) -> bool:
    """Loose city match — works for either bare city names or IATA codes."""
    if actual is None or expected is None:
        return False
    if actual == expected:
        return True
    a_str = str(actual)
    e_str = str(expected)
    if a_str.upper() == e_str.upper():
        return True
    if a_str in _IATA_TO_NAME and _IATA_TO_NAME[a_str] == e_str:
        return True
    if e_str in _IATA_TO_NAME and _IATA_TO_NAME[e_str] == a_str:
        return True
    return False


# --- Random args helpers --------------------------------------------------

def _random_flight_args(
    obs: dict, profile: PersonaProfile | None, rng: np.random.Generator,
) -> dict:
    clock = (obs.get("state") or {}).get("clock") or {}
    origin = (profile.origin_city if profile else None) or clock.get("origin") or "JFK"
    dest = (profile.dest_city if profile else None) or clock.get("dest") or "NRT"
    # Half the time, search the return route — gives random some chance of valid trips.
    if rng.random() < 0.5:
        origin, dest = dest, origin
        date = clock.get("return_date") or "2026-04-15"
    else:
        date = clock.get("depart_date") or "2026-04-10"
    return {"origin": origin, "dest": dest, "depart_date": date}


def _random_hotel_args(
    obs: dict, profile: PersonaProfile | None, rng: np.random.Generator,
) -> dict:
    clock = (obs.get("state") or {}).get("clock") or {}
    dest = (profile.dest_city if profile else None) or clock.get("dest") or "Tokyo"
    return {
        "city": dest,
        "checkin": clock.get("depart_date") or "2026-04-10",
        "checkout": clock.get("return_date") or "2026-04-15",
    }


def _random_activity_args(
    obs: dict, profile: PersonaProfile | None, rng: np.random.Generator,
) -> dict:
    clock = (obs.get("state") or {}).get("clock") or {}
    dest = (profile.dest_city if profile else None) or clock.get("dest") or "Tokyo"
    return {"city": dest}
