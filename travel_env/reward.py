"""Reward function: rule-based components + composition + optional LLM judge.

This is the single most consequential file in the project — the agent's
behavior IS the reward gradient. The plan and notes.md walk through the
exploit-defense thinking in detail; this docstring is a short tour.

## Anatomy of the reward

    total = hard_constraint_gate × Σᵢ (wᵢ · component_i)  -  step_penalty

  components (all bounded [0, 1] except step_penalty):
    hard_constraint_gate   {0, 1}     multiplicative — zeros the soft sum
    preference_coverage    [0, 1]     weighted score across persona.soft_prefs
    budget_efficiency      [0, 1]     concave bell over spent/cap, tolerance-aware
    coherence              [0, 1]     fraction of structural checks that pass
    recovery_quality       [0, 1]|None  conditional on a disruption firing
    step_penalty           ≥ 0        free for the first 5 turns, linear after

  If no disruption fired, recovery is None and the remaining weights are
  renormalized so the reward scale is preserved across disruption/no-disruption
  episodes.

## What we're NOT doing (and why)

- We do not blend the LLM judge into the training reward. It's logged
  separately in eval as a diagnostic, and we report corr(rule_reward, judge)
  to assess whether the rule-based proxy actually correlates with what a
  human-ish evaluator would say. Blending is bad: it's stochastic, expensive
  per rollout, and partially trusts a signal we can't audit.
- We do not give per-step rewards. There is no meaningful "good third turn"
  in travel planning — the trip is judged as a whole, scored on submit_final
  (or truncation). Sparse but principled.

## Exploit defenses, in plain terms

  "always pick the cheapest flight"
    → cheapest flights are typically overnight + multi-stop + economy.
      Many personas (luxury, business, family) set hard.no_overnight_flights
      and hard.max_stops; those flights fail the gate.
    → For personas without those hard prefs, preference_coverage
      penalizes mismatches on flight_cabin / flight_max_stops anyway.

  "always pick the highest-rated hotel"
    → 5-star Tokyo hotels for 5 nights × 2 people exceed most budget caps;
      the book tool refuses (env.py-side), so it never reaches reward.
    → If it does fit, budget_efficiency's upper-bound penalty (quadratic
      decay above the midpoint, reaching 0 at the hard cap) cuts the reward.

  "finish in one step"
    → Reward is only emitted at terminal (submit or truncation). With nothing
      booked, hard_constraint_gate = 0, multiplied through everything.

  "ignore disruptions"
    → recovery_quality has a weight comparable to budget; failing it costs
      ~25% of soft reward AND fails coherence (now missing a leg) AND fails
      the gate (incomplete itinerary). Triple penalty.

  "spam search to inflate confidence"
    → search idempotence in world.py means re-querying gives identical
      results; step_penalty grows the longer you dither.

  "submit with empty itinerary"
    → gate=0, reward=0 minus step penalty. Net negative.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from travel_env.persona import PersonaProfile
from travel_env.world import CITIES


# --- Weights -------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    """Per-component weights in the composition. Logged with every run.

    The four weights on the soft sum nominally add to 1.0 so the gated value
    stays in [0, 1]. step_penalty is in the same units as the other components
    so 0.005 × 20 turns ≈ 0.075 — meaningful but not dominant.
    """
    preference: float = 0.35
    budget: float = 0.20
    coherence: float = 0.20
    recovery: float = 0.25
    step_penalty: float = 0.005
    step_penalty_free_turns: int = 5


DEFAULT_WEIGHTS = RewardWeights()


# Numeric ranking for flight cabin (used by preference scoring & mismatch).
CABIN_RANK: dict[str, int] = {"economy": 0, "premium": 1, "business": 2}

# Activity categories the world generates and persona weights with `act_<cat>`.
ACTIVITY_CATEGORIES: tuple[str, ...] = ("food", "history", "nature", "nightlife", "family")

# Defaults that match persona.py tolerances if a profile omits them.
DEFAULT_TOLERANCES: dict[str, float] = {
    "hotel_stars": 1.0,
    "act_category_gap": 0.15,
    "flight_cabin_levels": 1.0,
    "budget_low_ratio": 0.55,
    "budget_high_ratio": 0.95,
}


# --- Result type ---------------------------------------------------------

@dataclass
class RewardBreakdown:
    hard_constraint_gate: float
    preference_coverage: float
    budget_efficiency: float
    coherence: float
    recovery_quality: float | None
    steps: int
    total: float
    weights: RewardWeights = DEFAULT_WEIGHTS
    extras: dict = field(default_factory=dict)


# --- Components ----------------------------------------------------------

def hard_constraint_gate(itinerary: list[dict], profile: PersonaProfile, spent: float) -> float:
    """1.0 iff every hard constraint passes; 0.0 otherwise.

    Hard constraints (any failure zeros the gate):
      - budget honored: spent ≤ budget_cap
      - structural: ≥1 booked hotel, ≥1 outbound flight, ≥1 return flight
      - route: outbound = origin→dest, return = dest→origin (any direction OK)
      - no_overnight_flights: no booked flight has overnight=True
      - max_stops: every booked flight's stops ≤ max_stops
      - required_amenities: every booked hotel has all required amenities
    """
    if spent > profile.hard.budget_cap:
        return 0.0

    booked = [s for s in itinerary if s.get("status") == "booked"]
    flights = [s for s in booked if _kind(s) == "flight"]
    hotels = [s for s in booked if _kind(s) == "hotel"]

    if not hotels or len(flights) < 2:
        return 0.0

    origin, dest = profile.origin_city, profile.dest_city
    has_outbound = any(
        _city_eq(_meta(f).get("origin"), origin) and _city_eq(_meta(f).get("dest"), dest)
        for f in flights
    )
    has_return = any(
        _city_eq(_meta(f).get("origin"), dest) and _city_eq(_meta(f).get("dest"), origin)
        for f in flights
    )
    if not (has_outbound and has_return):
        return 0.0

    if profile.hard.no_overnight_flights:
        if any(_meta(f).get("overnight") for f in flights):
            return 0.0

    if profile.hard.max_stops is not None:
        if any(int(_meta(f).get("stops", 0)) > profile.hard.max_stops for f in flights):
            return 0.0

    if profile.hard.required_amenities:
        required = set(profile.hard.required_amenities)
        for h in hotels:
            ams = set(_meta(h).get("amenities", []))
            if not required.issubset(ams):
                return 0.0

    return 1.0


def preference_coverage(itinerary: list[dict], profile: PersonaProfile) -> float:
    """Weighted score across persona soft prefs, normalized to [0, 1].

    Axes scored:
      act_<category>     activity-category coverage (count→[0,1] linear up to 2)
      hotel_stars        proximity to target int, tolerance-aware
      flight_cabin       proximity to target cabin level, tolerance-aware
      flight_max_stops   1.0 if all booked flights ≤ target, linear decay above

    A pref the persona didn't set contributes nothing. With no prefs at all,
    returns 0.5 (neutral) — the agent isn't punished for the persona being
    indifferent.
    """
    booked = [s for s in itinerary if s.get("status") == "booked"]
    activities = [s for s in booked if _kind(s) == "activity"]
    hotels = [s for s in booked if _kind(s) == "hotel"]
    flights = [s for s in booked if _kind(s) == "flight"]

    soft = profile.soft_prefs or {}
    tol = {**DEFAULT_TOLERANCES, **(profile.tolerances or {})}

    # Each axis contributes (score, weight) to the weighted mean.
    contributions: list[tuple[float, float]] = []

    # Activity categories — driven by act_<cat> weights the persona ships.
    for axis, raw_weight in soft.items():
        if not isinstance(axis, str) or not axis.startswith("act_"):
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        category = axis[len("act_"):]
        count = sum(1 for a in activities if _meta(a).get("category") == category)
        # Linear up to 2 of-category activities; further don't help. Two beats one,
        # two beats four (anti-greedy on the easiest pref).
        score = min(count / 2.0, 1.0)
        contributions.append((score, weight))

    # Hotel stars: target int, scored by distance with tolerance.
    if "hotel_stars" in soft and hotels:
        try:
            target = int(soft["hotel_stars"])
            tol_stars = float(tol.get("hotel_stars", 1.0))
            per_hotel = []
            for h in hotels:
                actual = int(_meta(h).get("stars", 0))
                diff = abs(actual - target)
                # Within tolerance: 1.0. Outside: linear decay over an extra (tol+1) range.
                per_hotel.append(max(0.0, 1.0 - max(0, diff - tol_stars) / max(tol_stars + 1.0, 1.0)))
            contributions.append((sum(per_hotel) / len(per_hotel), 0.5))
        except (TypeError, ValueError):
            pass

    # Flight cabin: convert to numeric ranks, score per booked flight.
    if "flight_cabin" in soft and flights:
        target_rank = CABIN_RANK.get(str(soft["flight_cabin"]), 1)
        tol_levels = float(tol.get("flight_cabin_levels", 1.0))
        per_flight = []
        for f in flights:
            actual_rank = CABIN_RANK.get(_meta(f).get("cabin", "economy"), 0)
            diff = abs(actual_rank - target_rank)
            per_flight.append(max(0.0, 1.0 - max(0, diff - tol_levels) / max(tol_levels + 1.0, 1.0)))
        contributions.append((sum(per_flight) / len(per_flight), 0.5))

    # Flight max stops (soft side; the hard side is enforced in the gate).
    if "flight_max_stops" in soft and flights:
        try:
            target = int(soft["flight_max_stops"])
            per_flight = []
            for f in flights:
                actual = int(_meta(f).get("stops", 0))
                if actual <= target:
                    per_flight.append(1.0)
                else:
                    # Each extra stop above the soft target costs 0.5.
                    per_flight.append(max(0.0, 1.0 - 0.5 * (actual - target)))
            contributions.append((sum(per_flight) / len(per_flight), 0.5))
        except (TypeError, ValueError):
            pass

    if not contributions:
        return 0.5

    total_weight = sum(w for _, w in contributions)
    if total_weight <= 0:
        return 0.5
    weighted = sum(s * w for s, w in contributions) / total_weight
    return max(0.0, min(weighted, 1.0))


def budget_efficiency(
    spent: float,
    cap: float,
    *,
    low_ratio: float = 0.55,
    high_ratio: float = 0.95,
) -> float:
    """Asymmetric bell: linear ramp up to midpoint, quadratic decay to 0 at the hard cap.

    The old symmetric bell hit zero at both `low_ratio` and `high_ratio`, which
    meant "spend almost nothing" and "spend near the cap" looked identical to
    the agent. Combined with the hard gate's cliff at ratio=1.0, that made
    undershooting strictly safer than aggressive spending — which is why
    "always pick the cheapest" was outscoring smarter policies. The asymmetry
    here flips that: cheapness now pays a continuous price (linear ramp from 0),
    while spending up toward the persona's natural ratio is the gradient
    direction (gentle quadratic decay only reaches 0 at the hard cap).

    Tolerances are persona-supplied (`budget_low_ratio`, `budget_high_ratio`)
    via DEFAULT_TOLERANCES. Their role is unchanged: the midpoint is still
    `(low_ratio + high_ratio) / 2`, so per-archetype tuning in
    `persona.ARCHETYPE_BUDGET_BANDS` keeps working without edits.

    Shape:
      - ratio = 0:        score = 0    (spent nothing; gate is failing anyway)
      - ratio = midpoint: score = 1.0  (sweet spot, unchanged from old bell)
      - ratio = 1.0:      score = 0    (hand off to the hard gate)
    """
    if cap <= 0:
        return 0.0
    ratio = spent / cap
    midpoint = (low_ratio + high_ratio) / 2.0
    # Degenerate tolerances (midpoint at or past the bounds): keep the old
    # exact-match behavior so callers don't get NaNs from div-by-zero below.
    if midpoint <= 0.0 or midpoint >= 1.0:
        return 1.0 if abs(ratio - midpoint) < 1e-6 else 0.0
    if ratio <= midpoint:
        # Below the sweet spot: linear ramp from 0 (spent nothing) to 1.0 (midpoint).
        return max(0.0, ratio / midpoint)
    # Above the sweet spot: quadratic decay from 1.0 (midpoint) to 0 (hard cap).
    distance = (ratio - midpoint) / (1.0 - midpoint)
    return max(0.0, 1.0 - distance ** 2)


def coherence(itinerary: list[dict], profile: PersonaProfile) -> float:
    """Fraction of structural checks that pass. Partial credit, not 0/1.

    Five checks:
      1. ≥1 outbound flight on the right route (origin → dest)
      2. ≥1 return flight (dest → origin)
      3. All booked hotels are in the destination city
      4. All booked activities are in the destination city (vacuous if none)
      5. Date alignment: outbound depart_date == profile.depart_date and
         return depart_date == profile.return_date
    """
    booked = [s for s in itinerary if s.get("status") == "booked"]
    flights = [s for s in booked if _kind(s) == "flight"]
    hotels = [s for s in booked if _kind(s) == "hotel"]
    activities = [s for s in booked if _kind(s) == "activity"]

    if not (flights or hotels or activities):
        return 0.0

    origin, dest = profile.origin_city, profile.dest_city
    checks: list[bool] = []

    outbound = next(
        (f for f in flights
         if _city_eq(_meta(f).get("origin"), origin) and _city_eq(_meta(f).get("dest"), dest)),
        None,
    )
    return_flight = next(
        (f for f in flights
         if _city_eq(_meta(f).get("origin"), dest) and _city_eq(_meta(f).get("dest"), origin)),
        None,
    )
    checks.append(outbound is not None)
    checks.append(return_flight is not None)

    if hotels:
        checks.append(all(_city_eq(_meta(h).get("city"), dest) for h in hotels))
    else:
        checks.append(False)

    if activities:
        checks.append(all(_city_eq(_meta(a).get("city"), dest) for a in activities))
    else:
        checks.append(True)

    date_ok = True
    if outbound and _iso_date(_meta(outbound).get("depart_iso")) != profile.hard.depart_date:
        date_ok = False
    if return_flight and _iso_date(_meta(return_flight).get("depart_iso")) != profile.hard.return_date:
        date_ok = False
    # If we don't have any matched flight to check against, leave date_ok True
    # so the missing-flight failure is captured by checks 1 and 2 above, not
    # double-counted here.
    checks.append(date_ok)

    return sum(checks) / len(checks)


def recovery_quality(disruption_info: dict | None) -> float | None:
    """None if no disruption fired; else [0, 1].

    Scored over three sub-components:
      price_efficiency (0.4): how close to the original flight price did the
        replacement come? Cheaper-than-original = full credit.
      time_efficiency  (0.4): how quickly did the agent rebook after the event?
        1 turn = 1.0, decays linearly to 0 over 8 turns.
      downstream_preservation (0.2): did non-flight bookings (hotel, activities)
        survive the recovery? Removing them mid-recovery is usually a mistake.

    If the agent never rebooked the cancelled route, score = 0.
    """
    if not disruption_info:
        return None

    pre = disruption_info.get("pre_itinerary", [])
    post = disruption_info.get("post_itinerary", [])
    fired_turn = int(disruption_info.get("fired_at_turn", 0))
    final_turn = int(disruption_info.get("final_turn", fired_turn))
    original_price = float(disruption_info.get("original_flight_price", 0.0))

    # Routes that were cancelled.
    cancelled_routes: set[tuple[str | None, str | None]] = set()
    cancelled_ids: set[str] = set()
    for s in post:
        if s.get("status") == "cancelled" and _kind(s) == "flight":
            m = _meta(s)
            cancelled_routes.add((m.get("origin"), m.get("dest")))
            cancelled_ids.add(s["item_id"])

    if not cancelled_routes:
        # No flight actually got cancelled (maybe the agent removed it before fire).
        # That counts as "no disruption to recover from" — neutral.
        return 0.5

    # New flights booked post-disruption with a matching route.
    pre_ids = {s["item_id"] for s in pre}
    replacements = [
        s for s in post
        if s.get("status") == "booked"
        and _kind(s) == "flight"
        and s["item_id"] not in pre_ids
        and (_meta(s).get("origin"), _meta(s).get("dest")) in cancelled_routes
    ]

    if not replacements:
        return 0.0

    # Price efficiency: cheaper-than-original is fine; pay-more is penalized
    # linearly up to 100% premium (where score reaches 0).
    new_price = sum(float(s["price"]) for s in replacements)
    if original_price > 0:
        delta_ratio = max(0.0, (new_price - original_price) / original_price)
        price_score = max(0.0, 1.0 - min(delta_ratio, 1.0))
    else:
        price_score = 1.0

    # Time efficiency: 1 turn = 1.0, 9+ turns = 0.0. The agent's `book` call
    # for the replacement is what closes the loop.
    turns_to_rebook = max(1, final_turn - fired_turn)
    time_score = max(0.0, 1.0 - (turns_to_rebook - 1) / 8.0)

    # Downstream preservation: non-flight booked items that existed pre-disruption
    # should still be booked post. Removing hotels/activities in a panic is bad.
    pre_non_flight = {
        s["item_id"] for s in pre
        if s.get("status") == "booked" and _kind(s) != "flight"
    }
    post_non_flight_booked = {
        s["item_id"] for s in post
        if s.get("status") == "booked" and _kind(s) != "flight"
    }
    if pre_non_flight:
        preservation = len(pre_non_flight & post_non_flight_booked) / len(pre_non_flight)
    else:
        preservation = 1.0

    return 0.4 * price_score + 0.4 * time_score + 0.2 * preservation


# --- Composition ---------------------------------------------------------

def compose(
    *,
    gate: float,
    pref: float,
    budget: float,
    coh: float,
    recovery: float | None,
    steps: int,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict]:
    """Combine components into a scalar reward + a small extras dict."""
    if recovery is None:
        # Renormalize the three remaining weights so the soft sum still spans [0, 1].
        wp, wb, wc = weights.preference, weights.budget, weights.coherence
        total_w = wp + wb + wc
        if total_w <= 0:
            soft_sum = 0.0
        else:
            soft_sum = (wp * pref + wb * budget + wc * coh) / total_w
        extras = {"renormalized_no_disruption": True}
    else:
        soft_sum = (
            weights.preference * pref
            + weights.budget * budget
            + weights.coherence * coh
            + weights.recovery * recovery
        )
        extras = {"renormalized_no_disruption": False}

    gated = gate * soft_sum
    penalty = weights.step_penalty * max(0, steps - weights.step_penalty_free_turns)
    total = gated - penalty
    extras["step_penalty"] = penalty
    return total, extras


def score_episode(
    *,
    itinerary: list[dict],
    profile: PersonaProfile,
    spent: float,
    steps: int,
    disruption_info: dict | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown:
    """Top-level entry point. Called by TravelEnv on submit_final or truncation."""
    tol = {**DEFAULT_TOLERANCES, **(profile.tolerances or {})}
    low_ratio = float(tol.get("budget_low_ratio", 0.55))
    high_ratio = float(tol.get("budget_high_ratio", 0.95))

    gate = hard_constraint_gate(itinerary, profile, spent)
    pref = preference_coverage(itinerary, profile)
    budg = budget_efficiency(spent, profile.hard.budget_cap, low_ratio=low_ratio, high_ratio=high_ratio)
    coh = coherence(itinerary, profile)
    rec = recovery_quality(disruption_info)

    total, extras = compose(
        gate=gate, pref=pref, budget=budg, coh=coh, recovery=rec,
        steps=steps, weights=weights,
    )

    extras.update({
        "budget_ratio": (spent / profile.hard.budget_cap) if profile.hard.budget_cap > 0 else 0.0,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
    })

    return RewardBreakdown(
        hard_constraint_gate=gate,
        preference_coverage=pref,
        budget_efficiency=budg,
        coherence=coh,
        recovery_quality=rec,
        steps=steps,
        total=total,
        weights=weights,
        extras=extras,
    )


# --- Optional LLM judge (diagnostic — never sums into training reward) ---

def llm_judge_score(
    request_text: str,
    itinerary: list[dict],
    *,
    model: str = "anthropic/claude-sonnet-4.5",
) -> float:
    """Use OpenRouter to ask an LLM to score the trip in [0, 1] given the request.

    Falls back to 0.5 (neutral) on any failure: missing API key, network error,
    parse error, malformed model output. This is a diagnostic signal in eval —
    we never want it to crash a training run.

    The judge sees ONLY (request_text, final itinerary). It does NOT see the
    structured profile. That's intentional: this is meant to approximate a
    human who only reads the request and inspects the result, the same way
    the rule-based reward judges it under different rules.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return 0.5

    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        items_text = "\n".join(
            f"- {s.get('slot', '?')}: {s.get('name', '?')} "
            f"(${float(s.get('price', 0)):.0f}, {s.get('status', '?')})"
            for s in itinerary
        ) or "(empty itinerary)"

        prompt = (
            "You are evaluating a travel agent's final itinerary against a client's "
            "request. Score from 0.0 to 1.0 based on how well the itinerary matches "
            "what the client would actually want — preference alignment, value, "
            "coherence, overall trip quality.\n\n"
            f"Client request:\n{request_text}\n\n"
            f"Final itinerary:\n{items_text}\n\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0. No other text."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d*\.?\d+", text)
        if not m:
            return 0.5
        val = float(m.group())
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5


# --- Internal helpers ----------------------------------------------------

# Build a small two-way map between IATA codes and canonical city names so
# coherence can match flights tagged with airport codes against personas
# carrying city names (and vice versa). Cheap to import — CITIES is already
# loaded by env.py / persona.py too.
_IATA_TO_CITY: dict[str, str] = {c.iata: name for name, c in CITIES.items()}


def _city_eq(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return False
    if actual == expected:
        return True
    a_str = str(actual)
    e_str = str(expected)
    if a_str.upper() == e_str.upper():
        return True
    if a_str in _IATA_TO_CITY and _IATA_TO_CITY[a_str] == e_str:
        return True
    if e_str in _IATA_TO_CITY and _IATA_TO_CITY[e_str] == a_str:
        return True
    return False


def _kind(slot: dict) -> str | None:
    meta = slot.get("meta") or {}
    return meta.get("kind")


def _meta(slot: dict) -> dict:
    return slot.get("meta") or {}


def _iso_date(iso: Any) -> str:
    """Coerce ISO datetime to date prefix YYYY-MM-DD; empty string on failure."""
    if not iso:
        return ""
    return str(iso)[:10]
