"""Persona: structured profile + LLM voice (request, feedback) with disk cache.

Two artifacts at episode reset:
  1. PersonaProfile  - deterministic, sampled from `seed`. Reward reads this.
  2. Request text    - LLM-voiced (or scripted) version. Agent reads this.

During the episode, on `propose_to_client` the env deterministically computes
the list of mismatches between the proposed itinerary and the profile; the
LLM (or scripted template) then voices those complaints in character. The
LLM never invents new complaints — that would break the training signal.

LLM calls go through OpenRouter via the openai SDK. Results are cached on
disk under .persona_cache/ keyed by sha256(method, model, payload). First
run pays the API cost; subsequent runs are deterministic and free.

If OPENROUTER_API_KEY is unset, the voice silently falls back to scripted
templates. This keeps `eval.py --persona-mode scripted` runnable without
external dependencies.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from travel_env.world_data import CITIES


Archetype = Literal["budget", "luxury", "foodie", "family", "history_buff", "business"]
CommStyle = Literal["terse", "chatty", "vague", "precise", "formal"]


# --- Profile -------------------------------------------------------------

@dataclass
class HardConstraints:
    budget_cap: float
    depart_date: str         # ISO date
    return_date: str
    group_size: int
    no_overnight_flights: bool = False
    max_stops: int | None = None
    required_amenities: tuple[str, ...] = ()


@dataclass
class PersonaProfile:
    seed: int
    archetype: Archetype
    origin_city: str
    dest_city: str
    hard: HardConstraints
    # Mixed-type bag. Activity weights live as act_<category> -> float in [0, 1].
    # Other entries: hotel_stars: int, flight_cabin: str, flight_max_stops: int.
    soft_prefs: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    communication_style: CommStyle = "chatty"


# --- Archetype table -----------------------------------------------------
#
# Each archetype is a coherent bundle: budget travelers tolerate stops and
# red-eyes, business travelers don't, luxury travelers want spa amenities,
# etc. Coherence here is the whole point: random profiles produce a flat
# agent. These bundles are what give reward shaping something to bite on.

@dataclass(frozen=True)
class ArchetypeSpec:
    act_weights: dict[str, float]            # raw category weights; jittered + renormalized at sample time
    hotel_stars: int                          # target star rating
    flight_cabin: str                         # preferred cabin
    flight_max_stops: int                     # preferred max; promoted to hard constraint if <= 1
    no_overnight: bool                        # hard constraint
    required_amenities: tuple[str, ...]       # hard constraint
    nightly_per_person: tuple[float, float]   # $ per night per person at cost_multiplier=1.0
    flight_per_person: tuple[float, float]    # $ per person round-trip
    group_size: tuple[int, int]               # inclusive
    comm_styles: tuple[CommStyle, ...]


# Per-archetype budget tolerance bands (low, high) for the concave budget
# bell in reward.py. A budget traveler expects to spend ~half the cap; a
# luxury traveler expects to spend most of it. Hardcoded defaults (0.55,
# 0.95) systematically under-rewarded budget personas because their actual
# spend ratio was below the lower bound.
ARCHETYPE_BUDGET_BANDS: dict[str, tuple[float, float]] = {
    "budget":       (0.40, 0.80),
    "luxury":       (0.80, 0.98),
    "foodie":       (0.55, 0.92),
    "family":       (0.60, 0.95),
    "history_buff": (0.50, 0.90),
    "business":     (0.70, 0.98),
}


ARCHETYPES: dict[Archetype, ArchetypeSpec] = {
    "budget": ArchetypeSpec(
        act_weights={"history": 0.35, "nature": 0.30, "food": 0.20, "family": 0.10, "nightlife": 0.05},
        hotel_stars=2, flight_cabin="economy", flight_max_stops=2,
        no_overnight=False, required_amenities=(),
        nightly_per_person=(60.0, 110.0), flight_per_person=(300.0, 650.0),
        group_size=(1, 2), comm_styles=("chatty", "vague"),
    ),
    "luxury": ArchetypeSpec(
        act_weights={"food": 0.35, "history": 0.25, "nature": 0.20, "nightlife": 0.15, "family": 0.05},
        hotel_stars=5, flight_cabin="business", flight_max_stops=0,
        no_overnight=True, required_amenities=("spa",),
        nightly_per_person=(500.0, 900.0), flight_per_person=(3000.0, 6000.0),
        group_size=(1, 2), comm_styles=("precise", "formal"),
    ),
    "foodie": ArchetypeSpec(
        act_weights={"food": 0.55, "nightlife": 0.20, "history": 0.15, "nature": 0.10, "family": 0.0},
        hotel_stars=3, flight_cabin="economy", flight_max_stops=1,
        no_overnight=False, required_amenities=(),
        nightly_per_person=(160.0, 280.0), flight_per_person=(500.0, 1000.0),
        group_size=(1, 2), comm_styles=("chatty", "precise"),
    ),
    "family": ArchetypeSpec(
        act_weights={"family": 0.40, "nature": 0.25, "history": 0.15, "food": 0.15, "nightlife": 0.05},
        hotel_stars=4, flight_cabin="premium", flight_max_stops=1,
        no_overnight=True, required_amenities=("breakfast", "wifi"),
        nightly_per_person=(180.0, 320.0), flight_per_person=(400.0, 900.0),
        group_size=(3, 5), comm_styles=("precise", "chatty"),
    ),
    "history_buff": ArchetypeSpec(
        act_weights={"history": 0.55, "food": 0.20, "nature": 0.15, "family": 0.05, "nightlife": 0.05},
        hotel_stars=3, flight_cabin="economy", flight_max_stops=2,
        no_overnight=False, required_amenities=(),
        nightly_per_person=(120.0, 200.0), flight_per_person=(400.0, 850.0),
        group_size=(1, 2), comm_styles=("precise", "chatty"),
    ),
    "business": ArchetypeSpec(
        act_weights={"food": 0.40, "nature": 0.30, "history": 0.20, "nightlife": 0.10, "family": 0.0},
        hotel_stars=4, flight_cabin="business", flight_max_stops=0,
        no_overnight=True, required_amenities=("wifi", "gym"),
        nightly_per_person=(300.0, 480.0), flight_per_person=(900.0, 1500.0),
        group_size=(1, 1), comm_styles=("terse", "precise"),
    ),
}


# --- Profile sampling ----------------------------------------------------

_TRIP_ANCHOR = dt.date(2026, 3, 1)


def sample_profile(seed: int) -> PersonaProfile:
    """Deterministic profile sampling. Same seed -> identical profile."""
    rng = np.random.default_rng(seed)
    archetype: Archetype = str(rng.choice(list(ARCHETYPES.keys())))  # type: ignore[assignment]
    spec = ARCHETYPES[archetype]

    city_names = list(CITIES.keys())
    origin = str(rng.choice(city_names))
    dest = str(rng.choice([c for c in city_names if c != origin]))

    advance_days = int(rng.integers(7, 91))
    nights = int(rng.integers(3, 11))
    depart = _TRIP_ANCHOR + dt.timedelta(days=advance_days)
    return_d = depart + dt.timedelta(days=nights)

    g_lo, g_hi = spec.group_size
    group_size = int(rng.integers(g_lo, g_hi + 1))

    dest_mult = CITIES[dest].cost_multiplier
    nightly = float(rng.uniform(*spec.nightly_per_person)) * dest_mult
    flight = float(rng.uniform(*spec.flight_per_person))
    budget_cap = round((nightly * group_size * nights + flight * group_size) * 1.10, -1)

    # Jitter activity weights and renormalize.
    act = {k: max(0.0, v + float(rng.uniform(-0.05, 0.05))) for k, v in spec.act_weights.items()}
    total = sum(act.values()) or 1.0
    act = {k: v / total for k, v in act.items()}

    soft_prefs: dict[str, Any] = {f"act_{k}": v for k, v in act.items()}
    soft_prefs["hotel_stars"] = spec.hotel_stars
    soft_prefs["flight_cabin"] = spec.flight_cabin
    soft_prefs["flight_max_stops"] = spec.flight_max_stops

    budget_low, budget_high = ARCHETYPE_BUDGET_BANDS.get(archetype, (0.55, 0.95))
    tolerances = {
        "hotel_stars": 1.0,
        "act_category_gap": 0.15,
        "flight_cabin_levels": 1.0,
        "budget_low_ratio": budget_low,
        "budget_high_ratio": budget_high,
    }

    comm_style: CommStyle = str(rng.choice(list(spec.comm_styles)))  # type: ignore[assignment]

    # Promote tight stop preferences to a hard constraint. <=1 is "I really
    # mean it"; >=2 is just a soft preference.
    hard_max_stops: int | None = spec.flight_max_stops if spec.flight_max_stops <= 1 else None

    return PersonaProfile(
        seed=seed,
        archetype=archetype,
        origin_city=origin,
        dest_city=dest,
        hard=HardConstraints(
            budget_cap=budget_cap,
            depart_date=depart.isoformat(),
            return_date=return_d.isoformat(),
            group_size=group_size,
            no_overnight_flights=spec.no_overnight,
            max_stops=hard_max_stops,
            required_amenities=spec.required_amenities,
        ),
        soft_prefs=soft_prefs,
        tolerances=tolerances,
        communication_style=comm_style,
    )


# --- Mismatch detection (deterministic, no LLM) --------------------------

@dataclass
class Mismatch:
    axis: str         # e.g. "hotel_stars_too_low", "activity_under_food"
    severity: float   # [0, 1]; used for sorting and LLM emphasis
    detail: dict      # structured info the voicer can drop into a sentence


_CABIN_RANK = {"economy": 0, "premium": 1, "business": 2}


def compute_mismatches(itinerary: list[dict], profile: PersonaProfile) -> list[Mismatch]:
    """Inspect the proposed itinerary against the profile; return structured complaints.

    Each itinerary item is a dict with at minimum:
      slot:    str
      item_id: str
      price:   float
      status:  "tentative" | "booked" | "cancelled"
      meta:    dict — populated by env from world inventory. Expected fields:
                 kind:     "flight" | "hotel" | "activity"
                 flights:  stops, cabin, overnight
                 hotels:   stars, amenities (iterable of strings)
                 acts:     category
    """
    active = [s for s in itinerary if s.get("status") != "cancelled"]
    flights = [s for s in active if s.get("meta", {}).get("kind") == "flight"]
    hotels = [s for s in active if s.get("meta", {}).get("kind") == "hotel"]
    activities = [s for s in active if s.get("meta", {}).get("kind") == "activity"]

    out: list[Mismatch] = []

    # Hard constraints — voiced as well as gated. A real client tells you
    # "no, I said no overnight flights"; they don't silently zero your reward.
    if profile.hard.no_overnight_flights:
        for f in flights:
            if f.get("meta", {}).get("overnight"):
                out.append(Mismatch(
                    axis="flight_overnight",
                    severity=1.0,
                    detail={"slot": f["slot"], "item_id": f["item_id"]},
                ))

    if profile.hard.max_stops is not None:
        for f in flights:
            stops = int(f.get("meta", {}).get("stops", 0))
            if stops > profile.hard.max_stops:
                out.append(Mismatch(
                    axis="flight_stops_exceeded",
                    severity=0.9,
                    detail={"slot": f["slot"], "stops": stops, "max": profile.hard.max_stops},
                ))

    # Hotel: stars vs target, required amenities.
    target_stars = profile.soft_prefs.get("hotel_stars")
    star_tol = profile.tolerances.get("hotel_stars", 1.0)
    for h in hotels:
        meta = h.get("meta", {})
        stars = meta.get("stars")
        if target_stars is not None and stars is not None and stars < target_stars - star_tol:
            out.append(Mismatch(
                axis="hotel_stars_too_low",
                severity=min(1.0, (target_stars - stars) / 5.0),
                detail={"actual": int(stars), "target": int(target_stars)},
            ))
        amenities = set(meta.get("amenities", ()))
        missing = [a for a in profile.hard.required_amenities if a not in amenities]
        if missing:
            out.append(Mismatch(
                axis="hotel_amenities_missing",
                severity=0.8,
                detail={"missing": missing, "slot": h["slot"]},
            ))

    # Activity category coverage. Underweight = "not enough X" for categories
    # the persona cares about. Overweight = "too much Y" for categories the
    # persona avoids (e.g. family persona getting nightlife).
    if activities:
        counts: dict[str, int] = {}
        for a in activities:
            cat = a.get("meta", {}).get("category")
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        total = sum(counts.values()) or 1
        actual_share = {k: v / total for k, v in counts.items()}
        gap_tol = profile.tolerances.get("act_category_gap", 0.15)
        for k, target in profile.soft_prefs.items():
            if not k.startswith("act_") or not isinstance(target, (int, float)):
                continue
            cat = k[4:]
            actual = actual_share.get(cat, 0.0)
            if target - actual > gap_tol:
                out.append(Mismatch(
                    axis=f"activity_under_{cat}",
                    severity=min(1.0, target - actual),
                    detail={"category": cat, "target": round(float(target), 2), "actual": round(actual, 2)},
                ))
            if target < 0.1 and actual > 0.25:
                out.append(Mismatch(
                    axis=f"activity_over_{cat}",
                    severity=min(1.0, actual - target),
                    detail={"category": cat, "target": round(float(target), 2), "actual": round(actual, 2)},
                ))

    # Flight cabin vs preference.
    pref_cabin = profile.soft_prefs.get("flight_cabin", "economy")
    pref_rank = _CABIN_RANK.get(pref_cabin, 0)
    cabin_tol = profile.tolerances.get("flight_cabin_levels", 1.0)
    for f in flights:
        actual_cabin = f.get("meta", {}).get("cabin", "economy")
        actual_rank = _CABIN_RANK.get(actual_cabin, 0)
        if pref_rank - actual_rank > cabin_tol:
            out.append(Mismatch(
                axis="flight_cabin_too_low",
                severity=min(1.0, (pref_rank - actual_rank) / 3.0),
                detail={"slot": f["slot"], "actual": actual_cabin, "preferred": pref_cabin},
            ))

    # Budget pacing. Only luxury complains about *underspending* (feels stingy);
    # everyone complains if we're right at the cap (no buffer for surprises).
    spent = sum(float(s.get("price", 0.0)) for s in active)
    cap = profile.hard.budget_cap
    if cap > 0:
        ratio = spent / cap
        low_thr = profile.tolerances.get("budget_low_ratio", 0.55)
        high_thr = profile.tolerances.get("budget_high_ratio", 0.95)
        if profile.archetype == "luxury" and ratio < low_thr:
            out.append(Mismatch(
                axis="budget_too_low",
                severity=min(1.0, low_thr - ratio),
                detail={"spent": round(spent, 2), "cap": cap, "ratio": round(ratio, 2)},
            ))
        if ratio > high_thr:
            out.append(Mismatch(
                axis="budget_too_high",
                severity=min(1.0, (ratio - high_thr) * 5.0),
                detail={"spent": round(spent, 2), "cap": cap, "ratio": round(ratio, 2)},
            ))

    out.sort(key=lambda m: -m.severity)
    return out


# --- LLM voice + scripted fallback ---------------------------------------

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


_STYLE_HINTS: dict[CommStyle, str] = {
    "terse": "Keep it under 50 words. Short sentences. No small talk.",
    "chatty": "Friendly and conversational. 80-150 words. A bit of personality.",
    "vague": "Be fuzzy on specifics. Don't pin down exact numbers unless asked directly.",
    "precise": "Use specific numbers and clear constraints. 60-120 words.",
    "formal": "Polite and professional. Complete sentences. 80-120 words.",
}


class PersonaVoice:
    """LLM voice with on-disk JSON cache and a scripted fallback.

    mode="llm"      : call OpenRouter (via openai SDK), cache the response.
    mode="scripted" : pure templates; deterministic; no API key needed.

    Auto-downgrades to "scripted" if OPENROUTER_API_KEY is unset.
    """

    def __init__(
        self,
        *,
        mode: Literal["llm", "scripted"] = "llm",
        model: str = "anthropic/claude-haiku-4.5",
        cache_dir: str = ".persona_cache",
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if mode == "llm" and not api_key:
            mode = "scripted"
        self.mode: Literal["llm", "scripted"] = mode
        self._client = None
        if mode == "llm":
            from openai import OpenAI
            self._client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key)

    # --- public API ---

    def render_request(self, profile: PersonaProfile) -> str:
        payload = self._profile_signature(profile)
        return self._cached_or_call(
            method="render_request",
            payload=payload,
            llm_fn=lambda: self._llm_request(profile),
            scripted_fn=lambda: self._scripted_request(profile),
        )

    def voice_feedback(self, profile: PersonaProfile, mismatches: list[Mismatch]) -> str:
        sig = [{"axis": m.axis, "severity": round(m.severity, 2), "detail": m.detail}
               for m in mismatches]
        payload = {"profile": self._profile_signature(profile), "mismatches": sig}
        return self._cached_or_call(
            method="voice_feedback",
            payload=payload,
            llm_fn=lambda: self._llm_feedback(profile, mismatches),
            scripted_fn=lambda: self._scripted_feedback(profile, mismatches),
        )

    def answer_question(self, profile: PersonaProfile, question: str) -> str:
        payload = {"profile": self._profile_signature(profile), "question": question.strip()}
        return self._cached_or_call(
            method="answer_question",
            payload=payload,
            llm_fn=lambda: self._llm_answer(profile, question),
            scripted_fn=lambda: self._scripted_answer(profile, question),
        )

    # --- cache + dispatch ---

    def _cached_or_call(self, *, method: str, payload: dict, llm_fn, scripted_fn) -> str:
        key = self._key(method, payload)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if self.mode == "llm":
            try:
                out = llm_fn()
                if not out:
                    raise ValueError("empty LLM response")
            except Exception as e:
                print(
                    f"[persona] LLM call failed ({type(e).__name__}: {e}); using scripted fallback",
                    file=sys.stderr,
                )
                out = scripted_fn()
        else:
            out = scripted_fn()

        self._cache_put(key, out, method=method, payload=payload)
        return out

    def _key(self, method: str, payload: dict) -> str:
        raw = json.dumps(
            {"method": method, "model": self.model, "mode": self.mode, "payload": payload},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text()).get("output")
        except Exception:
            return None

    def _cache_put(self, key: str, output: str, **meta: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        body = {"output": output, "model": self.model, "mode": self.mode, **meta}
        path.write_text(json.dumps(body, indent=2, default=str))

    def _profile_signature(self, p: PersonaProfile) -> dict:
        # A canonical, JSON-friendly view used for both cache keys and the
        # system prompt content. Rounding floats keeps cache hits stable.
        return {
            "seed": p.seed,
            "archetype": p.archetype,
            "origin": p.origin_city,
            "dest": p.dest_city,
            "depart": p.hard.depart_date,
            "return": p.hard.return_date,
            "group_size": p.hard.group_size,
            "budget_cap": p.hard.budget_cap,
            "no_overnight": p.hard.no_overnight_flights,
            "max_stops": p.hard.max_stops,
            "required_amenities": list(p.hard.required_amenities),
            "soft_prefs": {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in p.soft_prefs.items()},
            "comm_style": p.communication_style,
        }

    # --- LLM-mode prompts ---

    def _llm_call(self, system: str, user: str) -> str:
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()

    def _persona_system(self, p: PersonaProfile) -> str:
        top_acts = _top_acts(p, n=3)
        return (
            "You are roleplaying as a travel agency client. Stay in character. "
            "Output only what your character would say to the agent — no preamble, "
            "no signoff, no meta-commentary, no mention of being an AI or roleplay.\n\n"
            f"Archetype: {p.archetype}\n"
            f"Communication style: {p.communication_style} — {_STYLE_HINTS[p.communication_style]}\n"
            f"Trip: {p.origin_city} to {p.dest_city}, "
            f"{p.hard.depart_date} to {p.hard.return_date}\n"
            f"Group: {p.hard.group_size} traveler(s)\n"
            f"Budget cap: ${p.hard.budget_cap:.0f}\n"
            f"Hard constraints: {_format_hard(p.hard)}\n"
            f"Top interests: {top_acts}\n"
            f"Hotel preference: ~{p.soft_prefs.get('hotel_stars', 3)}-star\n"
            f"Cabin preference: {p.soft_prefs.get('flight_cabin', 'economy')}\n"
        )

    def _llm_request(self, p: PersonaProfile) -> str:
        system = self._persona_system(p)
        user = (
            "Write your opening message to the travel agent. Describe the trip you want, "
            "any constraints that matter to you, and your preferences. Match your "
            "communication style. Do not use bullet points unless your style is 'terse' "
            "or 'precise'."
        )
        return self._llm_call(system, user)

    def _llm_feedback(self, p: PersonaProfile, mismatches: list[Mismatch]) -> str:
        system = self._persona_system(p)
        if not mismatches:
            user = (
                "The agent has just shown you the proposed itinerary. You're happy with it. "
                "Tell them so, in character. Briefly."
            )
        else:
            bullets = "\n".join(f"- {_describe_mismatch(m)}" for m in mismatches[:6])
            user = (
                "The agent has shown you the proposed itinerary. Voice the following concerns "
                "in your own words. Do NOT invent new complaints — only voice these:\n\n"
                f"{bullets}\n\n"
                "Be specific about which item(s) you want swapped."
            )
        return self._llm_call(system, user)

    def _llm_answer(self, p: PersonaProfile, question: str) -> str:
        system = self._persona_system(p)
        user = f"The agent just asked you: {question!r}\n\nReply in character. Be honest about your preferences."
        return self._llm_call(system, user)

    # --- Scripted fallback ---

    def _scripted_request(self, p: PersonaProfile) -> str:
        nights = _nights(p.hard.depart_date, p.hard.return_date)
        top = _top_acts(p, n=2)
        cap = f"${p.hard.budget_cap:,.0f}"
        grp = f"{p.hard.group_size}" + (" of us" if p.hard.group_size > 1 else " person")

        base = {
            "budget": (
                f"Hi! Hoping to do {nights} days in {p.dest_city} from {p.origin_city}, "
                f"departing {p.hard.depart_date}. {grp}, trying to keep it under {cap}. "
                f"Mostly into {top}."
            ),
            "luxury": (
                f"I'd like to arrange a {nights}-day stay in {p.dest_city} for {grp}, "
                f"departing {p.hard.depart_date} from {p.origin_city}. Budget up to {cap}. "
                f"Expecting a 5-star hotel and business-class flights — no red-eyes. "
                f"Particular interest in {top}."
            ),
            "foodie": (
                f"Hey! Planning {nights} days in {p.dest_city} from {p.hard.depart_date} — "
                f"this is a food trip first. {grp}, budget around {cap}. Want to hit {top}. "
                f"Out of {p.origin_city}."
            ),
            "family": (
                f"Hi, family trip to {p.dest_city} for {grp}, "
                f"{p.hard.depart_date} to {p.hard.return_date} out of {p.origin_city}. "
                f"Budget about {cap}. No overnight flights, and we need breakfast and wifi at the hotel. "
                f"Looking for {top}."
            ),
            "history_buff": (
                f"Planning a {nights}-day trip to {p.dest_city} from {p.origin_city}, "
                f"departing {p.hard.depart_date}. {grp}. Budget around {cap}. "
                f"Main focus: {top}."
            ),
            "business": (
                f"{nights} nights in {p.dest_city}, {p.hard.depart_date} to {p.hard.return_date}. "
                f"One traveler from {p.origin_city}. Direct flights both ways, business class, "
                f"wifi and gym at the hotel. Budget {cap}."
            ),
        }
        text = base[p.archetype]

        if p.communication_style == "chatty":
            text += " Excited to hear what you come up with!"
        elif p.communication_style == "vague":
            text += " Flexible on most things — surprise me."
        elif p.communication_style == "precise" and p.hard.required_amenities:
            text += f" Required amenities: {', '.join(p.hard.required_amenities)}."
        elif p.communication_style == "formal":
            text = "Good afternoon. " + text + " Please let me know your recommendations."
        return text

    def _scripted_feedback(self, p: PersonaProfile, mismatches: list[Mismatch]) -> str:
        if not mismatches:
            return "This looks great. Let's go ahead and book it."
        intro = {
            "terse": "A few issues:",
            "chatty": "Thanks for putting this together! A few things to fix:",
            "vague": "Hmm, a few things feel off:",
            "precise": "Concerns with the proposal:",
            "formal": "Thank you. I have a few concerns:",
        }[p.communication_style]
        bullets = [_describe_mismatch(m)[:1].upper() + _describe_mismatch(m)[1:]
                   for m in mismatches[:5]]
        return intro + "\n" + "\n".join(f"- {b}" for b in bullets)

    def _scripted_answer(self, p: PersonaProfile, question: str) -> str:
        q = question.lower()
        h, sp = p.hard, p.soft_prefs

        if any(w in q for w in ("overnight", "red-eye", "red eye")):
            return ("No overnight flights, please." if h.no_overnight_flights
                    else "Overnight is fine if it saves money.")
        if any(w in q for w in ("layover", "stop", "connect")):
            if h.max_stops is not None:
                return f"At most {h.max_stops} stop(s)." if h.max_stops > 0 else "Direct only, please."
            return f"Up to {sp.get('flight_max_stops', 2)} stops is fine."
        if "cabin" in q or "class" in q:
            return f"I'd like {sp.get('flight_cabin', 'economy')}."
        if any(w in q for w in ("budget", "spend", "cost", "price")):
            return f"Trying to stay under ${h.budget_cap:,.0f} total."
        if any(w in q for w in ("food", "eat", "restaurant", "cuisine")):
            if sp.get("act_food", 0.0) > 0.3:
                return "Food is a huge priority — love local spots and food tours."
            return "Food's not a focus — just a few good meals."
        if any(w in q for w in ("neighborhood", "area", "where", "located")):
            return "Somewhere central if possible — close to the action."
        if any(w in q for w in ("amenity", "amenities", "hotel feature")):
            req = ", ".join(h.required_amenities) if h.required_amenities else "no strong feelings"
            return f"Need: {req}."
        if any(w in q for w in ("activity", "activities", "do", "see", "interest")):
            return f"Mostly into {_top_acts(p, n=3)}."
        return "No strong feelings — your call."


# --- Helpers (module-level) ----------------------------------------------

def _nights(d1: str, d2: str) -> int:
    a = dt.date.fromisoformat(d1)
    b = dt.date.fromisoformat(d2)
    return (b - a).days


def _top_acts(p: PersonaProfile, n: int = 2) -> str:
    items = [(k[4:], float(v)) for k, v in p.soft_prefs.items()
             if k.startswith("act_") and isinstance(v, (int, float))]
    items.sort(key=lambda x: -x[1])
    return ", ".join(c for c, _ in items[:n])


def _format_hard(hard: HardConstraints) -> str:
    parts = []
    if hard.no_overnight_flights:
        parts.append("no overnight flights")
    if hard.max_stops is not None:
        parts.append(f"max {hard.max_stops} stops")
    if hard.required_amenities:
        parts.append(f"required hotel amenities: {', '.join(hard.required_amenities)}")
    return "; ".join(parts) or "none beyond budget and dates"


def _describe_mismatch(m: Mismatch) -> str:
    d = m.detail
    axis = m.axis
    if axis == "flight_overnight":
        return "the flight is an overnight, which I don't want"
    if axis == "flight_stops_exceeded":
        return f"the flight has {d['stops']} stops — I asked for at most {d['max']}"
    if axis == "hotel_stars_too_low":
        return f"the hotel is only {d['actual']} stars (I wanted at least {d['target']})"
    if axis == "hotel_amenities_missing":
        return f"the hotel is missing amenities I need: {', '.join(d['missing'])}"
    if axis.startswith("activity_under_"):
        return (f"there's not enough {d['category']} content "
                f"(I'd like about {int(d['target']*100)}%, got {int(d['actual']*100)}%)")
    if axis.startswith("activity_over_"):
        return f"there's too much {d['category']} content for my taste"
    if axis == "flight_cabin_too_low":
        return f"the {d['slot']} is in {d['actual']} class — I'd prefer {d['preferred']}"
    if axis == "budget_too_low":
        return (f"the total (${d['spent']:.0f}) feels stingy against my "
                f"budget of ${d['cap']:.0f}")
    if axis == "budget_too_high":
        return (f"the total (${d['spent']:.0f}) is right at my cap of ${d['cap']:.0f} "
                "— no buffer for surprises")
    return f"issue: {axis}"
