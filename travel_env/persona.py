"""Persona: structured profile + LLM voice (request, feedback) with disk cache.

Two artifacts at episode reset:
  1. PersonaProfile  — deterministic, sampled from seed. Reward reads this.
  2. Request text    — LLM-voiced (or scripted) version. Agent reads this.

During the episode, on `propose_to_client` we deterministically compute the
list of mismatches between the proposed itinerary and the profile, then the
LLM (or template) voices those complaints in character.

Reproducibility: every LLM call is cached on disk under .persona_cache/,
keyed by sha256(seed || event_payload). First run pays the API cost; later
runs are deterministic and free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Archetype = Literal["budget", "luxury", "foodie", "family", "history_buff", "business"]


# --- Profile -------------------------------------------------------------

@dataclass
class HardConstraints:
    budget_cap: float
    depart_date: str       # ISO date
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
    # Soft preferences are a vector in [0, 1]^N over feature axes.
    soft_prefs: dict[str, float] = field(default_factory=dict)
    # Tolerances: how far we'll drift from a soft pref before complaining.
    tolerances: dict[str, float] = field(default_factory=dict)
    communication_style: Literal["terse", "chatty", "vague", "precise", "formal"] = "chatty"


def sample_profile(seed: int) -> PersonaProfile:
    """Deterministic profile sampling. Same seed -> identical profile."""
    raise NotImplementedError


# --- Mismatch detection (deterministic, no LLM) --------------------------

@dataclass
class Mismatch:
    axis: str              # e.g. "hotel_location", "flight_overnight", "budget_overshoot"
    severity: float        # 0..1
    detail: dict           # tool-readable info for voicing


def compute_mismatches(itinerary: list[dict], profile: PersonaProfile) -> list[Mismatch]:
    """Inspect the proposed itinerary against the profile; return structured complaints."""
    raise NotImplementedError


# --- LLM voice (request generation + feedback) ---------------------------

class PersonaVoice:
    """Wraps the OpenRouter client + on-disk JSON cache.

    Two modes:
      - mode="llm"      : calls OpenRouter, caches result.
      - mode="scripted" : template-based; deterministic; no API key needed.

    The mode is decided at construction. The env falls back to "scripted"
    automatically if OPENROUTER_API_KEY is unset.
    """

    def __init__(
        self,
        *,
        mode: Literal["llm", "scripted"] = "llm",
        model: str = "anthropic/claude-haiku-4.5",
        cache_dir: str = ".persona_cache",
    ) -> None:
        raise NotImplementedError

    def render_request(self, profile: PersonaProfile) -> str:
        """Produce the free-text client request the agent sees at episode start."""
        raise NotImplementedError

    def voice_feedback(
        self,
        profile: PersonaProfile,
        mismatches: list[Mismatch],
    ) -> str:
        """Voice a list of structured complaints in the persona's style."""
        raise NotImplementedError

    def answer_question(self, profile: PersonaProfile, question: str) -> str:
        """Respond to a clarifying question from the agent."""
        raise NotImplementedError
