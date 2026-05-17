"""Reward function: rule-based components + composition + optional LLM judge.

Design notes (mirrored in README):
  - Training reward is rule-based and computed from state. Cheap, reproducible.
  - Components are bounded [0,1] (except step_penalty) and composed with
    transparent weights logged with each run.
  - LLM judge runs at episode end and is logged *separately* — never blended
    into the training reward. Its purpose is diagnostic: correlation between
    rule-based reward and LLM-judge score tells you whether your proxy
    is measuring what humans care about. Low correlation => blind spot.

Exploit defenses (called out in plan.md):
  - hard_constraint_gate is multiplicative: failing it zeros the soft terms.
  - budget_efficiency is concave with a peak < 1.0; "always cheapest" cannot
    dominate because it leaves money on the table.
  - coherence is a 0/1 gate inside the soft sum; geometric/temporal
    infeasibility kills the proposal.
  - recovery_quality is conditional on a disruption firing; ignoring
    disruptions both fails coherence AND loses the recovery term.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from travel_env.persona import PersonaProfile


# --- Weights (logged with every run) -------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    preference: float = 0.35
    budget: float = 0.20
    coherence: float = 0.20
    recovery: float = 0.25
    step_penalty: float = 0.01


DEFAULT_WEIGHTS = RewardWeights()


# --- Component outputs ---------------------------------------------------

@dataclass
class RewardBreakdown:
    hard_constraint_gate: float          # 0 or 1
    preference_coverage: float           # [0, 1]
    budget_efficiency: float             # [0, 1]
    coherence: float                     # 0 or 1
    recovery_quality: float | None       # [0, 1] or None if no disruption
    steps: int
    total: float
    weights: RewardWeights = DEFAULT_WEIGHTS
    extras: dict = field(default_factory=dict)  # for debug / logging


# --- Components ----------------------------------------------------------

def hard_constraint_gate(itinerary: list[dict], profile: PersonaProfile, spent: float) -> float:
    """1.0 if budget cap honored, dates match, group size honored, slots filled. Else 0.0."""
    raise NotImplementedError


def preference_coverage(itinerary: list[dict], profile: PersonaProfile) -> float:
    """Weighted dot product of itinerary attributes against profile.soft_prefs."""
    raise NotImplementedError


def budget_efficiency(spent: float, cap: float, peak_ratio: float = 0.85) -> float:
    """Concave; peaks at peak_ratio * cap. Penalizes both 'way under' and 'right at cap'."""
    raise NotImplementedError


def coherence(itinerary: list[dict]) -> float:
    """Geometric/temporal feasibility check. 1.0 if all checks pass, else 0.0."""
    raise NotImplementedError


def recovery_quality(
    pre_disruption_itinerary: list[dict],
    post_recovery_itinerary: list[dict],
    *,
    turns_to_rebook: int,
    price_delta: float,
) -> float:
    """Score the rebook: faster + closer-priced + preserves downstream = better."""
    raise NotImplementedError


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
) -> float:
    """gate * (w1*pref + w2*budget + w3*coh + w4*recovery) - w5*steps"""
    raise NotImplementedError


def score_episode(
    *,
    itinerary: list[dict],
    profile: PersonaProfile,
    spent: float,
    steps: int,
    disruption_info: dict | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown:
    """Top-level entry point called by TravelEnv on submit_final."""
    raise NotImplementedError


# --- Optional LLM judge (diagnostic only) --------------------------------

def llm_judge_score(
    request_text: str,
    itinerary: list[dict],
    *,
    model: str = "anthropic/claude-sonnet-4.5",
) -> float:
    """Run an LLM over (request, final itinerary) and return a 0..1 score.

    Logged alongside the rule-based reward in eval output. Never summed into
    the training reward.
    """
    raise NotImplementedError
