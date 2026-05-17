"""Baseline policies for the eval harness.

All three are pure functions: (obs) -> action dict. They never call an LLM,
so they're cheap and deterministic. They're the floor (random), the
"naive cheap exploit" probe (cheapest), and the ceiling for non-learned
policies (heuristic).
"""

from __future__ import annotations

import numpy as np


def random_policy(obs: dict, *, rng: np.random.Generator) -> dict:
    """Pick uniformly from currently-valid tool calls.

    Floor benchmark. We restrict to the subset of tools whose preconditions
    are met (e.g. don't call `book` with no tentative items, don't `swap`
    on an empty itinerary). Establishes the "random valid action" baseline.
    """
    raise NotImplementedError


def cheapest_policy(obs: dict) -> dict:
    """Always pick the cheapest option that passes hard constraints.

    Designed to probe whether our reward function structurally resists the
    'always cheapest' exploit. If reward design works, cheapest should
    underperform heuristic by a clear margin on preference_coverage and
    recovery, even though it wins on raw spend.
    """
    raise NotImplementedError


def heuristic_policy(obs: dict, *, rng: np.random.Generator) -> dict:
    """Persona-aware heuristic. Respects soft prefs when ranking options.

    Phases:
      1. Search outbound flight, hotel, return flight matching profile.
      2. Add tentatively, propose, react to feedback (swap on complaint).
      3. Book.
      4. Watch for disruptions; rebook with same heuristic if one fires.
      5. Submit.

    Ceiling for non-learned policies. Not optimal — leaves headroom for a
    trained agent to beat.
    """
    raise NotImplementedError
