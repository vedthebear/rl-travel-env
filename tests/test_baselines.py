"""baselines.py — each policy runs to completion + the ordering holds."""

from __future__ import annotations

import numpy as np
import pytest

from travel_env.baselines import cheapest_policy, heuristic_policy, random_policy
from travel_env.env import TravelEnv


def _run_episode(policy_fn, seed, max_turns=40):
    env = TravelEnv(seed=seed, persona_mode="scripted", max_turns=max_turns)
    obs, info = env.reset()
    terminated = truncated = False
    last_info = info
    n_steps = 0
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, _r, terminated, truncated, last_info = env.step(action)
        n_steps += 1
    return last_info.get("reward_breakdown") or {}, n_steps


# --- Each policy runs without crashing across many seeds -----------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_random_policy_runs_to_completion(seed):
    rng = np.random.default_rng(seed)
    breakdown, n = _run_episode(lambda o: random_policy(o, rng=rng), seed)
    assert "total" in breakdown
    assert n > 0


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_cheapest_policy_runs_to_completion(seed):
    breakdown, n = _run_episode(lambda o: cheapest_policy(o), seed)
    assert "total" in breakdown
    assert n > 0


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_heuristic_policy_runs_to_completion(seed):
    rng = np.random.default_rng(seed)
    breakdown, n = _run_episode(lambda o: heuristic_policy(o, rng=rng), seed)
    assert "total" in breakdown
    assert n > 0


# --- Aggregate ordering: heuristic > cheapest > random -------------------

def test_baseline_ordering_holds_in_aggregate():
    """Over 15 seeds, heuristic mean reward should exceed cheapest, which
    should exceed random. This is the headline empirical claim of the reward
    design — if it doesn't hold, the reward function isn't differentiating."""
    seeds = list(range(15))
    r_random, r_cheap, r_heur = [], [], []
    for seed in seeds:
        rng_r = np.random.default_rng(seed)
        rng_h = np.random.default_rng(seed)
        r_random.append(_run_episode(lambda o: random_policy(o, rng=rng_r), seed)[0].get("total", 0))
        r_cheap.append(_run_episode(lambda o: cheapest_policy(o), seed)[0].get("total", 0))
        r_heur.append(_run_episode(lambda o: heuristic_policy(o, rng=rng_h), seed)[0].get("total", 0))

    m_random = sum(r_random) / len(r_random)
    m_cheap = sum(r_cheap) / len(r_cheap)
    m_heur = sum(r_heur) / len(r_heur)

    assert m_heur >= m_cheap, f"heuristic {m_heur:.3f} should beat cheapest {m_cheap:.3f}"
    assert m_cheap > m_random, f"cheapest {m_cheap:.3f} should beat random {m_random:.3f}"


def test_random_passes_gate_rarely():
    """Random policy should almost never produce a coherent trip — that's
    the floor we want to demonstrate."""
    gates = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        b, _ = _run_episode(lambda o: random_policy(o, rng=rng), seed)
        gates.append(b.get("hard_constraint_gate", 0))
    rate = sum(gates) / len(gates)
    assert rate <= 0.20, f"random gate pass rate {rate:.0%} is suspiciously high"


def test_heuristic_passes_gate_often():
    """Heuristic should produce coherent trips on the majority of seeds.
    The remaining failures are typically tight-budget archetype/destination
    combinations that are infeasible by design."""
    gates = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        b, _ = _run_episode(lambda o: heuristic_policy(o, rng=rng), seed)
        gates.append(b.get("hard_constraint_gate", 0))
    rate = sum(gates) / len(gates)
    assert rate >= 0.60, f"heuristic gate pass rate {rate:.0%} is too low"
