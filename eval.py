"""Evaluation harness.

Runs N episodes per baseline; emits a metrics table and a per-component
reward decomposition. Optionally invokes the LLM judge to log correlation
between rule-based reward and judge score (the headline reward-design
diagnostic).

CLI:
    uv run python eval.py --episodes 50 --seed 42 \\
        --baselines random,cheapest,heuristic \\
        [--persona-mode llm|scripted] \\
        [--llm-judge] \\
        [--render]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from travel_env.env import TravelEnv


@dataclass
class EpisodeResult:
    baseline: str
    seed: int
    reward_total: float
    pref_cov: float
    budget_eff: float
    coherence: float
    recovery: float | None
    constraint: float
    steps: int
    judge_score: float | None
    trajectory: list[dict]


def run_episode(*, baseline: str, env: TravelEnv, seed: int, with_judge: bool) -> EpisodeResult:
    """Reset env at `seed`, drive it with the named baseline, return the result."""
    raise NotImplementedError


def aggregate(results: list[EpisodeResult]) -> dict:
    """Compute per-baseline means + a correlation(rule, judge) if judge ran."""
    raise NotImplementedError


def print_table(agg: dict) -> None:
    """Pretty-print the aggregated metrics table to stdout."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        type=str,
        default="random,cheapest,heuristic",
        help="comma-separated subset of {random, cheapest, heuristic}",
    )
    parser.add_argument(
        "--persona-mode",
        choices=["llm", "scripted"],
        default="scripted",
        help="LLM voicing for persona (request + feedback) vs templated fallback",
    )
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run LLM judge on every episode; log corr(rule, judge)")
    parser.add_argument("--render", action="store_true",
                        help="Print one full trajectory per baseline")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
