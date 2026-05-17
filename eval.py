"""Evaluation harness.

Runs N episodes per baseline; emits a metrics table and a per-component
reward decomposition. Optionally invokes the LLM judge to log correlation
between rule-based reward and judge score — the headline reward-design
diagnostic ("is my proxy reward measuring what humans care about?").

The harness drives the env. There are two kinds of drivers:
  * Policy baselines (random, cheapest, heuristic) — pure (obs)->action
    callables from `travel_env.baselines`. Stateless across turns.
  * LLM rollout (baseline name "llm") — defers to `travel_env.rollout.rollout`
    with an OpenRouter-backed client. Stateful across turns; requires
    OPENROUTER_API_KEY.

CLI:
    python eval.py --episodes 50 --seed 42 \\
        --baselines random,cheapest,heuristic \\
        [--persona-mode llm|scripted] \\
        [--llm-judge] \\
        [--render]
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from travel_env.baselines import cheapest_policy, heuristic_policy, random_policy
from travel_env.env import TravelEnv
from travel_env.reward import DEFAULT_WEIGHTS, llm_judge_score


# --- Baseline registry --------------------------------------------------
#
# Each entry is a (obs, **kwargs) -> action dict callable. We inspect each
# callable's signature once and pass `rng` only if it accepts one.

POLICY_REGISTRY: dict[str, Callable[..., dict]] = {
    "random": random_policy,
    "cheapest": cheapest_policy,
    "heuristic": heuristic_policy,
}

LLM_BASELINE = "llm"
ALL_BASELINES: tuple[str, ...] = (*POLICY_REGISTRY.keys(), LLM_BASELINE)


# --- Result type --------------------------------------------------------

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
    extras: dict = field(default_factory=dict)


# --- Episode runner -----------------------------------------------------

def run_episode(
    *,
    baseline: str,
    env: TravelEnv,
    seed: int,
    with_judge: bool,
    llm_model: str = "anthropic/claude-haiku-4.5",
) -> EpisodeResult:
    """Reset env at `seed`, drive it with the named baseline, return the result."""
    if baseline == LLM_BASELINE:
        breakdown, obs_final, trajectory, info = _drive_llm(env, seed, llm_model)
    elif baseline in POLICY_REGISTRY:
        breakdown, obs_final, trajectory, info = _drive_policy(
            POLICY_REGISTRY[baseline], env, seed,
        )
    else:
        raise ValueError(f"unknown baseline {baseline!r}. Available: {ALL_BASELINES}")

    judge_score: float | None = None
    if with_judge:
        request_text = obs_final.get("request", "") or ""
        itinerary = (obs_final.get("state") or {}).get("itinerary", []) or []
        judge_score = llm_judge_score(request_text, itinerary)

    recovery = breakdown.get("recovery_quality")
    return EpisodeResult(
        baseline=baseline,
        seed=seed,
        reward_total=float(breakdown.get("total", 0.0)),
        pref_cov=float(breakdown.get("preference_coverage", 0.0)),
        budget_eff=float(breakdown.get("budget_efficiency", 0.0)),
        coherence=float(breakdown.get("coherence", 0.0)),
        recovery=None if recovery is None else float(recovery),
        constraint=float(breakdown.get("hard_constraint_gate", 0.0)),
        steps=int(breakdown.get("steps", 0)),
        judge_score=judge_score,
        trajectory=trajectory,
        extras={"archetype": info.get("archetype")},
    )


def _drive_policy(
    policy: Callable[..., dict],
    env: TravelEnv,
    seed: int,
) -> tuple[dict, dict, list[dict], dict]:
    """Drive env with a step-level policy until terminated/truncated."""
    accepts_rng = "rng" in inspect.signature(policy).parameters
    rng = np.random.default_rng(seed)
    obs, reset_info = env.reset(seed=seed)

    trajectory: list[dict] = []
    step_info: dict = {}
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs, rng=rng) if accepts_rng else policy(obs)
        obs, _reward, terminated, truncated, step_info = env.step(action)
        last = obs.get("last_result") or {}
        trajectory.append({
            "turn": step_info.get("turn"),
            "tool": action.get("tool") if isinstance(action, dict) else None,
            "ok": last.get("ok") if isinstance(last, dict) else None,
        })
    breakdown = step_info.get("reward_breakdown") or {}
    # Episode-level info (archetype, weights, seed) lives in reset_info; merge.
    return breakdown, obs, trajectory, {**reset_info, **step_info}


def _drive_llm(
    env: TravelEnv,
    seed: int,
    model: str,
) -> tuple[dict, dict, list[dict], dict]:
    """Drive env via rollout.rollout against OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--baselines llm requires OPENROUTER_API_KEY in the environment."
        )
    # Defer these imports so the harness works without `openai` installed
    # as long as the LLM baseline isn't selected.
    from openai import OpenAI

    from travel_env.rollout import rollout

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    traj = rollout(env, client, model=model, seed=seed)
    breakdown = (traj.get("info") or {}).get("reward_breakdown") or {}
    obs_final = traj["observations"][-1]
    short_traj = [
        {"turn": i + 1, "tool": a["tool"], "ok": None}
        for i, a in enumerate(traj["actions"])
    ]
    merged_info = {**(traj.get("reset_info") or {}), **(traj.get("info") or {})}
    return breakdown, obs_final, short_traj, merged_info


# --- Aggregation --------------------------------------------------------

def aggregate(results: list[EpisodeResult]) -> dict:
    """Per-baseline means + corr(rule_reward, judge_score) when judge ran."""
    by_baseline: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_baseline.setdefault(r.baseline, []).append(r)

    out: dict[str, dict] = {}
    for baseline, rs in by_baseline.items():
        recoveries = [r.recovery for r in rs if r.recovery is not None]
        judges = [r.judge_score for r in rs if r.judge_score is not None]
        rules_for_judge = [r.reward_total for r in rs if r.judge_score is not None]

        out[baseline] = {
            "n": len(rs),
            "reward": _mean(r.reward_total for r in rs),
            "pref_cov": _mean(r.pref_cov for r in rs),
            "budget_eff": _mean(r.budget_eff for r in rs),
            "coherence": _mean(r.coherence for r in rs),
            "recovery": _mean(recoveries) if recoveries else None,
            "n_disruptions": len(recoveries),
            "constraint": _mean(r.constraint for r in rs),
            "steps": _mean(r.steps for r in rs),
            "judge_score": _mean(judges) if judges else None,
            "judge_corr": _pearson(rules_for_judge, judges) if len(judges) >= 2 else None,
        }
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else None


# --- Table printing -----------------------------------------------------

def print_table(agg: dict) -> None:
    """Pretty-print the aggregated metrics table to stdout."""
    if not agg:
        print("(no results)")
        return

    headers = ["baseline", "n", "reward", "pref_cov", "budget_eff",
               "coherence", "recovery", "constraint", "steps"]
    has_judge = any(v.get("judge_score") is not None for v in agg.values())
    if has_judge:
        headers += ["judge", "judge_corr"]

    rows: list[list[str]] = []
    for baseline, v in agg.items():
        recovery_cell = (
            _fmt(v["recovery"], 3) if v["recovery"] is not None
            else f"-/{v['n_disruptions']}"
        )
        row = [
            baseline,
            str(v["n"]),
            _fmt(v["reward"], 4),
            _fmt(v["pref_cov"], 3),
            _fmt(v["budget_eff"], 3),
            _fmt(v["coherence"], 3),
            recovery_cell,
            _fmt(v["constraint"], 3),
            _fmt(v["steps"], 1),
        ]
        if has_judge:
            row.append(_fmt(v["judge_score"], 3) if v["judge_score"] is not None else "-")
            row.append(_fmt(v["judge_corr"], 3) if v["judge_corr"] is not None else "-")
        rows.append(row)

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))


def _fmt(x: Any, digits: int) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


# --- CLI ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation harness for the travel-env baselines.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        type=str,
        default="random,cheapest,heuristic",
        help=f"Comma-separated subset of {{{', '.join(ALL_BASELINES)}}}.",
    )
    parser.add_argument(
        "--persona-mode",
        choices=["llm", "scripted"],
        default="scripted",
    )
    parser.add_argument("--persona-model", type=str, default="anthropic/claude-haiku-4.5")
    parser.add_argument("--llm-model", type=str, default="anthropic/claude-haiku-4.5",
                        help="Model used for the 'llm' baseline rollout (via OpenRouter).")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--llm-judge", action="store_true",
                        help="Score every episode with the LLM judge and report corr(rule, judge).")
    parser.add_argument("--render", action="store_true",
                        help="Print one full trajectory + final state per baseline.")
    args = parser.parse_args(argv)

    selected = [b.strip() for b in args.baselines.split(",") if b.strip()]
    unknown = [b for b in selected if b not in ALL_BASELINES]
    if unknown:
        print(f"unknown baseline(s): {unknown}. Available: {ALL_BASELINES}",
              file=sys.stderr)
        return 2

    weights = DEFAULT_WEIGHTS
    print(
        f"episodes={args.episodes} seed={args.seed} baselines={selected} "
        f"persona_mode={args.persona_mode} judge={args.llm_judge}",
        file=sys.stderr,
    )
    print(
        f"weights: pref={weights.preference} budget={weights.budget} "
        f"coherence={weights.coherence} recovery={weights.recovery} "
        f"step_penalty={weights.step_penalty}",
        file=sys.stderr,
    )

    results: list[EpisodeResult] = []
    skipped: list[str] = []
    t0 = time.time()

    for baseline in selected:
        for i in range(args.episodes):
            ep_seed = args.seed + i
            env = TravelEnv(
                seed=ep_seed,
                persona_mode=args.persona_mode,
                persona_model=args.persona_model,
                max_turns=args.max_turns,
                weights=weights,
            )
            try:
                r = run_episode(
                    baseline=baseline,
                    env=env,
                    seed=ep_seed,
                    with_judge=args.llm_judge,
                    llm_model=args.llm_model,
                )
                results.append(r)
                if args.render and i == 0:
                    print(f"\n--- {baseline} | seed={ep_seed} | "
                          f"archetype={r.extras.get('archetype')} ---",
                          file=sys.stderr)
                    print(env.render(), file=sys.stderr)
            except NotImplementedError as e:
                print(
                    f"baseline {baseline!r} raised NotImplementedError "
                    f"({e or 'no message'}); skipping remaining episodes for it.",
                    file=sys.stderr,
                )
                skipped.append(baseline)
                env.close()
                break
            except Exception as e:
                print(f"baseline {baseline!r} seed={ep_seed} crashed: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                env.close()
                continue
            finally:
                # Already closed in except branches; safe to call again.
                env.close()

    elapsed = time.time() - t0
    agg = aggregate(results)
    print(f"\nResults ({len(results)} episodes total, {elapsed:.1f}s):")
    print_table(agg)
    if skipped:
        print(f"\nskipped (not implemented): {skipped}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
