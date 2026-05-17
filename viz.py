"""Live terminal visualization of a baseline running against TravelEnv.

Usage:
    uv run python viz.py [--seed 42] [--baseline heuristic] [--turn-delay 0.4]

Renders a 4-panel live dashboard (rich):
  - header   : seed, archetype, turn, request snippet
  - budget   : progress bar (spent / cap)
  - itinerary: table of slots with status color-coding
  - action   : last tool call + last_result snippet
At episode end the action panel is replaced with horizontal reward-component
bars + the total.

Standalone — does not modify or import from eval.py. Adds no required path
for the rest of the env; this is purely a demo surface.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from travel_env.baselines import cheapest_policy, heuristic_policy, random_policy
from travel_env.env import TravelEnv
from travel_env.persona import sample_profile
from travel_env.rollout import (
    parse_tool_call,
    render_system_prompt,
    render_turn_user_message,
)


# --- Per-status color so the eye tracks transitions ----------------------

_STATUS_STYLE = {
    "tentative": "yellow",
    "booked": "green",
    "cancelled": "red",
}


# --- Panel renderers -----------------------------------------------------

def _header_panel(seed: int, obs: dict, info: dict, total_turns: int) -> Panel:
    clock = obs["state"]["clock"]
    archetype = info.get("archetype", "?")
    turn = clock.get("turn", 0)
    request = obs.get("request", "")
    snippet = request if len(request) <= 240 else request[:237] + "..."
    title = (
        f"[bold]Travel Agent[/bold]  |  seed={seed}  |  "
        f"archetype=[cyan]{archetype}[/cyan]  |  "
        f"turn [bold]{turn}/{total_turns}[/bold]  |  "
        f"{clock.get('origin','?')} → {clock.get('dest','?')}  "
        f"({clock.get('depart_date','?')} → {clock.get('return_date','?')}, "
        f"group={clock.get('group_size', '?')})"
    )
    body = Text(snippet, style="italic dim")
    return Panel(body, title=title, border_style="blue")


def _budget_panel(obs: dict) -> Panel:
    b = obs["state"]["budget"]
    cap = float(b.get("cap", 0) or 0)
    spent = float(b.get("spent", 0) or 0)
    pct = (spent / cap * 100) if cap > 0 else 0.0

    bar = ProgressBar(total=max(cap, 1), completed=min(spent, cap), width=40)
    summary = Text(
        f" ${spent:,.0f} / ${cap:,.0f}  ({pct:.0f}% used; ${cap - spent:,.0f} left)",
        style="bold",
    )
    return Panel(Group(bar, summary), title="Budget", border_style="magenta")


def _itinerary_panel(obs: dict) -> Panel:
    items = obs["state"].get("itinerary", []) or []
    pending = obs["state"].get("pending_events", []) or []

    table = Table(show_header=True, header_style="bold", expand=True, padding=(0, 1))
    table.add_column("Status", width=11)
    table.add_column("Slot", width=22)
    table.add_column("Item", overflow="ellipsis")
    table.add_column("Price", justify="right", width=9)

    if items:
        for s in items:
            status = str(s.get("status", "?"))
            color = _STATUS_STYLE.get(status, "white")
            table.add_row(
                Text(status, style=color),
                str(s.get("slot", "?")),
                str(s.get("name", "?"))[:60],
                f"${float(s.get('price', 0)):,.0f}",
            )
    else:
        table.add_row(
            Text("(empty)", style="dim italic"), "—", "no items yet", "—",
        )

    body: list = [table]
    if pending:
        body.append(Text(""))
        body.append(Text(f"⚠ Pending events: {len(pending)}", style="bold red"))
        for ev in pending:
            body.append(Text(
                f"   {ev.get('type', '?')} fires@turn={ev.get('fires_at_turn', '?')}  "
                f"{ev.get('payload', {})}",
                style="red",
            ))
    return Panel(Group(*body), title="Itinerary", border_style="green")


def _action_panel(last_action: dict | None, obs: dict) -> Panel:
    if last_action is None:
        return Panel(Text("(no action yet)", style="dim"), title="Last action",
                     border_style="yellow")

    tool = last_action.get("tool", "?")
    args = last_action.get("args", {})
    args_str = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    if len(args_str) > 80:
        args_str = args_str[:77] + "..."
    head = Text(f"{tool}({args_str})", style="bold")

    last = obs.get("last_result") or {}
    ok = last.get("ok")
    lines: list = [head, Text("")]

    if "disruption_fired" in last:
        lines.append(Text("⚠ Disruption fired!", style="bold red"))
        for ev in last["disruption_fired"]:
            lines.append(Text(f"   {ev.get('type')}: {ev.get('payload', {})}",
                              style="red"))
        last = last.get("tool_result") or {}

    if ok is True:
        if last.get("tool", "").startswith("search_"):
            lines.append(Text(
                f"✓ returned {last.get('count', '?')} results", style="green",
            ))
        elif "booked" in last:
            booked = last.get("booked", [])
            for b in booked:
                lines.append(Text(
                    f"✓ booked {b.get('slot', '?')} @ ${b.get('price', 0):,.0f}",
                    style="green",
                ))
            lines.append(Text(
                f"  total debited ${last.get('total_debited', 0):,.0f}; "
                f"remaining ${last.get('remaining_budget', 0):,.0f}",
                style="dim",
            ))
        elif "feedback_text" in last:
            mismatches = last.get("mismatches", [])
            lines.append(Text(
                f"✓ proposal received ({len(mismatches)} mismatch{'es' if len(mismatches)!=1 else ''})",
                style="green",
            ))
            voice = last.get("feedback_text", "")[:160]
            if voice:
                lines.append(Text(f'  "{voice}"', style="italic"))
        elif "final_reward" in last:
            lines.append(Text(
                f"✓ submitted; reward {last.get('final_reward', 0):+.3f}",
                style="bold green",
            ))
        else:
            lines.append(Text(f"✓ ok", style="green"))
    elif ok is False:
        lines.append(Text(f"✗ {last.get('error', 'failed')}", style="red"))
    return Panel(Group(*lines), title="Last action", border_style="yellow")


def _reward_panel(breakdown: dict, total: float) -> Panel:
    """End-of-episode panel: horizontal bars for each component + total."""
    rows: list = []
    rows.append(Text(f"TOTAL: {total:+.3f}",
                     style="bold green" if total >= 0 else "bold red"))
    rows.append(Text(""))

    def bar(label: str, value: float | None, scale: float = 1.0):
        if value is None:
            rows.append(Text(f"  {label:24s}  —  (n/a, no disruption fired)",
                             style="dim"))
            return
        bar_width = 30
        filled = int(max(0.0, min(1.0, value / scale)) * bar_width)
        bar_str = "█" * filled + "░" * (bar_width - filled)
        rows.append(Text(f"  {label:24s}  [{bar_str}]  {value:+.3f}"))

    rows.append(Text("Components:", style="bold"))
    bar("hard_constraint_gate", breakdown.get("hard_constraint_gate"))
    bar("preference_coverage",  breakdown.get("preference_coverage"))
    bar("budget_efficiency",    breakdown.get("budget_efficiency"))
    bar("coherence",            breakdown.get("coherence"))
    bar("recovery_quality",     breakdown.get("recovery_quality"))
    rows.append(Text(""))
    rows.append(Text(
        f"  Steps: {breakdown.get('steps', 0)}  |  "
        f"weights: pref=0.35, budget=0.20, coh=0.20, recov=0.25",
        style="dim",
    ))
    return Panel(Group(*rows), title="EPISODE END · reward breakdown",
                 border_style="green")


# --- Composer ------------------------------------------------------------

def _compose(seed: int, obs: dict, info: dict, last_action: dict | None,
             total_turns: int, ended: bool, breakdown: dict | None,
             total_reward: float) -> Group:
    parts = [_header_panel(seed, obs, info, total_turns), _budget_panel(obs),
             _itinerary_panel(obs)]
    if ended and breakdown:
        parts.append(_reward_panel(breakdown, total_reward))
    else:
        parts.append(_action_panel(last_action, obs))
    return Group(*parts)


# --- Driver --------------------------------------------------------------

_POLICY_FACTORY = {
    "random":    lambda seed: (lambda o, rng=np.random.default_rng(seed),
                               p=sample_profile(seed):
                               random_policy(o, rng=rng, profile=p)),
    "cheapest":  lambda seed: (lambda o, p=sample_profile(seed):
                               cheapest_policy(o, profile=p)),
    "heuristic": lambda seed: (lambda o, rng=np.random.default_rng(seed),
                               p=sample_profile(seed):
                               heuristic_policy(o, rng=rng, profile=p)),
}


def visualize_episode(*, seed: int, baseline: str, max_turns: int,
                      turn_delay: float, persona_mode: str = "scripted",
                      llm_model: str = "anthropic/claude-haiku-4.5") -> None:
    if baseline == "llm":
        _next_action = _make_llm_driver(llm_model)
    elif baseline in _POLICY_FACTORY:
        policy = _POLICY_FACTORY[baseline](seed)
        def _next_action(_obs, _messages):
            return policy(_obs)
    else:
        raise SystemExit(
            f"unknown baseline {baseline!r}. "
            f"Available: {list(_POLICY_FACTORY) + ['llm']}")

    env = TravelEnv(seed=seed, persona_mode=persona_mode, max_turns=max_turns)
    obs, reset_info = env.reset()
    # reset_info carries archetype + weights; step_info doesn't. Keep a merged
    # view so the header stays informative every turn.
    display_info = dict(reset_info)
    last_action: dict | None = None
    total_reward = 0.0

    # LLM driver maintains chat history across turns; baselines ignore it.
    messages: list[dict] = []
    if baseline == "llm":
        messages = [
            {"role": "system", "content": render_system_prompt(obs)},
            {"role": "user", "content": render_turn_user_message(obs, first_turn=True)},
        ]

    initial = _compose(seed, obs, display_info, None, max_turns,
                       ended=False, breakdown=None, total_reward=0.0)
    with Live(initial, refresh_per_second=20, screen=False) as live:
        time.sleep(turn_delay)
        terminated = truncated = False
        breakdown: dict | None = None
        while not (terminated or truncated):
            action = _next_action(obs, messages)
            last_action = action if isinstance(action, dict) else {
                "tool": getattr(action, "tool", "?"),
                "args": getattr(action, "args", {}),
            }
            obs, reward, terminated, truncated, step_info = env.step(action)
            display_info.update(step_info)
            total_reward = reward if (terminated or truncated) else total_reward
            if terminated or truncated:
                breakdown = step_info.get("reward_breakdown") or {}
            elif baseline == "llm":
                messages.append({
                    "role": "user",
                    "content": render_turn_user_message(obs),
                })
            live.update(_compose(
                seed, obs, display_info, last_action, max_turns,
                ended=(terminated or truncated),
                breakdown=breakdown, total_reward=total_reward,
            ))
            time.sleep(turn_delay)


def _make_llm_driver(model: str):
    """Per-turn OpenRouter call. The Live panel re-renders between turns;
    we don't try to retry on parse failure here — viz is a demo, not a
    benchmark. If parsing fails, force submit_final so the episode ends
    cleanly instead of getting stuck in the same broken state.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "--baseline llm requires OPENROUTER_API_KEY (load .env first).")
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def step_fn(_obs: dict, messages: list[dict]) -> dict:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": content})
        action = parse_tool_call(content)
        if action is None:
            return {"tool": "submit_final", "args": {}}
        return {"tool": action.tool, "args": action.args}

    return step_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline",
                   choices=list(_POLICY_FACTORY) + ["llm"],
                   default="heuristic")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--turn-delay", type=float, default=0.4,
                   help="Seconds to sleep between turns (so a human can follow along).")
    p.add_argument("--persona-mode", choices=["scripted", "llm"], default="scripted")
    p.add_argument("--llm-model", type=str, default="anthropic/claude-haiku-4.5",
                   help="Model for the 'llm' baseline rollout (via OpenRouter).")
    args = p.parse_args()

    # If LLM mode requested without key, downgrade.
    if args.persona_mode == "llm" and not os.environ.get("OPENROUTER_API_KEY"):
        args.persona_mode = "scripted"

    visualize_episode(
        seed=args.seed, baseline=args.baseline,
        max_turns=args.max_turns, turn_delay=args.turn_delay,
        persona_mode=args.persona_mode,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
