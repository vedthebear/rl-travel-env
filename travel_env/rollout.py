"""Verifiers-style LLM driver loop.

The env is policy-agnostic. This module shows one way to drive it: parse
JSON tool calls from an OpenAI-SDK-compatible chat client (works against
OpenRouter out of the box, since OpenRouter is OpenAI-API-compatible).

Trajectory shape:

    {
        "messages":       [...],          # full chat transcript (system + alternating)
        "actions":        [...],          # parsed tool calls in order
        "observations":   [obs, ...],     # env returns, len = len(actions) + 1
        "rewards":        [...],          # per-step (0 except at terminal)
        "terminated":     bool,
        "truncated":      bool,
        "total_reward":   float,
        "info":           {...},          # last step's info dict
        "reset_info":     {...},          # info from reset()
        "parse_failures": int,
    }

The caller owns the OpenAI client (so they control auth, headers, timeouts):

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    traj = rollout(env, client, model="anthropic/claude-haiku-4.5")

On parse failure, the rollout surfaces the error back to the model and
retries up to `max_parse_retries`. If retries are exhausted, it forces
`submit_final` to cleanly close the episode. API errors propagate to
the caller — this module does not try to mask infrastructure failures.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, NamedTuple

from travel_env.env import Action, TravelEnv


# --- Public API ----------------------------------------------------------

def rollout(
    env: TravelEnv,
    client: Any,
    *,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_completion_tokens: int = 1024,
    seed: int | None = None,
    max_parse_retries: int = 2,
    verbose: bool = False,
) -> dict:
    """Run one episode of `env` driven by `client`. Returns a trajectory dict."""
    obs, reset_info = env.reset(seed=seed) if seed is not None else env.reset()

    sys_content = system_prompt if system_prompt is not None else render_system_prompt(obs)
    messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": render_turn_user_message(obs, first_turn=True)},
    ]

    observations: list[dict] = [obs]
    actions: list[Action] = []
    rewards: list[float] = []
    parse_failures = 0
    terminated = False
    truncated = False
    info: dict = reset_info

    while not (terminated or truncated):
        result = _solicit_action(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            max_parse_retries=max_parse_retries,
        )
        parse_failures += result.failures
        if verbose:
            print(
                f"[rollout] turn {len(actions)+1}: {result.action.tool}({result.action.args})",
                file=sys.stderr,
            )

        obs, reward, terminated, truncated, step_info = env.step(result.action)
        observations.append(obs)
        actions.append(result.action)
        rewards.append(reward)
        info = step_info

        if not (terminated or truncated):
            messages.append({"role": "user", "content": render_turn_user_message(obs)})

    return {
        "messages": messages,
        "actions": [{"tool": a.tool, "args": a.args} for a in actions],
        "observations": observations,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": float(sum(rewards)),
        "info": info,
        "reset_info": reset_info,
        "parse_failures": parse_failures,
    }


# --- Prompt rendering ----------------------------------------------------

def render_system_prompt(obs: dict) -> str:
    """Render the static layer (client request + tool catalogue) into a system prompt."""
    request = obs.get("request", "(missing)")
    tools = obs.get("tools", [])

    tool_lines: list[str] = []
    for t in tools:
        name = t.get("name", "?")
        args = t.get("args", {})
        sig = ", ".join(f"{k}: {v}" for k, v in args.items()) if args else ""
        desc = t.get("description", "")
        tool_lines.append(f"- {name}({sig})\n    {desc}")
    tools_section = "\n".join(tool_lines) or "(none)"

    return (
        "You are an AI travel agent helping a client plan a trip. You act by "
        "calling tools — that is your only way to affect the world. The client "
        "cannot see this conversation; they only see what you send via "
        "propose_to_client or message_client.\n\n"
        f"CLIENT REQUEST:\n{request}\n\n"
        f"TOOLS:\n{tools_section}\n\n"
        "OUTPUT FORMAT:\n"
        "Every turn, end your response with exactly one tool call as a JSON "
        "object in a fenced ```json block. Brief reasoning before the block "
        "is allowed; only the JSON block is parsed.\n\n"
        "```json\n"
        '{\"tool\": \"search_flights\", \"args\": {\"origin\": \"NRT\", \"dest\": \"CDG\", \"depart_date\": \"2026-04-12\"}}\n'
        "```\n\n"
        "STRATEGY:\n"
        "- Search inventory first, then add_to_itinerary, then book to commit.\n"
        "- Use propose_to_client to receive the client's structured feedback on a draft.\n"
        "- Watch pending_events for flight cancellations and rebook quickly.\n"
        "- Call submit_final only when the trip is complete and the client is satisfied.\n"
    )


def render_turn_user_message(obs: dict, *, first_turn: bool = False) -> str:
    """Render the per-turn state snapshot the model sees as a user message."""
    state = obs.get("state", {}) or {}
    budget = state.get("budget", {}) or {}
    clock = state.get("clock", {}) or {}
    itinerary = state.get("itinerary", []) or []
    pending = state.get("pending_events", []) or []
    last_result = obs.get("last_result")

    parts: list[str] = []
    turn = int(clock.get("turn", 0))
    rem = int(clock.get("turns_remaining", 0))
    parts.append(
        f"Turn {turn}/{turn + rem}. "
        f"Budget: spent ${float(budget.get('spent', 0)):.0f} / "
        f"cap ${float(budget.get('cap', 0)):.0f} "
        f"(remaining ${float(budget.get('remaining', 0)):.0f})."
    )
    parts.append(
        f"Trip dates: {clock.get('depart_date', '?')} -> {clock.get('return_date', '?')}."
    )

    if itinerary:
        parts.append("Itinerary:")
        for s in itinerary:
            parts.append(
                f"  [{str(s.get('status', '?')):<9}] "
                f"{str(s.get('slot', '?')):<24} "
                f"{str(s.get('name', '?'))[:40]:<40} "
                f"${float(s.get('price', 0)):.0f}"
            )
    else:
        parts.append("Itinerary: (empty)")

    if pending:
        parts.append("Pending events:")
        for e in pending:
            parts.append(
                f"  {e.get('type')} fires@turn={e.get('fires_at_turn')}: {e.get('payload')}"
            )

    if last_result is not None and not first_turn:
        parts.append("Last action result:")
        parts.append(json.dumps(last_result, indent=2, default=str))

    parts.append("What is your next tool call?")
    return "\n".join(parts)


# --- Tool-call parsing ---------------------------------------------------

def parse_tool_call(text: str) -> Action | None:
    """Best-effort extraction of {"tool": ..., "args": ...} from model output.

    Priority order:
      1. JSON inside a ```json fenced block (last one if multiple).
      2. JSON inside any ``` fenced block.
      3. Any balanced top-level {...} object containing a "tool" key
         (last one wins, since most models put scratch reasoning first).
    """
    for lang in ("json", None):
        obj = _try_fenced(text, lang)
        if obj is not None:
            action = _action_from_obj(obj)
            if action is not None:
                return action

    for cand in reversed(_scan_json_objects(text)):
        action = _action_from_obj(cand)
        if action is not None:
            return action

    return None


# --- Internals -----------------------------------------------------------

class _ActionResult(NamedTuple):
    action: Action
    failures: int


def _solicit_action(
    *,
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_parse_retries: int,
) -> _ActionResult:
    """Call the LLM until it emits a parseable tool call or retries are spent."""
    failures = 0
    for attempt in range(max_parse_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": content})

        action = parse_tool_call(content)
        if action is not None:
            return _ActionResult(action, failures)

        failures += 1
        if attempt < max_parse_retries:
            messages.append({"role": "user", "content": (
                "Your last response did not contain a parseable tool call. "
                "Respond with a single JSON object wrapped in a ```json fenced "
                "block, shaped {\"tool\": \"<name>\", \"args\": {...}}. Try again."
            )})

    return _ActionResult(Action(tool="submit_final", args={}), failures)


_FENCE_PATTERNS = {
    "json": re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE),
    None: re.compile(r"```[a-zA-Z]*\s*([\s\S]*?)```"),
}


def _try_fenced(text: str, lang: str | None) -> Any:
    """Pull JSON out of a fenced block. Brace-aware so nested JSON works."""
    pat = _FENCE_PATTERNS[lang]
    matches = pat.findall(text)
    if not matches:
        return None
    for chunk in reversed(matches):
        objs = _scan_json_objects(chunk)
        if objs:
            return objs[-1]
    return None


def _scan_json_objects(text: str) -> list[Any]:
    """Find all balanced top-level {...} blocks; return those that parse as JSON.

    Handles nested braces and string-literal contents (so {"text": "{x}"} parses
    as one object, not three).
    """
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        matched = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[i:j + 1]
                        try:
                            out.append(json.loads(chunk))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        matched = True
                        break
            j += 1
        if not matched:
            break
    return out


def _action_from_obj(obj: Any) -> Action | None:
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    args = obj.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return Action(tool=tool, args=dict(args))
