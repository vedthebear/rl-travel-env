"""Verifiers-style LLM driver loop.

The env itself is policy-agnostic. This module is one example of a policy
adapter: drive an LLM client (OpenAI-SDK-compatible, pointed at OpenRouter)
through a TravelEnv episode by parsing JSON tool calls from its output.

Trajectory shape returned:
    {
        "messages": [...],            # chat-format transcript
        "actions": [Action, ...],     # parsed tool calls in order
        "observations": [obs, ...],   # what the env returned
        "reward": RewardBreakdown,    # final reward
        "info": {...},                # env info (seed, persona, etc.)
    }
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from travel_env.env import Action, TravelEnv


class ChatClient(Protocol):
    """OpenAI-SDK shape we depend on (works for OpenRouter)."""

    def chat_completions_create(self, *, model: str, messages: list[dict], **kwargs) -> Any:
        ...


def rollout(
    env: TravelEnv,
    client: Any,
    *,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_completion_tokens: int = 1024,
) -> dict:
    """Run one episode of `env` driven by `client`. Returns a trajectory dict.

    Parsing: we look for a fenced ```json block or a top-level JSON object in
    the assistant's message. The expected shape is {"tool": str, "args": {...}}.
    On parse failure we surface that back to the model as a system message and
    let it retry (capped).
    """
    raise NotImplementedError


def render_system_prompt(obs: dict) -> str:
    """Render the static layer (request + tool catalogue) into a system prompt."""
    raise NotImplementedError


def parse_tool_call(text: str) -> Action | None:
    """Best-effort extraction of {"tool": ..., "args": ...} from model output."""
    raise NotImplementedError
