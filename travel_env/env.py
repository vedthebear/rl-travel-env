"""TravelEnv — Gymnasium-style simulator (reset/step/close).

Action space:  JSON tool calls. ~10 tools defined in TOOL_REGISTRY.
Observation:   Structured dict (see docstring of `_render_obs`).

The env is intentionally framework-agnostic: it does not import gymnasium,
because gym.spaces.Box / Discrete / Dict are a poor fit for variable-length
text and structured inventory. Any framework that can call reset/step/close
(Verifiers, OpenEnv, RLLib, our own rollout helper) can drive this env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from travel_env.persona import PersonaProfile, PersonaVoice
from travel_env.reward import RewardBreakdown, RewardWeights, DEFAULT_WEIGHTS
from travel_env.world import PendingEvent, World


# --- Tool registry --------------------------------------------------------
#
# Each tool has:
#   name        — what the agent emits in {"tool": "<name>", "args": {...}}
#   description — one-line summary for the system prompt
#   schema      — JSON-schema-ish args (kept simple; full validation is loose)
#
# This catalogue is rendered into the static portion of every observation
# so the agent always knows what's available.

TOOL_REGISTRY: dict[str, dict] = {
    "search_flights": {
        "description": "Find top-k flights for a route on the requested date(s).",
        "schema": {
            "origin": "str (IATA)",
            "dest": "str (IATA)",
            "depart_date": "str (YYYY-MM-DD)",
            "return_date": "str (YYYY-MM-DD) | None",
            "max_stops": "int | None",
            "max_price": "float | None",
        },
    },
    "search_hotels": {
        "description": "Find top-k hotels in a city for the date range.",
        "schema": {
            "city": "str",
            "checkin": "str (YYYY-MM-DD)",
            "checkout": "str (YYYY-MM-DD)",
            "neighborhood": "str | None",
            "min_stars": "int | None",
            "max_price": "float | None",
        },
    },
    "search_activities": {
        "description": "Find top-k activities in a city, optionally filtered by category.",
        "schema": {
            "city": "str",
            "date": "str (YYYY-MM-DD) | None",
            "categories": "list[str] | None",
            "max_price": "float | None",
        },
    },
    "get_details": {
        "description": "Look up the full record for any search-result id.",
        "schema": {"item_id": "str"},
    },
    "add_to_itinerary": {
        "description": "Tentatively add an item to a slot. No commit, no budget debit.",
        "schema": {"item_id": "str", "slot": "str"},
    },
    "remove_from_itinerary": {
        "description": "Remove a tentative or booked item from a slot.",
        "schema": {"slot": "str"},
    },
    "swap": {
        "description": "Replace the item in a slot with a different one.",
        "schema": {"slot": "str", "new_item_id": "str"},
    },
    "book": {
        "description": "Commit one or more tentative items. Debits budget, locks inventory, "
                       "and makes flights eligible for disruption.",
        "schema": {"item_ids": "list[str]"},
    },
    "propose_to_client": {
        "description": "Show the client the current itinerary; receive structured + voiced feedback.",
        "schema": {},
    },
    "message_client": {
        "description": "Ask the client a clarifying question.",
        "schema": {"text": "str"},
    },
    "submit_final": {
        "description": "Terminate the episode. Reward is computed on the final state.",
        "schema": {},
    },
}


# --- Action / observation types ------------------------------------------

@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


# --- Itinerary slots -----------------------------------------------------

ItineraryStatus = Literal["tentative", "booked", "cancelled"]


@dataclass
class ItinerarySlot:
    slot: str                 # e.g. "outbound_flight", "hotel_night_1", "activity_day_2"
    item_id: str
    name: str
    price: float
    status: ItineraryStatus
    meta: dict = field(default_factory=dict)


# --- Env -----------------------------------------------------------------

class TravelEnv:
    """A multi-turn travel-planning environment.

    Usage:
        env = TravelEnv(seed=42, persona_mode="llm")
        obs = env.reset()
        while not obs["done"]:
            action = my_policy(obs)        # {"tool": ..., "args": {...}}
            obs = env.step(action)
        # obs["info"]["reward"] holds the RewardBreakdown
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        persona_mode: Literal["llm", "scripted"] = "llm",
        persona_model: str = "anthropic/claude-haiku-4.5",
        max_turns: int = 40,
        weights: RewardWeights = DEFAULT_WEIGHTS,
        render_mode: Literal["none", "text"] = "none",
    ) -> None:
        raise NotImplementedError

    # --- Gymnasium-style API ---

    def reset(self, *, seed: int | None = None) -> dict:
        """Sample profile, render request, init world. Returns initial observation."""
        raise NotImplementedError

    def step(self, action: dict | Action) -> dict:
        """Apply one tool call. Returns the next observation (including done + reward)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (LLM clients, file handles)."""
        raise NotImplementedError

    def render(self) -> str:
        """Pretty-print the current state for debugging / --render mode."""
        raise NotImplementedError

    # --- Internals (tool dispatch + observation rendering) ---

    def _dispatch(self, action: Action) -> dict:
        """Route the tool call to its handler, return the last_result payload."""
        raise NotImplementedError

    def _render_obs(self, last_result: dict | None = None) -> dict:
        """Three-layer observation:
            1. static       : request text, tool catalogue
            2. state        : itinerary, budget, clock, pending_events
            3. last_result  : varies by the tool that just fired
          + history (with prior search results pruned to summaries)
          + done flag
          + info dict (debug only; not shown to agent)
        """
        raise NotImplementedError
