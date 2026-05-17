"""TravelEnv — Gymnasium-style simulator (reset/step/close).

Action space:  JSON tool calls. 11 tools defined in TOOL_REGISTRY.
Observation:   Structured dict (see docstring of `_render_obs`).

Conforms to the modern Gymnasium API (post-v0.26):
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)

The env intentionally does not import `gymnasium`: `gym.spaces.Box / Discrete /
Dict` are a poor fit for variable-length text and structured inventory. Any
framework that can call reset/step/close (Verifiers, OpenEnv, RLLib, our own
rollout helper) can drive this env without further glue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

from travel_env.persona import (
    PersonaProfile,
    PersonaVoice,
    compute_mismatches,
    sample_profile,
)
from travel_env.reward import (
    DEFAULT_WEIGHTS,
    RewardBreakdown,
    RewardWeights,
    score_episode,
)
from travel_env.world import (
    Activity,
    Flight,
    Hotel,
    PendingEvent,
    World,
    get_details as world_get_details,
    make_world,
    maybe_schedule_disruption,
    search_activities,
    search_flights,
    search_hotels,
)


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
            "origin": "str (IATA or city name)",
            "dest": "str (IATA or city name)",
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
        "description": "Tentatively add an item to a slot. No commit, no budget debit. "
                       "If the slot is already occupied, the existing item is replaced.",
        "schema": {"item_id": "str", "slot": "str (free-form, e.g. 'outbound_flight')"},
    },
    "remove_from_itinerary": {
        "description": "Remove the item in the given slot (tentative or booked).",
        "schema": {"slot": "str"},
    },
    "swap": {
        "description": "Replace the item in a slot with a different one. Sugar over "
                       "remove + add. Status resets to tentative.",
        "schema": {"slot": "str", "new_item_id": "str"},
    },
    "book": {
        "description": "Commit one or more tentative items. Debits budget, locks "
                       "inventory, and makes flights eligible for disruption.",
        "schema": {"item_ids": "list[str] (must be in itinerary as tentative)"},
    },
    "propose_to_client": {
        "description": "Show the client the current itinerary; receive structured + "
                       "voiced feedback. No state change beyond a turn tick.",
        "schema": {},
    },
    "message_client": {
        "description": "Ask the client a clarifying question; receive their answer.",
        "schema": {"text": "str"},
    },
    "submit_final": {
        "description": "Terminate the episode. Reward is computed on the final state.",
        "schema": {},
    },
}


# --- Action / itinerary types --------------------------------------------

@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


ItineraryStatus = Literal["tentative", "booked", "cancelled"]


@dataclass
class ItinerarySlot:
    slot: str
    item_id: str
    name: str
    price: float       # full cost for this slot (hotel: per-night × nights; else single price)
    status: ItineraryStatus
    meta: dict = field(default_factory=dict)


# --- Env -----------------------------------------------------------------

class TravelEnv:
    """A multi-turn travel-planning environment.

    Usage (Gymnasium-style):
        env = TravelEnv(seed=42, persona_mode="scripted")
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = my_policy(obs)             # {"tool": ..., "args": {...}}
            obs, reward, terminated, truncated, info = env.step(action)
        # info["reward_breakdown"] holds the full per-component decomposition
    """

    metadata = {"render_modes": ["none", "text"]}

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
        # Auto-downgrade to scripted if LLM mode requested without an API key.
        if persona_mode == "llm" and not os.environ.get("OPENROUTER_API_KEY"):
            persona_mode = "scripted"

        self._init_seed = seed
        self._persona_mode = persona_mode
        self._persona_model = persona_model
        self._max_turns = max_turns
        self._weights = weights
        self.render_mode = render_mode

        # Per-episode state — populated/reset by reset().
        self._world: World | None = None
        self._profile: PersonaProfile | None = None
        self._voice: PersonaVoice | None = None
        self._request_text: str = ""
        self._itinerary: list[ItinerarySlot] = []
        self._spent: float = 0.0
        self._turn: int = 0
        self._pending_events: list[PendingEvent] = []
        self._history: list[dict] = []
        self._terminated: bool = False
        self._truncated: bool = False
        self._reward_total: float = 0.0
        self._reward_breakdown: RewardBreakdown | None = None
        self._last_result: dict | None = None

        # Disruption recovery bookkeeping (None until a disruption fires).
        self._disruption_fired_turn: int | None = None
        self._pre_disruption_itinerary: list[ItinerarySlot] | None = None
        self._original_flight_price: float | None = None

        # Dispatch table populated below.
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "search_flights": self._tool_search_flights,
            "search_hotels": self._tool_search_hotels,
            "search_activities": self._tool_search_activities,
            "get_details": self._tool_get_details,
            "add_to_itinerary": self._tool_add_to_itinerary,
            "remove_from_itinerary": self._tool_remove_from_itinerary,
            "swap": self._tool_swap,
            "book": self._tool_book,
            "propose_to_client": self._tool_propose_to_client,
            "message_client": self._tool_message_client,
            "submit_final": self._tool_submit_final,
        }

    # --- Gymnasium-style API ---

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        """Sample profile + world, render the client request. Returns (obs, info)."""
        episode_seed = seed if seed is not None else self._init_seed
        self._init_seed = episode_seed  # subsequent resets without seed use the latest

        self._world = make_world(episode_seed)
        self._profile = sample_profile(episode_seed)
        self._voice = PersonaVoice(
            mode=self._persona_mode,
            model=self._persona_model,
        )
        self._request_text = self._voice.render_request(self._profile)

        self._itinerary = []
        self._spent = 0.0
        self._turn = 0
        self._pending_events = []
        self._history = []
        self._terminated = False
        self._truncated = False
        self._reward_total = 0.0
        self._reward_breakdown = None
        self._last_result = None
        self._disruption_fired_turn = None
        self._pre_disruption_itinerary = None
        self._original_flight_price = None

        obs = self._render_obs(last_result=None)
        info = {
            "seed": episode_seed,
            "persona_mode": self._persona_mode,
            "archetype": self._profile.archetype,
            "weights": _weights_to_dict(self._weights),
        }
        return obs, info

    def step(
        self, action: dict | Action
    ) -> tuple[dict, float, bool, bool, dict]:
        """Apply one tool call. Returns (obs, reward, terminated, truncated, info).

        Reward is 0 except on the terminating step. Use info["reward_breakdown"]
        for the full per-component decomposition.
        """
        if self._terminated or self._truncated:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if self._world is None or self._profile is None:
            raise RuntimeError("step() called before reset()")

        action_obj = _action_from_input(action)
        self._turn += 1

        # Fire any pending disruptions whose time has come (must precede dispatch
        # so the agent sees the new state in last_result if relevant).
        disruption_results = self._tick_disruptions()

        # Dispatch
        if action_obj.tool not in self._handlers:
            last_result = {
                "ok": False,
                "error": f"unknown tool: {action_obj.tool!r}",
                "valid_tools": list(self._handlers.keys()),
            }
        else:
            try:
                last_result = self._handlers[action_obj.tool](action_obj.args)
            except Exception as e:
                # Tool-level errors are surfaced to the agent rather than crashing
                # the env. Distinguishes "you used the tool wrong" from infra failures.
                last_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        # If a disruption just fired this turn, prepend the notice so the agent
        # sees it even if they were doing something unrelated.
        if disruption_results:
            last_result = {
                "ok": last_result.get("ok", True),
                "disruption_fired": disruption_results,
                "tool_result": last_result,
            }

        self._last_result = last_result
        self._history.append({
            "turn": self._turn,
            "action": {"tool": action_obj.tool, "args": action_obj.args},
            "result": last_result,
        })

        # Truncation check
        if not self._terminated and self._turn >= self._max_turns:
            self._truncated = True
            self._finalize_reward()

        step_reward = (
            float(self._reward_total)
            if (self._terminated or self._truncated)
            else 0.0
        )
        obs = self._render_obs(last_result=last_result)
        info = {
            "turn": self._turn,
            "reward_breakdown": (
                _breakdown_to_dict(self._reward_breakdown)
                if self._reward_breakdown is not None else None
            ),
            "disruption_fired_turn": self._disruption_fired_turn,
        }
        return obs, step_reward, self._terminated, self._truncated, info

    def close(self) -> None:
        """Release any held resources."""
        self._world = None
        self._profile = None
        if self._voice is not None:
            closer = getattr(self._voice, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        self._voice = None

    def render(self) -> str:
        """Pretty-print the current state (used for --render in eval)."""
        if self._world is None or self._profile is None:
            return "<TravelEnv: not reset>"
        lines: list[str] = []
        lines.append(f"=== Turn {self._turn}/{self._max_turns} ===")
        lines.append(f"Archetype: {self._profile.archetype}  "
                     f"Dates: {self._profile.hard.depart_date} -> "
                     f"{self._profile.hard.return_date}  Group: {self._profile.hard.group_size}")
        lines.append(f"Budget: spent ${self._spent:.0f} / cap ${self._profile.hard.budget_cap:.0f} "
                     f"(remaining ${self._profile.hard.budget_cap - self._spent:.0f})")
        lines.append(f"Itinerary ({len(self._itinerary)} slot{'s' if len(self._itinerary) != 1 else ''}):")
        for s in self._itinerary:
            lines.append(f"  [{s.status:9s}] {s.slot:24s} {s.name[:40]:40s} ${s.price:.0f}")
        if self._pending_events:
            lines.append(f"Pending events ({len(self._pending_events)}):")
            for e in self._pending_events:
                lines.append(f"  {e.type} fires@turn={e.fires_at_turn}: {e.payload}")
        if self._terminated or self._truncated:
            lines.append("--- EPISODE END ---")
            lines.append(f"Total reward: {self._reward_total:.3f}")
            if self._reward_breakdown is not None:
                b = self._reward_breakdown
                lines.append(f"  gate={b.hard_constraint_gate} pref={b.preference_coverage:.2f} "
                             f"budget={b.budget_efficiency:.2f} coh={b.coherence:.2f} "
                             f"recov={b.recovery_quality} steps={b.steps}")
        return "\n".join(lines)

    # --- Tool handlers (one per entry in TOOL_REGISTRY) ---

    def _tool_search_flights(self, args: dict) -> dict:
        results = search_flights(self._world, **_clean_args(args))
        return _search_envelope("search_flights", [self._flight_to_summary(f) for f in results])

    def _tool_search_hotels(self, args: dict) -> dict:
        results = search_hotels(self._world, **_clean_args(args))
        return _search_envelope("search_hotels", [self._hotel_to_summary(h) for h in results])

    def _tool_search_activities(self, args: dict) -> dict:
        results = search_activities(self._world, **_clean_args(args))
        return _search_envelope("search_activities", [self._activity_to_summary(a) for a in results])

    def _tool_get_details(self, args: dict) -> dict:
        item_id = args.get("item_id")
        if not item_id:
            return {"ok": False, "error": "get_details requires non-empty item_id"}
        item = world_get_details(self._world, item_id)
        if item is None:
            return {"ok": False, "error": f"unknown item_id: {item_id}"}
        return {"ok": True, "item": _item_to_full_dict(item)}

    def _tool_add_to_itinerary(self, args: dict) -> dict:
        item_id = args.get("item_id")
        slot = args.get("slot")
        if not item_id or not slot:
            return {"ok": False, "error": "add_to_itinerary requires item_id and slot"}
        item = world_get_details(self._world, item_id)
        if item is None:
            return {"ok": False, "error": f"unknown item_id: {item_id}"}

        # Compute the slot's effective price + the meta the reward function reads.
        if isinstance(item, Hotel):
            nights = _nights_between(
                self._profile.hard.depart_date, self._profile.hard.return_date,
            )
            slot_price = item.price_per_night * max(nights, 1)
            meta = {
                "kind": "hotel",
                "city": item.city,
                "neighborhood": item.neighborhood,
                "stars": item.stars,
                "price_per_night": item.price_per_night,
                "amenities": list(item.amenities),
                "nights": nights,
            }
        elif isinstance(item, Flight):
            slot_price = item.price
            meta = {
                "kind": "flight",
                "origin": item.origin,
                "dest": item.dest,
                "depart_iso": item.depart_iso,
                "arrive_iso": item.arrive_iso,
                "stops": item.stops,
                "cabin": item.cabin,
                "overnight": item.overnight,
            }
        elif isinstance(item, Activity):
            slot_price = item.price
            meta = {
                "kind": "activity",
                "city": item.city,
                "category": item.category,
                "duration_hours": item.duration_hours,
            }
        else:
            return {"ok": False, "error": f"unsupported item type: {type(item).__name__}"}

        # Replace whatever was in this slot. If the existing slot was booked, refund.
        existing = [s for s in self._itinerary if s.slot == slot]
        for s in existing:
            if s.status == "booked":
                self._spent -= s.price
        self._itinerary = [s for s in self._itinerary if s.slot != slot]

        self._itinerary.append(ItinerarySlot(
            slot=slot, item_id=item_id, name=getattr(item, "name", item_id),
            price=slot_price, status="tentative", meta=meta,
        ))
        return {
            "ok": True, "slot": slot, "item_id": item_id,
            "price": round(slot_price, 2), "replaced": len(existing) > 0,
        }

    def _tool_remove_from_itinerary(self, args: dict) -> dict:
        slot = args.get("slot")
        if not slot:
            return {"ok": False, "error": "remove_from_itinerary requires slot"}
        before = list(self._itinerary)
        # Refund anything booked we're removing.
        for s in before:
            if s.slot == slot and s.status == "booked":
                self._spent -= s.price
        self._itinerary = [s for s in before if s.slot != slot]
        removed = len(before) - len(self._itinerary)
        return {"ok": removed > 0, "slot": slot, "removed": removed}

    def _tool_swap(self, args: dict) -> dict:
        slot = args.get("slot")
        new_item_id = args.get("new_item_id")
        if not slot or not new_item_id:
            return {"ok": False, "error": "swap requires slot and new_item_id"}
        # Implementation is just remove + add. Status resets to tentative.
        self._tool_remove_from_itinerary({"slot": slot})
        return self._tool_add_to_itinerary({"item_id": new_item_id, "slot": slot})

    def _tool_book(self, args: dict) -> dict:
        item_ids = args.get("item_ids") or []
        if not item_ids:
            return {"ok": False, "error": "book requires a non-empty item_ids list"}

        to_book: list[ItinerarySlot] = []
        for iid in item_ids:
            matches = [s for s in self._itinerary if s.item_id == iid and s.status == "tentative"]
            if not matches:
                return {
                    "ok": False,
                    "error": f"item {iid} is not in itinerary as tentative "
                             f"(must add_to_itinerary first)",
                }
            to_book.extend(matches)

        total_cost = sum(s.price for s in to_book)
        cap = self._profile.hard.budget_cap
        if self._spent + total_cost > cap:
            return {
                "ok": False,
                "error": "would exceed budget cap",
                "would_spend": round(self._spent + total_cost, 2),
                "cap": cap,
                "shortfall": round(self._spent + total_cost - cap, 2),
            }

        # Commit.
        self._spent += total_cost
        for s in to_book:
            s.status = "booked"

        # Roll disruption for any booked flights. Resulting events are scheduled
        # for a future turn; the agent does NOT see them at booking time.
        for s in to_book:
            if s.meta.get("kind") == "flight":
                flight = world_get_details(self._world, s.item_id)
                if isinstance(flight, Flight):
                    event = maybe_schedule_disruption(
                        self._world, flight, current_turn=self._turn,
                    )
                    if event is not None:
                        self._pending_events.append(event)

        return {
            "ok": True,
            "booked": [
                {"slot": s.slot, "item_id": s.item_id, "price": round(s.price, 2)}
                for s in to_book
            ],
            "total_debited": round(total_cost, 2),
            "remaining_budget": round(cap - self._spent, 2),
        }

    def _tool_propose_to_client(self, args: dict) -> dict:
        mismatches = compute_mismatches(
            [self._slot_to_dict(s) for s in self._itinerary], self._profile,
        )
        feedback_text = self._voice.voice_feedback(self._profile, mismatches)
        return {
            "ok": True,
            "feedback_text": feedback_text,
            "mismatches": [
                {"axis": m.axis, "severity": m.severity, "detail": m.detail}
                for m in mismatches
            ],
        }

    def _tool_message_client(self, args: dict) -> dict:
        text = args.get("text", "")
        if not text:
            return {"ok": False, "error": "message_client requires non-empty text"}
        answer = self._voice.answer_question(self._profile, text)
        return {"ok": True, "client_says": answer}

    def _tool_submit_final(self, args: dict) -> dict:
        self._terminated = True
        self._finalize_reward()
        return {
            "ok": True,
            "final_reward": round(self._reward_total, 4),
            "breakdown": (
                _breakdown_to_dict(self._reward_breakdown)
                if self._reward_breakdown is not None else None
            ),
        }

    # --- Reward / disruption internals ---

    def _finalize_reward(self) -> None:
        disruption_info: dict | None = None
        if self._disruption_fired_turn is not None:
            disruption_info = {
                "fired_at_turn": self._disruption_fired_turn,
                "pre_itinerary": [
                    self._slot_to_dict(s)
                    for s in (self._pre_disruption_itinerary or [])
                ],
                "post_itinerary": [self._slot_to_dict(s) for s in self._itinerary],
                "original_flight_price": self._original_flight_price,
                "final_turn": self._turn,
            }
        self._reward_breakdown = score_episode(
            itinerary=[self._slot_to_dict(s) for s in self._itinerary],
            profile=self._profile,
            spent=self._spent,
            steps=self._turn,
            disruption_info=disruption_info,
            weights=self._weights,
        )
        self._reward_total = self._reward_breakdown.total

    def _tick_disruptions(self) -> list[dict]:
        """Fire any events whose `fires_at_turn` has arrived. Returns summaries."""
        if not self._pending_events:
            return []
        fired: list[dict] = []
        remaining: list[PendingEvent] = []
        for event in self._pending_events:
            if event.fires_at_turn <= self._turn:
                if self._disruption_fired_turn is None:
                    self._disruption_fired_turn = self._turn
                    self._pre_disruption_itinerary = [
                        ItinerarySlot(**vars(s)) for s in self._itinerary
                    ]
                    if event.type == "flight_cancelled":
                        self._original_flight_price = event.payload.get(
                            "original_price", 0.0,
                        )
                if event.type == "flight_cancelled":
                    fid = event.payload.get("flight_id")
                    for s in self._itinerary:
                        if s.item_id == fid and s.status == "booked":
                            s.status = "cancelled"
                            self._spent -= s.price  # full refund on cancellation
                fired.append({
                    "type": event.type,
                    "payload": dict(event.payload),
                })
            else:
                remaining.append(event)
        self._pending_events = remaining
        return fired

    # --- Item / state -> dict serialization helpers ---

    def _slot_to_dict(self, s: ItinerarySlot) -> dict:
        return {
            "slot": s.slot,
            "item_id": s.item_id,
            "name": s.name,
            "price": round(s.price, 2),
            "status": s.status,
            "meta": dict(s.meta),
        }

    def _flight_to_summary(self, f: Flight) -> dict:
        return {
            "id": f.id,
            "origin": f.origin, "dest": f.dest,
            "depart": f.depart_iso, "arrive": f.arrive_iso,
            "stops": f.stops, "airline": f.airline, "cabin": f.cabin,
            "price": round(f.price, 2),
            "overnight": f.overnight,
        }

    def _hotel_to_summary(self, h: Hotel) -> dict:
        return {
            "id": h.id, "city": h.city, "name": h.name,
            "neighborhood": h.neighborhood, "stars": h.stars,
            "price_per_night": round(h.price_per_night, 2),
            "amenities": list(h.amenities),
        }

    def _activity_to_summary(self, a: Activity) -> dict:
        return {
            "id": a.id, "city": a.city, "name": a.name,
            "category": a.category, "price": round(a.price, 2),
            "duration_hours": a.duration_hours,
        }

    # --- Observation rendering ---

    def _render_obs(self, last_result: dict | None) -> dict:
        """Three-layer observation:
            1. static       : request text + tool catalogue (constant within episode)
            2. state        : itinerary + budget + clock + pending events
            3. last_result  : varies by the tool that just fired
          + history (older search results collapsed to summaries to bound token growth)

        Note: `done` is intentionally NOT included here — that's what
        terminated/truncated in the step return tuple are for. The rollout
        helper can compose them however it likes.
        """
        prof = self._profile
        hard = prof.hard if prof else None
        cap = hard.budget_cap if hard else 0.0
        return {
            "request": self._request_text,
            "tools": _tool_catalogue(),
            "state": {
                "itinerary": [self._slot_to_dict(s) for s in self._itinerary],
                "budget": {
                    "cap": cap,
                    "spent": round(self._spent, 2),
                    "remaining": round(cap - self._spent, 2),
                },
                "clock": {
                    "turn": self._turn,
                    "turns_remaining": max(0, self._max_turns - self._turn),
                    "depart_date": hard.depart_date if hard else None,
                    "return_date": hard.return_date if hard else None,
                },
                "pending_events": [
                    {"type": e.type, "payload": dict(e.payload),
                     "fires_at_turn": e.fires_at_turn}
                    for e in self._pending_events
                ],
            },
            "last_result": last_result,
            "history": self._pruned_history(),
        }

    def _pruned_history(self) -> list[dict]:
        """Keep the most recent `full_window` turns in full; for older turns,
        collapse search-result lists down to one-line summaries. Bookings,
        feedback, and other non-search results stay in full because they're
        cheap and the agent needs them for context."""
        full_window = 3
        out: list[dict] = []
        n = len(self._history)
        for i, h in enumerate(self._history):
            if i >= n - full_window:
                out.append(h)
                continue
            result = h.get("result", {})
            tool = result.get("tool")
            if tool in ("search_flights", "search_hotels", "search_activities"):
                pruned = {
                    "turn": h["turn"],
                    "action": h["action"],
                    "result": {
                        "ok": result.get("ok"),
                        "tool": tool,
                        "summary": f"returned {len(result.get('results', []))} results",
                    },
                }
                out.append(pruned)
            else:
                out.append(h)
        return out


# --- Module-level helpers -------------------------------------------------

def _action_from_input(action: dict | Action) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, dict):
        return Action(tool=action.get("tool", ""), args=dict(action.get("args", {})))
    raise TypeError(f"action must be dict or Action, got {type(action).__name__}")


def _clean_args(args: dict) -> dict:
    """Drop keys with None values so kwargs forwarding to search_* doesn't
    override defaults with None (which most of the world.py search APIs
    intentionally treat as 'no filter')."""
    return {k: v for k, v in args.items() if v is not None}


def _tool_catalogue() -> list[dict]:
    return [
        {"name": name, "description": spec["description"], "args": spec["schema"]}
        for name, spec in TOOL_REGISTRY.items()
    ]


def _item_to_full_dict(item: Any) -> dict:
    out: dict[str, Any] = {"kind": type(item).__name__.lower()}
    for k in vars(item):
        v = getattr(item, k)
        if isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


def _search_envelope(tool: str, results: list[dict]) -> dict:
    return {"ok": True, "tool": tool, "count": len(results), "results": results}


def _nights_between(checkin_iso: str, checkout_iso: str) -> int:
    a = date.fromisoformat(checkin_iso)
    b = date.fromisoformat(checkout_iso)
    return max(1, (b - a).days)


def _breakdown_to_dict(b: RewardBreakdown) -> dict:
    return {
        "total": b.total,
        "hard_constraint_gate": b.hard_constraint_gate,
        "preference_coverage": b.preference_coverage,
        "budget_efficiency": b.budget_efficiency,
        "coherence": b.coherence,
        "recovery_quality": b.recovery_quality,
        "steps": b.steps,
    }


def _weights_to_dict(w: RewardWeights) -> dict:
    return {
        "preference": w.preference,
        "budget": w.budget,
        "coherence": w.coherence,
        "recovery": w.recovery,
        "step_penalty": w.step_penalty,
    }
