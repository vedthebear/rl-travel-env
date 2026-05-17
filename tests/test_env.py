"""env.py — TravelEnv reset/step contract + tool dispatch + state transitions."""

from __future__ import annotations

import pytest

from travel_env.env import TOOL_REGISTRY, TravelEnv


# --- reset / step contract -----------------------------------------------

def test_reset_returns_obs_info_tuple():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert "request" in obs and "tools" in obs and "state" in obs


def test_reset_state_shape():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    state = obs["state"]
    assert "itinerary" in state
    assert "budget" in state
    assert "clock" in state
    assert "pending_events" in state
    assert state["itinerary"] == []
    assert state["budget"]["spent"] == 0
    assert state["clock"]["turn"] == 0


def test_reset_info_has_archetype_and_weights():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    _, info = env.reset()
    assert info["seed"] == 42
    assert info["persona_mode"] == "scripted"
    assert "archetype" in info
    assert "weights" in info


def test_clock_exposes_trip_and_constraints():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    clock = obs["state"]["clock"]
    # Trip params publicly visible (baselines/agents need them).
    assert clock["origin"] is not None
    assert clock["dest"] is not None
    assert clock["depart_date"] is not None
    assert clock["return_date"] is not None
    assert clock["group_size"] >= 1
    # Hard constraints exposed (publicly stated by the client in the request).
    assert "no_overnight_flights" in clock
    assert "max_stops" in clock
    assert "required_amenities" in clock


def test_step_returns_5_tuple():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    ret = env.step({"tool": "search_flights", "args": {
        "origin": obs["state"]["clock"]["origin"],
        "dest": obs["state"]["clock"]["dest"],
        "depart_date": obs["state"]["clock"]["depart_date"],
    }})
    assert len(ret) == 5
    obs2, reward, terminated, truncated, info = ret
    assert isinstance(obs2, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_on_finished_episode_raises():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    env.reset()
    env.step({"tool": "submit_final", "args": {}})
    with pytest.raises(RuntimeError):
        env.step({"tool": "search_flights", "args": {}})


# --- Tool dispatch -------------------------------------------------------

def test_all_registered_tools_dispatched():
    """Every tool in TOOL_REGISTRY must have a handler."""
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    assert set(env._handlers.keys()) == set(TOOL_REGISTRY.keys())


def test_unknown_tool_returns_error_envelope():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    env.reset()
    obs, _, _, _, _ = env.step({"tool": "do_the_thing", "args": {}})
    lr = obs["last_result"]
    assert lr["ok"] is False
    assert "unknown tool" in lr["error"]


def test_malformed_args_returned_as_tool_error_not_crash():
    """Even garbage args should be caught and returned as ok=False, not crash."""
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    env.reset()
    obs, _, _, _, _ = env.step({"tool": "search_flights", "args": {"origin": "ZZZ", "dest": "QQQ", "depart_date": "garbage"}})
    lr = obs["last_result"]
    assert lr["ok"] is False
    assert "error" in lr


# --- Itinerary state transitions -----------------------------------------

def test_add_to_itinerary_creates_tentative_slot():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step({"tool": "search_flights", "args": {
        "origin": obs["state"]["clock"]["origin"],
        "dest": obs["state"]["clock"]["dest"],
        "depart_date": obs["state"]["clock"]["depart_date"],
    }})
    fid = obs["last_result"]["results"][0]["id"]
    obs, _, _, _, _ = env.step({"tool": "add_to_itinerary", "args": {"item_id": fid, "slot": "outbound_flight"}})
    items = obs["state"]["itinerary"]
    assert len(items) == 1
    assert items[0]["status"] == "tentative"
    assert items[0]["item_id"] == fid


def test_book_transitions_tentative_to_booked_and_debits():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step({"tool": "search_flights", "args": {
        "origin": obs["state"]["clock"]["origin"],
        "dest": obs["state"]["clock"]["dest"],
        "depart_date": obs["state"]["clock"]["depart_date"],
    }})
    fid = obs["last_result"]["results"][0]["id"]
    obs, _, _, _, _ = env.step({"tool": "add_to_itinerary", "args": {"item_id": fid, "slot": "outbound_flight"}})
    before_spent = obs["state"]["budget"]["spent"]
    obs, _, _, _, _ = env.step({"tool": "book", "args": {"item_ids": [fid]}})
    items = obs["state"]["itinerary"]
    assert items[0]["status"] == "booked"
    assert obs["state"]["budget"]["spent"] > before_spent


def test_book_rejected_on_budget_overshoot():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    # Force budget cap absurdly low so any flight overshoots.
    env._profile.hard.budget_cap = 1.0
    obs, _, _, _, _ = env.step({"tool": "search_flights", "args": {
        "origin": obs["state"]["clock"]["origin"],
        "dest": obs["state"]["clock"]["dest"],
        "depart_date": obs["state"]["clock"]["depart_date"],
    }})
    fid = obs["last_result"]["results"][0]["id"]
    env.step({"tool": "add_to_itinerary", "args": {"item_id": fid, "slot": "x"}})
    obs, _, _, _, _ = env.step({"tool": "book", "args": {"item_ids": [fid]}})
    lr = obs["last_result"]
    assert lr["ok"] is False
    assert "budget" in lr["error"].lower()
    # And state must not have been mutated.
    assert all(s["status"] != "booked" for s in obs["state"]["itinerary"])


def test_remove_from_itinerary_refunds_booked():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step({"tool": "search_flights", "args": {
        "origin": obs["state"]["clock"]["origin"],
        "dest": obs["state"]["clock"]["dest"],
        "depart_date": obs["state"]["clock"]["depart_date"],
    }})
    fid = obs["last_result"]["results"][0]["id"]
    env.step({"tool": "add_to_itinerary", "args": {"item_id": fid, "slot": "x"}})
    env.step({"tool": "book", "args": {"item_ids": [fid]}})
    spent_before = obs["state"]["budget"]["spent"]  # captured after add, before book
    obs, _, _, _, _ = env.step({"tool": "remove_from_itinerary", "args": {"slot": "x"}})
    # After remove: refund happens, spent back to pre-book level (0 here).
    assert obs["state"]["budget"]["spent"] == 0


# --- Termination / truncation --------------------------------------------

def test_submit_final_terminates_episode():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    env.reset()
    obs, reward, terminated, truncated, info = env.step({"tool": "submit_final", "args": {}})
    assert terminated is True
    assert truncated is False
    assert info["reward_breakdown"] is not None


def test_truncation_at_max_turns():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=3)
    env.reset()
    # Take 3 no-op-ish steps; the third should truncate.
    for _ in range(2):
        obs, r, term, trunc, _ = env.step({"tool": "get_details", "args": {"item_id": "x"}})
        assert not (term or trunc)
    obs, r, term, trunc, info = env.step({"tool": "get_details", "args": {"item_id": "x"}})
    assert truncated_or_terminated(term, trunc)
    assert info["reward_breakdown"] is not None  # finalized on truncation too


def truncated_or_terminated(term, trunc):
    return term or trunc


def test_empty_itinerary_submit_gives_zero_reward():
    env = TravelEnv(seed=42, persona_mode="scripted", max_turns=20)
    env.reset()
    _, reward, _, _, info = env.step({"tool": "submit_final", "args": {}})
    b = info["reward_breakdown"]
    assert b["hard_constraint_gate"] == 0
    assert reward <= 0  # at worst small step penalty; gate=0 zeros soft sum
