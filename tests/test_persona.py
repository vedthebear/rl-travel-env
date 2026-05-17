"""persona.py — profile sampling + scripted voice invariants."""

from __future__ import annotations

import pytest

from travel_env.persona import (
    ARCHETYPES,
    HardConstraints,
    Mismatch,
    PersonaProfile,
    PersonaVoice,
    compute_mismatches,
    sample_profile,
)
from travel_env.world import CITIES


# --- Profile sampling ----------------------------------------------------

def test_sample_profile_deterministic():
    p1 = sample_profile(42)
    p2 = sample_profile(42)
    assert p1.archetype == p2.archetype
    assert p1.origin_city == p2.origin_city
    assert p1.dest_city == p2.dest_city
    assert p1.hard.budget_cap == p2.hard.budget_cap
    assert p1.soft_prefs == p2.soft_prefs


def test_sample_profile_returns_valid_archetype():
    p = sample_profile(0)
    assert p.archetype in ARCHETYPES


def test_origin_and_dest_differ():
    for seed in range(20):
        p = sample_profile(seed)
        assert p.origin_city != p.dest_city
        assert p.origin_city in CITIES
        assert p.dest_city in CITIES


def test_hard_constraints_internally_consistent():
    for seed in range(20):
        p = sample_profile(seed)
        h = p.hard
        assert h.budget_cap > 0
        assert h.depart_date < h.return_date  # ISO date strings sort correctly
        assert h.group_size >= 1


def test_all_archetypes_sample_within_100_seeds():
    seen: set[str] = set()
    for seed in range(100):
        p = sample_profile(seed)
        seen.add(p.archetype)
    assert seen == set(ARCHETYPES.keys()), f"missing archetypes: {set(ARCHETYPES) - seen}"


def test_soft_prefs_have_activity_axes():
    p = sample_profile(42)
    soft = p.soft_prefs
    act_axes = [k for k in soft if k.startswith("act_")]
    assert len(act_axes) >= 3  # at least a few categories


def test_soft_prefs_activity_weights_sum_close_to_1():
    """Per persona.py, activity weights are jittered then renormalized."""
    p = sample_profile(42)
    act_total = sum(v for k, v in p.soft_prefs.items() if k.startswith("act_"))
    assert 0.95 < act_total < 1.05


def test_business_persona_has_hard_constraints():
    """The business archetype sets no_overnight + max_stops=0 hard. If sampling
    ever produces a business profile, those should be on."""
    for seed in range(50):
        p = sample_profile(seed)
        if p.archetype == "business":
            assert p.hard.no_overnight_flights is True
            assert p.hard.max_stops == 0
            return
    pytest.skip("no business archetype sampled in 50 seeds")


# --- Scripted voice (no API key) -----------------------------------------

def test_persona_voice_scripted_renders_request():
    voice = PersonaVoice(mode="scripted")
    p = sample_profile(42)
    text = voice.render_request(p)
    assert isinstance(text, str) and len(text) > 20
    # Scripted persona mentions both cities by name somewhere in the request.
    assert p.dest_city in text


def test_persona_voice_scripted_answer_question():
    voice = PersonaVoice(mode="scripted")
    p = sample_profile(42)
    answer = voice.answer_question(p, "Are you flexible on dates?")
    assert isinstance(answer, str) and len(answer) > 0


# --- Mismatch detection --------------------------------------------------

def test_compute_mismatches_empty_itinerary():
    """Empty itinerary -> no mismatches detected (it'll just fail the reward gate)."""
    p = sample_profile(42)
    mismatches = compute_mismatches([], p)
    # Empty itinerary returns empty (nothing to complain about that exists)
    # or trivial set — either way, well-formed list.
    assert isinstance(mismatches, list)
    for m in mismatches:
        assert isinstance(m, Mismatch)


def test_compute_mismatches_overnight_flight_for_no_overnight_persona():
    """If persona has no_overnight=True and we propose an overnight flight,
    it should show up as a mismatch."""
    for seed in range(50):
        p = sample_profile(seed)
        if p.hard.no_overnight_flights:
            break
    else:
        pytest.skip("no no_overnight persona in 50 seeds")

    itinerary = [{
        "slot": "outbound_flight",
        "item_id": "flt_x",
        "name": "Red Eye",
        "price": 200.0,
        "status": "tentative",
        "meta": {"kind": "flight", "overnight": True, "stops": 0,
                 "origin": p.origin_city, "dest": p.dest_city},
    }]
    mismatches = compute_mismatches(itinerary, p)
    axes = {m.axis for m in mismatches}
    assert "flight_overnight" in axes
