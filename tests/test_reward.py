"""reward.py — components + composition + exploit defenses.

The README flagged reward as "the hardest part" so the test suite leans
heavily here. Every exploit defense documented in notes.md has at least
one test asserting the defense holds.
"""

from __future__ import annotations

import pytest

from travel_env.persona import sample_profile
from travel_env.reward import (
    DEFAULT_WEIGHTS,
    RewardWeights,
    budget_efficiency,
    coherence,
    compose,
    hard_constraint_gate,
    preference_coverage,
    recovery_quality,
    score_episode,
)


# --- Helpers -------------------------------------------------------------

def make_slot(slot, item_id, name, price, status, **meta):
    return {
        "slot": slot, "item_id": item_id, "name": name,
        "price": price, "status": status,
        "meta": {"kind": meta.pop("kind"), **meta},
    }


def make_complete_itinerary(profile):
    """A coherent minimal trip: outbound + return + hotel, all booked."""
    origin = profile.origin_city
    dest = profile.dest_city
    out = make_slot(
        "outbound_flight", "flt_o", "Out", 500, "booked",
        kind="flight", origin=origin, dest=dest,
        depart_iso=profile.hard.depart_date + "T10:00",
        arrive_iso=profile.hard.depart_date + "T20:00",
        stops=0, cabin=profile.soft_prefs.get("flight_cabin", "economy"),
        overnight=False,
    )
    ret = make_slot(
        "return_flight", "flt_r", "Ret", 500, "booked",
        kind="flight", origin=dest, dest=origin,
        depart_iso=profile.hard.return_date + "T10:00",
        arrive_iso=profile.hard.return_date + "T20:00",
        stops=0, cabin=profile.soft_prefs.get("flight_cabin", "economy"),
        overnight=False,
    )
    htl = make_slot(
        "hotel", "htl_h", "Hotel", 1000, "booked",
        kind="hotel", city=dest, neighborhood="Center",
        stars=profile.soft_prefs.get("hotel_stars", 3),
        price_per_night=200,
        amenities=list(profile.hard.required_amenities) + ["wifi"],
        nights=5,
    )
    return [out, ret, htl]


# --- hard_constraint_gate ------------------------------------------------

def test_gate_zero_for_empty_itinerary():
    p = sample_profile(42)
    assert hard_constraint_gate([], p, 0) == 0.0


def test_gate_zero_for_budget_overshoot():
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    spent = p.hard.budget_cap * 1.5
    assert hard_constraint_gate(it, p, spent) == 0.0


def test_gate_one_for_minimal_coherent_trip():
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    spent = sum(s["price"] for s in it)
    if spent > p.hard.budget_cap:
        pytest.skip("synthetic spend exceeds cap for this seed")
    assert hard_constraint_gate(it, p, spent) == 1.0


def test_gate_zero_on_overnight_for_no_overnight_persona():
    for seed in range(50):
        p = sample_profile(seed)
        if p.hard.no_overnight_flights:
            break
    else:
        pytest.skip("no no_overnight persona in 50 seeds")
    it = make_complete_itinerary(p)
    it[0]["meta"]["overnight"] = True
    assert hard_constraint_gate(it, p, 1000) == 0.0


def test_gate_zero_on_amenity_violation():
    for seed in range(50):
        p = sample_profile(seed)
        if p.hard.required_amenities:
            break
    else:
        pytest.skip("no required_amenities persona in 50 seeds")
    it = make_complete_itinerary(p)
    it[2]["meta"]["amenities"] = []  # strip amenities
    assert hard_constraint_gate(it, p, 1000) == 0.0


# --- budget_efficiency ---------------------------------------------------

def test_budget_efficiency_peaks_at_midpoint():
    val = budget_efficiency(0.75 * 1000, 1000, low_ratio=0.55, high_ratio=0.95)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_budget_efficiency_partial_credit_below_low_bound():
    # The asymmetric bell ramps linearly from 0 (spent=0) up to 1.0 at the
    # midpoint, so spending exactly at low_ratio is partway up the ramp —
    # not flat-zero like the old symmetric bell.
    val = budget_efficiency(0.55 * 1000, 1000, low_ratio=0.55, high_ratio=0.95)
    midpoint = (0.55 + 0.95) / 2.0
    assert val == pytest.approx(0.55 / midpoint, abs=1e-6)


def test_budget_efficiency_zero_at_hard_cap():
    # The high side decays quadratically from the midpoint, reaching 0 only at
    # the hard cap (ratio=1.0) — not at high_ratio. The hard gate handles
    # everything past the cap, so this is the natural place to bottom out.
    assert budget_efficiency(1000, 1000, low_ratio=0.55, high_ratio=0.95) == pytest.approx(0.0, abs=1e-6)


def test_budget_efficiency_monotone_around_midpoint():
    # Linear ramp on the low side, quadratic decay on the high side — both
    # strictly monotone toward the peak at the midpoint (ratio=0.75 here).
    f = budget_efficiency
    assert f(750, 1000) > f(700, 1000) > f(600, 1000)
    assert f(750, 1000) > f(800, 1000) > f(900, 1000)


def test_budget_efficiency_handles_zero_cap():
    assert budget_efficiency(0, 0) == 0.0


# --- preference_coverage -------------------------------------------------

def test_preference_coverage_bounded():
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    score = preference_coverage(it, p)
    assert 0.0 <= score <= 1.0


def test_preference_coverage_higher_with_matching_activities():
    p = sample_profile(42)
    # Find top activity category for this persona
    top_cat = max(
        ((k[4:], v) for k, v in p.soft_prefs.items() if k.startswith("act_")),
        key=lambda x: x[1],
    )[0]
    base_it = make_complete_itinerary(p)
    enriched = base_it + [make_slot(
        f"act_{i}", f"act_{i}", "Activity", 80, "booked",
        kind="activity", city=p.dest_city, category=top_cat, duration_hours=2,
    ) for i in range(2)]
    assert preference_coverage(enriched, p) > preference_coverage(base_it, p)


# --- coherence -----------------------------------------------------------

def test_coherence_full_for_complete_trip():
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    assert coherence(it, p) == pytest.approx(1.0)


def test_coherence_partial_for_missing_return():
    p = sample_profile(42)
    it = make_complete_itinerary(p)[:2]  # drop return
    it = [it[0], it[1]] if len(it) == 2 else it  # outbound + hotel
    # Actually drop just the return flight
    p_it = [s for s in make_complete_itinerary(p) if s["slot"] != "return_flight"]
    score = coherence(p_it, p)
    assert 0 < score < 1


def test_coherence_zero_for_empty():
    p = sample_profile(42)
    assert coherence([], p) == 0.0


# --- recovery_quality ----------------------------------------------------

def test_recovery_quality_none_without_disruption():
    assert recovery_quality(None) is None
    assert recovery_quality({}) is None


def test_recovery_quality_zero_when_not_rebooked():
    """Disruption fired but agent never booked a replacement → 0."""
    p = sample_profile(42)
    pre = make_complete_itinerary(p)
    post = [dict(s) for s in pre]
    post[0]["status"] = "cancelled"  # outbound got cancelled
    info = {
        "fired_at_turn": 5, "final_turn": 15,
        "pre_itinerary": pre, "post_itinerary": post,
        "original_flight_price": 500,
    }
    assert recovery_quality(info) == 0.0


def test_recovery_quality_positive_when_rebooked():
    """Disruption fired and agent rebooked at similar price → > 0."""
    p = sample_profile(42)
    pre = make_complete_itinerary(p)
    post = [dict(s) for s in pre]
    post[0] = dict(post[0])
    post[0]["status"] = "cancelled"
    # Replacement on same route, similar price, ~1 turn later
    replacement = make_slot(
        "outbound_flight_rebook", "flt_r2", "Rebook", 550, "booked",
        kind="flight", origin=p.origin_city, dest=p.dest_city,
        depart_iso=p.hard.depart_date + "T11:00",
        arrive_iso=p.hard.depart_date + "T21:00",
        stops=0, cabin="economy", overnight=False,
    )
    post.append(replacement)
    info = {
        "fired_at_turn": 8, "final_turn": 10,
        "pre_itinerary": pre, "post_itinerary": post,
        "original_flight_price": 500,
    }
    score = recovery_quality(info)
    assert score is not None and score > 0.5


# --- compose -------------------------------------------------------------

def test_compose_renormalizes_when_no_disruption():
    """Without a disruption, the three remaining weights should sum to 1.0
    inside the gated bracket — so a perfect non-disruption trip can still
    hit total ≈ 1.0 (minus step penalty)."""
    total, extras = compose(
        gate=1.0, pref=1.0, budget=1.0, coh=1.0, recovery=None, steps=5,
        weights=DEFAULT_WEIGHTS,
    )
    assert extras["renormalized_no_disruption"] is True
    assert total > 0.99


def test_compose_uses_recovery_when_present():
    total_with_recovery, _ = compose(
        gate=1.0, pref=1.0, budget=1.0, coh=1.0, recovery=1.0, steps=5,
        weights=DEFAULT_WEIGHTS,
    )
    total_no_recovery, _ = compose(
        gate=1.0, pref=1.0, budget=1.0, coh=1.0, recovery=0.0, steps=5,
        weights=DEFAULT_WEIGHTS,
    )
    # When recovery exists and = 1.0, total should be ~1.0.
    # When recovery = 0.0, total should be significantly lower (we lose w_recovery).
    assert total_with_recovery > total_no_recovery + 0.1


def test_compose_gate_zero_zeros_soft_sum():
    total, _ = compose(
        gate=0.0, pref=1.0, budget=1.0, coh=1.0, recovery=1.0, steps=5,
        weights=DEFAULT_WEIGHTS,
    )
    assert total <= 0.0  # only the step penalty remains


def test_step_penalty_kicks_in_after_free_turns():
    """First N turns are free; penalty grows linearly after."""
    w = DEFAULT_WEIGHTS
    free = w.step_penalty_free_turns
    t1, _ = compose(gate=1, pref=1, budget=1, coh=1, recovery=None, steps=free, weights=w)
    t2, _ = compose(gate=1, pref=1, budget=1, coh=1, recovery=None, steps=free + 10, weights=w)
    assert t1 > t2  # more turns => more penalty


# --- score_episode (top-level) -------------------------------------------

def test_score_episode_handles_minimal_trip():
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    spent = sum(s["price"] for s in it)
    b = score_episode(itinerary=it, profile=p, spent=spent, steps=10)
    assert 0 <= b.preference_coverage <= 1
    assert 0 <= b.budget_efficiency <= 1
    assert 0 <= b.coherence <= 1
    assert b.recovery_quality is None  # no disruption
    assert isinstance(b.total, float)


# --- Exploit defenses (the headline tests) -------------------------------

def test_exploit_cheapest_overnight_gated_for_no_overnight_persona():
    """The 'always cheapest' exploit: overnight + multi-stop. Should be gated."""
    for seed in range(50):
        p = sample_profile(seed)
        if p.hard.no_overnight_flights:
            break
    else:
        pytest.skip("no no_overnight persona")
    it = make_complete_itinerary(p)
    it[0]["meta"]["overnight"] = True
    it[0]["price"] = 100  # super cheap
    b = score_episode(itinerary=it, profile=p, spent=1500, steps=10)
    assert b.hard_constraint_gate == 0.0
    assert b.total <= 0.0


def test_exploit_luxury_hotel_overshoot_gated():
    """5-star hotel that blows the budget should fail the gate."""
    p = sample_profile(42)
    it = make_complete_itinerary(p)
    it[2]["price"] = p.hard.budget_cap * 2  # blow it
    spent = sum(s["price"] for s in it)
    b = score_episode(itinerary=it, profile=p, spent=spent, steps=10)
    assert b.hard_constraint_gate == 0.0


def test_exploit_finish_in_one_step_gives_zero():
    """Submit with nothing booked → gate=0 → total ≤ 0."""
    p = sample_profile(42)
    b = score_episode(itinerary=[], profile=p, spent=0, steps=1)
    assert b.hard_constraint_gate == 0.0
    assert b.total <= 0.0
