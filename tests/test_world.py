"""world.py — synthetic generator invariants.

The world is the foundation. If it's not deterministic + idempotent, every
downstream test becomes flaky.
"""

from __future__ import annotations

import pytest

from travel_env.world import (
    Activity,
    CITIES,
    Flight,
    Hotel,
    get_details,
    make_world,
    maybe_schedule_disruption,
    search_activities,
    search_flights,
    search_hotels,
)


# --- CITIES table sanity --------------------------------------------------

def test_cities_have_required_fields():
    assert len(CITIES) >= 15
    iatas = set()
    for name, city in CITIES.items():
        assert city.name == name
        assert len(city.iata) == 3 and city.iata.isupper()
        assert -90 <= city.lat <= 90
        assert -180 <= city.lon <= 180
        assert len(city.neighborhoods) >= 3
        assert 0.3 <= city.cost_multiplier <= 2.0
        assert city.iata not in iatas, f"duplicate IATA {city.iata}"
        iatas.add(city.iata)


# --- Determinism + idempotence -------------------------------------------

def test_make_world_deterministic_seed():
    w1 = make_world(42)
    w2 = make_world(42)
    f1 = search_flights(w1, origin="JFK", dest="NRT", depart_date="2026-04-10")
    f2 = search_flights(w2, origin="JFK", dest="NRT", depart_date="2026-04-10")
    assert [f.id for f in f1] == [f.id for f in f2]
    assert [f.price for f in f1] == [f.price for f in f2]


def test_make_world_different_seeds_different_inventory():
    w1 = make_world(1)
    w2 = make_world(2)
    f1 = search_flights(w1, origin="JFK", dest="NRT", depart_date="2026-04-10")
    f2 = search_flights(w2, origin="JFK", dest="NRT", depart_date="2026-04-10")
    # Prices should differ; deterministic generators with different seeds shouldn't collide.
    assert [f.price for f in f1] != [f.price for f in f2]


def test_search_flights_idempotent():
    w = make_world(7)
    a = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10")
    b = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10")
    assert [f.id for f in a] == [f.id for f in b]


def test_search_hotels_idempotent():
    w = make_world(7)
    a = search_hotels(w, city="Tokyo", checkin="2026-04-10", checkout="2026-04-15")
    b = search_hotels(w, city="Tokyo", checkin="2026-04-10", checkout="2026-04-15")
    assert [h.id for h in a] == [h.id for h in b]


def test_search_activities_idempotent():
    w = make_world(7)
    a = search_activities(w, city="Tokyo")
    b = search_activities(w, city="Tokyo")
    assert [x.id for x in a] == [x.id for x in b]


# --- Search filters work --------------------------------------------------

def test_search_flights_max_stops_filter():
    w = make_world(42)
    direct = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10", max_stops=0)
    assert direct, "no direct flights returned"
    assert all(f.stops == 0 for f in direct)


def test_search_flights_max_price_filter():
    w = make_world(42)
    cheap = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10", max_price=1500)
    assert cheap
    assert all(f.price <= 1500 for f in cheap)


def test_search_hotels_min_stars_filter():
    w = make_world(42)
    lux = search_hotels(w, city="Tokyo", checkin="2026-04-10", checkout="2026-04-15", min_stars=4)
    assert lux
    assert all(h.stars >= 4 for h in lux)


def test_search_hotels_neighborhood_filter():
    w = make_world(42)
    target = "Shibuya"
    rs = search_hotels(w, city="Tokyo", checkin="2026-04-10", checkout="2026-04-15", neighborhood=target)
    assert rs
    assert all(h.neighborhood == target for h in rs)


def test_search_activities_categories_filter():
    w = make_world(42)
    food = search_activities(w, city="Tokyo", categories=["food"])
    assert food
    assert all(a.category == "food" for a in food)


# --- Item registry --------------------------------------------------------

def test_get_details_round_trip():
    w = make_world(42)
    flights = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10")
    for f in flights:
        d = get_details(w, f.id)
        assert isinstance(d, Flight)
        assert d.id == f.id


def test_get_details_unknown_returns_none():
    w = make_world(42)
    assert get_details(w, "flt_does_not_exist") is None


def test_item_ids_have_kind_prefix():
    w = make_world(42)
    f = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-10")[0]
    h = search_hotels(w, city="Tokyo", checkin="2026-04-10", checkout="2026-04-15")[0]
    a = search_activities(w, city="Tokyo")[0]
    assert f.id.startswith("flt_")
    assert h.id.startswith("htl_")
    assert a.id.startswith("act_")


# --- Disruption sampler ---------------------------------------------------

def test_disruption_distribution_in_expected_range():
    """p ≈ 0.15 base; over 100 trials, expect 5-30 fires on a non-early non-stop flight."""
    fires = 0
    n = 100
    for seed in range(n):
        w = make_world(seed)
        flights = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-15")
        # Pick a daytime direct flight (low-risk profile)
        candidates = [f for f in flights if f.stops == 0 and "T0" not in f.depart_iso[:13]]
        if not candidates:
            continue
        ev = maybe_schedule_disruption(w, candidates[0], current_turn=5)
        if ev is not None:
            fires += 1
    # Generous range — Bernoulli p=0.15 has std ≈ 3.6 over n=100, so [5, 30] is ~3σ envelope
    assert 5 <= fires <= 30, f"disruption fires {fires}/100 outside expected envelope"


def test_disruption_event_well_formed():
    """When a disruption fires, the PendingEvent has the right shape."""
    seen_event = False
    for seed in range(50):
        w = make_world(seed)
        flights = search_flights(w, origin="JFK", dest="NRT", depart_date="2026-04-15")
        ev = maybe_schedule_disruption(w, flights[0], current_turn=5)
        if ev is not None:
            assert ev.type == "flight_cancelled"
            assert "flight_id" in ev.payload
            assert ev.payload["flight_id"] == flights[0].id
            assert ev.fires_at_turn > 5
            seen_event = True
            break
    assert seen_event, "no disruption fired across 50 seeds — generator may be broken"
