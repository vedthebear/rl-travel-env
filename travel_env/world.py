"""Synthetic world: cities, flights, hotels, activities, disruptions.

All sampling is deterministic given the episode seed via numpy.random.Generator.
Real city names give the reviewer something concrete; inventory is fully synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# --- Static city table (hardcoded, ~15 cities) ----------------------------

@dataclass(frozen=True)
class City:
    name: str
    iata: str
    lat: float
    lon: float
    neighborhoods: tuple[str, ...]
    cost_multiplier: float  # 1.0 = baseline; Tokyo/NYC > 1, Bangkok < 1


CITIES: dict[str, City] = {}  # populated in world.py implementation


# --- Inventory items ------------------------------------------------------

@dataclass
class Flight:
    id: str
    origin: str
    dest: str
    depart_iso: str          # e.g. "2026-03-15T08:30"
    arrive_iso: str
    stops: int
    airline: str
    cabin: Literal["economy", "premium", "business"]
    price: float
    overnight: bool          # red-eye? used by persona hard prefs


@dataclass
class Hotel:
    id: str
    city: str
    name: str
    neighborhood: str
    stars: int               # 1..5
    price_per_night: float
    amenities: tuple[str, ...]  # e.g. ("wifi", "gym", "breakfast")


@dataclass
class Activity:
    id: str
    city: str
    name: str
    category: Literal["food", "history", "nature", "nightlife", "family"]
    price: float
    duration_hours: float


# --- Disruption events ----------------------------------------------------

@dataclass
class PendingEvent:
    type: Literal["flight_cancelled"]
    payload: dict
    fires_at_turn: int


# --- World handle ---------------------------------------------------------

@dataclass
class World:
    """Holds the deterministic RNG and any cached per-episode inventory state."""
    rng: np.random.Generator
    booked_flight_ids: set[str] = field(default_factory=set)
    booked_hotel_ids: set[str] = field(default_factory=set)


def make_world(seed: int) -> World:
    """Create a fresh world with a seeded RNG."""
    raise NotImplementedError


# --- Search APIs (called by the env's tool dispatch) ----------------------

def search_flights(
    world: World,
    *,
    origin: str,
    dest: str,
    depart_date: str,
    return_date: str | None = None,
    max_stops: int | None = None,
    max_price: float | None = None,
    k: int = 8,
) -> list[Flight]:
    """Generate top-k candidate flights for the route on the requested date(s)."""
    raise NotImplementedError


def search_hotels(
    world: World,
    *,
    city: str,
    checkin: str,
    checkout: str,
    neighborhood: str | None = None,
    min_stars: int | None = None,
    max_price: float | None = None,
    k: int = 8,
) -> list[Hotel]:
    """Generate top-k candidate hotels in the city for the date range."""
    raise NotImplementedError


def search_activities(
    world: World,
    *,
    city: str,
    date: str | None = None,
    categories: list[str] | None = None,
    max_price: float | None = None,
    k: int = 8,
) -> list[Activity]:
    """Generate top-k candidate activities in the city, optionally filtered."""
    raise NotImplementedError


def get_details(world: World, item_id: str) -> Flight | Hotel | Activity | None:
    """Look up the full record for any search-result id."""
    raise NotImplementedError


# --- Disruption sampler ---------------------------------------------------

def maybe_schedule_disruption(
    world: World,
    flight: Flight,
    *,
    current_turn: int,
    base_cancel_rate: float = 0.15,
) -> PendingEvent | None:
    """After booking a flight, roll for cancellation; if hit, schedule the event."""
    raise NotImplementedError
