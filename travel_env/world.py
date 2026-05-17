"""Synthetic world: cities, flights, hotels, activities, disruptions.

All sampling is deterministic given the episode seed via numpy.random.Generator.
Real city names give the reviewer something concrete; inventory is fully synthetic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal

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


CITIES: dict[str, City] = {
    "Tokyo": City(
        "Tokyo", "NRT", 35.7720, 140.3929,
        ("Shibuya", "Shinjuku", "Ginza", "Asakusa", "Roppongi"),
        1.30,
    ),
    "Paris": City(
        "Paris", "CDG", 49.0097, 2.5479,
        ("Le Marais", "Saint-Germain", "Montmartre", "Latin Quarter", "Pigalle"),
        1.20,
    ),
    "New York": City(
        "New York", "JFK", 40.6413, -73.7781,
        ("SoHo", "Midtown", "Williamsburg", "Upper East Side", "Harlem"),
        1.40,
    ),
    "London": City(
        "London", "LHR", 51.4700, -0.4543,
        ("Soho", "Shoreditch", "Camden", "Notting Hill", "Mayfair"),
        1.30,
    ),
    "Bangkok": City(
        "Bangkok", "BKK", 13.6900, 100.7501,
        ("Sukhumvit", "Silom", "Khao San", "Chinatown", "Thonglor"),
        0.50,
    ),
    "Rome": City(
        "Rome", "FCO", 41.8003, 12.2389,
        ("Trastevere", "Centro Storico", "Monti", "Testaccio", "Prati"),
        1.05,
    ),
    "Barcelona": City(
        "Barcelona", "BCN", 41.2974, 2.0833,
        ("El Born", "Gracia", "Barceloneta", "Eixample", "Gothic Quarter"),
        1.00,
    ),
    "Sydney": City(
        "Sydney", "SYD", -33.9399, 151.1753,
        ("Bondi", "Surry Hills", "Newtown", "The Rocks", "Manly"),
        1.20,
    ),
    "Lisbon": City(
        "Lisbon", "LIS", 38.7813, -9.1359,
        ("Alfama", "Bairro Alto", "Chiado", "Belem", "Principe Real"),
        0.85,
    ),
    "Reykjavik": City(
        "Reykjavik", "KEF", 63.9850, -22.6056,
        ("Laugavegur", "Old Harbour", "Vesturbaer", "Hlemmur"),
        1.45,
    ),
    "Mexico City": City(
        "Mexico City", "MEX", 19.4361, -99.0719,
        ("Roma Norte", "Condesa", "Polanco", "Coyoacan", "Centro"),
        0.60,
    ),
    "Cape Town": City(
        "Cape Town", "CPT", -33.9690, 18.6017,
        ("City Bowl", "Sea Point", "Camps Bay", "Woodstock", "V&A Waterfront"),
        0.70,
    ),
    "Istanbul": City(
        "Istanbul", "IST", 41.2753, 28.7519,
        ("Beyoglu", "Sultanahmet", "Kadikoy", "Karakoy", "Besiktas"),
        0.65,
    ),
    "Vancouver": City(
        "Vancouver", "YVR", 49.1967, -123.1815,
        ("Gastown", "Yaletown", "Kitsilano", "Mount Pleasant", "Commercial Drive"),
        1.15,
    ),
    "Singapore": City(
        "Singapore", "SIN", 1.3644, 103.9915,
        ("Tiong Bahru", "Chinatown", "Kampong Glam", "Orchard", "Tanjong Pagar"),
        1.25,
    ),
    "San Francisco": City(
        "San Francisco", "SFO", 37.6213, -122.3790,
        ("Mission", "SoMa", "Hayes Valley", "North Beach", "Castro"),
        1.45,
    ),
}


# IATA -> city-name index, for routes specified as IATA codes.
_IATA_TO_CITY: dict[str, str] = {c.iata: name for name, c in CITIES.items()}


def _resolve_city(key: str) -> tuple[str, City]:
    """Accept either a city name or an IATA code; return (name, City)."""
    if key in CITIES:
        return key, CITIES[key]
    if key in _IATA_TO_CITY:
        name = _IATA_TO_CITY[key]
        return name, CITIES[name]
    raise KeyError(f"unknown city or IATA code: {key!r}")


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
    # Caches: search-arg fingerprint -> result list (for idempotence).
    _search_cache: dict[str, list] = field(default_factory=dict)
    # Per-city procedural hotel pools (lazy).
    _hotel_pools: dict[str, list[Hotel]] = field(default_factory=dict)
    # Per-city activity pools (lazy).
    _activity_pools: dict[str, list[Activity]] = field(default_factory=dict)
    # Inventory registry for get_details lookups.
    _inventory: dict[str, Any] = field(default_factory=dict)


def make_world(seed: int) -> World:
    """Create a fresh world with a seeded RNG."""
    return World(rng=np.random.default_rng(seed))


# --- Internal helpers -----------------------------------------------------

_AIRLINES = (
    "Aeris", "BlueArc", "Cirrus", "Delta-Star", "Equinox",
    "Falcon", "Globe", "Helix", "Iris", "Junction",
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def _cache_key(tool: str, args: dict) -> str:
    # Normalize: drop Nones, sort keys, freeze list values.
    norm = []
    for k in sorted(args.keys()):
        v = args[k]
        if v is None:
            continue
        if isinstance(v, list):
            v = tuple(v)
        norm.append((k, v))
    return f"{tool}:{_stable_hash(tuple(norm))}"


def _seeded_rng_for(world: World, *parts: Any) -> np.random.Generator:
    """Derive a sub-RNG from the world seed + a key. Used so cached searches
    don't depend on global RNG draw order between unrelated calls."""
    # Mix world's RNG state into the key so different seeds give different sub-streams.
    # We extract one int from world.rng's bit_generator state for stable per-world salt.
    state = world.rng.bit_generator.state
    # default_rng's state under PCG64 holds 'state' -> {'state': int, 'inc': int}
    inner = state.get("state", {})
    salt = inner.get("state", 0) if isinstance(inner, dict) else 0
    h = _stable_hash(salt, *parts)
    seed_int = int(h[:16], 16)
    return np.random.default_rng(seed_int)


# --- Flights --------------------------------------------------------------

def _time_mult(hour: int) -> float:
    if 6 <= hour <= 9 or 17 <= hour <= 19:
        return 1.15
    if hour >= 22 or hour <= 5:
        return 0.85
    return 1.0


_CABIN_MULT = {"economy": 1.0, "premium": 1.8, "business": 3.5}


def _is_overnight(depart_hour: int, total_hours: float) -> bool:
    # True if any segment of the flight (a sample of hours along the way)
    # falls into 22:00-05:00. We approximate by checking depart, mid, arrive.
    hours = [
        depart_hour % 24,
        (depart_hour + total_hours / 2) % 24,
        (depart_hour + total_hours) % 24,
    ]
    for h in hours:
        if h >= 22 or h < 5:
            return True
    return False


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
    args = {
        "origin": origin, "dest": dest, "depart_date": depart_date,
        "return_date": return_date, "max_stops": max_stops,
        "max_price": max_price, "k": k,
    }
    key = _cache_key("search_flights", args)
    if key in world._search_cache:
        return world._search_cache[key]

    o_name, o_city = _resolve_city(origin)
    d_name, d_city = _resolve_city(dest)
    distance_km = _haversine_km(o_city.lat, o_city.lon, d_city.lat, d_city.lon)
    great_circle_hours = max(1.0, distance_km / 800.0)  # ~800 km/h cruise

    # Advance days. Cheap parse; assumes ISO dates. Anchor to a fixed "today" so
    # repeat searches with same args are stable across calls.
    advance_days = _days_between("2026-01-01", depart_date)

    rng = _seeded_rng_for(
        world, "search_flights", origin, dest, depart_date, return_date,
        max_stops, max_price, k,
    )

    base_per_km = 0.10
    n_candidates = int(rng.integers(25, 41))
    candidates: list[Flight] = []

    for i in range(n_candidates):
        noise_p = float(rng.uniform(-0.18, 0.18))
        # Anti-correlated quality noise: when price is below average (noise_p < 0),
        # we push quality choices toward the worse end (more stops, worse hours).
        # This breaks the dominated-option pattern.
        q_skew = -noise_p  # in [-0.18, 0.18], opposite sign to price

        # Stops: bias toward more stops when q_skew > 0 (cheap option -> more stops).
        stops_roll = float(rng.uniform()) + q_skew * 2.0
        if stops_roll < 0.55:
            stops = 0
        elif stops_roll < 0.90:
            stops = 1
        else:
            stops = 2

        # Cabin: cheap options bias to economy; pricey options bias to higher cabins.
        cabin_roll = float(rng.uniform()) - q_skew * 1.5
        if cabin_roll < 0.70:
            cabin = "economy"
        elif cabin_roll < 0.92:
            cabin = "premium"
        else:
            cabin = "business"

        # Depart hour: cheap options bias to red-eye; expensive bias to prime.
        hour_roll = float(rng.uniform()) - q_skew * 1.2
        if hour_roll < 0.35:
            depart_hour = int(rng.choice([22, 23, 0, 1, 5]))
        elif hour_roll < 0.75:
            depart_hour = int(rng.choice([6, 7, 8, 9, 17, 18, 19]))
        else:
            depart_hour = int(rng.choice([10, 11, 12, 13, 14, 15, 16, 20, 21]))

        depart_minute = int(rng.choice([0, 15, 30, 45]))
        total_hours = great_circle_hours + stops * 1.5

        t_mult = _time_mult(depart_hour)
        adv_mult = 1.0 + max(0.0, (14 - advance_days) / 14.0) * 0.6
        stops_mult = 1.0 - 0.20 * stops
        cab_mult = _CABIN_MULT[cabin]
        price = (
            base_per_km * distance_km * t_mult * adv_mult * stops_mult * cab_mult
            * (1.0 + noise_p)
        )
        price = max(50.0, round(price, 2))

        airline = _AIRLINES[int(rng.integers(0, len(_AIRLINES)))]
        flight_no = int(rng.integers(100, 9999))

        depart_iso = f"{depart_date}T{depart_hour:02d}:{depart_minute:02d}"
        arrive_iso = _add_hours_iso(depart_iso, total_hours)
        overnight = _is_overnight(depart_hour, total_hours)

        id_hash = _stable_hash("flt", o_city.iata, d_city.iata, depart_date, i,
                               stops, cabin, depart_hour, depart_minute, airline,
                               flight_no, key)
        fid = f"flt_{id_hash[:10]}"

        f = Flight(
            id=fid,
            origin=o_city.iata,
            dest=d_city.iata,
            depart_iso=depart_iso,
            arrive_iso=arrive_iso,
            stops=stops,
            airline=f"{airline} {flight_no}",
            cabin=cabin,
            price=price,
            overnight=overnight,
        )
        candidates.append(f)

    # Filters.
    if max_stops is not None:
        candidates = [c for c in candidates if c.stops <= max_stops]
    if max_price is not None:
        candidates = [c for c in candidates if c.price <= max_price]

    # Top-k selection targets a Pareto frontier on (price, stops, cabin-rank, overnight)
    # rather than pure price. Sort the candidate pool by price, then greedily keep
    # any flight not dominated by something already in `top` — this preserves the
    # "no dominated options" property even when the cheapest n are all overnight/economy.
    candidates.sort(key=lambda x: x.price)
    _cabin_rank = {"economy": 0, "premium": 1, "business": 2}

    def _axes(f: Flight) -> tuple[float, int, int, int]:
        return (f.price, f.stops, -_cabin_rank[f.cabin], int(f.overnight))

    def _dominates(a: Flight, b: Flight) -> bool:
        ax_a, ax_b = _axes(a), _axes(b)
        return all(x <= y for x, y in zip(ax_a, ax_b)) and any(x < y for x, y in zip(ax_a, ax_b))

    pareto: list[Flight] = []
    for c in candidates:
        if any(_dominates(p, c) for p in pareto):
            continue
        pareto.append(c)
        if len(pareto) >= k:
            break
    # If filters were tight and we have fewer than k Pareto candidates, fall back
    # to the next-cheapest items to fill out the list (sortable, ranked by price).
    if len(pareto) < k:
        for c in candidates:
            if c in pareto:
                continue
            pareto.append(c)
            if len(pareto) >= k:
                break

    top = sorted(pareto[:k], key=lambda x: x.price)
    for f in top:
        world._inventory[f.id] = f
    world._search_cache[key] = top
    return top


# --- Hotels ---------------------------------------------------------------

_HOTEL_PREFIXES = (
    "Grand", "Royal", "Park", "Plaza", "Garden", "Sky", "Harbor",
    "Central", "Old Town", "Imperial", "Velvet", "Lantern",
)
_HOTEL_SUFFIXES = (
    "Hotel", "Inn", "Suites", "Lodge", "House", "Residences",
)


_TIER_AMENITIES = {
    5: ("wifi", "gym", "breakfast", "pool", "spa", "concierge"),
    4: ("wifi", "gym", "breakfast", "pool"),
    3: ("wifi", "breakfast"),
    2: ("wifi",),
    1: (),
}


def _build_hotel_pool(world: World, city_name: str, city: City) -> list[Hotel]:
    rng = _seeded_rng_for(world, "hotel_pool", city.iata)
    n = int(rng.integers(25, 41))

    # Per-neighborhood premium: pick 1-2 premium and 1 budget per city, others 1.0.
    nbhds = list(city.neighborhoods)
    n_premium = 2 if len(nbhds) >= 4 else 1
    perm = rng.permutation(len(nbhds))
    premium_idx = set(perm[:n_premium].tolist())
    budget_idx = int(perm[-1])
    nbhd_premium: dict[str, float] = {}
    for i, nb in enumerate(nbhds):
        if i in premium_idx:
            nbhd_premium[nb] = 1.5
        elif i == budget_idx and i not in premium_idx:
            nbhd_premium[nb] = 0.8
        else:
            nbhd_premium[nb] = 1.0

    star_weights = np.array([0.05, 0.15, 0.35, 0.30, 0.15])
    pool: list[Hotel] = []
    for i in range(n):
        stars = int(rng.choice([1, 2, 3, 4, 5], p=star_weights))
        nb = nbhds[int(rng.integers(0, len(nbhds)))]
        prem = nbhd_premium[nb]
        noise = float(rng.uniform(-0.12, 0.12))
        price = city.cost_multiplier * (40 + 35 * stars * stars) * prem * (1.0 + noise)
        price = round(max(25.0, price), 2)

        # Amenity bag: start with tier's defaults, then add a couple of randoms
        # from the next-higher tier with some probability (a 3-star with a pool exists).
        amenities = set(_TIER_AMENITIES[stars])
        if stars < 5:
            for extra in _TIER_AMENITIES[stars + 1]:
                if extra not in amenities and rng.uniform() < 0.20:
                    amenities.add(extra)

        prefix = _HOTEL_PREFIXES[int(rng.integers(0, len(_HOTEL_PREFIXES)))]
        suffix = _HOTEL_SUFFIXES[int(rng.integers(0, len(_HOTEL_SUFFIXES)))]
        name = f"{prefix} {nb} {suffix}"

        id_hash = _stable_hash("htl", city.iata, i, stars, nb, prefix, suffix)
        hid = f"htl_{id_hash[:10]}"

        h = Hotel(
            id=hid,
            city=city_name,
            name=name,
            neighborhood=nb,
            stars=stars,
            price_per_night=price,
            amenities=tuple(sorted(amenities)),
        )
        pool.append(h)
        world._inventory[h.id] = h
    return pool


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
    args = {
        "city": city, "checkin": checkin, "checkout": checkout,
        "neighborhood": neighborhood, "min_stars": min_stars,
        "max_price": max_price, "k": k,
    }
    key = _cache_key("search_hotels", args)
    if key in world._search_cache:
        return world._search_cache[key]

    name, city_obj = _resolve_city(city)
    if name not in world._hotel_pools:
        world._hotel_pools[name] = _build_hotel_pool(world, name, city_obj)
    pool = world._hotel_pools[name]

    out = list(pool)
    if neighborhood is not None:
        out = [h for h in out if h.neighborhood.lower() == neighborhood.lower()]
    if min_stars is not None:
        out = [h for h in out if h.stars >= min_stars]
    if max_price is not None:
        out = [h for h in out if h.price_per_night <= max_price]

    # Rank by best-value (stars per dollar). Stable order on ties via id.
    out.sort(key=lambda h: (-(h.stars / max(1.0, h.price_per_night)), h.id))
    top = out[:k]
    world._search_cache[key] = top
    return top


# --- Activities -----------------------------------------------------------

_CATEGORY_BASE = {"food": 80.0, "history": 30.0, "nature": 50.0,
                  "nightlife": 60.0, "family": 40.0}
_CATEGORY_DUR = {"food": 3.0, "history": 2.5, "nature": 3.5,
                 "nightlife": 4.0, "family": 3.0}


# Hand-curated anchors for the most-recognized cities. Less-iconic cities get
# generic procedural fill only — that's fine for v1.
_NAMED_ACTIVITIES: dict[str, list[tuple[str, str]]] = {
    "Tokyo": [
        ("Senso-ji Temple visit", "history"),
        ("Tsukiji Outer Market food tour", "food"),
        ("Yoyogi Park stroll", "nature"),
        ("Shibuya nightlife crawl", "nightlife"),
        ("teamLab Planets", "family"),
    ],
    "Paris": [
        ("Louvre Museum", "history"),
        ("Le Marais food tour", "food"),
        ("Bois de Boulogne walk", "nature"),
        ("Pigalle nightlife tour", "nightlife"),
        ("Disneyland Paris", "family"),
    ],
    "New York": [
        ("Metropolitan Museum of Art", "history"),
        ("Greenwich Village food tour", "food"),
        ("Central Park bike loop", "nature"),
        ("Lower East Side bar crawl", "nightlife"),
        ("American Museum of Natural History", "family"),
    ],
    "London": [
        ("British Museum", "history"),
        ("Borough Market food tour", "food"),
        ("Hampstead Heath walk", "nature"),
        ("Shoreditch pub crawl", "nightlife"),
        ("Tower of London", "family"),
    ],
    "Rome": [
        ("Colosseum and Forum tour", "history"),
        ("Trastevere food tour", "food"),
        ("Villa Borghese gardens", "nature"),
        ("Campo de Fiori nightlife", "nightlife"),
        ("Vatican Museums", "family"),
    ],
    "Barcelona": [
        ("Sagrada Familia", "history"),
        ("Tapas tour in El Born", "food"),
        ("Parc Guell", "nature"),
        ("Gothic Quarter bar crawl", "nightlife"),
        ("Barcelona Aquarium", "family"),
    ],
    "Bangkok": [
        ("Grand Palace tour", "history"),
        ("Street food tour in Chinatown", "food"),
        ("Lumpini Park visit", "nature"),
        ("Khao San Road nightlife", "nightlife"),
        ("Safari World", "family"),
    ],
    "Sydney": [
        ("Sydney Opera House tour", "history"),
        ("Surry Hills food walk", "food"),
        ("Bondi to Coogee coastal walk", "nature"),
        ("The Rocks pub crawl", "nightlife"),
        ("Taronga Zoo", "family"),
    ],
    "Lisbon": [
        ("Jeronimos Monastery", "history"),
        ("Time Out Market food tour", "food"),
        ("Sintra day hike", "nature"),
        ("Bairro Alto bar crawl", "nightlife"),
        ("Oceanario de Lisboa", "family"),
    ],
    "Reykjavik": [
        ("National Museum of Iceland", "history"),
        ("Icelandic food tour", "food"),
        ("Golden Circle tour", "nature"),
        ("Laugavegur bar crawl", "nightlife"),
        ("Whales of Iceland exhibition", "family"),
    ],
    "Mexico City": [
        ("Teotihuacan pyramids", "history"),
        ("Roma Norte taco crawl", "food"),
        ("Chapultepec Park", "nature"),
        ("Condesa cantina hop", "nightlife"),
        ("Papalote Children's Museum", "family"),
    ],
    "Cape Town": [
        ("Robben Island tour", "history"),
        ("Bo-Kaap cooking class", "food"),
        ("Table Mountain hike", "nature"),
        ("Long Street bar crawl", "nightlife"),
        ("Two Oceans Aquarium", "family"),
    ],
    "Istanbul": [
        ("Hagia Sophia", "history"),
        ("Karakoy food tour", "food"),
        ("Bosphorus ferry cruise", "nature"),
        ("Beyoglu nightlife crawl", "nightlife"),
        ("Miniaturk", "family"),
    ],
    "Vancouver": [
        ("Museum of Anthropology", "history"),
        ("Granville Island food tour", "food"),
        ("Stanley Park seawall", "nature"),
        ("Gastown pub crawl", "nightlife"),
        ("Vancouver Aquarium", "family"),
    ],
    "Singapore": [
        ("National Museum of Singapore", "history"),
        ("Hawker centre food tour", "food"),
        ("Gardens by the Bay", "nature"),
        ("Clarke Quay bar crawl", "nightlife"),
        ("Singapore Zoo", "family"),
    ],
    "San Francisco": [
        ("Alcatraz Island tour", "history"),
        ("Mission taqueria crawl", "food"),
        ("Golden Gate Park bike loop", "nature"),
        ("North Beach bar crawl", "nightlife"),
        ("Exploratorium", "family"),
    ],
}


def _build_activity_pool(world: World, city_name: str, city: City) -> list[Activity]:
    rng = _seeded_rng_for(world, "activity_pool", city.iata)
    pool: list[Activity] = []

    # 1) Hand-curated anchors (or generic stubs if missing).
    named = _NAMED_ACTIVITIES.get(city_name, [])
    if not named:
        # Generic per-category stubs so every city has at least one of each.
        named = [
            (f"{city_name} historic walking tour", "history"),
            (f"{city_name} street food sampler", "food"),
            (f"{city_name} parks loop", "nature"),
            (f"{city_name} nightlife sampler", "nightlife"),
            (f"{city_name} family day out", "family"),
        ]
    for i, (nm, cat) in enumerate(named):
        noise = float(rng.uniform(-0.15, 0.15))
        price = _CATEGORY_BASE[cat] * city.cost_multiplier * (1.0 + noise)
        price = round(max(5.0, price), 2)
        dur = _CATEGORY_DUR[cat] + float(rng.uniform(-0.5, 0.5))
        id_hash = _stable_hash("act", city.iata, "named", i, nm, cat)
        aid = f"act_{id_hash[:10]}"
        a = Activity(
            id=aid, city=city_name, name=nm, category=cat,
            price=price, duration_hours=round(dur, 2),
        )
        pool.append(a)
        world._inventory[a.id] = a

    # 2) Procedural fill (5-10 extras).
    n_extra = int(rng.integers(5, 11))
    nbhds = list(city.neighborhoods)
    templates = [
        ("Walking tour of {nb}", "history"),
        ("Cooking class in {nb}", "food"),
        ("Food market visit in {nb}", "food"),
        ("Bike ride through {nb}", "nature"),
        ("Wine bar evening in {nb}", "nightlife"),
        ("Live music night in {nb}", "nightlife"),
        ("Family-friendly tour of {nb}", "family"),
        ("Museum visit in {nb}", "history"),
        ("Garden visit near {nb}", "nature"),
    ]
    for i in range(n_extra):
        tmpl, cat = templates[int(rng.integers(0, len(templates)))]
        nb = nbhds[int(rng.integers(0, len(nbhds)))]
        nm = tmpl.format(nb=nb)
        noise = float(rng.uniform(-0.15, 0.15))
        price = _CATEGORY_BASE[cat] * city.cost_multiplier * (1.0 + noise)
        price = round(max(5.0, price), 2)
        dur = _CATEGORY_DUR[cat] + float(rng.uniform(-0.5, 0.5))
        id_hash = _stable_hash("act", city.iata, "proc", i, nm, cat)
        aid = f"act_{id_hash[:10]}"
        a = Activity(
            id=aid, city=city_name, name=nm, category=cat,
            price=price, duration_hours=round(dur, 2),
        )
        pool.append(a)
        world._inventory[a.id] = a

    return pool


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
    args = {
        "city": city, "date": date, "categories": categories,
        "max_price": max_price, "k": k,
    }
    key = _cache_key("search_activities", args)
    if key in world._search_cache:
        return world._search_cache[key]

    name, city_obj = _resolve_city(city)
    if name not in world._activity_pools:
        world._activity_pools[name] = _build_activity_pool(world, name, city_obj)
    pool = world._activity_pools[name]

    out = list(pool)
    if categories:
        cat_set = {c.lower() for c in categories}
        out = [a for a in out if a.category in cat_set]
    if max_price is not None:
        out = [a for a in out if a.price <= max_price]

    # Rank for category diversity: interleave per-category by price, then
    # truncate to k. This way a foodie's max_price=200 still surfaces a mix.
    by_cat: dict[str, list[Activity]] = {}
    for a in out:
        by_cat.setdefault(a.category, []).append(a)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: (x.price, x.id))
    interleaved: list[Activity] = []
    ordered_cats = sorted(by_cat.keys())
    while ordered_cats:
        for cat in list(ordered_cats):
            if by_cat[cat]:
                interleaved.append(by_cat[cat].pop(0))
            else:
                ordered_cats.remove(cat)
    top = interleaved[:k]
    world._search_cache[key] = top
    return top


# --- get_details ----------------------------------------------------------

def get_details(world: World, item_id: str) -> Flight | Hotel | Activity | None:
    """Look up the full record for any search-result id."""
    return world._inventory.get(item_id)


# --- Disruption sampler ---------------------------------------------------

def maybe_schedule_disruption(
    world: World,
    flight: Flight,
    *,
    current_turn: int,
    base_cancel_rate: float = 0.15,
) -> PendingEvent | None:
    """After booking a flight, roll for cancellation; if hit, schedule the event."""
    world.booked_flight_ids.add(flight.id)

    depart_hour = int(flight.depart_iso[11:13])
    early_morning = depart_hour < 7
    p = base_cancel_rate
    if early_morning:
        p *= 1.3
    if flight.stops > 0:
        p *= 1.2
    p = min(1.0, p)

    if float(world.rng.uniform()) >= p:
        return None

    offset = int(world.rng.integers(1, 5))
    return PendingEvent(
        type="flight_cancelled",
        payload={
            "flight_id": flight.id,
            "original_price": flight.price,
            "route": (flight.origin, flight.dest),
        },
        fires_at_turn=current_turn + offset,
    )


# --- Date helpers (tiny, no external deps) -------------------------------

def _days_between(d1: str, d2: str) -> int:
    """Return calendar days from d1 to d2 (YYYY-MM-DD). Negative if d2 before d1."""
    import datetime as _dt
    a = _dt.date.fromisoformat(d1)
    b = _dt.date.fromisoformat(d2)
    return (b - a).days


def _add_hours_iso(iso: str, hours: float) -> str:
    """Add fractional hours to an ISO 'YYYY-MM-DDTHH:MM' string; return same format."""
    import datetime as _dt
    base = _dt.datetime.fromisoformat(iso)
    delta = _dt.timedelta(minutes=int(round(hours * 60)))
    out = base + delta
    return out.strftime("%Y-%m-%dT%H:%M")
