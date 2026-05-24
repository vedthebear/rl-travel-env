"""Static data tables for the synthetic travel world.

Pure data: no functions, no RNG, no logic. Pulled out of world.py so the
generator/search code stays focused on logic, and the dataset can grow
without churning the module that consumes it.

Everything here is hardcoded and generated. The procedural generators in world.py
read from these tables and combine them with seeded noise to produce the
per-episode inventory the agent searches over. This is NOT the final dataset the agent will see.

`CITIES` lives here as the single source of truth for the static city table.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- City schema ----------------------------------------------------------

@dataclass(frozen=True)
class City:
    name: str
    iata: str
    lat: float
    lon: float
    neighborhoods: tuple[str, ...]
    cost_multiplier: float  # 1.0 = baseline; Tokyo/NYC > 1, Bangkok/Marrakech < 1


# --- Cities (~23 real cities) ---------------------------------------------
# Coordinates are airport-anchored so haversine distance produces flight-route-y
# times. Cost multipliers reflect 4-star-equivalent ballparks so a $40k trip
# looks absurd in Marrakech and a $1k trip looks tight in Reykjavik.

CITIES: dict[str, City] = {
    "Tokyo": City(
        "Tokyo", "NRT", 35.7720, 140.3929,
        ("Shibuya", "Shinjuku", "Ginza", "Asakusa", "Roppongi"), 1.30,
    ),
    "Paris": City(
        "Paris", "CDG", 49.0097, 2.5479,
        ("Le Marais", "Saint-Germain", "Montmartre", "Latin Quarter", "Pigalle"), 1.20,
    ),
    "New York": City(
        "New York", "JFK", 40.6413, -73.7781,
        ("SoHo", "Midtown", "Williamsburg", "Upper East Side", "Harlem"), 1.40,
    ),
    "London": City(
        "London", "LHR", 51.4700, -0.4543,
        ("Soho", "Shoreditch", "Camden", "Notting Hill", "Mayfair"), 1.30,
    ),
    "Bangkok": City(
        "Bangkok", "BKK", 13.6900, 100.7501,
        ("Sukhumvit", "Silom", "Khao San", "Chinatown", "Thonglor"), 0.50,
    ),
    "Rome": City(
        "Rome", "FCO", 41.8003, 12.2389,
        ("Trastevere", "Centro Storico", "Monti", "Testaccio", "Prati"), 1.05,
    ),
    "Barcelona": City(
        "Barcelona", "BCN", 41.2974, 2.0833,
        ("El Born", "Gracia", "Barceloneta", "Eixample", "Gothic Quarter"), 1.00,
    ),
    "Sydney": City(
        "Sydney", "SYD", -33.9399, 151.1753,
        ("Bondi", "Surry Hills", "Newtown", "The Rocks", "Manly"), 1.20,
    ),
    "Lisbon": City(
        "Lisbon", "LIS", 38.7813, -9.1359,
        ("Alfama", "Bairro Alto", "Chiado", "Belem", "Principe Real"), 0.85,
    ),
    "Reykjavik": City(
        "Reykjavik", "KEF", 63.9850, -22.6056,
        ("Laugavegur", "Old Harbour", "Vesturbaer", "Hlemmur"), 1.45,
    ),
    "Mexico City": City(
        "Mexico City", "MEX", 19.4361, -99.0719,
        ("Roma Norte", "Condesa", "Polanco", "Coyoacan", "Centro"), 0.60,
    ),
    "Cape Town": City(
        "Cape Town", "CPT", -33.9690, 18.6017,
        ("City Bowl", "Sea Point", "Camps Bay", "Woodstock", "V&A Waterfront"), 0.70,
    ),
    "Istanbul": City(
        "Istanbul", "IST", 41.2753, 28.7519,
        ("Beyoglu", "Sultanahmet", "Kadikoy", "Karakoy", "Besiktas"), 0.65,
    ),
    "Vancouver": City(
        "Vancouver", "YVR", 49.1967, -123.1815,
        ("Gastown", "Yaletown", "Kitsilano", "Mount Pleasant", "Commercial Drive"), 1.15,
    ),
    "Singapore": City(
        "Singapore", "SIN", 1.3644, 103.9915,
        ("Tiong Bahru", "Chinatown", "Kampong Glam", "Orchard", "Tanjong Pagar"), 1.25,
    ),
    "San Francisco": City(
        "San Francisco", "SFO", 37.6213, -122.3790,
        ("Mission", "SoMa", "Hayes Valley", "North Beach", "Castro"), 1.45,
    ),
    "Berlin": City(
        "Berlin", "BER", 52.3667, 13.5033,
        ("Kreuzberg", "Mitte", "Prenzlauer Berg", "Friedrichshain", "Charlottenburg"), 1.05,
    ),
    "Dubai": City(
        "Dubai", "DXB", 25.2528, 55.3644,
        ("Downtown", "Marina", "Jumeirah", "Deira", "Al Quoz"), 1.50,
    ),
    "Buenos Aires": City(
        "Buenos Aires", "EZE", -34.8222, -58.5358,
        ("Palermo", "Recoleta", "San Telmo", "Belgrano", "La Boca"), 0.55,
    ),
    "Seoul": City(
        "Seoul", "ICN", 37.4602, 126.4407,
        ("Gangnam", "Hongdae", "Itaewon", "Insadong", "Myeongdong"), 1.10,
    ),
    "Marrakech": City(
        "Marrakech", "RAK", 31.6069, -8.0363,
        ("Medina", "Gueliz", "Hivernage", "Kasbah", "Palmeraie"), 0.50,
    ),
    "Amsterdam": City(
        "Amsterdam", "AMS", 52.3105, 4.7683,
        ("Jordaan", "De Pijp", "Oud-Zuid", "Oost", "Noord"), 1.25,
    ),
    "Athens": City(
        "Athens", "ATH", 37.9364, 23.9445,
        ("Plaka", "Kolonaki", "Psiri", "Koukaki", "Exarchia"), 0.85,
    ),
}


# --- Airlines (synthetic; 20 carriers) ------------------------------------
# All made-up. Names are deliberately neutral so a real airline brand never
# accidentally appears in agent-visible search results.

AIRLINES: tuple[str, ...] = (
    "Aeris", "BlueArc", "Cirrus", "Delta-Star", "Equinox",
    "Falcon", "Globe", "Helix", "Iris", "Junction",
    "Kestrel", "Lumen", "Meridian", "Nova", "Orion",
    "Polaris", "Quill", "Strata", "Vector", "Zenith",
)


# --- Hotel naming, stratified by star tier --------------------------------
# Cheap hotels should feel cheap, luxury hotels should feel luxe. With a single
# flat prefix/suffix list, we'd produce "Backpacker Mayfair Residences" and
# similar nonsense. Stratifying by tier keeps the vibe consistent at both
# extremes; tier 3 (modal) overlaps with neighbors to avoid an artificial gap.

HOTEL_PREFIXES_BY_TIER: dict[int, tuple[str, ...]] = {
    5: ("The Imperial", "The Grand", "Palace", "The Royal", "The St. Regis",
        "The Mandarin", "Aurelius", "Belvedere"),
    4: ("Grand", "Park", "Plaza", "Garden", "Harbor", "Royal", "The Capital",
        "Lantern"),
    3: ("Central", "Park", "Lantern", "Old Town", "Harbor", "The Capital"),
    2: ("Sky", "Velvet", "Central", "Old Town"),
    1: ("Easy", "Sleep", "Backpacker", "Budget", "City"),
}

HOTEL_SUFFIXES_BY_TIER: dict[int, tuple[str, ...]] = {
    5: ("Hotel", "Residences", "Suites", "Palace"),
    4: ("Hotel", "Suites", "Inn", "Residences"),
    3: ("Hotel", "Inn", "House"),
    2: ("Inn", "House", "Lodge"),
    1: ("Hostel", "Inn", "Lodge"),
}


# --- Hotel amenities by tier ---------------------------------------------
# Each tier's default amenity bag. _build_hotel_pool also sprinkles 1-2
# next-tier amenities on top with low probability (a 3-star with a pool
# exists in real life). Amenity strings here are the same lowercase tokens
# the persona's required_amenities lists target — keep them in sync.

TIER_AMENITIES: dict[int, tuple[str, ...]] = {
    5: ("wifi", "gym", "breakfast", "pool", "spa", "concierge",
        "room_service", "minibar"),
    4: ("wifi", "gym", "breakfast", "pool", "room_service"),
    3: ("wifi", "breakfast", "gym"),
    2: ("wifi", "breakfast"),
    1: ("wifi",),
}


# --- Activity pricing / duration baselines --------------------------------

CATEGORY_BASE: dict[str, float] = {
    "food": 80.0,
    "history": 30.0,
    "nature": 50.0,
    "nightlife": 60.0,
    "family": 40.0,
}

CATEGORY_DUR: dict[str, float] = {
    "food": 3.0,
    "history": 2.5,
    "nature": 3.5,
    "nightlife": 4.0,
    "family": 3.0,
}


# --- Named activity anchors (hand-curated, ~10-11 per city) --------------
# Each entry: (display_name, category). Mixed real-world references and
# composed-but-plausible names. Categories are balanced (≥2 per category for
# most cities) but leaning into the city's actual strengths — Reykjavik gets
# extra nature, Tokyo and Dubai get extra family, etc. Procedural fill
# (5–10 templated extras per city) is added on top in world.py.

NAMED_ACTIVITIES: dict[str, list[tuple[str, str]]] = {
    "Tokyo": [
        ("Senso-ji Temple visit", "history"),
        ("Meiji Shrine walk", "history"),
        ("Tokyo National Museum", "history"),
        ("Tsukiji Outer Market food tour", "food"),
        ("Ramen-making class in Shinjuku", "food"),
        ("Yoyogi Park stroll", "nature"),
        ("Mt Takao day hike", "nature"),
        ("Shibuya nightlife crawl", "nightlife"),
        ("Golden Gai bar tour", "nightlife"),
        ("teamLab Planets", "family"),
        ("Studio Ghibli Museum", "family"),
    ],
    "Paris": [
        ("Louvre Museum", "history"),
        ("Musee d'Orsay", "history"),
        ("Notre-Dame walking tour", "history"),
        ("Le Marais food tour", "food"),
        ("Saint-Germain pastry crawl", "food"),
        ("Bois de Boulogne walk", "nature"),
        ("Luxembourg Gardens", "nature"),
        ("Pigalle nightlife tour", "nightlife"),
        ("Latin Quarter wine bar tour", "nightlife"),
        ("Disneyland Paris", "family"),
        ("Jardin d'Acclimatation", "family"),
    ],
    "New York": [
        ("Metropolitan Museum of Art", "history"),
        ("9/11 Memorial and Museum", "history"),
        ("Statue of Liberty tour", "history"),
        ("Greenwich Village food tour", "food"),
        ("Lower East Side deli crawl", "food"),
        ("Central Park bike loop", "nature"),
        ("High Line walk", "nature"),
        ("Lower East Side bar crawl", "nightlife"),
        ("Broadway show", "nightlife"),
        ("American Museum of Natural History", "family"),
        ("Bronx Zoo", "family"),
    ],
    "London": [
        ("British Museum", "history"),
        ("Westminster Abbey", "history"),
        ("Borough Market food tour", "food"),
        ("East End curry crawl", "food"),
        ("Hampstead Heath walk", "nature"),
        ("Kew Gardens", "nature"),
        ("Shoreditch pub crawl", "nightlife"),
        ("Soho cocktail tour", "nightlife"),
        ("Tower of London", "family"),
        ("London Eye", "family"),
        ("Natural History Museum", "family"),
    ],
    "Bangkok": [
        ("Grand Palace tour", "history"),
        ("Wat Pho temple visit", "history"),
        ("Street food tour in Chinatown", "food"),
        ("Floating market tour", "food"),
        ("Lumpini Park visit", "nature"),
        ("Chao Phraya river cruise", "nature"),
        ("Khao San Road nightlife", "nightlife"),
        ("Sukhumvit rooftop bar crawl", "nightlife"),
        ("Safari World", "family"),
        ("Siam Park family day", "family"),
    ],
    "Rome": [
        ("Colosseum and Forum tour", "history"),
        ("Pantheon visit", "history"),
        ("Vatican Museums", "history"),
        ("Trastevere food tour", "food"),
        ("Testaccio market crawl", "food"),
        ("Villa Borghese gardens", "nature"),
        ("Appian Way bike tour", "nature"),
        ("Campo de Fiori nightlife", "nightlife"),
        ("Trastevere wine bar crawl", "nightlife"),
        ("Cinecitta family tour", "family"),
        ("Bioparco di Roma", "family"),
    ],
    "Barcelona": [
        ("Sagrada Familia", "history"),
        ("Picasso Museum", "history"),
        ("Gothic Quarter walking tour", "history"),
        ("Tapas tour in El Born", "food"),
        ("Boqueria market food walk", "food"),
        ("Parc Guell", "nature"),
        ("Montjuic hike", "nature"),
        ("Gothic Quarter bar crawl", "nightlife"),
        ("El Born cocktail tour", "nightlife"),
        ("Barcelona Aquarium", "family"),
        ("Tibidabo amusement park", "family"),
    ],
    "Sydney": [
        ("Sydney Opera House tour", "history"),
        ("Hyde Park Barracks", "history"),
        ("Surry Hills food walk", "food"),
        ("Newtown food crawl", "food"),
        ("Bondi to Coogee coastal walk", "nature"),
        ("Royal Botanic Garden", "nature"),
        ("The Rocks pub crawl", "nightlife"),
        ("Kings Cross bar tour", "nightlife"),
        ("Taronga Zoo", "family"),
        ("Sydney Aquarium", "family"),
        ("Manly ferry day", "family"),
    ],
    "Lisbon": [
        ("Jeronimos Monastery", "history"),
        ("Castelo de Sao Jorge", "history"),
        ("Time Out Market food tour", "food"),
        ("Alfama pastel de nata crawl", "food"),
        ("Sintra day hike", "nature"),
        ("Belem riverside walk", "nature"),
        ("Bairro Alto bar crawl", "nightlife"),
        ("Pink Street nightlife tour", "nightlife"),
        ("Oceanario de Lisboa", "family"),
        ("Lisbon Zoo", "family"),
    ],
    "Reykjavik": [
        ("National Museum of Iceland", "history"),
        ("Saga Museum", "history"),
        ("Icelandic food tour", "food"),
        ("Reykjavik bakery crawl", "food"),
        ("Golden Circle tour", "nature"),
        ("Blue Lagoon visit", "nature"),
        ("South Coast waterfalls tour", "nature"),
        ("Northern Lights night tour", "nature"),
        ("Laugavegur bar crawl", "nightlife"),
        ("Whales of Iceland exhibition", "family"),
        ("Family Park and Zoo", "family"),
    ],
    "Mexico City": [
        ("Teotihuacan pyramids", "history"),
        ("National Museum of Anthropology", "history"),
        ("Roma Norte taco crawl", "food"),
        ("Mercado de la Merced food walk", "food"),
        ("Chapultepec Park", "nature"),
        ("Xochimilco floating gardens", "nature"),
        ("Condesa cantina hop", "nightlife"),
        ("Zona Rosa bar tour", "nightlife"),
        ("Papalote Children's Museum", "family"),
        ("Six Flags Mexico", "family"),
    ],
    "Cape Town": [
        ("Robben Island tour", "history"),
        ("District Six Museum", "history"),
        ("Bo-Kaap cooking class", "food"),
        ("V&A Waterfront food tour", "food"),
        ("Table Mountain hike", "nature"),
        ("Cape Point day trip", "nature"),
        ("Boulders Beach penguins", "nature"),
        ("Long Street bar crawl", "nightlife"),
        ("Camps Bay sundowner tour", "nightlife"),
        ("Two Oceans Aquarium", "family"),
    ],
    "Istanbul": [
        ("Hagia Sophia", "history"),
        ("Topkapi Palace", "history"),
        ("Blue Mosque visit", "history"),
        ("Karakoy food tour", "food"),
        ("Grand Bazaar food walk", "food"),
        ("Bosphorus ferry cruise", "nature"),
        ("Princes' Islands ferry", "nature"),
        ("Beyoglu nightlife crawl", "nightlife"),
        ("Karakoy rooftop bar tour", "nightlife"),
        ("Miniaturk", "family"),
        ("Istanbul Aquarium", "family"),
    ],
    "Vancouver": [
        ("Museum of Anthropology", "history"),
        ("Steveston village walk", "history"),
        ("Granville Island food tour", "food"),
        ("Richmond night market", "food"),
        ("Stanley Park seawall", "nature"),
        ("Lynn Canyon suspension bridge", "nature"),
        ("Capilano hike", "nature"),
        ("Gastown pub crawl", "nightlife"),
        ("Yaletown cocktail tour", "nightlife"),
        ("Vancouver Aquarium", "family"),
        ("Science World", "family"),
    ],
    "Singapore": [
        ("National Museum of Singapore", "history"),
        ("Chinatown heritage walk", "history"),
        ("Hawker centre food tour", "food"),
        ("Little India food crawl", "food"),
        ("Gardens by the Bay", "nature"),
        ("Sentosa beach day", "nature"),
        ("MacRitchie reservoir hike", "nature"),
        ("Clarke Quay bar crawl", "nightlife"),
        ("Marina Bay rooftop tour", "nightlife"),
        ("Singapore Zoo", "family"),
        ("Universal Studios Singapore", "family"),
    ],
    "San Francisco": [
        ("Alcatraz Island tour", "history"),
        ("de Young Museum", "history"),
        ("Mission taqueria crawl", "food"),
        ("Ferry Building food walk", "food"),
        ("Golden Gate Park bike loop", "nature"),
        ("Lands End coastal hike", "nature"),
        ("Muir Woods day trip", "nature"),
        ("North Beach bar crawl", "nightlife"),
        ("Mission cocktail tour", "nightlife"),
        ("Exploratorium", "family"),
        ("California Academy of Sciences", "family"),
    ],
    "Berlin": [
        ("Brandenburg Gate walking tour", "history"),
        ("Berlin Wall Memorial", "history"),
        ("Pergamon Museum", "history"),
        ("Kreuzberg Turkish food tour", "food"),
        ("Prenzlauer Berg brunch crawl", "food"),
        ("Tiergarten walk", "nature"),
        ("Tempelhofer Feld bike loop", "nature"),
        ("Friedrichshain club night", "nightlife"),
        ("Kreuzberg bar crawl", "nightlife"),
        ("Museum Island family tour", "family"),
        ("Berlin Zoo", "family"),
    ],
    "Dubai": [
        ("Dubai Museum", "history"),
        ("Al Fahidi historical district walk", "history"),
        ("Old Souk food tour", "food"),
        ("Marina dinner cruise", "food"),
        ("Desert dune safari", "nature"),
        ("Palm Jumeirah beach day", "nature"),
        ("Marina sundowner cruise", "nightlife"),
        ("Downtown rooftop bar tour", "nightlife"),
        ("Burj Khalifa observation", "family"),
        ("IMG Worlds of Adventure", "family"),
        ("Dubai Aquarium", "family"),
    ],
    "Buenos Aires": [
        ("Recoleta Cemetery tour", "history"),
        ("Casa Rosada visit", "history"),
        ("Palermo steakhouse tour", "food"),
        ("San Telmo Sunday market crawl", "food"),
        ("Bosques de Palermo walk", "nature"),
        ("Tigre delta day trip", "nature"),
        ("Palermo Soho bar crawl", "nightlife"),
        ("San Telmo tango bar night", "nightlife"),
        ("Buenos Aires Zoo", "family"),
        ("Temaiken Biopark", "family"),
    ],
    "Seoul": [
        ("Gyeongbokgung Palace", "history"),
        ("War Memorial of Korea", "history"),
        ("Bukchon Hanok Village walk", "history"),
        ("Gwangjang Market food tour", "food"),
        ("Itaewon BBQ crawl", "food"),
        ("Bukhansan day hike", "nature"),
        ("Han River bike loop", "nature"),
        ("Hongdae nightlife crawl", "nightlife"),
        ("Gangnam cocktail bar tour", "nightlife"),
        ("Lotte World", "family"),
        ("N Seoul Tower", "family"),
    ],
    "Marrakech": [
        ("Bahia Palace tour", "history"),
        ("Saadian Tombs visit", "history"),
        ("Jemaa el-Fnaa food crawl", "food"),
        ("Medina spice market tour", "food"),
        ("Majorelle Garden", "nature"),
        ("Atlas Mountains day trip", "nature"),
        ("Gueliz rooftop bar tour", "nightlife"),
        ("Medina cabaret night", "nightlife"),
        ("Camel ride in the Palmeraie", "family"),
        ("Medina family treasure hunt", "family"),
    ],
    "Amsterdam": [
        ("Anne Frank House", "history"),
        ("Rijksmuseum", "history"),
        ("Van Gogh Museum", "history"),
        ("Jordaan food tour", "food"),
        ("De Pijp pancake crawl", "food"),
        ("Vondelpark bike loop", "nature"),
        ("Canal cruise", "nature"),
        ("Red Light District guided night", "nightlife"),
        ("Jordaan brown bar crawl", "nightlife"),
        ("NEMO Science Museum", "family"),
        ("Artis Royal Zoo", "family"),
    ],
    "Athens": [
        ("Acropolis and Parthenon tour", "history"),
        ("National Archaeological Museum", "history"),
        ("Ancient Agora walk", "history"),
        ("Plaka taverna crawl", "food"),
        ("Psiri street food tour", "food"),
        ("Lycabettus Hill hike", "nature"),
        ("Cape Sounion day trip", "nature"),
        ("Psiri bar crawl", "nightlife"),
        ("Gazi rooftop bar tour", "nightlife"),
        ("Athens National Garden", "family"),
        ("Hellenic Children's Museum", "family"),
    ],
}
