#!/usr/bin/env bash
# L2: hand-crafted episode through every relevant tool. No LLM, scripted persona.
# search → add → search → add → search → add → book → propose → submit.
set -e
. "$(dirname "$0")/_lib.sh"

banner "L2: hand-crafted happy-path episode"

$PY << 'PY'
from travel_env.env import TravelEnv
from travel_env.world import CITIES

env = TravelEnv(seed=42, persona_mode='scripted', max_turns=40)
obs, info = env.reset()
prof = env._profile
o_iata = CITIES[prof.origin_city].iata
d_iata = CITIES[prof.dest_city].iata
print(f"  trip: {prof.origin_city} -> {prof.dest_city}, cap=${prof.hard.budget_cap:.0f}")

def step(a, label):
    obs2, r, term, trunc, info = env.step(a)
    last = obs2['last_result']
    ok = last.get('ok') if isinstance(last, dict) else None
    assert ok, f"{label} failed: {last}"
    return obs2

obs = step({'tool': 'search_flights', 'args': {
    'origin': o_iata, 'dest': d_iata, 'depart_date': prof.hard.depart_date}}, 'search_flights out')
fid = obs['last_result']['results'][0]['id']
obs = step({'tool': 'add_to_itinerary', 'args': {'item_id': fid, 'slot': 'outbound_flight'}}, 'add out')
obs = step({'tool': 'search_flights', 'args': {
    'origin': d_iata, 'dest': o_iata, 'depart_date': prof.hard.return_date}}, 'search_flights return')
rfid = obs['last_result']['results'][0]['id']
obs = step({'tool': 'add_to_itinerary', 'args': {'item_id': rfid, 'slot': 'return_flight'}}, 'add return')
obs = step({'tool': 'search_hotels', 'args': {
    'city': prof.dest_city, 'checkin': prof.hard.depart_date, 'checkout': prof.hard.return_date}}, 'search hotels')
hid = obs['last_result']['results'][0]['id']
obs = step({'tool': 'add_to_itinerary', 'args': {'item_id': hid, 'slot': 'hotel'}}, 'add hotel')
obs = step({'tool': 'book', 'args': {'item_ids': [fid, rfid, hid]}}, 'book')

obs2, r, term, trunc, _ = env.step({'tool': 'propose_to_client', 'args': {}})
mm = obs2['last_result'].get('mismatches', [])
print(f"  propose: {len(mm)} mismatches voiced")

obs2, r, term, trunc, info = env.step({'tool': 'submit_final', 'args': {}})
assert term, "submit_final didn't terminate"
b = info['reward_breakdown']
assert b['hard_constraint_gate'] in (0.0, 1.0), f"gate not boolean: {b['hard_constraint_gate']}"
print(f"  final reward={b['total']:.3f}, gate={b['hard_constraint_gate']}, coh={b['coherence']:.2f}")
env.close()
PY

pass "hand-crafted episode completed end-to-end; reward populated"
