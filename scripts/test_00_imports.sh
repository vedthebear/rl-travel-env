#!/usr/bin/env bash
# L0: imports — every module loads cleanly, key constants are sensible.
set -e
. "$(dirname "$0")/_lib.sh"

banner "L0: imports"

$PY -c "
from travel_env import TravelEnv, rollout
from travel_env.world import CITIES, search_flights, search_hotels, search_activities, make_world
from travel_env.persona import sample_profile, compute_mismatches, PersonaProfile, PersonaVoice, Mismatch
from travel_env.reward import score_episode, DEFAULT_WEIGHTS, llm_judge_score, RewardWeights
from travel_env.baselines import random_policy, cheapest_policy, heuristic_policy
from travel_env.env import TOOL_REGISTRY, Action
assert len(CITIES) >= 15, f'expected >=15 cities, got {len(CITIES)}'
assert len(TOOL_REGISTRY) == 11, f'expected 11 tools, got {len(TOOL_REGISTRY)}'
print(f'  cities: {len(CITIES)}')
print(f'  tools:  {len(TOOL_REGISTRY)}')
"

pass "all modules import; city + tool counts sane"
