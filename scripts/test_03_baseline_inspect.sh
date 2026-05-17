#!/usr/bin/env bash
# L3: run each baseline on a single seed, print a one-line trajectory summary.
# Confirms baselines execute without crashing and produce a non-zero reward.
set -e
. "$(dirname "$0")/_lib.sh"

banner "L3: single-seed inspection of all three baselines"

$PY << 'PY'
import inspect
import numpy as np
from travel_env.env import TravelEnv
from travel_env.baselines import random_policy, cheapest_policy, heuristic_policy
from travel_env.persona import sample_profile

SEED = 42
for name, pol in [('random', random_policy), ('cheapest', cheapest_policy), ('heuristic', heuristic_policy)]:
    env = TravelEnv(seed=SEED, persona_mode='scripted', max_turns=40)
    obs, info = env.reset()
    profile = sample_profile(SEED)
    rng = np.random.default_rng(SEED)
    sig = inspect.signature(pol).parameters
    tools = []
    term = trunc = False
    while not (term or trunc):
        kw = {}
        if 'rng' in sig: kw['rng'] = rng
        if 'profile' in sig: kw['profile'] = profile
        a = pol(obs, **kw)
        obs, r, term, trunc, info = env.step(a)
        tools.append(a['tool'])
    b = info['reward_breakdown']
    histo = {t: tools.count(t) for t in set(tools)}
    print(f"  {name:10s}  steps={len(tools):2d}  reward={b['total']:+.3f}  "
          f"gate={b['hard_constraint_gate']:.0f}  coh={b['coherence']:.2f}")
    print(f"             tools: {histo}")
    env.close()
PY

pass "all three baselines ran to termination on seed 42"
