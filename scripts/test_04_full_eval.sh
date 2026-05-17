#!/usr/bin/env bash
# L4: full eval across 20 episodes, 3 baselines. Asserts the headline ordering:
# random < cheapest, and heuristic ≥ cheapest (the reward-design claim).
set -e
. "$(dirname "$0")/_lib.sh"

banner "L4: 20-episode eval × 3 baselines (scripted persona)"

# Run eval and capture output for assertion.
out=$($PY eval.py --episodes 20 --seed 42 \
  --baselines random,cheapest,heuristic --persona-mode scripted 2>&1)
echo "$out"

# Parse the three reward column means (column 3 in the table).
random_r=$(echo "$out" | awk '/^random / {print $3}')
cheapest_r=$(echo "$out" | awk '/^cheapest / {print $3}')
heuristic_r=$(echo "$out" | awk '/^heuristic / {print $3}')

[ -z "$random_r" ] && fail "couldn't parse random reward"
[ -z "$cheapest_r" ] && fail "couldn't parse cheapest reward"
[ -z "$heuristic_r" ] && fail "couldn't parse heuristic reward"

# Use python for float comparison.
$PY -c "
r, c, h = $random_r, $cheapest_r, $heuristic_r
assert c > r, f'cheapest {c} should beat random {r}'
assert h >= c - 0.05, f'heuristic {h} should be ≥ cheapest {c} (within 0.05)'
print(f'  random < cheapest: {r:.3f} < {c:.3f}  ✓')
print(f'  heuristic ≥ cheapest: {h:.3f} ≥ {c:.3f}  ✓')
"
pass "baseline ordering holds: random < cheapest ≤ heuristic"
