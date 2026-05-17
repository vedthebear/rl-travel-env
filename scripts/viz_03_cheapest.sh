#!/usr/bin/env bash
# Demo 3/5 — cheapest baseline at default seed. ~8 turns, watchable pace.
# Trip should complete with gate=1, watch tentative -> booked transitions.
set -e
. "$(dirname "$0")/_lib.sh"
banner "DEMO 3 — cheapest baseline, watchable pace"
$PY viz.py --baseline cheapest --seed 42 --turn-delay 0.4
