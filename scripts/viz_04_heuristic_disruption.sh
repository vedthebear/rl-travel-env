#!/usr/bin/env bash
# Demo 4/5 — heuristic at seed 102: flight cancellation fires mid-episode.
# Watch for a red 'cancelled' row, then a fresh *_rebook slot appearing.
# This is the recovery flow visualized. ~18 turns.
set -e
. "$(dirname "$0")/_lib.sh"
banner "DEMO 4 — heuristic, seed 102 (disruption + rebook)"
$PY viz.py --baseline heuristic --seed 102 --turn-delay 0.5
