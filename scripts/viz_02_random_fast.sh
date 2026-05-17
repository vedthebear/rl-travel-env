#!/usr/bin/env bash
# Demo 2/5 — fastest possible live run. Random baseline, no delay.
# Confirms the rich panels actually draw. Episode terminates in 3-5 turns.
set -e
. "$(dirname "$0")/_lib.sh"
banner "DEMO 2 — random baseline, no delay (panels draw test)"
$PY viz.py --baseline random --seed 42 --turn-delay 0 --max-turns 10
