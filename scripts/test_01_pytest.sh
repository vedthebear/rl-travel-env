#!/usr/bin/env bash
# L1: the existing pytest suite (env, world, persona, reward, baselines).
set -e
. "$(dirname "$0")/_lib.sh"

banner "L1: pytest suite"
$PY -m pytest tests/ --tb=short -q
pass "all pytest tests green"
