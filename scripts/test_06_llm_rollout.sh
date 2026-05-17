#!/usr/bin/env bash
# L6: full LLM-driven rollout (agent + env). One episode, capped at 20 turns.
# Slow (~45s). Validates the full agent loop end-to-end.
set -e
. "$(dirname "$0")/_lib.sh"
need_key "L6: LLM rollout"

banner "L6: LLM agent rollout (one episode, ~45s)"

$PY eval.py --episodes 1 --seed 42 --baselines llm \
  --persona-mode scripted --max-turns 20 \
  --llm-model anthropic/claude-haiku-4.5

pass "LLM rollout completed; reward populated"
