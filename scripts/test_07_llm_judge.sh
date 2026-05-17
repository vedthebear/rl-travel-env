#!/usr/bin/env bash
# L7: LLM judge correlation. The headline diagnostic: does the rule-based reward
# track what an LLM evaluator would say? Reports corr(rule_reward, judge_score).
set -e
. "$(dirname "$0")/_lib.sh"
need_key "L7: LLM judge"

banner "L7: LLM judge correlation (4 episodes, ~10s)"

$PY eval.py --episodes 4 --seed 42 --baselines heuristic \
  --persona-mode scripted --llm-judge

pass "LLM judge ran; correlation reported"
