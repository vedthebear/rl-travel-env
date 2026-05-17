#!/usr/bin/env bash
# Demo 6 — the agent itself is an LLM (Claude Haiku via OpenRouter).
# This is the headline demo: real LLM driving the env, panels update each
# turn with whatever tool call the model decided to make. ~1-3s per turn
# of API latency on top of --turn-delay.
#
# Tip: bump --turn-delay to 0 if you don't need extra pacing — the LLM
# itself is the bottleneck.
set -e
. "$(dirname "$0")/_lib.sh"
need_key "DEMO 6 (LLM agent)"
banner "DEMO 6 — LLM agent (real OpenRouter rollout)"
$PY viz.py --baseline llm --seed 42 --turn-delay 0.2 --max-turns 20
