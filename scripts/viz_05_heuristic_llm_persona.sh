#!/usr/bin/env bash
# Demo 5/5 — heuristic agent + LLM persona. Real OpenRouter call voices the
# client request in the header panel. ~3s latency at episode start, then the
# heuristic runs with persona-aware filters (min_stars, top categories).
set -e
. "$(dirname "$0")/_lib.sh"
need_key "DEMO 5 (LLM persona)"
banner "DEMO 5 — heuristic agent, LLM-voiced persona"
$PY viz.py --baseline heuristic --seed 42 --persona-mode llm --turn-delay 0.4
