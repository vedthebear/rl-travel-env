#!/usr/bin/env bash
# Shared helpers for the test scripts.
# Source this from each script: . scripts/_lib.sh
set -u

# cd to repo root regardless of where the script is invoked from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

# Load .env if present so OPENROUTER_API_KEY is available.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

PY=".venv/bin/python"

# ANSI colors (skip if not a TTY).
if [ -t 1 ]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; BOLD='\033[1m'; RESET='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; BOLD=''; RESET=''
fi

banner() {
  printf "${BOLD}=== %s ===${RESET}\n" "$1"
}

pass() {
  printf "${GREEN}✓ PASS${RESET}  %s\n" "$1"
}

fail() {
  printf "${RED}✗ FAIL${RESET}  %s\n" "$1"
  exit 1
}

skip() {
  printf "${YELLOW}~ SKIP${RESET}  %s\n" "$1"
}

need_key() {
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    skip "$1 (OPENROUTER_API_KEY not set)"
    exit 0
  fi
}
