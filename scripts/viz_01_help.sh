#!/usr/bin/env bash
# Demo 1/5 — sanity: CLI parses, no rich UI.
set -e
. "$(dirname "$0")/_lib.sh"
banner "DEMO 1 — viz --help"
$PY viz.py --help
