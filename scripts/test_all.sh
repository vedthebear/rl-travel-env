#!/usr/bin/env bash
# Run every test script in order. Stops on first failure.
# Use --skip-llm to skip the slow OpenRouter-dependent ones.
set -e
. "$(dirname "$0")/_lib.sh"

SKIP_LLM=false
if [ "${1:-}" = "--skip-llm" ]; then
  SKIP_LLM=true
fi

scripts=(
  scripts/test_00_imports.sh
  scripts/test_01_pytest.sh
  scripts/test_02_happy_path.sh
  scripts/test_03_baseline_inspect.sh
  scripts/test_04_full_eval.sh
)
if [ "$SKIP_LLM" = "false" ]; then
  scripts+=(
    scripts/test_05_llm_persona.sh
    scripts/test_06_llm_rollout.sh
    scripts/test_07_llm_judge.sh
  )
fi

start=$(date +%s)
for s in "${scripts[@]}"; do
  printf "\n"
  bash "$s"
done
elapsed=$(($(date +%s) - start))
printf "\n${BOLD}${GREEN}All %d scripts passed in %ds.${RESET}\n" "${#scripts[@]}" "$elapsed"
