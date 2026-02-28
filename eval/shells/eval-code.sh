#!/usr/bin/env bash
# Evaluate Ouro-Thinking on code benchmarks.
#
# All tasks:     ./eval/shells/eval-code.sh
# Smoke test:    LIMIT=20 ./eval/shells/eval-code.sh

set -euo pipefail
cd ~/ouro_rl
source eval/shells/eval-common.sh

TASKS="${TASKS:-mbpp}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo ""

run_all "$TASKS"
