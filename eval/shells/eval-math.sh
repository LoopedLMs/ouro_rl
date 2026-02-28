#!/usr/bin/env bash
# Evaluate Ouro-Thinking on math benchmarks.
#
# All tasks:     ./eval/shells/eval-math.sh
# Pick tasks:    TASKS=gsm8k_thinking,aime24_thinking ./eval/shells/eval-math.sh
# Smoke test:    LIMIT=20 ./eval/shells/eval-math.sh
# Fine-tuned:    MODEL=path/to/checkpoint ./eval/shells/eval-math.sh
# + pass@10:     TASKS=gsm8k_thinking,aime24_thinking,aime24_thinking_pass10 ./eval/shells/eval-math.sh
# Slurm:         NUM_GPUS=2 ./shells/_submit.sh eval/shells/eval-math.sh -- --partition=eagle --account=eagle --nodelist=eagle --gres=gpu:h100:2

set -euo pipefail

cd ~/ouro_rl
source eval/shells/eval-common.sh

TASKS="${TASKS:-gsm8k_thinking,math500_thinking,aime24_thinking,aime25_thinking}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo ""

run_all "$TASKS"
