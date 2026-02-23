#!/usr/bin/env bash
# Evaluate Ouro-Thinking on math benchmarks.
#
# All tasks:     ./shells/eval-math.sh
# Single task:   TASKS=gsm8k_thinking ./shells/eval-math.sh
# Smoke test:    LIMIT=20 ./shells/eval-math.sh
# Fine-tuned:    MODEL=path/to/checkpoint ./shells/eval-math.sh
# + pass@10:     PASS_AT=10 ./shells/eval-math.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="Solve the math problem. Put your final answer in \\boxed{}."
TASKS="${TASKS:-gsm8k_thinking minerva_math500 aime24_thinking}"
PASS_AT="${PASS_AT:-}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo "Max gen tokens: $MAX_GEN_TOKS"
if [[ -n "$PASS_AT" ]]; then
    echo "Pass@k:         $PASS_AT (aime24)"
fi
echo ""

for task in $TASKS; do
    run_eval "$task" --system_instruction "$SYSTEM_INSTRUCTION"
done

# --- pass@k (sampled) ---
if [[ -n "$PASS_AT" ]]; then
    echo ""
    echo "=== aime24 pass@${PASS_AT} ==="
    run_eval "aime24_thinking_pass${PASS_AT}" \
        --system_instruction "$SYSTEM_INSTRUCTION" \
        --log_samples

    SAMPLES=$(ls -t "$OUTPUT_DIR"/*/samples_aime24_thinking_pass${PASS_AT}_*.jsonl 2>/dev/null | head -1)
    if [[ -n "$SAMPLES" ]]; then
        uv run python scripts/compute_pass_at_k.py "$SAMPLES" --k "$PASS_AT"
    fi
fi
