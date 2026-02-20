#!/usr/bin/env bash
# Evaluate Ouro-Thinking with CoT generation using lm-eval + vLLM.
# Triggers thinking mode via enable_thinking, then strips <think>...</think>
# traces via think_end_token before answer extraction.
#
# Quick smoke test:   ./shells/eval.sh --quick
# Full evaluation:    ./shells/eval.sh
# Math only:          ./shells/eval.sh --math
# QA only:            ./shells/eval.sh --qa
# Single task:        TASKS=gsm8k_cot ./shells/eval.sh
# Fine-tuned model:   MODEL=path/to/checkpoint ./shells/eval.sh
# Multi-GPU:          TP=2 ./shells/eval.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-ByteDance/Ouro-2.6B-Thinking}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-3796}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"
LIMIT="${LIMIT:-}"

# --- Patched tokenizer ---
# Upstream Ouro tokenizer has wrong bos/eos tokens and no enable_thinking
# support in the chat template. We save a patched copy on first run.
TOKENIZER_DIR="models/tokenizer"
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "Setting up patched tokenizer (enable_thinking + fixed bos/eos)..."
    uv run python -c "
from transformers import AutoTokenizer
from pathlib import Path
t = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
t.chat_template = Path('templates/ouro_chat.j2').read_text()
t.bos_token = '<|im_start|>'
t.eos_token = '<|im_end|>'
Path('$TOKENIZER_DIR').mkdir(parents=True, exist_ok=True)
t.save_pretrained('$TOKENIZER_DIR')
print('Saved to $TOKENIZER_DIR')
"
fi

# --- Task definitions ---
# All tasks use generate_until (CoT / generative variants).
# Thinking traces are stripped by think_end_token before filters extract answers.
MATH_TASKS="gsm8k_cot_zeroshot minerva_math500 aime24"
QA_TASKS="arc_challenge_chat mmlu_flan_cot_zeroshot_stem gpqa_diamond_cot_zeroshot"
CODE_TASKS="mbpp"
ALL_TASKS="$MATH_TASKS $QA_TASKS $CODE_TASKS"

if [[ "${1:-}" == "--quick" ]]; then
    TASKS="gsm8k_cot_zeroshot"
    LIMIT=20
    echo "[Quick mode] tasks=$TASKS limit=$LIMIT"
elif [[ "${1:-}" == "--math" ]]; then
    TASKS="$MATH_TASKS"
    echo "[Math benchmarks only]"
elif [[ "${1:-}" == "--qa" ]]; then
    TASKS="$QA_TASKS"
    echo "[QA benchmarks only]"
fi

TASKS="${TASKS:-$ALL_TASKS}"

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

MODEL_ARGS="pretrained=$MODEL,tokenizer=$TOKENIZER_DIR,trust_remote_code=True,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,tensor_parallel_size=$TP,enable_thinking=True,think_end_token=</think>"

mkdir -p "$OUTPUT_DIR"

echo "Model:          $MODEL"
echo "Tokenizer:      $TOKENIZER_DIR"
echo "Tasks:          $TASKS"
echo "Max gen tokens: $MAX_GEN_TOKS"
echo ""

for task in $TASKS; do
    echo "=== $task ==="

    uv run lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size auto \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path "$OUTPUT_DIR" \
        --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0" \
        $LIMIT_ARG
done
