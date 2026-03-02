#!/bin/bash
set -e

cd ~/ouro_rl
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config || exit 1

cd dev/analysis

N_LOOPS=4
MODEL=ByteDance/Ouro-1.4B
OUTPUT_DIR=outputs

ARGS=("$@")
[ -n "$N_LOOPS" ] && ARGS+=("--n-loops" "$N_LOOPS")
[ -n "$MODEL" ] && ARGS+=("--model" "$MODEL")
[ -n "$OUTPUT_DIR" ] && ARGS+=("--output-dir" "$OUTPUT_DIR")

uv run python visualize_angular_distance.py "${ARGS[@]}"
uv run python visualize_latent_state_dist_per_loop.py "${ARGS[@]}"
uv run python visualize_latent_state_dist_tokens.py "${ARGS[@]}"
