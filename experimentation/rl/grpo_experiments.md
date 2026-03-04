# GRPO Training Experiments — Ouro-1.4B-Thinking

## Motivation

Two goals: (1) explore whether RL (GRPO) can push Ouro-1.4B-Thinking beyond its SFT performance on math reasoning, and (2) establish a solid vanilla-RL baseline before experimenting with architectural changes (latent tokens, KV sharing, etc.) so we can isolate the effect of those modifications.

## Background

Ouro-1.4B-Thinking was SFT'd on long reasoning chains. Evaluation shows strong sensitivity to context length: AIME24 accuracy jumps from 30% → 53% and MATH500 from 65% → 74% when increasing max tokens from 8K to 30K. This makes RL training expensive — short sequence budgets understate the model's true capability and may distort the reward signal.

## Shared Config

| Param | Value | Rationale |
|-------|-------|-----------|
| model | ByteDance/Ouro-1.4B-Thinking | |
| loss_type | cispo | Truncated IS policy gradient, robust to large ratios |
| clip_eps / truncation_max | 0.2 / 5.0 | |
| lr | 1e-6 | |
| max_grad_norm | 0.1 | |
| prompts_per_step | 32 | |
| rollouts_per_prompt | 8 | |
| temperature / top_p | 1.0 / 0.7 | |
| enable_thinking | true | |
| enable_interruptions | true | ScaleRL-style forced answer after budget |
| thinking_budget | 3072–4096 (random) | |
| answer_budget | 512 | |
| gradient_checkpointing | true | |
| hardware | 2×A100-SXM4-80GB | |

**Training script:** `scripts/grpo_train.py`

---

## Run 1: competition_math (highest difficulty)

**Dataset:** competition_math (filtered to highest difficulty)
**max_new_tokens:** ~4096

### Results

- Mean reward: 0.6–0.8 (high)
- Zero-std fraction: 0.5–0.6 (majority of prompt groups have no reward variance → no gradient signal)
- No upward trend in reward over training

### Takeaways

- Dataset is too easy at this difficulty filter for this model — most prompts are either always solved or never solved, leaving little room for the policy gradient to act.
- Switched to a harder dataset.

---

## Run 2: DeepMath-103K (min_level=6)

**Dataset:** zwhe99/DeepMath-103K, min_level=6
**max_new_tokens:** 4608
**max_model_len:** 6144

### Results

- Mean reward: 0.1–0.2 (low)
- Zero-std fraction: 0.7–0.8
- Interrupted rollouts: >60% at ~4096 max length — model runs out of token budget before finishing reasoning
- No upward trend in reward over training

### Takeaways

- This difficulty level is likely too hard given the current token budget. The model needs more thinking tokens to solve these problems but gets cut off.
- Two levers to explore:
  1. **Increase max tokens** — let the model think longer (costly: more VRAM, slower steps)
  2. **Lower min_level** — use easier problems where the model can succeed within the budget

---

## Run 3: DeepMath-103K (min_level=6, thinking_budget=7680)

**Dataset:** zwhe99/DeepMath-103K, min_level=6
**thinking_budget:** 7680
**max_new_tokens:** ~8192
**max_model_len:** ~10240
**WandB:** [DeepMath-Level6-8k](https://wandb.ai/medtum/ouro-rl/runs/j9uu4d8z)

### Results (7 steps before GPU error)

- Mean reward: 0.13–0.37 (avg ~0.22, slightly higher than Run 2's 0.1–0.2)
- Zero-std fraction: 0.69–0.88 (worse than Run 2's 0.7–0.8)
- Interrupted rollouts: ~35–43% (down from >60% in Run 2)
- Step time: ~40–58 min per step (gen ~30–50 min, train ~5–7 min) — much slower than Run 2
- No upward trend in reward over 7 steps
- Run crashed after ~6 hours due to GPU error

### Takeaways

- Doubling the token budget cut interruptions significantly (35–43% vs >60%) but didn't translate into meaningful reward improvement.
- Zero-std fraction actually worsened — more token budget lets the model attempt more problems but most are still consistently solved or unsolved.
- The core issue is dataset difficulty, not token budget. min_level=6 is too hard for 1.4B regardless of thinking time.
- Training is prohibitively slow at this sequence length (~45 min/step vs the target of 140 steps).

---

## Next Steps

- Reduce token budget back to ~4K and use a **max_level** filter instead of min_level — target mid-difficulty problems (e.g. level 3–5) where the model can realistically solve some but not all within budget.
- Goal: mean reward ~0.3–0.5 with zero-std fraction <0.5.

---

## Open Questions

- What is the sweet spot for difficulty × token budget that gives ~0.3–0.5 mean reward with low zero-std fraction?
- Can we afford longer sequences (e.g. 8K–16K max_new_tokens) on 2×A100, or do we need to reduce batch size?
- Would curriculum difficulty (start easy, ramp up) help bootstrap learning?
