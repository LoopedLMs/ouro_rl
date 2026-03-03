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

## Open Questions

- What is the sweet spot for difficulty × token budget that gives ~0.3–0.5 mean reward with low zero-std fraction?
- Can we afford longer sequences (e.g. 8K–16K max_new_tokens) on 2×A100, or do we need to reduce batch size?
- Would curriculum difficulty (start easy, ramp up) help bootstrap learning?
