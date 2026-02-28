# Ouro Reinforcement Learning

RL fine-tuning for [Ouro](https://ouro-llm.github.io) models using GRPO. Evaluation via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Evaluation Results

0-shot, max-tokens=30768, temperature=1.0, top-p=0.7

| Model | [GSM8K](eval/tasks/gsm8k_thinking.yaml) | [AIME24](eval/tasks/aime24_thinking.yaml) | [AIME25](eval/tasks/aime25_thinking.yaml) | [MATH500](eval/tasks/math500_thinking.yaml) | [GPQA](eval/tasks/gpqa_main_thinking.yaml) |
|-------|--------|---------|---------|----------|------|
| Ouro-1.4B-Thinking | 93.40 ± 0.68 | 53.33 ± 9.26 | 43.33 ± 9.20 | 74.20 ± 1.96 | 22.99 ± 1.99 |
