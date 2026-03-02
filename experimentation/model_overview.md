# Ouro architecture overview

## Shared architecture (all models)

| Property | Value |
|---|---|
| **Model stage** | Chat model (ChatML template) |
| **Architecture** | Decoder-only LoopLM (weight-tied recurrence) |
| **Hidden size** | 2048 |
| **Attention type** | Full MHA (no sliding window), 16 heads, head dim 128 |
| **KV heads** | 16 (no GQA) |
| **FFN** | SwiGLU, intermediate size 5632 |
| **Normalization** | Sandwich RMSNorm (pre + post on both attn and FFN) |
| **Position encoding** | RoPE, θ = 1,000,000 |
| **Recurrent steps (T)** | 4 |
| **Context length** | 64K tokens (`max_position_embeddings` = 65536) |
| **Vocab size** | 49,152 (SmolLM2 tokenizer) |
| **dtype** | bfloat16 |

## Per-model differences

- **Ouro-1.4B** (`ByteDance/Ouro-1.4B`) — 24 layers, 96 KV cache entries, chat (mid-trained on 7.7T tokens)
- **Ouro-1.4B-Thinking** (`ByteDance/Ouro-1.4B-Thinking`) — same architecture, Thinking SFT on 8.3M examples with `<think>` scaffolding
- **Ouro-2.6B** (`ByteDance/Ouro-2.6B`) — 48 layers, 192 KV cache entries, upcycled from 1.4B via layer duplication, chat (mid-trained on 7.7T tokens)
- **Ouro-2.6B-Thinking** (`ByteDance/Ouro-2.6B-Thinking`) — same architecture, Thinking SFT on 8.3M examples with `<think>` scaffolding

## Notes

- The 2.6B model was created by **upcycling** the 24-layer 1.4B model to 48 layers via layer duplication mid-training (at Stage 1b), not trained from scratch.
- Both base models were trained across 5 stages: Pre-train I (3T, 4K ctx, T=8→4), Pre-train II (3T, 4K, T=4), CT Annealing (1.4T, 16K, T=4), LongCT (20B, 64K, T=4), Mid-training (300B, 32K, T=4).
- **Thinking** variants are SFT fine-tunes on 8.3M examples (math 3.5M, code 3.2M, science 808K, chat 767K) using `enable_thinking=True` in the chat template.
- Context length of 64K comes from the LongCT stage (ProLong-64K data); stored as `max_position_embeddings=65536` in config.
