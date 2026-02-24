"""Lightweight Ouro constants — no torch/transformers imports.

Safe to import from any context (vLLM workers, scripts, etc.) without
triggering CUDA initialization.
"""

from pathlib import Path

# Correct token IDs for ChatML (upstream config has all three set to 0).
BOS_TOKEN_ID = 1  # <|im_start|>
EOS_TOKEN_ID = 2  # <|im_end|>
PAD_TOKEN_ID = 0  # <|endoftext|> — safe for padding since it's NOT the real EOS

CHAT_TEMPLATE = (Path(__file__).parent / "chat_template.jinja").read_text()

DEFAULT_MODEL = "ByteDance/Ouro-1.4B-Thinking"
