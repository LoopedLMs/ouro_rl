"""Save a patched Ouro tokenizer with corrected bos/eos and enable_thinking support.

Upstream Ouro tokenizer ships with bos/eos/pad all set to <|endoftext|> (id=0).
This script fixes them and injects a chat template that supports enable_thinking.

Usage:
    uv run python eval/scripts/setup_tokenizer.py [--model MODEL] [--output DIR]
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from ouro_rl.modeling import BOS_TOKEN_ID, CHAT_TEMPLATE, EOS_TOKEN_ID, PAD_TOKEN_ID


def setup_tokenizer(
    model: str = "ByteDance/Ouro-1.4B-Thinking",
    output: str = "models/tokenizer",
) -> None:
    output_path = Path(output)
    if output_path.exists():
        print(f"Tokenizer already exists at {output_path}, skipping.")
        return

    print(f"Loading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenizer.bos_token_id = BOS_TOKEN_ID
    tokenizer.eos_token_id = EOS_TOKEN_ID
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.chat_template = CHAT_TEMPLATE

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved patched tokenizer to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    parser.add_argument("--output", default="models/tokenizer")
    args = parser.parse_args()
    setup_tokenizer(model=args.model, output=args.output)
