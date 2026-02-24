"""Test batched generation for Ouro models with HF Transformers and vLLM.

Reports suggest only the longest sequence in a batch produces correct output.
This script generates responses individually and in a batch, then compares them
to see if shorter prompts get corrupted by padding.

Run HF test:
    uv run python dev/test_batched_generation.py --backend hf  --attn-impl eager
Run vLLM test:
    uv run python dev/test_batched_generation.py --backend vllm
"""

import argparse

from ouro_rl.modeling.constants import CHAT_TEMPLATE, EOS_TOKEN_ID, PAD_TOKEN_ID

# Prompts of deliberately different lengths to expose padding issues.
# Ordered short -> long so prompt[0] is most likely to be corrupted if
# batched generation only works for the longest sequence.
PROMPTS = [
    "What is 2+2?",
    "Name three primary colors.",
    "Explain why the sky is blue in one sentence.",
    "Write a short poem about a cat sitting on a windowsill watching the rain.",
]

SYSTEM_PROMPT = "You are a helpful assistant."


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def strip_thinking(text: str) -> str:
    end_idx = text.find("</think>")
    if end_idx != -1:
        return text[end_idx + 8 :].strip()
    return text.strip()


def truncate_at_eos(text: str) -> str:
    """Truncate at first EOS token and strip padding artifacts."""
    eos_idx = text.find("<|im_end|>")
    if eos_idx != -1:
        text = text[:eos_idx]
    # Strip any leaked pad tokens (<|endoftext|>)
    text = text.replace("<|endoftext|>", "")
    return text.strip()


# ---------------------------------------------------------------------------
# HF Transformers
# ---------------------------------------------------------------------------


def test_hf(model_name: str, max_new_tokens: int, attn_impl: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from ouro_rl.modeling import OuroForCausalLM

    print("=" * 70)
    print(f"HF TRANSFORMERS — BATCHED GENERATION TEST (attn={attn_impl})")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = EOS_TOKEN_ID
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"

    model = OuroForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    # --- individual generation (reference) ---
    print("\n--- Individual generation (reference) ---")
    individual_outputs: list[str] = []
    for i, prompt in enumerate(PROMPTS):
        messages = build_messages(prompt)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=CHAT_TEMPLATE,
            enable_thinking=False,
        )
        input_ids = torch.tensor([prompt_ids], device=model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=False,  # greedy for reproducibility
                eos_token_id=EOS_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
            )

        new_tokens = output_ids[0][input_ids.shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        answer = truncate_at_eos(strip_thinking(text))
        individual_outputs.append(answer)
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Input length: {len(prompt_ids)} tokens")
        print(f"    Answer: {answer[:200]}")

    # --- batched generation ---
    print("\n--- Batched generation ---")
    all_prompt_texts: list[str] = []
    all_prompt_lengths: list[int] = []
    for prompt in PROMPTS:
        messages = build_messages(prompt)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=CHAT_TEMPLATE,
            enable_thinking=False,
        )
        all_prompt_texts.append(text)
        token_len = len(tokenizer.encode(text))
        all_prompt_lengths.append(token_len)

    batch = tokenizer(
        all_prompt_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Individual token lengths: {all_prompt_lengths}")

    with torch.no_grad():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=False,  # greedy for reproducibility
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )

    batched_outputs: list[str] = []
    for i, prompt in enumerate(PROMPTS):
        # Strip the input portion (including padding)
        input_len = batch["input_ids"].shape[1]
        new_tokens = output_ids[i][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        answer = truncate_at_eos(strip_thinking(text))
        batched_outputs.append(answer)
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Answer: {answer[:200]}")

    # --- comparison ---
    print("\n--- Comparison ---")
    for i, prompt in enumerate(PROMPTS):
        match = individual_outputs[i] == batched_outputs[i]
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Match: {match}")
        if not match:
            print(f"    Individual: {individual_outputs[i][:150]}")
            print(f"    Batched:    {batched_outputs[i][:150]}")


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


def test_vllm(model_name: str, max_new_tokens: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("vLLM — BATCHED GENERATION TEST")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
    )

    # Build all prompts
    all_prompts: list[str] = []
    for prompt in PROMPTS:
        messages = build_messages(prompt)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=CHAT_TEMPLATE,
            enable_thinking=False,
        )
        all_prompts.append(text)

    # --- individual generation (one at a time) ---
    print("\n--- Individual generation (reference) ---")
    individual_outputs: list[str] = []
    for i, (prompt, full_prompt) in enumerate(zip(PROMPTS, all_prompts)):
        outputs = llm.generate([full_prompt], sampling_params)
        text = outputs[0].outputs[0].text
        answer = truncate_at_eos(strip_thinking(text))
        individual_outputs.append(answer)
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Answer: {answer[:200]}")

    # --- batched generation (all at once) ---
    print("\n--- Batched generation (all prompts at once) ---")
    outputs = llm.generate(all_prompts, sampling_params)
    batched_outputs: list[str] = []
    for i, (prompt, output) in enumerate(zip(PROMPTS, outputs)):
        text = output.outputs[0].text
        answer = truncate_at_eos(strip_thinking(text))
        batched_outputs.append(answer)
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Answer: {answer[:200]}")

    # --- comparison ---
    print("\n--- Comparison ---")
    for i, prompt in enumerate(PROMPTS):
        match = individual_outputs[i] == batched_outputs[i]
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Match: {match}")
        if not match:
            print(f"    Individual: {individual_outputs[i][:150]}")
            print(f"    Batched:    {batched_outputs[i][:150]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Test batched generation for Ouro")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    p.add_argument("--backend", choices=["hf", "vllm", "both"], default="both")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for HF (sdpa crashes on batched input due to "
        "non-contiguous mask in Ouro's custom modeling code)",
    )
    args = p.parse_args()

    if args.backend in ("hf", "both"):
        test_hf(args.model, args.max_new_tokens, args.attn_impl)

    if args.backend in ("vllm", "both"):
        test_vllm(args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
