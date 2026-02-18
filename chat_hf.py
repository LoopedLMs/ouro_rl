"""Interactive chat with ByteDance/Ouro-1.4B-Thinking via HuggingFace Transformers.

Run with:
    uv run python chat_hf.py
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ByteDance/Ouro-1.4B-Thinking"
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = "You are a helpful assistant."


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split <think>...</think> from the final answer."""
    if text.startswith("<think>"):
        end_idx = text.find("</think>")
        if end_idx != -1:
            thinking = text[7:end_idx].strip()
            answer = text[end_idx + 8 :].strip()
            return thinking, answer
    return None, text.strip()


def main() -> None:
    print(f"Loading {MODEL} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"\nReady. Chatting with {MODEL}.")
    print("Type 'exit' or 'quit' to stop, '/clear' to reset conversation history.\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        if user_input == "/clear":
            messages = []
            print("[Conversation history cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        inputs = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        thinking, answer = split_thinking(raw_response)

        if thinking:
            print(f"\n\033[2m[Thinking]\n{thinking}\033[0m\n")

        print(f"Ouro: {answer}\n")

        messages.append({"role": "assistant", "content": raw_response})


if __name__ == "__main__":
    main()
