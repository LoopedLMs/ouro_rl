# Ouro RL - Exploration of Ouro LM

## Philosophy
Research code optimized for rapid iteration and debugging:
- Simple, hackable implementations > frameworks
- Missing error handling is GOOD (faster bug discovery)
- Understand every component > black-box abstractions

## Code Standards
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Self-documenting names > comments
- Run ruff after changes: `uv run ruff format . && uv run ruff check --fix .`

## Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Package Management (CRITICAL)
- ✅ ALWAYS: `uv add <package>`
- ❌ NEVER: manually edit pyproject.toml
- ❌ NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment:
- **Option 1**: `uv run python script.py` (recommended for one-off commands)
- **Option 2**: Activate environment first with `source .venv/bin/activate`, then run normally

## Key Dependencies
- `torch==2.9.0` — pinned, cu128
- `flash-attn` — for efficient attention

## Debugging
Check `.venv` source code directly for library implementation details

## Background Knowledge
Paper summaries and research notes live in `./knowledge/`. Check there for context on relevant prior work (e.g. layer redundancy, recurrence retrofitting). The paper behind this repository is summarized in knowledge/summary_retrofitting_recurrence.md.

## Ouro-Thinking Model Quirks (Huggingface)
- **Wrong bos/eos upstream**: Both 1.4B and 2.6B Thinking models ship with bos/eos/pad all set to `<|endoftext|>` (id=0). Correct: bos=`<|im_start|>` (1), eos=`<|im_end|>` (2).
- **enable_thinking**: Ouro-Thinking won't emit `<think>` on its own — it must be prepended in the prompt. Upstream chat template lacks `enable_thinking` support. We use a local template at `templates/ouro_chat.j2` that appends `<think>\n` after `<|im_start|>assistant\n` when `enable_thinking=True`.

## Research Stack
- Framework: PyTorch + HuggingFace Transformers + vLLM
- Testing: pytest for core components only (skip for exploratory code)
