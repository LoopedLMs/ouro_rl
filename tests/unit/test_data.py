"""Unit tests for ouro_rl/data.py — dataset loading, prompt formatting, answer extraction."""

from unittest.mock import MagicMock, patch

from datasets import Dataset

from ouro_rl.data import CHAT_TEMPLATE, extract_boxed_answer, format_prompt, load_math_train

# ---------------------------------------------------------------------------
# load_math_train (DeepMath-103K)
# ---------------------------------------------------------------------------


def _make_deepmath_dataset() -> Dataset:
    """Minimal fake DeepMath-103K rows with the upstream column names."""
    return Dataset.from_dict(
        {
            "question": ["Prove that sqrt(2) is irrational.", "Find all primes p such that p+2 is also prime."],
            "final_answer": ["Assume for contradiction...", "p=3"],
            "difficulty": [7.0, 5.0],  # float64 in the real dataset
            "topic": ["Number Theory", "Number Theory"],
            "r1_solution_1": ["solution a", "solution b"],
        }
    )


class TestLoadMathTrain:
    def test_default_dataset_name(self):
        """load_math_train calls load_dataset with zwhe99/DeepMath-103K by default."""
        fake_ds = _make_deepmath_dataset()
        with patch("ouro_rl.data.load_dataset", return_value=fake_ds) as mock_load:
            load_math_train()
        mock_load.assert_called_once_with("zwhe99/DeepMath-103K", split="train")

    def test_columns_remapped(self):
        """question→problem and answer→solution after loading."""
        fake_ds = _make_deepmath_dataset()
        with patch("ouro_rl.data.load_dataset", return_value=fake_ds):
            ds = load_math_train()
        assert "problem" in ds.column_names
        assert "solution" in ds.column_names
        assert "question" not in ds.column_names
        assert "answer" not in ds.column_names

    def test_difficulty_column_preserved(self):
        """The numeric difficulty column (1-9) is retained after remapping."""
        fake_ds = _make_deepmath_dataset()
        with patch("ouro_rl.data.load_dataset", return_value=fake_ds):
            ds = load_math_train()
        assert "difficulty" in ds.column_names
        assert all(isinstance(d, float) for d in ds["difficulty"])

    def test_problem_solution_values_correct(self):
        """Remapped values match the original question/answer fields."""
        fake_ds = _make_deepmath_dataset()
        with patch("ouro_rl.data.load_dataset", return_value=fake_ds):
            ds = load_math_train()
        assert ds["problem"][0] == "Prove that sqrt(2) is irrational."
        assert ds["solution"][1] == "p=3"

    def test_min_level_filter(self):
        """Filtering by difficulty >= min_level (as grpo_train.py does) works correctly."""
        fake_ds = _make_deepmath_dataset()
        with patch("ouro_rl.data.load_dataset", return_value=fake_ds):
            ds = load_math_train()
        filtered = ds.filter(lambda x: x["difficulty"] >= 6)
        assert len(filtered) == 1
        assert filtered["problem"][0] == "Prove that sqrt(2) is irrational."


# ---------------------------------------------------------------------------
# extract_boxed_answer
# ---------------------------------------------------------------------------


class TestExtractBoxedAnswer:
    def test_simple_boxed(self):
        assert extract_boxed_answer("The answer is \\boxed{42}") == "42"

    def test_no_boxed(self):
        assert extract_boxed_answer("The answer is 42") is None

    def test_nested_braces(self):
        r"""Handles nested braces: \boxed{\frac{1}{2}}."""
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed_takes_last(self):
        r"""rfind selects the last \boxed{} occurrence."""
        solution = "First \\boxed{wrong} then \\boxed{right}"
        assert extract_boxed_answer(solution) == "right"

    def test_unclosed_boxed(self):
        """Unclosed \\boxed{ → None (no matching closing brace)."""
        assert extract_boxed_answer("\\boxed{42") is None


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def _make_tokenizer(self) -> MagicMock:
        """Mock tokenizer that captures apply_chat_template calls."""
        tok = MagicMock()
        tok.apply_chat_template.side_effect = lambda messages, **kwargs: (
            f"formatted:{len(messages)}:{kwargs.get('enable_thinking', False)}"
        )
        return tok

    def test_user_message_only_no_system(self):
        """With system_prompt=None, messages should only contain user turn."""
        tok = self._make_tokenizer()
        format_prompt("What is 2+2?", tok, system_prompt=None)

        call_args = tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"

    def test_with_system_prompt(self):
        """With a system prompt, messages should have system + user."""
        tok = self._make_tokenizer()
        format_prompt("What is 2+2?", tok, system_prompt="You are helpful.")

        call_args = tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    def test_enable_thinking_passed(self):
        """enable_thinking kwarg is forwarded to apply_chat_template."""
        tok = self._make_tokenizer()
        format_prompt("test", tok, enable_thinking=True)
        kwargs = tok.apply_chat_template.call_args[1]
        assert kwargs["enable_thinking"] is True

        format_prompt("test", tok, enable_thinking=False)
        kwargs = tok.apply_chat_template.call_args[1]
        assert kwargs["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Chat template rendering (using real Jinja2)
# ---------------------------------------------------------------------------


class TestChatTemplateRendering:
    """Test the actual Jinja2 template produces correct ChatML."""

    def _render(self, messages: list[dict], enable_thinking: bool = True) -> str:
        from jinja2 import Environment

        env = Environment()
        template = env.from_string(CHAT_TEMPLATE)
        return template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def test_basic_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self._render(messages, enable_thinking=False)

        assert "<|im_start|>user\nHello<|im_end|>\n" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_thinking_enabled(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self._render(messages, enable_thinking=True)

        assert result.endswith("<|im_start|>assistant\n<think>\n")

    def test_system_message(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        result = self._render(messages, enable_thinking=True)

        assert "<|im_start|>system\nBe concise.<|im_end|>\n" in result
        assert "<|im_start|>user\nHi<|im_end|>\n" in result
        assert result.strip().endswith("<think>")
