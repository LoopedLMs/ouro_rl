"""GPQA utilities for custom task YAMLs."""

import re

from lm_eval.tasks.gpqa.n_shot.utils import process_docs

__all__ = ["process_docs", "process_results"]

# Matches (A), (B), ... or standalone A, B, ... at word boundary.
_ANSWER_RE = re.compile(r"\(?([A-D])\)?(?:\s|$|[.,;:])", re.IGNORECASE)


def process_results(doc: dict, results: list[str]) -> dict:
    """Extract last ABCD letter from model output, compare to gold."""
    response = results[0]

    # Strip thinking tags if present.
    if "</think>" in response:
        response = response.split("</think>")[-1]

    matches = _ANSWER_RE.findall(response)
    pred = f"({matches[-1].upper()})" if matches else ""
    gold = doc["answer"]
    return {"exact_match": float(pred == gold)}
