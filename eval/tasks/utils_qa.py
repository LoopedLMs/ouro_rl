"""Re-export upstream GPQA utilities for custom task YAMLs."""

from lm_eval.tasks.gpqa.n_shot.utils import process_docs

__all__ = ["process_docs"]
