"""Monkey-patches for upstream Ouro model bugs on HuggingFace.

Apply once at import time before loading the model:

    from ouro_rl.patches import patch_ouro
    patch_ouro()

Fixes:
    1. UniversalTransformerCache.get_mask_sizes — returns wrong KV length
       during autoregressive steps (always returns query_length instead of
       cached_length + query_length). This makes the 4D attention mask too
       small, so padding positions get broadcasted away and batched generation
       is corrupted for all sequences except the longest (unpadded) one.

    2. Wrong eos/bos/pad token IDs — upstream config sets all three to
       <|endoftext|> (id=0). Correct: bos=<|im_start|> (1), eos=<|im_end|> (2).
       We fix these on the tokenizer after loading.
"""

import importlib
import logging

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Correct token IDs for ChatML (upstream has all three set to 0).
CORRECT_BOS_TOKEN_ID = 1  # <|im_start|>
CORRECT_EOS_TOKEN_ID = 2  # <|im_end|>
PAD_TOKEN_ID = 0  # <|endoftext|> — safe for padding since it's NOT the real EOS

_PATCHED = False


def _patched_get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
    """Return (kv_length, kv_offset) accounting for cached tokens.

    The original inherited Cache.get_mask_sizes falls through to
    ``return cache_position.shape[0], 0`` because
    UniversalTransformerCache.layers is always empty.  During autoregressive
    decoding cache_position has length 1, so the mask is built for
    kv_length=1 instead of the full cached sequence length + 1.
    """
    query_length = cache_position.shape[0]
    seq_length = self.get_seq_length(layer_idx)
    kv_length = seq_length + query_length
    return kv_length, 0


def patch_ouro() -> None:
    """Apply all Ouro model patches. Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    # --- Patch 1: UniversalTransformerCache.get_mask_sizes ---
    # The module is auto-downloaded by trust_remote_code, so we need to handle
    # both the case where it's already imported and where it isn't yet.
    try:
        mod = importlib.import_module(
            "transformers_modules.ByteDance."
            "Ouro_hyphen_1_dot_4B_hyphen_Thinking."
            "9b5b209ac9659127e330672162e821f05a9131bb."
            "modeling_ouro"
        )
        _patch_cache_class(mod.UniversalTransformerCache)
        logger.info("Patched UniversalTransformerCache.get_mask_sizes (1.4B)")
    except (ImportError, AttributeError):
        logger.debug("1.4B modeling module not yet loaded, will patch on first access")

    try:
        mod = importlib.import_module("transformers_modules.ByteDance.Ouro_hyphen_2_dot_6B_hyphen_Thinking.modeling_ouro")
        _patch_cache_class(mod.UniversalTransformerCache)
        logger.info("Patched UniversalTransformerCache.get_mask_sizes (2.6B)")
    except (ImportError, AttributeError):
        logger.debug("2.6B modeling module not yet loaded")

    _PATCHED = True


def _patch_cache_class(cache_cls: type) -> None:
    """Patch get_mask_sizes on a UniversalTransformerCache class."""
    cache_cls.get_mask_sizes = _patched_get_mask_sizes


def patch_ouro_post_load(model: object, tokenizer: AutoTokenizer) -> None:
    """Patches that need the model/tokenizer instances (call after from_pretrained).

    - Fixes UniversalTransformerCache if the module was loaded by trust_remote_code
      after patch_ouro() was called.
    - Fixes tokenizer eos/bos/pad token IDs.
    """
    # Patch the cache class from the actually-loaded module
    model_mod = type(model).__module__
    try:
        mod = importlib.import_module(model_mod.rsplit(".", 1)[0] + ".modeling_ouro")
        _patch_cache_class(mod.UniversalTransformerCache)
        logger.info("Patched UniversalTransformerCache.get_mask_sizes (post-load)")
    except (ImportError, AttributeError):
        pass

    # Fix token IDs
    tokenizer.bos_token_id = CORRECT_BOS_TOKEN_ID
    tokenizer.eos_token_id = CORRECT_EOS_TOKEN_ID
    tokenizer.pad_token_id = PAD_TOKEN_ID
    logger.info(
        "Fixed tokenizer token IDs: bos=%d, eos=%d, pad=%d",
        CORRECT_BOS_TOKEN_ID,
        CORRECT_EOS_TOKEN_ID,
        PAD_TOKEN_ID,
    )
