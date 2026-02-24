"""Local Ouro model implementation with upstream bug fixes.

Light imports (constants, chat template) are available without torch::

    from ouro_rl.modeling.constants import EOS_TOKEN_ID, CHAT_TEMPLATE

Heavy imports (OuroForCausalLM, etc.) are lazy-loaded on first access
so that ``import ouro_rl.modeling`` alone does not pull in torch.
"""

# Light — no torch/transformers
from ouro_rl.modeling.constants import (
    BOS_TOKEN_ID,
    CHAT_TEMPLATE,
    DEFAULT_MODEL,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
)

__all__ = [
    "BOS_TOKEN_ID",
    "CHAT_TEMPLATE",
    "DEFAULT_MODEL",
    "EOS_TOKEN_ID",
    "OuroConfig",
    "OuroForCausalLM",
    "OuroModel",
    "PAD_TOKEN_ID",
    "UniversalTransformerCache",
]

# Heavy imports — lazy-loaded to avoid pulling in torch on package import.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OuroConfig": ("ouro_rl.modeling.configuration_ouro", "OuroConfig"),
    "OuroForCausalLM": ("ouro_rl.modeling.modeling_ouro", "OuroForCausalLM"),
    "OuroModel": ("ouro_rl.modeling.modeling_ouro", "OuroModel"),
    "UniversalTransformerCache": ("ouro_rl.modeling.modeling_ouro", "UniversalTransformerCache"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        # Cache on the module so __getattr__ isn't called again.
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
