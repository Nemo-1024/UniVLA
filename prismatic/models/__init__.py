"""
prismatic.models

This package is pruned to support UniVLA's LatentVLAModel / LatentWorldVLA workflows.

Legacy OpenVLA registry/materialize utilities are intentionally removed as part of cleanup.
"""

from .vlm_auto import (
    load_vlm_auto,
    freeze_vlm_generic,
    freeze_qwen3vl,
    freeze_internvl,
    _get_nested_attr,
    _resolve_llm_module,
    _unfreeze_last_n_llm_layers,
)

__all__ = [
    "load_vlm_auto",
    "freeze_vlm_generic",
    "freeze_qwen3vl",
    "freeze_internvl",
    "_get_nested_attr",
    "_resolve_llm_module",
    "_unfreeze_last_n_llm_layers",
]

