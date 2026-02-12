"""
Prismatic (pruned)

This repo keeps a minimal subset of the original `prismatic` package to support UniVLA's
LatentVLAModel / LatentWorldVLA workflows.
"""

from .models import freeze_qwen3vl, freeze_vlm_generic, load_vlm_auto

__all__ = ["load_vlm_auto", "freeze_vlm_generic", "freeze_qwen3vl"]

__version__ = "0.0.1"
