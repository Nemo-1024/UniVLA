"""
vlm_auto.py

Minimal HuggingFace VLM loader + freezing utilities used by:
- `prismatic/vla/latent_vla_model.py`
- `prismatic/models/vlas/latent_world_vla.py`
- `prismatic/training/accelerate_fsdp_trainer.py`

This file is intentionally self-contained and must NOT depend on OpenVLA/registry/conf so that
the `prismatic/` tree can be pruned aggressively.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from prismatic.overwatch import initialize_overwatch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    InternVLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


overwatch = initialize_overwatch(__name__)


def load_vlm_auto(model_id, cache_dir=None, dtype: torch.dtype = torch.bfloat16):
    """
    Generic VLM loader.

    Returns: (vlm, processor)
    - Always loads the processor via AutoProcessor(trust_remote_code=True).
    - Tries a small set of common model classes (Qwen3-VL, InternVL, Vision2Seq, Seq2Seq, CausalLM).
    """
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
    )

    last_err = None
    for loader in (
        Qwen3VLForConditionalGeneration,
        InternVLForConditionalGeneration,
        AutoModelForVision2Seq,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    ):
        try:
            vlm = loader.from_pretrained(
                model_id,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=dtype,
            )
            return vlm, processor
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load VLM `{model_id}` via generic loaders") from last_err


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _resolve_llm_module(vlm):
    """
    Try to locate the language/backbone module across common nesting schemes.
    """
    candidate_llm_paths = [
        "language_model",
        "model.language_model",
        "text_model",
        "model.text_model",
        "transformer",
        "model.decoder",
        "decoder",
        "model",
    ]
    for path in candidate_llm_paths:
        llm = _get_nested_attr(vlm, path)
        if llm is not None:
            return llm
    return None


def _unfreeze_last_n_llm_layers(llm_module, n: int) -> bool:
    """
    Unfreeze the last N transformer layers if they can be located on the LLM module.
    Returns True if any layers were unfrozen, otherwise False.
    """
    if n is None or n <= 0:
        return False

    candidate_layer_paths = [
        "layers",
        "h",
        "language_model.layers",
        "language_model.model.layers",
        "language_model.decoder.layers",
        "language_model.decoder.layer",
        "language_model.transformer.h",
        "model.layers",
        "model.decoder.layers",
        "model.encoder.layers",
        "decoder.layers",
        "decoder.layer",
        "encoder.layers",
        "encoder.layer",
        "transformer.h",
        "transformer.layers",
        "transformer.blocks",
        "transformer.block",
        "blocks",
        "block",
    ]

    layers_container = None
    for path in candidate_layer_paths:
        candidate = _get_nested_attr(llm_module, path)
        if isinstance(candidate, (list, nn.ModuleList)):
            layers_container = candidate
            break

    if layers_container is None:
        try:
            overwatch.warning(
                f"[vlm_auto._unfreeze_last_n_llm_layers] Failed to locate LLM layers; tried paths: {candidate_layer_paths}"
            )
        except Exception:
            pass
        return False

    num_layers = len(layers_container)
    n_layers = min(int(n), num_layers)
    if n_layers <= 0:
        return False

    for layer in list(layers_container)[-n_layers:]:
        try:
            layer.requires_grad_(True)
        except Exception:
            continue

    try:
        overwatch.info(f"[vlm_auto._unfreeze_last_n_llm_layers] Unfroze last {n_layers}/{num_layers} LLM layers")
    except Exception:
        pass
    return True


def freeze_qwen3vl(
    vlm,
    freeze_vision_backbone,
    freeze_llm_backbone,
    freeze_last_llm_layer,
    freeze_embedding: bool = False,
    unfreeze_vision_merger: bool = False,
):
    """
    Qwen3-VL specific freezing logic with explicit module paths.
    """
    visual = _get_nested_attr(vlm, "model.visual") or _get_nested_attr(vlm, "visual")
    if freeze_vision_backbone and visual is not None:
        try:
            visual.requires_grad_(False)
        except Exception:
            pass

        if unfreeze_vision_merger:
            unfroze_any = False
            try:
                if hasattr(visual, "merger"):
                    visual.merger.requires_grad_(True)
                    unfroze_any = True
            except Exception:
                pass
            try:
                if hasattr(visual, "deepstack_merger_list"):
                    visual.deepstack_merger_list.requires_grad_(True)
                    unfroze_any = True
            except Exception:
                pass
            if unfroze_any:
                try:
                    overwatch.info("[freeze_qwen3vl] Kept Qwen3VL vision merger trainable (unfreeze_vision_merger=True)")
                except Exception:
                    pass

    language_model = _get_nested_attr(vlm, "model.language_model") or _get_nested_attr(vlm, "language_model")
    if freeze_llm_backbone:
        if language_model is not None:
            try:
                language_model.requires_grad_(False)
            except Exception:
                pass
        else:
            try:
                vlm.requires_grad_(False)
            except Exception:
                pass

        if not freeze_embedding:
            try:
                emb = None
                if hasattr(vlm, "get_input_embeddings"):
                    emb = vlm.get_input_embeddings()
                if emb is None and language_model is not None and hasattr(language_model, "embed_tokens"):
                    emb = language_model.embed_tokens
                if emb is not None:
                    emb.requires_grad_(True)
            except Exception:
                pass

        if not freeze_last_llm_layer:
            try:
                if hasattr(vlm, "lm_head") and vlm.lm_head is not None:
                    vlm.lm_head.requires_grad_(True)
            except Exception:
                pass

    if freeze_embedding:
        try:
            emb = None
            if hasattr(vlm, "get_input_embeddings"):
                emb = vlm.get_input_embeddings()
            if emb is None and language_model is not None and hasattr(language_model, "embed_tokens"):
                emb = language_model.embed_tokens
            if emb is not None:
                emb.requires_grad_(False)
        except Exception:
            pass

    if freeze_last_llm_layer:
        try:
            if hasattr(vlm, "lm_head") and vlm.lm_head is not None:
                vlm.lm_head.requires_grad_(False)
        except Exception:
            pass


def freeze_internvl(
    vlm,
    freeze_vision_backbone,
    freeze_projector,
    freeze_llm_backbone,
    freeze_last_llm_layer,
):
    if freeze_vision_backbone and hasattr(vlm, "vision_tower"):
        vlm.vision_tower.requires_grad_(False)
    if freeze_projector and hasattr(vlm, "multi_modal_projector"):
        vlm.multi_modal_projector.requires_grad_(False)
    llm_module = _resolve_llm_module(vlm)
    if freeze_llm_backbone and llm_module is not None:
        llm_module.requires_grad_(False)
    if freeze_last_llm_layer and hasattr(vlm, "lm_head"):
        vlm.lm_head.requires_grad_(False)


def freeze_vlm_generic(
    vlm,
    freeze_vision_backbone,
    freeze_projector,
    freeze_llm_backbone,
    freeze_last_llm_layer,
    freeze_embedding: bool = False,
    unfreeze_vision_merger: bool = False,
):
    """
    Generic freezing logic for HF VLMs. Best-effort; skips unknown submodules.
    """
    if freeze_vision_backbone:
        for name in ["vision_tower", "visual", "vision_model", "vision_encoder", "vision_modules"]:
            if hasattr(vlm, name):
                try:
                    getattr(vlm, name).requires_grad_(False)
                except Exception:
                    pass
        if hasattr(vlm, "model"):
            try:
                if hasattr(vlm.model, "vision_tower"):
                    vlm.model.vision_tower.requires_grad_(False)
            except Exception:
                pass
            visual = _get_nested_attr(vlm, "model.visual")
            if visual is not None:
                try:
                    visual.requires_grad_(False)
                except Exception:
                    pass

    if unfreeze_vision_merger:
        try:
            vision_candidates = [
                "vision_model",
                "model.vision_model",
                "visual",
                "model.visual",
                "vision_tower",
                "model.vision_tower",
                "vision_encoder",
                "model.vision_encoder",
            ]
            vision_module = None
            for path in vision_candidates:
                vision_module = _get_nested_attr(vlm, path)
                if vision_module is not None:
                    break
            if vision_module is not None:
                if hasattr(vision_module, "merger"):
                    try:
                        vision_module.merger.requires_grad_(True)
                    except Exception:
                        pass
                if hasattr(vision_module, "deepstack_merger_list"):
                    try:
                        vision_module.deepstack_merger_list.requires_grad_(True)
                    except Exception:
                        pass
        except Exception:
            pass

    if freeze_projector:
        for name in ["multi_modal_projector", "mm_projector", "projector"]:
            if hasattr(vlm, name):
                try:
                    getattr(vlm, name).requires_grad_(False)
                except Exception:
                    pass
        proj = _get_nested_attr(vlm, "model.multi_modal_projector")
        if proj is not None:
            try:
                proj.requires_grad_(False)
            except Exception:
                pass

    if freeze_llm_backbone:
        llm_module = _resolve_llm_module(vlm)
        if llm_module is not None:
            try:
                llm_module.requires_grad_(False)
            except Exception:
                pass
        else:
            try:
                vlm.requires_grad_(False)
            except Exception:
                pass

        if not freeze_embedding:
            try:
                emb = vlm.get_input_embeddings() if hasattr(vlm, "get_input_embeddings") else None
                if emb is not None:
                    emb.requires_grad_(True)
            except Exception:
                pass

        if not freeze_last_llm_layer:
            try:
                for name in ["lm_head", "generator", "cls"]:
                    if hasattr(vlm, name):
                        getattr(vlm, name).requires_grad_(True)
            except Exception:
                pass

    if freeze_embedding:
        try:
            emb = vlm.get_input_embeddings() if hasattr(vlm, "get_input_embeddings") else None
            if emb is not None:
                emb.requires_grad_(False)
        except Exception:
            pass

    if freeze_last_llm_layer:
        for name in ["lm_head", "generator", "cls"]:
            if hasattr(vlm, name):
                try:
                    getattr(vlm, name).requires_grad_(False)
                except Exception:
                    pass

