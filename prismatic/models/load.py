"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn as nn
from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA

from prismatic.overwatch import initialize_overwatch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    InternVLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    # 返回注册表中可读名称列表
    return list(GLOBAL_REGISTRY.keys())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load_InternVL(model_id, cache_dir=None, dtype=torch.bfloat16):
    """加载预训练 VLM。"""

    vlm = InternVLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
        device_map="cpu",  # 避免多进程默认加载到 cuda:0；后续由 Accelerate 迁移到各自 GPU
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
    )

    return vlm, processor


def load_vlm_auto(model_id, cache_dir=None, dtype=torch.bfloat16):
    """
    通用加载接口：优先按 Vision2Seq，其次 Seq2Seq/CAUSAL LM，均允许 trust_remote_code。
    返回 (vlm, processor)。
    """
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
    )

    last_err = None
    for loader in (Qwen3VLForConditionalGeneration, InternVLForConditionalGeneration, AutoModelForVision2Seq, AutoModelForSeq2SeqLM, AutoModelForCausalLM):
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

def load_Qwen3VL(model_id, cache_dir=None, dtype=torch.bfloat16):
    """加载预训练 VLM。"""

    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
        device_map="cpu",  # 避免多进程默认加载到 cuda:0；后续由 Accelerate 迁移到各自 GPU
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
    )

    return vlm, processor    
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

    Architecture (HF):
      - vlm.model.visual: Qwen3VLVisionModel (patch_embed/pos_embed/blocks/merger/deepstack_merger_list)
      - vlm.model.language_model: Qwen3VLTextModel (embed_tokens/layers/...)
      - vlm.lm_head: Linear(...)

    This is intentionally explicit (less "generic") to avoid brittle heuristics for Qwen3-VL.
    """

    # ---- Vision ----
    visual = _get_nested_attr(vlm, "model.visual") or _get_nested_attr(vlm, "visual")
    if freeze_vision_backbone and visual is not None:
        try:
            # Freeze everything in visual by default
            visual.requires_grad_(False)
        except Exception:
            pass

        # Optionally unfreeze only merger modules (cheap adaptation)
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

    # ---- Language ----
    # Explicit path for Qwen3-VL
    language_model = _get_nested_attr(vlm, "model.language_model") or _get_nested_attr(vlm, "language_model")
    if freeze_llm_backbone:
        if language_model is not None:
            try:
                language_model.requires_grad_(False)
            except Exception:
                pass
        else:
            # fallback (shouldn't happen for Qwen3-VL)
            try:
                vlm.requires_grad_(False)
            except Exception:
                pass

        # Keep embeddings trainable if requested (embedding is under language_model.embed_tokens)
        if not freeze_embedding:
            try:
                emb = None
                if hasattr(vlm, "get_input_embeddings"):
                    emb = vlm.get_input_embeddings()
                if emb is None and language_model is not None and hasattr(language_model, "embed_tokens"):
                    emb = language_model.embed_tokens
                if emb is not None:
                    emb.requires_grad_(True)
                    try:
                        overwatch.info("[freeze_qwen3vl] Kept Qwen3VL embed_tokens trainable (freeze_llm_backbone=True, freeze_embedding=False)")
                    except Exception:
                        pass
            except Exception:
                pass

        # Keep lm_head trainable unless explicitly requested to freeze it
        if not freeze_last_llm_layer:
            try:
                if hasattr(vlm, "lm_head") and vlm.lm_head is not None:
                    vlm.lm_head.requires_grad_(True)
            except Exception:
                pass

    # Explicit embedding freeze if requested (works even when freeze_llm_backbone=False)
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


def _freeze_first_n_llm_layers(llm_module, freeze_llm_first_n_layers: Optional[int]) -> bool:
    """
    Freeze the first N transformer layers if they can be located on the LLM module.
    Returns True if any layers were frozen, otherwise False.
    """
    if freeze_llm_first_n_layers is None or freeze_llm_first_n_layers <= 0:
        return False

    # Common layer container attribute paths across popular HF LLMs
    candidate_layer_paths = [
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
        "layers",
        "h",
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
        overwatch.warning(
            f"[freeze_vlm_generic] Failed to locate LLM layers; tried paths: {candidate_layer_paths}"
        )
        return False

    num_layers = len(layers_container)
    n = min(int(freeze_llm_first_n_layers), num_layers)
    if n <= 0:
        return False

    for layer in list(layers_container)[:n]:
        try:
            layer.requires_grad_(False)
        except Exception:
            continue

    overwatch.info(f"[freeze_vlm_generic] Froze first {n}/{num_layers} LLM layers")
    return True


def _unfreeze_last_n_llm_layers(llm_module, n: int) -> bool:
    """
    Unfreeze the last N transformer layers if they can be located on the LLM module.
    Returns True if any layers were unfrozen, otherwise False.
    """
    if n is None or n <= 0:
        return False

    # Common layer container attribute paths across popular HF LLMs
    # 优先检查直接属性（最常见的情况，如 Qwen3VLTextModel.layers）
    candidate_layer_paths = [
        "layers",  # 最常见：直接属性（Qwen3VL, InternVL, LLaMA等）
        "h",  # GPT-2, GPT-J 等
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
        overwatch.warning(
            f"[unfreeze_vlm_generic] Failed to locate LLM layers; tried paths: {candidate_layer_paths}"
        )
        return False

    num_layers = len(layers_container)
    n_layers = min(int(n), num_layers)
    if n_layers <= 0:
        return False

    # 解冻最后n层
    for layer in list(layers_container)[-n_layers:]:
        try:
            layer.requires_grad_(True)
        except Exception:
            continue

    overwatch.info(f"[unfreeze_vlm_generic] Unfroze last {n_layers}/{num_layers} LLM layers")
    return True


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
    针对通用 HF VLM 的冻结逻辑：按常见子模块名称尝试冻结，未找到则跳过。
    """
    if freeze_vision_backbone:
        for name in ["vision_tower", "visual", "vision_model", "vision_encoder", "vision_modules"]:
            if hasattr(vlm, name):
                getattr(vlm, name).requires_grad_(False)
        if hasattr(vlm, "model"):
            if hasattr(vlm.model, "vision_tower"):
                vlm.model.vision_tower.requires_grad_(False)
            visual = _get_nested_attr(vlm, "model.visual")
            if visual is not None:
                visual.requires_grad_(False)

    # Optionally keep only the "merger" trainable inside the (otherwise frozen) vision backbone.
    # This is useful for Qwen3-VL style vision model where `vision_model.merger` / `deepstack_merger_list`
    # performs patch merging and can adapt cheaply while keeping `blocks` frozen.
    if unfreeze_vision_merger:
        try:
            # Try to locate the vision module across common nesting schemes.
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

            if vision_module is None:
                try:
                    overwatch.warning(
                        "[freeze_vlm_generic] unfreeze_vision_merger=True but failed to locate vision module; "
                        f"tried paths={vision_candidates}"
                    )
                except Exception:
                    pass
            else:
                unfroze_any = False
                # main merger
                if hasattr(vision_module, "merger"):
                    try:
                        vision_module.merger.requires_grad_(True)
                        unfroze_any = True
                    except Exception:
                        pass
                # deepstack mergers (if present)
                if hasattr(vision_module, "deepstack_merger_list"):
                    try:
                        vision_module.deepstack_merger_list.requires_grad_(True)
                        unfroze_any = True
                    except Exception:
                        pass
                if not unfroze_any:
                    try:
                        overwatch.warning(
                            "[freeze_vlm_generic] unfreeze_vision_merger=True but vision module has no "
                            "`merger` / `deepstack_merger_list` attributes."
                        )
                    except Exception:
                        pass
                else:
                    try:
                        overwatch.info("[freeze_vlm_generic] Kept vision merger trainable (unfreeze_vision_merger=True)")
                    except Exception:
                        pass
        except Exception:
            # Best-effort; do not break training if model structure differs.
            pass

    if freeze_projector:
        for name in ["multi_modal_projector", "vision_proj", "projector"]:
            if hasattr(vlm, name):
                getattr(vlm, name).requires_grad_(False)

    # Freeze embedding layer
    if freeze_embedding:
        frozen = False
        # First try get_input_embeddings() method (most common in HF models, including Qwen3VL)
        if hasattr(vlm, "get_input_embeddings"):
            try:
                emb = vlm.get_input_embeddings()
                if emb is not None:
                    emb.requires_grad_(False)
                    frozen = True
                    overwatch.info("[freeze_vlm_generic] Froze embedding via get_input_embeddings()")
            except Exception as e:
                overwatch.debug(f"[freeze_vlm_generic] get_input_embeddings() failed: {e}")
        
        # If not frozen yet, try direct attribute access
        # Order: most common paths first (Qwen3VL uses model.language_model.embed_tokens)
        if not frozen:
            embedding_candidates = [
                "model.language_model.embed_tokens",  # Qwen3VL, InternVL, etc.
                "model.embed_tokens",  # Common in many models
                "language_model.embed_tokens",  # Alternative nesting
                "embed_tokens",  # Direct access
                "model.text_model.embed_tokens",  # Some models use text_model
                "text_model.embed_tokens",
                "model.embedding",  # Alternative name
                "embedding",
            ]
            for path in embedding_candidates:
                emb = _get_nested_attr(vlm, path)
                if emb is not None:
                    try:
                        emb.requires_grad_(False)
                        frozen = True
                        overwatch.info(f"[freeze_vlm_generic] Froze embedding via {path}")
                        break
                    except Exception as e:
                        overwatch.debug(f"[freeze_vlm_generic] Failed to freeze via {path}: {e}")
                        continue
        
        if not frozen:
            overwatch.warning(
                "[freeze_vlm_generic] Failed to locate embedding layer. "
                "Tried get_input_embeddings() and common attribute paths. "
                "Please check model architecture manually."
            )

    # Identify LLM backbone once for reuse
    llm_module = _resolve_llm_module(vlm)

    if freeze_llm_backbone:
        if llm_module is not None:
            llm_module.requires_grad_(False)
        else:
            vlm.requires_grad_(False)

        # If the user explicitly wants embeddings trainable, make sure freezing the backbone didn't
        # inadvertently freeze them (many architectures place embed_tokens under the LLM module).
        if not freeze_embedding:
            try:
                emb = None
                if hasattr(vlm, "get_input_embeddings"):
                    emb = vlm.get_input_embeddings()
                if emb is None:
                    # Common nesting for Qwen3-VL / InternVL style models
                    emb = _get_nested_attr(vlm, "model.language_model.embed_tokens")
                    if emb is None:
                        emb = _get_nested_attr(vlm, "language_model.embed_tokens")
                    if emb is None:
                        emb = _get_nested_attr(vlm, "model.embed_tokens")
                if emb is not None:
                    emb.requires_grad_(True)
                    try:
                        overwatch.info("[freeze_vlm_generic] Kept input embeddings trainable (freeze_llm_backbone=True, freeze_embedding=False)")
                    except Exception:
                        pass
            except Exception:
                pass

        # Similarly, keep the output head trainable unless explicitly requested to freeze it.
        # Some models attach lm_head under the LLM module, so freezing the backbone would freeze it too.
        if not freeze_last_llm_layer:
            try:
                for name in ["lm_head", "generator", "cls"]:
                    if hasattr(vlm, name):
                        getattr(vlm, name).requires_grad_(True)
                # Also try common nested paths
                nested_heads = [
                    "model.language_model.lm_head",
                    "language_model.lm_head",
                    "model.lm_head",
                ]
                for path in nested_heads:
                    head = _get_nested_attr(vlm, path)
                    if head is not None:
                        head.requires_grad_(True)
                try:
                    overwatch.info("[freeze_vlm_generic] Kept lm_head trainable (freeze_llm_backbone=True, freeze_last_llm_layer=False)")
                except Exception:
                    pass
            except Exception:
                pass

    if freeze_last_llm_layer:
        for name in ["lm_head", "generator", "cls"]:
            if hasattr(vlm, name):
                getattr(vlm, name).requires_grad_(False)




# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
    action_codebook_size: int = 32,
) -> OpenVLA:
    """Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify Checkpoint to Load (via `step_to_load`)
        step_to_load_str = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(
            f"{hf_path}/checkpoints/step-{step_to_load_str if step_to_load_str is not None else ''}*.pt"
        )
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with checkpoint `{checkpoint_pt}`"
    )

    # 直接通过 HF Auto 类加载 VLA（通过其 trust_remote_code 实现）
    from transformers import AutoModelForVision2Seq, AutoProcessor

    AutoProcessor.register(model_cfg.__class__, None)  # 兼容占位，实际项目中应注册自定义 Processor
    vla = AutoModelForVision2Seq.from_pretrained(
        run_dir.as_posix() if os.path.isfile(model_id_or_path) else str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path),
        trust_remote_code=True,
    )

    return vla
