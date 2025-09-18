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
from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA

from prismatic.overwatch import initialize_overwatch
from transformers import AutoProcessor, InternVLForConditionalGeneration

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
def load_vlm(model_id, cache_dir=None,dtype=torch.bfloat16):
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
    tokenizer = processor.tokenizer 

    return vlm, tokenizer

def freeze_internvl(vlm, freeze_vision_backbone, freeze_projector, freeze_llm_backbone, freeze_last_llm_layer):
    if freeze_vision_backbone and hasattr(vlm, "vision_tower"):
        vlm.vision_tower.requires_grad_(False)
    if freeze_projector and hasattr(vlm, "multi_modal_projector"):
        vlm.multi_modal_projector.requires_grad_(False)
    if freeze_llm_backbone and hasattr(vlm, "language_model"):
        vlm.language_model.requires_grad_(False)
    if freeze_last_llm_layer and hasattr(vlm, "lm_head"):
        vlm.lm_head.requires_grad_(False)




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
