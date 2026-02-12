import enum
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model
from prismatic.models import freeze_qwen3vl, _resolve_llm_module, _unfreeze_last_n_llm_layers
from prismatic.models.vlm_auto import load_vlm_auto

from .flowmatching_expert import ConditionalFlowMatchingConfig, ConditionalFlowMatchingHead


# ============================================================================
# Config & Constants
# ============================================================================

HOME_PATH = "/mnt/mnt/public/jlchen"


class FutureFeatureMode(str, enum.Enum):
    """Reserved for future extension of future-feature sources."""

    LAM_FROM_VLM = "lam_from_vlm"
    LAM_FROM_GT = "lam_from_gt"
    VJEPA_GT = "vjepa_gt"


@dataclass
class LatentWorldVLAConfig:
    """Independent LatentWorldVLA config."""

    # Inference controls
    cfg_guidance_scale: float = 1.5
    num_inference_steps: int = 50

    # Flow head config
    flow_cfg: ConditionalFlowMatchingConfig = field(default_factory=ConditionalFlowMatchingConfig)

    # Base checkpoints
    model_id: str = (
        HOME_PATH
        + "/code/UniVLA/vla_scripts/vla_log/0107_103411+vla_emb_unfreeze_llm4_libero_90/checkpoints/checkpoint-10000"
    )
    hf_cache_dir: Optional[Union[str, Path]] = None
    lam_ckpt_path: str = (
        HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bridge/version_0/checkpoints/epoch=39.ckpt"
    )
    lam_yaml_path: str = HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bridge/version_0/dino_base_ae.yaml"

    # Optional YAML override used by from_config
    yaml_path: Optional[Union[str, Path]] = None

    # VLM dtype
    vlm_dtype: torch.dtype = torch.bfloat16

    # Freeze policy
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False
    unfreeze_llm_last_n_layers: Optional[int] = None
    unfreeze_lam_decoder: bool = False

    # Placeholder token
    latent_action_placeholder_token: str = "<ACT_PH>"

    # World / distill losses
    perceptual_weight: float = 0.1
    enable_loss_distill: bool = True
    latent_loss_type: str = "mse"  # "mse" | "cosine"
    lam_encoder_distill_weight: float = 1.0

    # Flow conditioning variants
    future_prediction: bool = False
    repeated_diffusion_steps: int = 4
    enable_wrist_view: bool = False
    # Flow-only 梯度模式：仅允许 flow placeholder 位置的 VLM hidden 接收 flow loss 梯度
    flow_only_mode: bool = False

    # Independent additions
    num_action_queries: int = 8
    extra_ckpt_name: str = "latent_world_vla_extra.pt"


# ============================================================================
# Lightweight Blocks
# ============================================================================


class VLMToLAMQFormer(nn.Module):
    """Refine VLM query hidden states into one latent action embedding."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("[VLMToLAMQFormer] num_layers must be > 0.")
        if num_heads <= 0:
            raise ValueError("[VLMToLAMQFormer] num_heads must be > 0.")
        if ffn_expansion_factor <= 0:
            raise ValueError("[VLMToLAMQFormer] ffn_expansion_factor must be > 0.")

        self.query = nn.Parameter(torch.randn(1, 1, int(lam_code_dim)) * 0.02)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=int(lam_code_dim),
                    kdim=int(vlm_hidden_dim),
                    vdim=int(vlm_hidden_dim),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_qs = nn.ModuleList([nn.LayerNorm(int(lam_code_dim)) for _ in range(int(num_layers))])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(int(vlm_hidden_dim)) for _ in range(int(num_layers))])
        hidden_dim = int(int(lam_code_dim) * float(ffn_expansion_factor))
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(lam_code_dim)),
                    nn.Linear(int(lam_code_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, int(lam_code_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() != 3:
            raise ValueError(f"[VLMToLAMQFormer] expected context [B, Q, D], got {tuple(context.shape)}")
        bsz = int(context.shape[0])
        queries = self.query.expand(bsz, -1, -1).to(device=context.device, dtype=context.dtype)
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        return self.final_norm(queries)


# ============================================================================
# VLM / LAM Load Helpers
# ============================================================================


def _apply_yaml_overrides(
    cfg: LatentWorldVLAConfig,
    *,
    preserve_checkpoint_model_id: bool,
) -> None:
    if cfg.yaml_path is None:
        return

    import yaml

    yaml_path = Path(cfg.yaml_path)
    if not yaml_path.exists():
        print(f"[LatentWorldVLA] yaml override skipped: file not found `{yaml_path}`")
        return

    try:
        with open(str(yaml_path), "r", encoding="utf-8") as f:
            ycfg = yaml.safe_load(f) or {}

        if isinstance(ycfg, dict):
            for nest_key in ("train", "config", "cfg"):
                if nest_key in ycfg and isinstance(ycfg[nest_key], dict):
                    ycfg = ycfg[nest_key]
                    break

        keep_model_id = False
        model_id_val = str(cfg.model_id or "")
        if preserve_checkpoint_model_id and model_id_val and os.path.isdir(model_id_val):
            keep_model_id = True

        def _cast_like(key: str, cur: Any, val: Any) -> Any:
            if key == "model_id" and keep_model_id:
                return cur
            if isinstance(cur, bool):
                return bool(val)
            if isinstance(cur, int) and not isinstance(cur, bool):
                return int(val)
            if isinstance(cur, float):
                return float(val)
            if isinstance(cur, Path):
                return Path(val)
            if cur is None and key in (
                "hf_cache_dir",
                "data_root_dir",
                "run_root_dir",
                "yaml_path",
            ) and isinstance(val, str):
                return Path(val) if val else None
            return val

        if isinstance(ycfg, dict):
            for k, v in ycfg.items():
                try:
                    cur = getattr(cfg, k, None)
                    setattr(cfg, k, _cast_like(str(k), cur, v))
                except Exception:
                    pass
        print(f"[LatentWorldVLA] Applied yaml overrides from `{yaml_path}` (preserve_model_id={keep_model_id})")
    except Exception as e:
        print(f"[LatentWorldVLA] Yaml override skipped/failed: {e}")


def _load_vlm_and_processor(
    cfg: LatentWorldVLAConfig,
) -> Tuple[nn.Module, Any, Any, int]:
    vlm, processor = load_vlm_auto(cfg.model_id, cfg.hf_cache_dir, dtype=cfg.vlm_dtype)
    tokenizer = processor.tokenizer

    placeholder_token = str(cfg.latent_action_placeholder_token)
    tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})  # type: ignore[attr-defined]
    placeholder_token_id = int(tokenizer.convert_tokens_to_ids(placeholder_token))
    if placeholder_token_id < 0:
        raise ValueError(f"[LatentWorldVLA] invalid placeholder token id for `{placeholder_token}`")

    if hasattr(vlm, "generation_config") and vlm.generation_config is not None:
        vlm.generation_config.max_new_tokens = 4
    if hasattr(vlm, "config") and vlm.config is not None:
        try:
            vlm.config.loss_type = "ForCausalLMLoss"
            vlm.config.use_cache = False
        except Exception:
            pass

    return vlm, processor, tokenizer, placeholder_token_id


def _load_lam(cfg: LatentWorldVLAConfig) -> nn.Module:
    return load_latent_action_model(cfg.lam_ckpt_path, cfg.lam_yaml_path)


def _apply_freeze_policy(vlm: nn.Module, cfg: LatentWorldVLAConfig) -> None:
    freeze_qwen3vl(
        vlm,
        freeze_vision_backbone=cfg.freeze_vision_backbone,
        freeze_llm_backbone=cfg.freeze_llm_backbone,
        freeze_last_llm_layer=cfg.freeze_last_llm_layer,
        freeze_embedding=cfg.freeze_embedding,
        unfreeze_vision_merger=cfg.unfreeze_vision_merger,
    )
    if cfg.freeze_llm_backbone and cfg.unfreeze_llm_last_n_layers:
        llm_module = _resolve_llm_module(vlm)
        if llm_module is not None:
            _unfreeze_last_n_llm_layers(llm_module, cfg.unfreeze_llm_last_n_layers)


# ============================================================================
# Query Injection Helpers
# ============================================================================


def _inject_queries_into_embeddings(
    *,
    inputs_embeds: torch.Tensor,
    placeholder_mask: torch.BoolTensor,
    queries: torch.Tensor,
    num_queries: int,
    name: str,
) -> None:
    device = inputs_embeds.device
    placeholder_mask = placeholder_mask.to(device=device, dtype=torch.bool)

    bsz, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
    expected = int(bsz * num_queries)
    got = int(placeholder_mask.sum().item())
    if got != expected:
        raise ValueError(
            f"[LatentWorldVLA] {name} placeholder count mismatch: got={got}, expected={expected}"
        )

    per_sample = placeholder_mask.sum(dim=1)
    if not torch.all(per_sample == int(num_queries)):
        bad = torch.nonzero(per_sample != int(num_queries), as_tuple=False).flatten()
        b = int(bad[0].item()) if bad.numel() > 0 else -1
        got_b = int(per_sample[b].item()) if b >= 0 else -1
        raise ValueError(
            f"[LatentWorldVLA] {name} placeholder count mismatch for sample {b}: got={got_b}, expected={num_queries}"
        )

    qvec = queries.to(device=device, dtype=inputs_embeds.dtype)
    idx = placeholder_mask.nonzero(as_tuple=False)
    flat = idx[:, 0] * seq_len + idx[:, 1]
    idx = idx[flat.argsort()]
    b_idx, p_idx = idx[:, 0], idx[:, 1]
    q_idx = torch.arange(int(num_queries), device=device).repeat(bsz)
    inputs_embeds[b_idx, p_idx, :] = qvec[q_idx]


def _vlm_forward_with_queries(
    *,
    vlm: nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    image_grid_thw: Optional[torch.LongTensor],
    act_placeholder_mask: torch.BoolTensor,
    act_query: torch.Tensor,
    act_num_queries: int,
    flow_placeholder_mask: torch.BoolTensor,
    flow_query: torch.Tensor,
    flow_num_queries: int,
):
    embed = vlm.get_input_embeddings()
    inputs_embeds = embed(input_ids)

    _inject_queries_into_embeddings(
        inputs_embeds=inputs_embeds,
        placeholder_mask=act_placeholder_mask,
        queries=act_query,
        num_queries=int(act_num_queries),
        name="act_query",
    )
    _inject_queries_into_embeddings(
        inputs_embeds=inputs_embeds,
        placeholder_mask=flow_placeholder_mask,
        queries=flow_query,
        num_queries=int(flow_num_queries),
        name="flow_query",
    )

    return vlm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=None,
        output_hidden_states=True,
    )


def _project_action_hidden_to_lam(
    *,
    hidden: torch.Tensor,
    act_placeholder_mask: torch.BoolTensor,
    num_queries: int,
    vlm_to_lam: nn.Module,
) -> torch.Tensor:
    bsz = int(hidden.shape[0])
    q = int(num_queries)
    h_act = hidden[act_placeholder_mask]
    if h_act.numel() == 0:
        raise ValueError("[LatentWorldVLA] empty action hidden selection; check act_placeholder_mask.")
    if h_act.shape[0] != bsz * q:
        raise ValueError(
            f"[LatentWorldVLA] action hidden count mismatch: got={h_act.shape[0]}, expected={bsz*q}"
        )

    h_act = h_act.view(bsz, q, -1)
    try:
        p0 = next(vlm_to_lam.parameters())
        target_dtype = p0.dtype
    except Exception:
        target_dtype = h_act.dtype
    if h_act.dtype != target_dtype:
        h_act = h_act.to(dtype=target_dtype)

    return vlm_to_lam(h_act)


def _infer_act_flow_masks(
    *,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    act_queries: int,
    flow_queries: int,
    act_placeholder_mask: Optional[torch.Tensor],
    flow_placeholder_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    act_q = int(act_queries)
    flow_q = int(flow_queries)
    if act_q <= 0 or flow_q <= 0:
        raise ValueError(f"[LatentWorldVLA] invalid query counts: act_q={act_q}, flow_q={flow_q}")

    if act_placeholder_mask is not None and flow_placeholder_mask is not None:
        act_mask = act_placeholder_mask.to(device=device, dtype=torch.bool)
        flow_mask = flow_placeholder_mask.to(device=device, dtype=torch.bool)
        if torch.any(act_mask & flow_mask):
            raise ValueError("[LatentWorldVLA] act_placeholder_mask and flow_placeholder_mask overlap.")
        if not torch.all(act_mask.sum(dim=1) == act_q):
            raise ValueError("[LatentWorldVLA] act_placeholder_mask count mismatch.")
        if not torch.all(flow_mask.sum(dim=1) == flow_q):
            raise ValueError("[LatentWorldVLA] flow_placeholder_mask count mismatch.")
        return act_mask, flow_mask

    placeholder_mask = (input_ids == int(placeholder_token_id)).to(dtype=torch.bool, device=device)
    bsz = int(placeholder_mask.shape[0])
    expected_total = int(act_q + flow_q)
    act_mask = torch.zeros_like(placeholder_mask, dtype=torch.bool)
    flow_mask = torch.zeros_like(placeholder_mask, dtype=torch.bool)

    for b in range(bsz):
        pos = torch.nonzero(placeholder_mask[b], as_tuple=False).flatten()
        if int(pos.numel()) != expected_total:
            raise ValueError(
                f"[LatentWorldVLA] placeholder count mismatch for sample {b}: "
                f"got={int(pos.numel())}, expected={expected_total}"
            )
        act_mask[b, pos[:act_q]] = True
        flow_mask[b, pos[act_q:expected_total]] = True

    return act_mask, flow_mask


def _apply_flow_only_grad_to_h_vlm(
    *,
    h_vlm: torch.Tensor,
    flow_placeholder_mask: torch.Tensor,
    enable_flow_only: bool,
) -> torch.Tensor:
    """
    Keep full VLM context values for flow, but restrict gradient to flow placeholder positions.
    """
    if not bool(enable_flow_only):
        return h_vlm
    if h_vlm.dim() != 3:
        raise ValueError(f"[LatentWorldVLA] expected h_vlm [B, L, D], got {tuple(h_vlm.shape)}")
    if flow_placeholder_mask is None:
        raise ValueError("[LatentWorldVLA] flow_only_mode=True requires flow_placeholder_mask.")
    if flow_placeholder_mask.dim() != 2:
        raise ValueError(
            f"[LatentWorldVLA] expected flow_placeholder_mask [B, L], got {tuple(flow_placeholder_mask.shape)}"
        )
    if flow_placeholder_mask.shape[0] != h_vlm.shape[0] or flow_placeholder_mask.shape[1] != h_vlm.shape[1]:
        raise ValueError(
            "[LatentWorldVLA] flow_placeholder_mask shape mismatch with h_vlm: "
            f"mask={tuple(flow_placeholder_mask.shape)} vs h_vlm={tuple(h_vlm.shape)}"
        )
    flow_mask = flow_placeholder_mask.to(device=h_vlm.device, dtype=torch.bool).unsqueeze(-1)
    flow_mask_f = flow_mask.to(dtype=h_vlm.dtype)
    return h_vlm * flow_mask_f + h_vlm.detach() * (1.0 - flow_mask_f)


# ============================================================================
# Loss & Distill Helpers
# ============================================================================


def _extract_lam_vision_features(lam: nn.Module, videos: torch.Tensor) -> torch.Tensor:
    return lam.extract_vision_features(videos)


def _build_lam_teacher_inputs_for_distill(
    *,
    lam: nn.Module,
    lam_videos: torch.Tensor,
    lam_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected_t = int(getattr(getattr(lam, "encoder", None), "num_frames", 0) or 0)
    if expected_t <= 0:
        return lam_videos, lam_states

    cur_t = int(lam_videos.shape[1])
    if cur_t == expected_t:
        return lam_videos, lam_states
    if cur_t < expected_t:
        raise ValueError(
            f"[LatentWorldVLA] distill teacher temporal mismatch: got T={cur_t}, "
            f"but LAM encoder expects num_frames={expected_t}."
        )

    idx = torch.linspace(0, cur_t - 1, steps=expected_t, device=lam_videos.device).round().long()
    return lam_videos.index_select(1, idx), lam_states.index_select(1, idx.to(lam_states.device))


def _run_lam_teacher(
    *,
    lam: nn.Module,
    lam_videos: torch.Tensor,
    lam_states: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    lam_videos_t, lam_states_t = _build_lam_teacher_inputs_for_distill(
        lam=lam,
        lam_videos=lam_videos,
        lam_states=lam_states,
    )
    with torch.no_grad():
        device_type = lam_videos_t.device.type if hasattr(lam_videos_t, "device") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            lam_out = lam.get_latent_action(
                videos=lam_videos_t,
                states=lam_states_t,
                dec_videos=None,
                predict_future_frame=False,
                dataset_ids=None,
            )
    teacher_latent = lam_out["quantized"].detach().clone()
    return lam_out, teacher_latent


def _compute_latent_loss(
    *,
    pred_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    latent_loss_type: str,
) -> torch.Tensor:
    if pred_latent.shape != teacher_latent.shape:
        raise ValueError(
            f"[LatentWorldVLA] latent shape mismatch: pred={tuple(pred_latent.shape)}, "
            f"teacher={tuple(teacher_latent.shape)}"
        )

    loss_type = str(latent_loss_type).lower()
    if loss_type == "mse":
        return F.mse_loss(pred_latent, teacher_latent)
    return 1 - F.cosine_similarity(pred_latent, teacher_latent, dim=-1).mean()


def _compute_distill_loss(
    *,
    lam: nn.Module,
    pred_latent: torch.Tensor,
    lam_videos: torch.Tensor,
    lam_states: torch.Tensor,
    latent_loss_type: str,
) -> torch.Tensor:
    _, teacher_latent = _run_lam_teacher(
        lam=lam,
        lam_videos=lam_videos,
        lam_states=lam_states,
    )
    teacher_latent = teacher_latent.to(device=pred_latent.device, dtype=pred_latent.dtype)
    return _compute_latent_loss(
        pred_latent=pred_latent,
        teacher_latent=teacher_latent,
        latent_loss_type=latent_loss_type,
    )


# ============================================================================
# Checkpoint IO Helpers
# ============================================================================


def _guard_forbidden_legacy_checkpoint(load_dir: Union[str, Path]) -> None:
    p = Path(str(load_dir))
    legacy = p / "latent_vla_extra.pt"
    if legacy.exists():
        raise ValueError(f"[LatentWorldVLA] forbidden legacy checkpoint detected: `{legacy}`")


def _save_extra_checkpoint(
    model: "LatentWorldVLA",
    save_dir: Union[str, Path],
    ckpt_name: str,
) -> Path:
    save_path = Path(str(save_dir)) / str(ckpt_name)
    lam_decoder = getattr(model.lam, "decoder", None)
    payload: Dict[str, Any] = {
        "version": 1,
        "act_query": model.act_query.detach().cpu(),
        "vlm_to_lam": model.vlm_to_lam.state_dict(),
        "lam_decoder": lam_decoder.state_dict() if lam_decoder is not None else None,
        "flow": model.flow.state_dict(),
        "meta": {
            "num_action_queries": int(model.num_action_queries),
            "placeholder_token": str(model.model_cfg.latent_action_placeholder_token),
            "latent_loss_type": str(model.model_cfg.latent_loss_type),
        },
    }
    torch.save(payload, save_path)
    return save_path


def _load_extra_checkpoint(
    model: "LatentWorldVLA",
    load_dir: Union[str, Path],
    ckpt_name: str,
    *,
    strict: bool = True,
    map_location: str = "cpu",
) -> bool:
    p = Path(str(load_dir)) / str(ckpt_name)
    if not p.exists():
        return False

    obj = torch.load(str(p), map_location=map_location)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[LatentWorldVLA] invalid extra checkpoint format: {type(obj)}")

    required = ("version", "act_query", "vlm_to_lam", "flow", "meta")
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(f"[LatentWorldVLA] extra checkpoint missing keys: {missing}")

    act_query = obj["act_query"]
    if not isinstance(act_query, torch.Tensor):
        raise RuntimeError("[LatentWorldVLA] `act_query` must be a Tensor in extra checkpoint.")
    if tuple(act_query.shape) != tuple(model.act_query.shape):
        raise ValueError(
            f"[LatentWorldVLA] act_query shape mismatch: ckpt={tuple(act_query.shape)} model={tuple(model.act_query.shape)}"
        )
    model.act_query.data.copy_(act_query.to(device=model.act_query.device, dtype=model.act_query.dtype))

    model.vlm_to_lam.load_state_dict(obj["vlm_to_lam"], strict=strict)
    model.flow.load_state_dict(obj["flow"], strict=strict)

    if "lam_decoder" not in obj:
        raise RuntimeError("[LatentWorldVLA] extra checkpoint missing key: `lam_decoder`")
    lam_dec_state = obj.get("lam_decoder", None)
    lam_decoder = getattr(model.lam, "decoder", None)
    if lam_decoder is not None and lam_dec_state is not None:
        lam_decoder.load_state_dict(lam_dec_state, strict=strict)

    return True


# ============================================================================
# LatentWorldVLA
# ============================================================================


class LatentWorldVLA(nn.Module):
    """Independent world-model VLA without LatentVLAModel dependency."""

    def __init__(self, model_cfg: LatentWorldVLAConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg

        # 1) Load VLM + processor/tokenizer and register placeholder token.
        self.vlm, self.processor, self.tokenizer, self.placeholder_token_id = _load_vlm_and_processor(self.model_cfg)

        # 2) Load LAM.
        self.lam = _load_lam(self.model_cfg)
        self.lam.eval()
        for p in self.lam.parameters():
            p.requires_grad = False
        if bool(self.model_cfg.unfreeze_lam_decoder):
            lam_decoder = getattr(self.lam, "decoder", None)
            if lam_decoder is not None:
                for p in lam_decoder.parameters():
                    p.requires_grad = True

        # 3) Create trainable query and mapping head.
        self.num_action_queries = int(self.model_cfg.num_action_queries)
        if self.num_action_queries <= 0:
            raise ValueError("[LatentWorldVLA] num_action_queries must be > 0")

        vlm_hidden_dim = int(self.vlm.config.text_config.hidden_size)
        lam_code_dim = int(self.lam.code_dim)
        self.act_query = nn.Parameter(torch.randn(self.num_action_queries, vlm_hidden_dim) * 0.02)
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden_dim,
            lam_code_dim=lam_code_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=4.0,
            dropout=0.0,
        )

        # 4) Apply freeze policy.
        _apply_freeze_policy(self.vlm, self.model_cfg)

        # 5) Align flow config to LAM output dimensions.
        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_size = int(getattr(self.lam.encoder, "grid_size", 0) or 0)
        lam_num_tokens = int(lam_grid_size * lam_grid_size) if lam_grid_size > 0 else int(
            self.model_cfg.flow_cfg.num_vision_tokens
        )
        if int(self.model_cfg.flow_cfg.vision_dim) != lam_vision_dim:
            print(
                f"[LatentWorldVLA] flow_cfg.vision_dim={self.model_cfg.flow_cfg.vision_dim} "
                f"!= LAM vision dim={lam_vision_dim}; auto-aligned."
            )
        self.model_cfg.flow_cfg.vision_dim = lam_vision_dim

        if int(self.model_cfg.flow_cfg.num_vision_tokens) != lam_num_tokens:
            print(
                f"[LatentWorldVLA] flow_cfg.num_vision_tokens={self.model_cfg.flow_cfg.num_vision_tokens} "
                f"!= LAM token count={lam_num_tokens}; auto-aligned."
            )
            self.model_cfg.flow_cfg.num_vision_tokens = lam_num_tokens

        # 6) Flow head.
        self.flow = ConditionalFlowMatchingHead(config=self.model_cfg.flow_cfg)

        # 7) Expose compatibility attributes expected by surrounding code.
        self.code_book_size = int(getattr(self.lam, "codebook_size", 0) or 0)

    @classmethod
    def from_config(
        cls,
        cfg: LatentWorldVLAConfig,
        *,
        preserve_checkpoint_model_id: bool = True,
    ) -> Tuple["LatentWorldVLA", Any]:
        _apply_yaml_overrides(cfg, preserve_checkpoint_model_id=preserve_checkpoint_model_id)

        model = cls(cfg)

        load_dir = str(cfg.model_id or "")
        if load_dir and os.path.isdir(load_dir):
            _guard_forbidden_legacy_checkpoint(load_dir)
            loaded = _load_extra_checkpoint(
                model,
                load_dir,
                cfg.extra_ckpt_name,
                strict=True,
                map_location="cpu",
            )
            if loaded:
                print(f"[LatentWorldVLA] Loaded extra checkpoint from `{Path(load_dir) / cfg.extra_ckpt_name}`")
            else:
                print(
                    f"[LatentWorldVLA] No `{cfg.extra_ckpt_name}` found in `{load_dir}`; "
                    "using initialized query/map/flow weights."
                )

        return model, model.processor

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        *,
        lam_ckpt_path: str,
        lam_yaml_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["LatentWorldVLA", Any]:
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

        model_cfg = LatentWorldVLAConfig(
            model_id=str(ckpt_dir),
            lam_ckpt_path=lam_ckpt_path,
            lam_yaml_path=lam_yaml_path,
        )
        print(f"[LatentWorldVLA] Loading from checkpoint: {ckpt_dir}")
        model, processor = cls.from_config(model_cfg)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        model = model.to(device=device, dtype=dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print(f"[LatentWorldVLA] Model loaded successfully on {device} with dtype {dtype}")
        return model, processor

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_module_train_mode_by_params(self.vlm)
            self._set_module_train_mode_by_params(self.lam)
            self._set_module_train_mode_by_params(self.flow)
            self._set_module_train_mode_by_params(self.vlm_to_lam)
        return self

    def _set_module_train_mode_by_params(self, module: nn.Module) -> None:
        for child in module.children():
            has_trainable = any(p.requires_grad for p in child.parameters())
            if has_trainable:
                child.train()
                self._set_module_train_mode_by_params(child)
            else:
                child.eval()

    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = self.vlm.save_pretrained(str(save_dir), **kwargs)
        extra_path = _save_extra_checkpoint(self, save_dir, self.model_cfg.extra_ckpt_name)
        print(f"[LatentWorldVLA] Extra weights saved to: {extra_path}")
        return out

    # ------------------
    # VLM helper pipeline
    # ------------------
    def _forward_vlm_queries_supervise_latent_with_flow(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        flow_placeholder_mask: torch.BoolTensor,
        flow_query: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        flow_q = int(flow_query.shape[0])
        vlm_out = _vlm_forward_with_queries(
            vlm=self.vlm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
            act_query=self.act_query,
            act_num_queries=int(self.num_action_queries),
            flow_placeholder_mask=flow_placeholder_mask,
            flow_query=flow_query,
            flow_num_queries=flow_q,
        )
        hidden = vlm_out.hidden_states[-1]
        pred_latent = _project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=int(self.num_action_queries),
            vlm_to_lam=self.vlm_to_lam,
        )
        return {
            "h_vlm": hidden,
            "pred_latent": pred_latent,
            "vlm_out": vlm_out,
        }

    # ------------------
    # Forward
    # ------------------
    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        act_placeholder_mask: torch.Tensor,
        flow_placeholder_mask: Optional[torch.Tensor] = None,
        lam_videos: torch.Tensor,
        lam_states: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        wrist_videos: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del labels

        with torch.no_grad():
            features = _extract_lam_vision_features(self.lam, lam_videos)
        if features is None:
            raise ValueError("[LatentWorldVLA] lam visual feature extraction returned None; check LAM config.")

        h_t = features[:, 0, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)
        h_t1_gt = features[:, -1, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)

        flow_query = getattr(self.flow, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldVLA] flow_action_query is None; check ConditionalFlowMatchingHead.")

        act_placeholder_mask, flow_placeholder_mask = _infer_act_flow_masks(
            input_ids=input_ids,
            placeholder_token_id=self.placeholder_token_id,
            act_queries=int(self.num_action_queries),
            flow_queries=int(flow_query.shape[0]),
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
        )

        vlm_out_dict = self._forward_vlm_queries_supervise_latent_with_flow(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
            flow_query=flow_query,
        )
        h_vlm = vlm_out_dict["h_vlm"]
        h_vlm_for_flow = _apply_flow_only_grad_to_h_vlm(
            h_vlm=h_vlm,
            flow_placeholder_mask=flow_placeholder_mask,
            enable_flow_only=bool(self.model_cfg.flow_only_mode),
        )
        pred_action_emb = vlm_out_dict["pred_latent"]

        loss_distill = torch.tensor(0.0, device=pixel_values.device, dtype=pred_action_emb.dtype)
        if bool(self.model_cfg.enable_loss_distill):
            loss_distill = _compute_distill_loss(
                lam=self.lam,
                pred_latent=pred_action_emb,
                lam_videos=lam_videos,
                lam_states=lam_states,
                latent_loss_type=self.model_cfg.latent_loss_type,
            )

        if self.model_cfg.future_prediction:
            h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
            if isinstance(h_t1_pred, tuple):
                h_t1_pred = h_t1_pred[0]
            if h_t1_pred.dim() == 4:
                h_t1_pred = h_t1_pred[:, 0, :, :] if h_t1_pred.shape[1] == 1 else h_t1_pred[:, -1, :, :]
            h_t1_pred = h_t1_pred.to(dtype=h_t1_gt.dtype)
            loss_perceptual = F.mse_loss(h_t1_pred, h_t1_gt)
        else:
            h_t1_pred = h_t
            loss_perceptual = torch.tensor(0.0, device=pixel_values.device, dtype=h_t1_gt.dtype)

        if self.model_cfg.enable_wrist_view and wrist_videos is not None:
            with torch.no_grad():
                features_w = _extract_lam_vision_features(self.lam, wrist_videos)
            if features_w is None:
                raise ValueError("[LatentWorldVLA] wrist visual feature extraction returned None.")
            h_t_w = features_w[:, 0, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)
            h_t = torch.cat([h_t_w, h_t], dim=1)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.float32)
            if pixel_values.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            repeat_steps = int(self.model_cfg.repeated_diffusion_steps)
            h_t_rep = h_t.repeat(repeat_steps, *([1] * (h_t.ndim - 1)))
            h_t1_rep = h_t1_pred.repeat(repeat_steps, *([1] * (h_t1_pred.ndim - 1)))
            h_vlm_rep = h_vlm_for_flow.repeat(repeat_steps, *([1] * (h_vlm_for_flow.ndim - 1)))
            proprio_rep = proprio.repeat(repeat_steps, *([1] * (proprio.ndim - 1)))
            actions_rep = actions.repeat(repeat_steps, *([1] * (actions.ndim - 1)))
            attn_rep = attention_mask.repeat(repeat_steps, *([1] * (attention_mask.ndim - 1)))

            loss_flow = self.flow(
                h_t=h_t_rep,
                h_t1_star=h_t1_rep.detach().clone(),
                h_vlm=h_vlm_rep,
                proprio=proprio_rep,
                actions=actions_rep,
                attention_mask=attn_rep == 1,
            )

        loss_total = (
            loss_flow
            + self.model_cfg.perceptual_weight * loss_perceptual
            + self.model_cfg.lam_encoder_distill_weight * loss_distill
        )

        zero = torch.tensor(0.0, device=pixel_values.device, dtype=h_t1_pred.dtype)
        return {
            "loss_flow": loss_flow,
            "loss_perceptual": loss_perceptual,
            "loss_distill": loss_distill,
            "loss_vlm": zero,
            "loss_total": loss_total,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lam_videos: Optional[torch.Tensor] = None,
        lam_states: Optional[torch.Tensor] = None,
        act_placeholder_mask: Optional[torch.Tensor] = None,
        flow_placeholder_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        wrist_videos: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        debug: bool = False,
        return_intermediates: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        del lam_states, debug, kwargs

        device = pixel_values.device
        dtype = pixel_values.dtype
        batch_size = input_ids.shape[0]

        if lam_videos is None:
            raise ValueError("predict_action requires `lam_videos`.")
        if lam_videos.dim() == 4:
            lam_videos = lam_videos.unsqueeze(1)
        if lam_videos.shape[0] != batch_size:
            raise ValueError(
                f"lam_videos batch size mismatch: {lam_videos.shape[0]} vs input_ids {batch_size}"
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if proprio is None:
            proprio_dim = int(getattr(self.flow.config, "proprio_dim", 8))
            proprio = torch.zeros(batch_size, proprio_dim, dtype=dtype, device=device)

        if guidance_scale is None:
            guidance_scale = float(self.model_cfg.cfg_guidance_scale)
        if num_inference_steps is None:
            num_inference_steps = int(self.model_cfg.num_inference_steps)

        features = _extract_lam_vision_features(self.lam, lam_videos)
        if features is None:
            raise ValueError("[predict_action] lam visual feature extraction returned None; check LAM config.")
        h_t_original = features[:, 0, :, :].to(device=device, dtype=dtype)
        h_t = h_t_original

        flow_query = getattr(self.flow, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldVLA] flow_action_query is None; check ConditionalFlowMatchingHead.")

        act_placeholder_mask, flow_placeholder_mask = _infer_act_flow_masks(
            input_ids=input_ids,
            placeholder_token_id=self.placeholder_token_id,
            act_queries=int(self.num_action_queries),
            flow_queries=int(flow_query.shape[0]),
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
        )

        vlm_out_dict = self._forward_vlm_queries_supervise_latent_with_flow(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
            flow_query=flow_query,
        )
        h_vlm = vlm_out_dict["h_vlm"]
        pred_action_emb = vlm_out_dict["pred_latent"]

        if self.model_cfg.future_prediction:
            h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
            if isinstance(h_t1_pred, tuple):
                h_t1_pred = h_t1_pred[0]
            if h_t1_pred.dim() == 4:
                h_t1_pred = h_t1_pred[:, 0, :, :] if h_t1_pred.shape[1] == 1 else h_t1_pred[:, -1, :, :]
            h_t1_pred = h_t1_pred.to(dtype=dtype)
        else:
            h_t1_pred = h_t

        if self.model_cfg.enable_wrist_view and wrist_videos is not None:
            features_w = _extract_lam_vision_features(self.lam, wrist_videos)
            if features_w is None:
                raise ValueError("[predict_action] wrist visual feature extraction returned None.")
            h_t_w = features_w[:, 0, :, :].to(device=device, dtype=dtype)
            h_t = torch.cat([h_t_w, h_t], dim=1)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.float32)
            if pixel_values.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            actions = self.flow.sample_actions_cfg(
                h_t=h_t,
                h_t1_star=h_t1_pred,
                h_vlm=h_vlm,
                proprio=proprio,
                cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                attention_mask=attention_mask == 1,
            )

        if not return_intermediates:
            return actions

        num_tokens = h_t_original.shape[1]
        hw = int(num_tokens ** 0.5)
        if hw * hw != num_tokens:
            hw = 16
        intermediates = {
            "h_t": h_t_original.detach().cpu(),
            "h_t1_pred": h_t1_pred.detach().cpu(),
            "vision_tokens_hw": (hw, hw),
        }
        return actions, intermediates
