import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List
from typing import Dict, Optional, Any, Tuple

from latent_action_model.core.lam_model import LatentLAMModel, load_latent_action_model
from prismatic.models import load_vlm_auto


class LatentVLAModel(nn.Module):
    """
    组合模型：封装 VLM 与 LAM

    - 在 forward 中调用 LAM.vq_encode 取得离散 latent（indices）与连续表征（quantized）
    - 将占位符 token (<ACT_PH>) 替换为真实 <ACT_i>，并同步替换 labels
    - 计算主 CE 损失（VLM 内部）与 encoder 蒸馏损失（VLM hidden 对齐 LAM quantized）
    - decoder 感知损失接口预留，默认关闭
    """

    def __init__(
        self,
        vlm: nn.Module,
        lam: LatentLAMModel,
        *,
        action_token_begin_id: int,
        codebook_size: int,
        placeholder_token_id: int,
        supervise_quantized: bool = False,
        quantized_loss_type: str = "cosine",
        enable_lam_decoder_perceptual: bool = False,
        lam_encoder_distill_weight: float = 1.0,
        lam_decoder_perceptual_weight: float = 1.0,
        vlm_hidden_dim: int = 1024,
        lam_code_dim: int = 256,
        processor: Optional[Any] = None,
        new_token_ids: Optional[Any] = None,
        new_token_lr_scale: float = 1.0,
        act_query_lr_scale: float = 1.0,
        vlm_to_lam_lr_scale: float = 1.0,
        enable_lam_kl_loss: bool = False,
        lam_kl_weight: float = 1.0,
        lam_kl_temperature: float = 0.1,
        debug_mode: bool = False,
        unfreeze_lam_decoder: bool = False,
        lam_decoder_target: str = "gt",
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.lam = lam.eval()
        for p in self.lam.parameters():
            p.requires_grad = False
        self.processor = processor

        self.action_token_begin_id = int(action_token_begin_id)
        self.codebook_size = int(codebook_size)
        self.placeholder_token_id = int(placeholder_token_id)

        # 当启用时：训练主监督从 token CE 切换为回归 vq_out["quantized"]（在动作位置“前一位”的 hidden 上预测）
        self.supervise_quantized = bool(supervise_quantized)
        self.quantized_loss_type = str(quantized_loss_type).lower()
        self.enable_lam_decoder_perceptual = bool(enable_lam_decoder_perceptual)
        # 历史字段名：lam_encoder_distill_weight。当前仅在 supervise_quantized=True 时生效，表示 quantized 回归 loss 的权重。
        self.lam_encoder_distill_weight = float(lam_encoder_distill_weight)
        self.lam_decoder_perceptual_weight = float(lam_decoder_perceptual_weight)
        self.new_token_lr_scale = float(new_token_lr_scale)
        self.act_query_lr_scale = float(act_query_lr_scale)
        self.vlm_to_lam_lr_scale = float(vlm_to_lam_lr_scale)
        self.enable_lam_kl_loss = bool(enable_lam_kl_loss)
        self.lam_kl_weight = float(lam_kl_weight)
        self.lam_kl_temperature = float(lam_kl_temperature)
        self.new_token_ids = list(new_token_ids) if new_token_ids is not None else []
        self.debug_mode = bool(debug_mode)
        self.unfreeze_lam_decoder = bool(unfreeze_lam_decoder)
        lam_decoder_target = str(lam_decoder_target).lower()
        if lam_decoder_target not in ("teacher", "gt"):
            raise ValueError(f"[LatentVLAModel] invalid lam_decoder_target={lam_decoder_target}, choose 'teacher' or 'gt'.")
        self.lam_decoder_target = lam_decoder_target

        # Safety: if we train the LAM decoder, using decoder-produced "teacher" targets is a moving-target objective
        # (both recon_pred and recon_target depend on the same trainable decoder weights), which can trivially collapse
        # `loss_perceptual` while letting the decoder drift in scale. Prefer a fixed target (`tgt`) in this mode.
        if self.enable_lam_decoder_perceptual and self.unfreeze_lam_decoder and self.lam_decoder_target == "teacher":
            try:
                import warnings

                warnings.warn(
                    "[LatentVLAModel] unfreeze_lam_decoder=True with lam_decoder_target='teacher' "
                    "creates a moving-target perceptual objective. For stability, switching lam_decoder_target -> 'gt'."
                )
            except Exception:
                pass
            self.lam_decoder_target = "gt"
        # 训练时的 latent 数量（Q）；推理沿用同一数目
        nq = getattr(lam, "num_queries", None)
        if nq is None and hasattr(lam, "vq"):
            nq = getattr(lam.vq, "num_queries", None)
        self.num_queries = int(nq) if nq is not None else None

        def _register_grad_scale(param: Optional[torch.nn.Parameter], scale: float, name: str) -> None:
            """Multiply gradient by `scale` via hook; safe when grad is None."""
            if param is None:
                return
            if scale == 1.0:
                return
            s = float(scale)

            def _hook(g):
                if g is None:
                    return None
                return g * s

            try:
                param.register_hook(_hook)
            except Exception:
                pass

        # 仅在 supervise_quantized 下需要投影头（量化向量回归为主监督）
        if self.supervise_quantized:
            self.vlm_to_lam = nn.Sequential(nn.Linear(vlm_hidden_dim, vlm_hidden_dim), nn.LayerNorm(vlm_hidden_dim), nn.GELU(), nn.Linear(vlm_hidden_dim, lam_code_dim))
        else:
            self.vlm_to_lam = None

        # supervise_quantized: 在 <ACT_PH> 位置注入可学习 query embedding（而不是“从头预测”动作 token）
        # 注意：这些 query 不在 VLM 内部，因此默认不会被 vlm.save_pretrained 保存；我们会在 save_pretrained 里额外落盘。
        if self.supervise_quantized:
            if self.num_queries is None:
                raise ValueError("[LatentVLAModel] supervise_quantized=True requires self.num_queries to be set.")
            q = int(self.num_queries)
            # float32 参数更稳定；前向时会按 inputs_embeds dtype 做 cast
            self.act_query = nn.Parameter(torch.randn(q, int(vlm_hidden_dim)) * 0.02)
        else:
            self.act_query = None

        # 在 supervise_quantized 下对 query / head 做梯度放大（类似 new_token_lr_scale，但作用于参数整体）
        if self.supervise_quantized:
            if self.act_query is not None and self.act_query_lr_scale != 1.0:
                _register_grad_scale(self.act_query, self.act_query_lr_scale, "act_query")
            if self.vlm_to_lam is not None and self.vlm_to_lam_lr_scale != 1.0:
                # `vlm_to_lam` may be a single Linear or a small MLP (e.g., nn.Sequential).
                # Scale gradients for all its parameters robustly.
                if hasattr(self.vlm_to_lam, "weight") and isinstance(getattr(self.vlm_to_lam, "weight"), torch.nn.Parameter):
                    _register_grad_scale(self.vlm_to_lam.weight, self.vlm_to_lam_lr_scale, "vlm_to_lam.weight")
                    if getattr(self.vlm_to_lam, "bias", None) is not None:
                        _register_grad_scale(self.vlm_to_lam.bias, self.vlm_to_lam_lr_scale, "vlm_to_lam.bias")
                else:
                    for n, p in self.vlm_to_lam.named_parameters(recurse=True):
                        _register_grad_scale(p, self.vlm_to_lam_lr_scale, f"vlm_to_lam.{n}")

        # 针对新增 token 行做梯度放大，实现"只对新增 embedding 提高有效学习率"
        if self.new_token_lr_scale != 1.0 and len(self.new_token_ids) > 0:
            try:
                emb = self.vlm.get_input_embeddings()
                # 检查 embedding 是否被冻结：如果被冻结，梯度放大hook无法生效，提前跳过
                if not emb.weight.requires_grad:
                    # Embedding已冻结，梯度放大hook无法工作，但仍可记录配置
                    self._new_token_factor = None
                    import warnings
                    warnings.warn(
                        "[LatentVLAModel] new_token_lr_scale is set but embedding is frozen. "
                        "Gradient scaling hook will not be applied. "
                        "Consider setting freeze_embedding=False if you want to scale new token gradients."
                    )
                else:
                    # 构建放大系数并注册为 buffer，避免 device 不一致
                    row_mask = torch.zeros(
                        emb.weight.size(0), device=emb.weight.device, dtype=emb.weight.dtype
                    )
                    for tid in self.new_token_ids:
                        tid_int = int(tid)
                        if 0 <= tid_int < row_mask.numel():
                            row_mask[tid_int] = 1.0
                    scale = self.new_token_lr_scale
                    factor = 1.0 + (scale - 1.0) * row_mask  # [vocab]
                    self.register_buffer("_new_token_factor", factor, persistent=False)

                    def _scale_grad(g):
                        # g: [vocab, dim] or None (if parameter is frozen)
                        # 处理梯度为None的情况（当参数被冻结时）
                        if g is None:
                            return None
                        if not hasattr(self, "_new_token_factor") or self._new_token_factor is None:
                            return g
                        f = self._new_token_factor
                        if f.device != g.device:
                            f = f.to(device=g.device, dtype=g.dtype)
                        else:
                            f = f.to(dtype=g.dtype)
                        return g * f.unsqueeze(1)

                    emb.weight.register_hook(_scale_grad)
            except Exception as e:
                self._new_token_factor = None
                import warnings
                warnings.warn(f"[LatentVLAModel] Failed to register new token gradient scaling hook: {e}")

    def train(self, mode: bool = True):
        """
        细粒度训练模式控制：
        - 对所有子模块（VLM、LAM），根据其是否有可训练参数来决定 train/eval：
          * 如果模块所有参数都被冻结（requires_grad=False）→ eval 模式
          * 如果模块有可训练参数（requires_grad=True）→ train 模式
        这样可以正确处理部分解冻的情况（如 unfreeze_llm_last_n_layers, 
        unfreeze_vision_merger, unfreeze_lam_decoder 等）
        """
        super().train(mode)
        
        # 只在训练模式下需要细粒度控制
        if mode:
            # 对 VLM 和 LAM 应用细粒度控制
            self._set_module_train_mode_by_params(self.vlm)
            self._set_module_train_mode_by_params(self.lam)
        
        return self
    
    def _set_module_train_mode_by_params(self, module: nn.Module):
        """
        递归地为模块设置 train/eval 模式：
        - 如果模块有任何可训练参数 → train 模式
        - 如果模块所有参数都冻结 → eval 模式
        
        这是一个辅助方法，用于处理部分解冻的情况
        """
        for child_name, child_module in module.named_children():
            # 检查该子模块是否有可训练参数
            has_trainable = any(p.requires_grad for p in child_module.parameters())
            
            if has_trainable:
                # 有可训练参数，设为 train 模式，并递归处理其子模块
                child_module.train()
                self._set_module_train_mode_by_params(child_module)
            else:
                # 所有参数都冻结，设为 eval 模式
                child_module.eval()
    @property
    def tokenizer(self):
        return self.processor.tokenizer if self.processor is not None else None

    # 仅保存/加载 VLM 权重，保持与原有管线一致的 checkpoint 行为
    def save_pretrained(self, save_directory, **kwargs):
        # 1) 先保存 VLM（与原管线一致）
        out = self.vlm.save_pretrained(save_directory, **kwargs)
        # 2) 额外保存 wrapper 侧参数（query / projection head），否则训练结果会丢失这部分可学习权重
        try:
            import os

            os.makedirs(save_directory, exist_ok=True)
            # 仅保存 wrapper 侧权重；不写任何 config-like 元信息，避免加载时覆盖用户配置
            extra: Dict[str, Any] = {}
            if getattr(self, "act_query", None) is not None:
                extra["act_query"] = self.act_query.detach().cpu()
            if getattr(self, "vlm_to_lam", None) is not None:
                # 保存线性投影头（用于 quantized 回归/encoder distill）
                extra["vlm_to_lam"] = self.vlm_to_lam.state_dict()
            lam_dec = getattr(self.lam, "decoder", None)
            if lam_dec is not None:
                extra["lam_decoder"] = lam_dec.state_dict()
            torch.save(extra, os.path.join(save_directory, "latent_vla_extra.pt"))
        except Exception:
            # 不让额外保存影响主保存流程
            pass
        return out

    def load_latent_vla_extra(self, load_directory: str, *, strict: bool = True, map_location: str = "cpu") -> bool:
        """
        Load wrapper-side trainable parameters saved by `save_pretrained()`:
        - act_query (when supervise_quantized=True)
        - vlm_to_lam projection head

        Returns True if an extra file existed and was loaded; False if file missing.
        """
        import os

        extra_path = os.path.join(str(load_directory), "latent_vla_extra.pt")
        if not os.path.exists(extra_path):
            return False

        extra = torch.load(extra_path, map_location=map_location)
        if not isinstance(extra, dict):
            if strict:
                raise ValueError(f"[LatentVLAModel] invalid extra checkpoint format: {type(extra)}")
            return False

        # act_query
        if "act_query" in extra and extra["act_query"] is not None:
            if getattr(self, "act_query", None) is None:
                if strict:
                    raise ValueError(
                        "[LatentVLAModel] extra contains act_query but current model has no act_query. "
                        "Did you forget to set supervise_quantized=True?"
                    )
            else:
                q = extra["act_query"]
                if tuple(q.shape) != tuple(self.act_query.shape):
                    raise ValueError(
                        f"[LatentVLAModel] act_query shape mismatch: ckpt={tuple(q.shape)} "
                        f"model={tuple(self.act_query.shape)}"
                    )
                # copy to preserve Parameter object
                self.act_query.data.copy_(q.to(device=self.act_query.device, dtype=self.act_query.dtype))

        # vlm_to_lam
        if "vlm_to_lam" in extra and extra["vlm_to_lam"] is not None:
            if getattr(self, "vlm_to_lam", None) is None:
                if strict:
                    raise ValueError(
                        "[LatentVLAModel] extra contains vlm_to_lam but current model has no vlm_to_lam. "
                        "Enable supervise_quantized."
                    )
            else:
                sd = extra["vlm_to_lam"]
                self.vlm_to_lam.load_state_dict(sd, strict=strict)

        # lam_decoder
        if "lam_decoder" in extra and extra["lam_decoder"] is not None:
            lam_dec = getattr(self.lam, "decoder", None)
            if lam_dec is None:
                if strict:
                    raise ValueError(
                        "[LatentVLAModel] extra contains lam_decoder but current LAM has no decoder."
                    )
            else:
                try:
                    lam_dec.load_state_dict(extra["lam_decoder"], strict=strict)
                except Exception as e:
                    if strict:
                        raise
                    else:
                        try:
                            import warnings
                            warnings.warn(f"[LatentVLAModel] lam_decoder load skipped (strict=False): {e}")
                        except Exception:
                            pass

        return True

    def state_dict(self, *args, **kwargs):
        return self.vlm.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.vlm.load_state_dict(state_dict, strict=strict)

    def _build_prefix_from_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        *,
        strict_placeholder_count: bool,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        截断到每个样本第一个 <ACT_PH> 之前作为 prefix，并对 batch 进行 padding。
        返回 (prefix_input_ids, prefix_attention_mask, placeholder_counts)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        device = input_ids.device
        placeholder_mask = input_ids == self.placeholder_token_id  # [B, L]
        counts = placeholder_mask.sum(dim=1)  # [B]

        if self.num_queries is not None and strict_placeholder_count:
            mismatch = counts != self.num_queries
            if mismatch.any():
                raise ValueError(
                    f"[LatentVLAModel] placeholder count mismatch: "
                    f"found={counts.tolist()}, expected={int(self.num_queries)}"
                )

        B, L = input_ids.shape
        idx_range = torch.arange(L, device=device)
        first_pos = torch.where(
            placeholder_mask, idx_range.unsqueeze(0).expand_as(input_ids), torch.full_like(input_ids, L)
        ).min(dim=1).values  # [B]

        prefixes: List[torch.Tensor] = []
        prefix_masks: List[torch.Tensor] = []
        for b in range(B):
            cut = int(first_pos[b].item())
            prefixes.append(input_ids[b, :cut])
            prefix_masks.append(attention_mask[b, :cut])

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.vlm.config, "pad_token_id", 0)

        prefix_input_ids = pad_sequence(prefixes, batch_first=True, padding_value=pad_token_id)
        prefix_attention_mask = pad_sequence(prefix_masks, batch_first=True, padding_value=0)
        return prefix_input_ids, prefix_attention_mask, counts

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        overwatch=None,
        debug_mode: bool = False,
        *,
        yaml_path: Optional[str] = None,
        preserve_checkpoint_model_id: bool = True,
    ) -> Tuple["LatentVLAModel", Any]:
        """
        内部加载 VLM 与 LAM，返回 (model, processor)
        占位符 id 可通过 model.placeholder_token_id 访问

        Args:
            yaml_path: 可选。训练时使用的 YAML 配置文件路径；若提供则会先读取并覆盖传入 cfg 的字段，
                       用于推理/继续训练时确保 `supervise_quantized` 等关键参数与训练一致。
            preserve_checkpoint_model_id: 若为 True，则当 cfg.model_id 指向一个本地目录（通常是 HF checkpoint dir）
                       时，不使用 YAML 中的 model_id 覆盖它，以避免把 checkpoint 路径改回 base 模型路径。
        """
        if overwatch is None:
            import logging
            overwatch = logging.getLogger(__name__)

        # --- Optional: override cfg from training YAML (similar to LAM loader) ---
        if yaml_path is not None:
            try:
                import os
                import yaml
                from pathlib import Path

                with open(str(yaml_path), "r", encoding="utf-8") as f:
                    ycfg = yaml.safe_load(f) or {}
                # Support either flat yaml or a nested {"train": {...}} / {"config": {...}} style.
                if isinstance(ycfg, dict):
                    for nest_key in ("train", "config", "cfg"):
                        if nest_key in ycfg and isinstance(ycfg[nest_key], dict):
                            ycfg = ycfg[nest_key]
                            break

                # Preserve checkpoint dir model_id if requested
                keep_model_id = False
                try:
                    model_id_val = str(getattr(cfg, "model_id", "") or "")
                    if preserve_checkpoint_model_id and model_id_val and os.path.isdir(model_id_val):
                        keep_model_id = True
                except Exception:
                    keep_model_id = False

                def _cast_like(key: str, cur, val):
                    # Preserve model_id (checkpoint path) if requested
                    if key == "model_id" and keep_model_id:
                        return cur
                    # Cast for common dataclass fields
                    if isinstance(cur, bool):
                        return bool(val)
                    if isinstance(cur, int) and not isinstance(cur, bool):
                        return int(val)
                    if isinstance(cur, float):
                        return float(val)
                    if isinstance(cur, Path):
                        return Path(val)
                    # Heuristic: some dirs are Paths in TrainConfig but may be None in cfg here
                    if cur is None and key in ("hf_cache_dir", "data_root_dir", "run_root_dir") and isinstance(val, str):
                        return Path(val)
                    return val

                if isinstance(ycfg, dict):
                    for k, v in ycfg.items():
                        try:
                            cur = getattr(cfg, k, None)
                            setattr(cfg, k, _cast_like(str(k), cur, v))
                        except Exception:
                            # Best-effort: do not break loading if a key cannot be applied
                            pass
                    try:
                        overwatch.info(f"[LatentVLAModel] applied yaml overrides from `{yaml_path}` (preserve_model_id={keep_model_id})")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    overwatch.info(f"[LatentVLAModel] yaml override skipped/failed: {e}")
                except Exception:
                    pass

        overwatch.info(f"🔄 加载基础 VLM `{cfg.model_id}`（HF from_pretrained, generic）")
        vlm, processor = load_vlm_auto(cfg.model_id, cfg.hf_cache_dir, dtype=torch.bfloat16)
        tokenizer = processor.tokenizer
        vlm.generation_config.max_new_tokens = int(getattr(cfg, "max_new_tokens", 4))
        vlm.config.loss_type = str(getattr(cfg, "loss_type", "ForCausalLMLoss"))
        vlm.config.use_cache = False

        overwatch.info(f"🔄 加载 LAM（yaml=`{cfg.lam_yaml_path}`）")
        lam = load_latent_action_model(cfg.lam_ckpt_path, cfg.lam_yaml_path)

        # 获取 codebook_size：优先从 vq 模块获取，如果 vq 为 None（AE 模式），则从模型属性获取
        if lam.vq is not None:
            codebook_size = lam.vq.codebook_size
        else:
            # AE 模式：vq 为 None，使用 code_book_size 属性
            # 兼容历史字段名：`code_book_size`（LatentLAMModel 内部设置）与可能的 `codebook_size`
            codebook_size = getattr(lam, "code_book_size", None)
            if codebook_size is None:
                codebook_size = getattr(lam, "codebook_size", None)
            if codebook_size is None:
                raise ValueError(
                    "[LatentVLAModel] LAM model has no VQ module (vq_type='ae') and no code_book_size attribute. "
                    "Cannot determine codebook size for action tokens."
                )
            overwatch.info(f"[LatentVLAModel] Using code_book_size={codebook_size} from LAM (AE mode, vq=None)")

        # 注册动作 token 与占位符
        special_tokens_dict = {"additional_special_tokens": [f"<ACT_{i}>" for i in range(codebook_size)]}
        tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
        placeholder_token = getattr(cfg, "latent_action_placeholder_token", "<ACT_PH>")
        tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})  # type: ignore[attr-defined]


        # 打印当前 tokenizer 词表大小与 VLM embedding 大小，检查是否超出预留长度
        try:
            vocab_size = len(tokenizer)
            emb = vlm.get_input_embeddings()
            num_embeddings = emb.num_embeddings if hasattr(emb, "num_embeddings") else None
            overwatch.info(
                f"[LatentVLAModel] tokenizer vocab_size={vocab_size}, "
                f"vlm.num_embeddings={num_embeddings}"
            )
            if num_embeddings is not None and vocab_size > num_embeddings:
                overwatch.info(
                    f"[LatentVLAModel] WARNING: vocab_size({vocab_size}) > num_embeddings({num_embeddings}); "
                    "请确认 VLM 是否为本地权重并已预留足够 embedding 空位。"
                )
        except Exception:
            pass

        act_tokens = [f"<ACT_{i}>" for i in range(codebook_size)]
        act_ids = tokenizer.convert_tokens_to_ids(act_tokens)
        action_token_begin_id = min(act_ids)
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        new_token_ids = act_ids

        model = cls(
            vlm=vlm,
            lam=lam,
            action_token_begin_id=action_token_begin_id,
            codebook_size=codebook_size,
            placeholder_token_id=placeholder_token_id,
            supervise_quantized=getattr(cfg, "supervise_quantized", False),
            quantized_loss_type=getattr(cfg, "quantized_loss_type", "cosine"),
            enable_lam_decoder_perceptual=getattr(cfg, "enable_lam_decoder_perceptual", False),
            lam_encoder_distill_weight=getattr(cfg, "lam_encoder_distill_weight", 1.0),
            lam_decoder_perceptual_weight=getattr(cfg, "lam_decoder_perceptual_weight", 0.0),
            enable_lam_kl_loss=getattr(cfg, "enable_lam_kl_loss", False),
            lam_kl_weight=getattr(cfg, "lam_kl_weight", 1.0),
            lam_kl_temperature=getattr(cfg, "lam_kl_temperature", 1.0),
            vlm_hidden_dim=vlm.config.text_config.hidden_size,
            # Important: vq_out["quantized"] is in VQ input_dim space (after out_proj), not code_dim.
            lam_code_dim=int(lam.feature_dim),
            processor=processor,
            new_token_ids=new_token_ids,
            new_token_lr_scale=getattr(cfg, "new_token_lr_scale", 1.0),
            act_query_lr_scale=getattr(cfg, "act_query_lr_scale", 1.0),
            vlm_to_lam_lr_scale=getattr(cfg, "vlm_to_lam_lr_scale", 1.0),
            debug_mode=bool(debug_mode),
            unfreeze_lam_decoder=bool(getattr(cfg, "unfreeze_lam_decoder", False)),
            lam_decoder_target=str(getattr(cfg, "lam_decoder_target", "teacher")),
        )

        # 同步 cfg 中的字段（便于后续使用）
        cfg.action_token_begin_id = action_token_begin_id
        cfg.codebook_size = codebook_size
        try:
            cfg.unfreeze_lam_decoder = bool(getattr(cfg, "unfreeze_lam_decoder", False))
        except Exception:
            pass
        try:
            cfg.lam_decoder_target = str(getattr(cfg, "lam_decoder_target", "teacher"))
        except Exception:
            pass

        # 自动加载 wrapper-side 参数（act_query / vlm_to_lam），当 model_id 指向已保存目录时生效
        try:
            import os

            load_dir = str(getattr(cfg, "model_id", ""))
            extra_path = os.path.join(load_dir, "latent_vla_extra.pt")
            if load_dir and os.path.isdir(load_dir) and os.path.exists(extra_path):
                overwatch.info(f"[LatentVLAModel] loading extra wrapper weights from `{extra_path}`")
                loaded = model.load_latent_vla_extra(load_dir, strict=False, map_location="cpu")
                overwatch.info(f"[LatentVLAModel] extra wrapper weights loaded={loaded}")
        except Exception as e:
            try:
                overwatch.info(f"[LatentVLAModel] extra wrapper weights load skipped/failed: {e}")
            except Exception:
                pass

        return model, processor

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        act_placeholder_mask: torch.BoolTensor,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        lam_dec_videos: Optional[torch.FloatTensor] = None,
        lam_dataset_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # === 1) LAM 编码（teacher）===
        vq_out = self._lam_vq_encode(
            lam_videos=lam_videos,
            lam_states=lam_states,
            lam_dec_videos=lam_dec_videos,
            lam_dataset_ids=lam_dataset_ids,
        )
        self._maybe_debug_check_lam_codes(vq_out)

        latent_codes = vq_out.get("indices", None)
        z_teacher = vq_out["quantized"].detach().clone()  # [B, Q, D_lam]
        # VAE path: indices may be None; require supervise_quantized to regress continuous latents
        if latent_codes is None:
            if not self.supervise_quantized:
                raise ValueError("[LatentVLAModel] VAE mode provides no discrete indices; set supervise_quantized=True to use continuous latents.")
            B, Q = z_teacher.shape[0], z_teacher.shape[1]
            latent_codes = torch.zeros(B, Q, device=z_teacher.device, dtype=torch.long)
        else:
            latent_codes = latent_codes.detach().clone()

        # === 2) 构造 VLM 输入（token teacher-forcing / query injection）===
        latent_codes = latent_codes.to(input_ids.device)
        z_teacher = z_teacher.to(input_ids.device)
        act_token_ids = self.action_token_begin_id + latent_codes  # [B, Q]
        input_ids_used, labels_used = self._prepare_vlm_token_inputs(
            input_ids=input_ids,
            labels=labels,
            act_placeholder_mask=act_placeholder_mask,
            act_token_ids=act_token_ids,
        )

        vlm_out = self._vlm_forward(
            input_ids=input_ids_used,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels_used,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=int(latent_codes.shape[1]),
        )
        hidden = vlm_out.hidden_states[-1]  # [B, L, D_vlm]  最后一层是norm过的特征！
        logits = vlm_out.logits
        loss_main = None if self.supervise_quantized else getattr(vlm_out, "loss", None)

        # === 3) losses ===
        pred_quantized = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=int(latent_codes.shape[1]),
        )
        loss_distill = self._compute_quantized_loss(pred_quantized=pred_quantized, z_teacher=z_teacher)
        loss_perceptual, delta_student = self._compute_decoder_perceptual_loss(
            vq_out=vq_out,
            pred_quantized=pred_quantized,
        )
        loss_kl = self._compute_kl_loss(
            vq_out=vq_out,
            logits=logits,
            act_placeholder_mask=act_placeholder_mask,
        )

        total_loss = self._sum_losses(
            base_dtype=hidden.dtype,
            base_device=input_ids.device,
            loss_main=loss_main,
            loss_distill=loss_distill,
            loss_perceptual=loss_perceptual,
            loss_kl=loss_kl,
            logits=logits,
        )

        # Compute identity shortcut metric for wandb logging
        identity_shortcut = None
        if delta_student is not None:
            with torch.no_grad():
                # Mean absolute difference of delta_student, quantifying decoder's tendency to copy f_t
                # Smaller value means decoder tends to directly copy f_t (recon_pred ≈ dec_in_f)
                identity_shortcut = delta_student.abs().mean().item()

        return {
            "loss": total_loss,
            "loss_main": loss_main,
            "loss_distill": loss_distill,
            "loss_perceptual": loss_perceptual,
            "loss_kl": loss_kl,
            "logits": logits,
            "identity_shortcut": identity_shortcut,
        }

    # -------------------------
    # Forward helper functions
    # -------------------------
    def _lam_vq_encode(
        self,
        *,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        lam_dec_videos: Optional[torch.FloatTensor],
        lam_dataset_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run frozen LAM vq_encode in fp32 (autocast disabled) and return vq_out dict."""
        with torch.no_grad():
            device_type = lam_videos.device.type if hasattr(lam_videos, "device") else "cuda"
            with torch.autocast(device_type=device_type, enabled=False):
                return self.lam.vq_encode(
                    videos=lam_videos,
                    states=lam_states,
                    dec_videos=lam_dec_videos,
                    predict_future_frame=bool(
                        self.enable_lam_decoder_perceptual and self.lam_decoder_perceptual_weight > 0
                    ),
                    dataset_ids=lam_dataset_ids,
                    return_teacher_probs=self.enable_lam_kl_loss,
                    teacher_temperature=self.lam_kl_temperature,
                )

    def _maybe_debug_check_lam_codes(self, vq_out: Dict[str, torch.Tensor]) -> None:
        """Optional debug hook for LAM indices under debug_repeat_batch.

        NOTE: We keep debug_mode for other debugging workflows, but we no longer
        perform the original "compare current codes vs a reference batch" logic.
        Instead, we only record the most recent indices for potential inspection.
        """
        if not self.debug_mode:
            return
        try:
            codes = vq_out.get("indices", None)
            if codes is None:
                print("[LamCodes Debug] codes is None; skipping diff (vq_training=True?)")
                return
            codes = codes.detach().clone()
            # Record latest codes (train/eval separated) for debugging/inspection.
            last_attr = "_last_lam_codes_train" if self.training else "_last_lam_codes_eval"
            setattr(self, last_attr, codes)
        except Exception as e:
            print(f"[LamCodes Debug] diff check failed: {e}")

    def _prepare_vlm_token_inputs(
        self,
        *,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        act_token_ids: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        """Optionally teacher-force <ACT_i> into input_ids/labels (CE mode)."""
        if self.supervise_quantized:
            return input_ids, labels

        input_ids_flat = input_ids.view(-1)
        mask_flat = act_placeholder_mask.view(-1)
        act_ids_flat = act_token_ids.view(-1)
        assert mask_flat.sum().item() == act_ids_flat.numel(), (
            "占位符数量与 latent 数量不匹配。"
            f"mask_flat.sum()={mask_flat.sum().item()}, act_ids_flat.numel()={act_ids_flat.numel()}"
        )

        input_ids_flat[mask_flat] = act_ids_flat
        input_ids_replaced = input_ids_flat.view_as(input_ids)

        if labels is None:
            return input_ids_replaced, None
        labels_flat = labels.view(-1)
        labels_flat[mask_flat] = act_ids_flat
        labels_replaced = labels_flat.view_as(labels)
        return input_ids_replaced, labels_replaced

    def _vlm_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        num_queries: int,
    ):
        """Forward VLM; in supervise_quantized mode inject learnable queries via inputs_embeds."""
        if not self.supervise_quantized:
            return self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                output_hidden_states=True,
            )

        if self.act_query is None:
            raise ValueError("[LatentVLAModel] supervise_quantized=True but act_query is None.")
        B = int(input_ids.shape[0])
        if int(act_placeholder_mask.sum().item()) != int(B * num_queries):
            raise ValueError(
                f"[LatentVLAModel] placeholder count mismatch under supervise_quantized: "
                f"mask_sum={int(act_placeholder_mask.sum().item())}, expected={int(B*num_queries)}"
            )
        embed = self.vlm.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        qvec = self.act_query.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)  # [Q, D]
        for b in range(B):
            pos = torch.nonzero(act_placeholder_mask[b], as_tuple=False).flatten()
            if pos.numel() != num_queries:
                raise ValueError(
                    f"[LatentVLAModel] placeholder count mismatch for sample {b}: got={int(pos.numel())}, expected={num_queries}"
                )
            inputs_embeds[b, pos, :] = qvec

        return self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            output_hidden_states=True,
        )

    def forward_vlm_queries_supervise_quantized(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        num_queries: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        提供可复用的 VLM-query 前向（supervise_quantized 路径）：
        - 注入 act_query
        - 返回占位符位置 hidden (h_vlm) 与投影后的 pred_quantized
        """
        Q = int(num_queries) if num_queries is not None else int(self.num_queries)
        vlm_out = self._vlm_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=Q,
        )
        hidden = vlm_out.hidden_states[-1]
        pred_quantized = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=Q,
        )
        return {
            "h_vlm": hidden,
            "pred_quantized": pred_quantized,
            "vlm_out": vlm_out,
        }

    def _project_action_hidden_to_lam(
        self,
        *,
        hidden: torch.Tensor,
        act_placeholder_mask: torch.BoolTensor,
        num_queries: int,
    ) -> Optional[torch.Tensor]:
        """Project action/query hidden states to LAM code space: returns [B,Q,D_lam] or None."""
        if not self.supervise_quantized:
            return None
        if self.vlm_to_lam is None:
            return None
        B = int(hidden.shape[0])
        Q = int(num_queries)
        h_act = hidden[act_placeholder_mask]  # [B*Q, D_vlm]
        if h_act.numel() == 0:
            raise ValueError("[LatentVLAModel] empty action hidden selection; check act_placeholder_mask / padding.")
        if h_act.shape[0] != B * Q:
            raise ValueError(
                f"[LatentVLAModel] action hidden count mismatch: got={h_act.shape[0]}, expected={B*Q}. "
                f"mask_sum={int(act_placeholder_mask.sum().item())}, B={B}, Q={Q}"
            )
        h_act = h_act.view(B, Q, -1)  # [B, Q, D_vlm]
        # `vlm_to_lam` may be Linear or an MLP (nn.Sequential). Infer a safe target dtype from its parameters.
        try:
            p0 = next(self.vlm_to_lam.parameters())
            target_dtype = p0.dtype
        except StopIteration:
            target_dtype = h_act.dtype
        except Exception:
            target_dtype = h_act.dtype
        if target_dtype != h_act.dtype:
            h_act = h_act.to(dtype=target_dtype)
        return self.vlm_to_lam(h_act)  # [B, Q, D_lam]

    def _compute_quantized_loss(
        self,
        *,
        pred_quantized: Optional[torch.Tensor],
        z_teacher: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute quantized regression/distill loss (cosine or mse)."""
        if pred_quantized is None:
            return None
        if pred_quantized.shape != z_teacher.shape:
            raise ValueError(
                f"[LatentVLAModel] quantized shape mismatch: pred={tuple(pred_quantized.shape)} "
                f"teacher={tuple(z_teacher.shape)}. "
                "This usually means `vlm_to_lam` out_features was set to VQ code_dim instead of VQ input_dim."
            )
        if self.quantized_loss_type == "mse":
            return F.mse_loss(pred_quantized, z_teacher)
        return 1 - F.cosine_similarity(pred_quantized, z_teacher, dim=-1).mean()

    def _compute_decoder_perceptual_loss(
        self,
        *,
        vq_out: Dict[str, torch.Tensor],
        pred_quantized: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute perceptual loss via LAM decoder recon_pred vs recon_teacher (from vq_encode).
        
        Returns:
            Tuple of (loss, delta_student):
            - loss: MSE loss between delta_student and delta_teacher, or None if disabled
            - delta_student: recon_pred - dec_in_f, used for identity shortcut metric, or None if disabled
        """
        if not (self.enable_lam_decoder_perceptual and self.lam_decoder_perceptual_weight > 0):
            return None, None
        if pred_quantized is None:
            raise ValueError(
                "[LatentVLAModel] enable_lam_decoder_perceptual requires pred_quantized "
                "(supervise_quantized must be True)."
            )
        dec_in = vq_out.get("dec_in", None)
        if dec_in is None:
            raise ValueError("[LatentVLAModel] vq_out['dec_in'] is None; cannot compute decoder perceptual loss.")
        # 目标类型：teacher 预测或真实 future 特征
        target_kind = self.lam_decoder_target
        if target_kind == "gt":
            recon_target = vq_out.get("tgt", None)
            if recon_target is None:
                raise ValueError("[LatentVLAModel] vq_out['tgt'] is None; cannot use ground_truth target for perceptual loss.")
        else:
            recon_target = vq_out.get("recon", None)
            if recon_target is None:
                raise ValueError(
                    "[LatentVLAModel] vq_out['recon'] is None; ensure vq_encode was called with predict_future_frame=True."
                )
        # vq_encode runs under torch.inference_mode(): returned tensors are "inference tensors" and cannot be
        # saved for backward in autograd ops. Clone them to normal tensors (still no grad) before using in loss/decoder.
        try:
            dec_in = dec_in.detach().clone()
        except Exception:
            pass
        try:
            recon_target = recon_target.detach().clone()
        except Exception:
            pass

        dec_in_f = dec_in.to(device=pred_quantized.device, dtype=pred_quantized.dtype)

        recon_pred = self.lam.decoder(
            features=dec_in_f,
            actions=pred_quantized,
        )
        if recon_pred is None:
            raise ValueError("[LatentVLAModel] decoder returned None recon; cannot compute perceptual loss.")
        if isinstance(recon_pred, (tuple, list)) and len(recon_pred) > 0:
            recon_pred = recon_pred[0]
        recon_target = recon_target.to(device=recon_pred.device, dtype=recon_pred.dtype)
        dec_in_f = dec_in_f.to(device=recon_pred.device, dtype=recon_pred.dtype)
        # ensure recon_teacher is a normal (non-inference) tensor for autograd save_for_backward
        recon_target = recon_target.detach().clone()

        loss = F.mse_loss(recon_pred, recon_target)
        return loss, recon_pred-dec_in_f

    def _compute_kl_loss(
        self,
        *,
        vq_out: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        act_placeholder_mask: torch.BoolTensor,
    ) -> Optional[torch.Tensor]:
        """KL(p_teacher || q_student) on action vocabulary, teacher from LAM soft probs."""
        if not self.enable_lam_kl_loss:
            return None
        teacher_probs = vq_out.get("vq_probs", None)
        if teacher_probs is None:
            return None

        act_token_range = torch.arange(self.codebook_size, device=logits.device) + self.action_token_begin_id
        student_logits_all = logits.index_select(dim=-1, index=act_token_range)

        mask_student = act_placeholder_mask[:, 1:]
        student_logits = student_logits_all[:, :-1, :]

        num_targets = teacher_probs.numel() // teacher_probs.shape[-1]
        if mask_student.sum().item() != num_targets:
            mask_student = act_placeholder_mask
            student_logits = student_logits_all

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        mask_flat = mask_student.reshape(-1)
        student_log_probs_flat = student_log_probs.reshape(-1, student_log_probs.shape[-1])
        student_log_probs_flat = student_log_probs_flat[mask_flat]

        teacher_probs = teacher_probs.to(device=student_log_probs_flat.device)
        teacher_probs_flat = teacher_probs.view(-1, teacher_probs.shape[-1]).to(student_log_probs_flat.dtype)
        if student_log_probs_flat.shape[0] != teacher_probs_flat.shape[0]:
            raise ValueError(
                f"[LatentVLAModel] KL shape mismatch: student={student_log_probs_flat.shape} "
                f"teacher={teacher_probs_flat.shape}, mask_sum={mask_flat.sum().item()}, "
                f"teacher_targets={teacher_probs_flat.shape[0]}"
            )
        return F.kl_div(student_log_probs_flat, teacher_probs_flat, reduction="mean")

    def _sum_losses(
        self,
        *,
        base_device: torch.device,
        base_dtype: torch.dtype,
        loss_main: Optional[torch.Tensor],
        loss_distill: Optional[torch.Tensor],
        loss_perceptual: Optional[torch.Tensor],
        loss_kl: Optional[torch.Tensor],
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum enabled loss terms with corresponding weights."""
        total = torch.tensor(0.0, device=base_device, dtype=base_dtype)
        if loss_main is not None:
            total = total + loss_main
        if loss_distill is not None:
            total = total + self.lam_encoder_distill_weight * loss_distill
        if loss_perceptual is not None:
            total = total + self.lam_decoder_perceptual_weight * loss_perceptual
        if loss_kl is not None:
            total = total + self.lam_kl_weight * loss_kl

        # DDP safety: supervise_quantized 下通常不使用 logits/lm_head 来计算 loss，
        # 会导致 lm_head（及其 bias）等参数在某些配置下被判定为 unused。
        # 加一个 0 * logits.sum()，使其进入 autograd 图但不改变数值。
        if self.supervise_quantized and logits is not None:
            total = total + logits.sum().to(dtype=total.dtype) * 0.0
        return total

    @torch.inference_mode()
    def predict_latent_actions(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        lam_dec_videos: Optional[torch.FloatTensor] = None,
        lam_dataset_ids: Optional[torch.Tensor] = None,
        act_placeholder_mask: Optional[torch.BoolTensor] = None,
        strict_placeholder_count: bool = True,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
    推理：
      - supervise_quantized=False（token CE）：返回
          lam_latent_idx: LAM vq_encode 的 teacher indices，形状 [B, Q]
          vlm_latent_idx: VLM generate 在动作词表上的 argmax indices，形状 [B, Q]
      - supervise_quantized=True（query 回归）：无需 vq 索引，直接回归连续 embedding，返回
          pred_quantized: VLM 回归的动作 embedding，形状 [B, Q, D_lam]
          （若传入 lam_videos/lam_states，则额外返回 lam_latent_idx/lam_quantized 供对比）
    Q 由训练时配置 self.num_queries 决定；若缺失则尝试从 LAM 输出推断。
        """
        placeholder_mask = (
            act_placeholder_mask.to(dtype=torch.bool, device=input_ids.device)
            if act_placeholder_mask is not None
            else input_ids == self.placeholder_token_id
        )
        if self.num_queries is None:
            raise ValueError("[LatentVLAModel] num_queries is None; ensure LAM提供了 num_queries")
        Q = int(self.num_queries)
        counts = placeholder_mask.sum(dim=1)  # [B]
        unique_counts = torch.unique(counts)
        if unique_counts.numel() != 1:
            raise ValueError(f"[LatentVLAModel] inconsistent placeholder counts across batch: {counts.tolist()}")
        Q_mask = int(unique_counts.item())
        if Q_mask != Q:
            raise ValueError(
                f"[LatentVLAModel] placeholder count mismatch: found={Q_mask}, expected={Q} "
                "(check data pipeline vs model.num_queries)"
            )

        # ---- supervise_quantized: 直接回归 embedding，不生成 token ----
        if self.supervise_quantized:
            vlm_out = self._vlm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
                act_placeholder_mask=placeholder_mask,
                num_queries=Q,
            )
            hidden = vlm_out.hidden_states[-1]
            pred_quantized = self._project_action_hidden_to_lam(
                hidden=hidden,
                act_placeholder_mask=placeholder_mask,
                num_queries=Q,
            )

            out: Dict[str, torch.Tensor] = {
                "pred_quantized": pred_quantized,
                "logits": vlm_out.logits,
            }

            # 可选：如仍提供 LAM 输入，返回 teacher embedding/indices 便于比较或评估
            if lam_videos is not None and lam_states is not None:
                vq_out = self._lam_vq_encode(
                    lam_videos=lam_videos,
                    lam_states=lam_states,
                    lam_dec_videos=lam_dec_videos,
                    lam_dataset_ids=lam_dataset_ids,
                )
                out["lam_latent_idx"] = vq_out["indices"].detach()
                out["lam_quantized"] = vq_out["quantized"].detach()

            return out

        # ---- token CE 路径：保持原生成/argmax 逻辑 ----
        # 1) LAM 编码（teacher）
        device_type = lam_videos.device.type if hasattr(lam_videos, "device") else "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=False):
                vq_out = self.lam.vq_encode(
                    videos=lam_videos,
                    states=lam_states,
                    dec_videos=lam_dec_videos,
                    predict_future_frame=False,
                    dataset_ids=lam_dataset_ids,
                    return_teacher_probs=False,
                )

        lam_latent_idx = vq_out.get("indices", None)
        if lam_latent_idx is None:
            raise ValueError("[LatentVLAModel] VAE mode provides no discrete indices; token-generation path is not supported. Use supervise_quantized=True or enable discrete VQ.")
        lam_latent_idx = lam_latent_idx.detach()
        if lam_latent_idx.shape[1] != Q:
            raise ValueError(
                f"[LatentVLAModel] LAM indices length mismatch: got {lam_latent_idx.shape[1]}, expected {Q}"
            )
        lam_latent_idx = lam_latent_idx.to(device=input_ids.device)

        # 2) 构造生成前缀（去掉占位符）
        placeholder_mask = (
            act_placeholder_mask.to(dtype=torch.bool, device=input_ids.device)
            if act_placeholder_mask is not None
            else input_ids == self.placeholder_token_id
        )
        if placeholder_mask.any():
            prefix_input_ids, prefix_attention_mask, _ = self._build_prefix_from_placeholders(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict_placeholder_count=True,
            )
        else:
            prefix_input_ids = input_ids
            prefix_attention_mask = attention_mask

        # 3) VLM 生成 Q 个动作 token，并在动作词表上取 argmax
        gen_kwargs = dict(generate_kwargs or {})
        gen_kwargs.setdefault("min_new_tokens", Q)
        gen_kwargs.setdefault("max_new_tokens", Q)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("return_dict_in_generate", True)
        gen_kwargs.setdefault("output_scores", True)

        gen_out = self.vlm.generate(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **gen_kwargs,
        )
        scores = gen_out.scores
        if scores is None or len(scores) < Q:
            raise ValueError(f"[LatentVLAModel] generate returned insufficient scores, len={0 if scores is None else len(scores)}")

        act_token_range = torch.arange(self.codebook_size, device=scores[0].device) + self.action_token_begin_id
        step_indices = []
        for s in scores[:Q]:
            act_logits = s.index_select(dim=-1, index=act_token_range)  # [B, K]
            step_indices.append(torch.argmax(act_logits, dim=-1))  # [B]
        vlm_latent_idx = torch.stack(step_indices, dim=0).transpose(0, 1).contiguous()  # [B, Q]

        return {
            "lam_latent_idx": lam_latent_idx,
            "vlm_latent_idx": vlm_latent_idx.to(device=input_ids.device),
        }