import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple

from latent_action_model.core.lam_model import LatentLAMModel, load_latent_action_model
from prismatic.models import load_vlm_auto


class VLMToLAMQFormer(nn.Module):
    """Refine VLM query hidden states into a single latent action."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.1,
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
                    # nn.Dropout(float(dropout)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() != 3:
            raise ValueError(f"[VLMToLAMQFormer] expected context [B, Q, D], got {tuple(context.shape)}")
        B = int(context.shape[0])
        queries = self.query.expand(B, -1, -1)
        queries = queries.to(device=context.device, dtype=context.dtype)
        if context.dtype != queries.dtype:
            context = context.to(dtype=queries.dtype)
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        queries = self.final_norm(queries)
        return queries


class LatentVLAModel(nn.Module):
    """
    组合模型：封装 VLM 与 LAM（使用可学习 token 作为 latent action）

    - 在 forward 中调用 LAM 编码器取得连续 latent 表征作为监督信号
    - 在 <ACT_PH> 位置注入可学习 query embedding（act_query）
    - 计算 latent 回归损失（VLM hidden 对齐 LAM latent）
    - 可选 decoder 感知损失

    注意：不再使用特殊 token 索引进行训练，完全基于可学习的 query embeddings
    """

    def __init__(
        self,
        vlm: nn.Module,
        lam: LatentLAMModel,
        *,
        placeholder_token_id: int,
        latent_loss_type: str = "cosine",
        enable_lam_decoder_perceptual: bool = False,
        lam_encoder_distill_weight: float = 1.0,
        lam_decoder_perceptual_weight: float = 1.0,
        vlm_hidden_dim: int = 2048,
        lam_code_dim: int = 256,
        processor: Optional[Any] = None,
        act_query_lr_scale: float = 1.0,
        vlm_to_lam_lr_scale: float = 1.0,
        num_queries: int = 8,
        qformer_layers: int = 1,
        qformer_heads: int = 8,
        qformer_ffn_expansion: float = 4.0,
        qformer_dropout: float = 0.0,
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

        self.placeholder_token_id = int(placeholder_token_id)

        # 使用 latent 回归监督
        self.latent_loss_type = str(latent_loss_type).lower()
        self.enable_lam_decoder_perceptual = bool(enable_lam_decoder_perceptual)
        self.lam_encoder_distill_weight = float(lam_encoder_distill_weight)
        self.lam_decoder_perceptual_weight = float(lam_decoder_perceptual_weight)
        self.act_query_lr_scale = float(act_query_lr_scale)
        self.vlm_to_lam_lr_scale = float(vlm_to_lam_lr_scale)
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
        # 训练时的 latent query 数量（Q）；由配置决定，不再从 LAM 获取
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError("[LatentVLAModel] num_queries must be > 0.")

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

        # latent 向量回归投影头：QFormer 将多个 VLM query 进一步提炼为单一 latent action
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden_dim,
            lam_code_dim=lam_code_dim,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
            ffn_expansion_factor=qformer_ffn_expansion,
            dropout=qformer_dropout,
        )

        # 在 <ACT_PH> 位置注入可学习 query embedding（而不是“从头预测”动作 token）
        # 注意：这些 query 不在 VLM 内部，因此默认不会被 vlm.save_pretrained 保存；我们会在 save_pretrained 里额外落盘。
        q = int(self.num_queries)
        # float32 参数更稳定；前向时会按 inputs_embeds dtype 做 cast
        self.act_query = nn.Parameter(torch.randn(q, int(vlm_hidden_dim)) * 0.02)

        # 对 query / head 做梯度放大
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

    # -------------------------
    # Checkpoint IO
    # -------------------------
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
                # 保存线性投影头（用于 latent 回归/encoder distill）
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
        - act_query
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
                        "[LatentVLAModel] extra contains act_query but current model has no act_query."
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
                        "[LatentVLAModel] extra contains vlm_to_lam but current model has no vlm_to_lam."
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

    # -------------------------
    # Construction / config
    # -------------------------
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
        
        该方法会：
        1. 加载基础 VLM 和 LAM 模型
        2. 注册占位符 token <ACT_PH>（用于标记可学习 query 的注入位置）
        3. 初始化可学习的 act_query 参数（替代特殊 token 索引）
        
        占位符 id 可通过 model.placeholder_token_id 访问

        Args:
            yaml_path: 可选。训练时使用的 YAML 配置文件路径；若提供则会先读取并覆盖传入 cfg 的字段，
                       用于推理/继续训练时确保关键参数与训练一致。
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

        # 注册占位符 token
        placeholder_token = getattr(cfg, "latent_action_placeholder_token", "<ACT_PH>")
        tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})  # type: ignore[attr-defined]

        # 打印当前 tokenizer 词表大小与 VLM embedding 大小
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

        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        model = cls(
            vlm=vlm,
            lam=lam,
            placeholder_token_id=placeholder_token_id,
            latent_loss_type=getattr(cfg, "latent_loss_type", "cosine"),
            enable_lam_decoder_perceptual=getattr(cfg, "enable_lam_decoder_perceptual", False),
            lam_encoder_distill_weight=getattr(cfg, "lam_encoder_distill_weight", 1.0),
            lam_decoder_perceptual_weight=getattr(cfg, "lam_decoder_perceptual_weight", 0.0),
            vlm_hidden_dim=vlm.config.text_config.hidden_size,
            lam_code_dim=int(lam.code_dim),
            processor=processor,
            act_query_lr_scale=getattr(cfg, "act_query_lr_scale", 1.0),
            vlm_to_lam_lr_scale=getattr(cfg, "vlm_to_lam_lr_scale", 1.0),
            debug_mode=bool(debug_mode),
            unfreeze_lam_decoder=bool(getattr(cfg, "unfreeze_lam_decoder", False)),
            lam_decoder_target=str(getattr(cfg, "lam_decoder_target", "teacher")),
        )
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

    # -------------------------
    # Public forward APIs
    # -------------------------
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
        lam_out, teacher_latent = self._run_lam_teacher(
            lam_videos=lam_videos,
            lam_states=lam_states,
            lam_dec_videos=lam_dec_videos,
            lam_dataset_ids=lam_dataset_ids,
        )
        teacher_latent = teacher_latent.to(input_ids.device)

        vlm_out, hidden, pred_latent = self._run_vlm_act_queries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
        )
        logits = vlm_out.logits
        loss_main = None
        loss_distill = self._compute_latent_loss(pred_latent=pred_latent, teacher_latent=teacher_latent)
        loss_perceptual, delta_student = self._compute_decoder_perceptual_loss(
            lam_out=lam_out,
            pred_latent=pred_latent,
        )
        total_loss = self._sum_losses(
            base_dtype=hidden.dtype,
            base_device=input_ids.device,
            loss_main=loss_main,
            loss_distill=loss_distill,
            loss_perceptual=loss_perceptual,
            logits=logits,
        )

        identity_shortcut = self._compute_identity_shortcut(delta_student)

        return {
            "loss": total_loss,
            "loss_main": loss_main,
            "loss_distill": loss_distill,
            "loss_perceptual": loss_perceptual,
            "logits": logits,
            "identity_shortcut": identity_shortcut,
        }

    # -------------------------
    # LAM helpers
    # -------------------------
    def _run_lam_teacher(
        self,
        *,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        lam_dec_videos: Optional[torch.FloatTensor],
        lam_dataset_ids: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        lam_out = self._lam_encode_teacher(
            lam_videos=lam_videos,
            lam_states=lam_states,
            lam_dec_videos=lam_dec_videos,
            lam_dataset_ids=lam_dataset_ids,
        )
        self._maybe_debug_check_lam_codes(lam_out)
        teacher_latent = lam_out["quantized"].detach().clone()
        return lam_out, teacher_latent

    def _lam_encode_teacher(
        self,
        *,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        lam_dec_videos: Optional[torch.FloatTensor],
        lam_dataset_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run frozen LAM encoder in fp32 (autocast disabled) and return latent dict."""
        with torch.no_grad():
            device_type = lam_videos.device.type if hasattr(lam_videos, "device") else "cuda"
            with torch.autocast(device_type=device_type, enabled=False):
                return self.lam.get_latent_action(
                    videos=lam_videos,
                    states=lam_states,
                    dec_videos=lam_dec_videos,
                    predict_future_frame=bool(
                        self.enable_lam_decoder_perceptual and self.lam_decoder_perceptual_weight > 0
                    ),
                    dataset_ids=lam_dataset_ids,
                )

    def _maybe_debug_check_lam_codes(self, lam_out: Dict[str, torch.Tensor]) -> None:
        """Optional debug hook for LAM indices under debug_repeat_batch.

        NOTE: We keep debug_mode for other debugging workflows, but we no longer
        perform the original "compare current codes vs a reference batch" logic.
        Instead, we only record the most recent indices for potential inspection.
        """
        if not self.debug_mode:
            return
        try:
            codes = lam_out.get("indices", None)
            if codes is None:
                print("[LamCodes Debug] codes is None; skipping diff (lam training=True?)")
                return
            codes = codes.detach().clone()
            # Record latest codes (train/eval separated) for debugging/inspection.
            last_attr = "_last_lam_codes_train" if self.training else "_last_lam_codes_eval"
            setattr(self, last_attr, codes)
        except Exception as e:
            print(f"[LamCodes Debug] diff check failed: {e}")

    # -------------------------
    # VLM helpers
    # -------------------------
    def _run_vlm_act_queries(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
    ) -> Tuple[Any, torch.Tensor, Optional[torch.Tensor]]:
        Q = int(self.num_queries)
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
        pred_latent = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=Q,
        )
        return vlm_out, hidden, pred_latent

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
        """Forward VLM; inject learnable queries via inputs_embeds."""
        if self.act_query is None:
            raise ValueError("[LatentVLAModel] act_query is None.")
        return self._vlm_forward_with_queries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            act_placeholder_mask=act_placeholder_mask,
            act_query=self.act_query,
            act_num_queries=num_queries,
            flow_placeholder_mask=None,
            flow_query=None,
            flow_num_queries=None,
        )

    def _inject_queries_into_embeddings(
        self,
        *,
        inputs_embeds: torch.Tensor,
        placeholder_mask: torch.BoolTensor,
        queries: torch.Tensor,
        num_queries: int,
        name: str,
    ) -> None:
        if placeholder_mask is None or queries is None:
            return
        device = inputs_embeds.device
        placeholder_mask = placeholder_mask.to(device=device, dtype=torch.bool)
        B, L = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        if int(placeholder_mask.sum().item()) != int(B * num_queries):
            raise ValueError(
                f"[LatentVLAModel] {name} placeholder count mismatch: "
                f"mask_sum={int(placeholder_mask.sum().item())}, expected={int(B*num_queries)}"
            )
        per_sample = placeholder_mask.sum(dim=1)
        if not torch.all(per_sample == int(num_queries)):
            bad = torch.nonzero(per_sample != int(num_queries), as_tuple=False).flatten()
            b = int(bad[0].item()) if bad.numel() > 0 else -1
            raise ValueError(
                f"[LatentVLAModel] {name} placeholder count mismatch for sample {b}: "
                f"got={int(per_sample[b].item()) if b >= 0 else 'unknown'}, expected={num_queries}"
            )

        qvec = queries.to(device=device, dtype=inputs_embeds.dtype)  # [Q, D]

        # Vectorized assignment: map the i-th placeholder in each sample to queries[i].
        # Sort indices to guarantee per-sample ascending position order.
        idx = placeholder_mask.nonzero(as_tuple=False)  # [B*num_queries, 2] => (b, pos)
        flat = idx[:, 0] * L + idx[:, 1]
        idx = idx[flat.argsort()]
        b_idx, p_idx = idx[:, 0], idx[:, 1]
        q_idx = torch.arange(int(num_queries), device=device).repeat(B)  # [B*num_queries]
        inputs_embeds[b_idx, p_idx, :] = qvec[q_idx]

    def _vlm_forward_with_queries(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        act_placeholder_mask: Optional[torch.BoolTensor],
        act_query: Optional[torch.Tensor],
        act_num_queries: Optional[int],
        flow_placeholder_mask: Optional[torch.BoolTensor],
        flow_query: Optional[torch.Tensor],
        flow_num_queries: Optional[int],
    ):
        """Forward VLM with custom query injection (supports dual query sets)."""
        embed = self.vlm.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        if act_placeholder_mask is not None and act_query is not None:
            if act_num_queries is None:
                raise ValueError("[LatentVLAModel] act_num_queries is None.")
            self._inject_queries_into_embeddings(
                inputs_embeds=inputs_embeds,
                placeholder_mask=act_placeholder_mask,
                queries=act_query,
                num_queries=int(act_num_queries),
                name="act_query",
            )
        if flow_placeholder_mask is not None and flow_query is not None:
            if flow_num_queries is None:
                raise ValueError("[LatentVLAModel] flow_num_queries is None.")
            self._inject_queries_into_embeddings(
                inputs_embeds=inputs_embeds,
                placeholder_mask=flow_placeholder_mask,
                queries=flow_query,
                num_queries=int(flow_num_queries),
                name="flow_query",
            )
        return self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            output_hidden_states=True,
        )

    def forward_vlm_queries_supervise_latent(
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
        提供可复用的 VLM-query 前向：
        - 注入 act_query
        - 返回占位符位置 hidden (h_vlm) 与投影后的 pred_latent
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
        pred_latent = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=Q,
        )
        return {
            "h_vlm": hidden,
            "pred_latent": pred_latent,
            "vlm_out": vlm_out,
        }

    def forward_vlm_queries_supervise_latent_with_flow(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        flow_placeholder_mask: torch.BoolTensor,
        flow_query: torch.Tensor,
        num_queries: Optional[int] = None,
        flow_num_queries: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        VLM 前向：注入 act_query + flow_query（两组占位符），
        返回：
            - pred_latent（来自 act_query 位置）
            - h_vlm（全序列 hidden）
        """
        Q = int(num_queries) if num_queries is not None else int(self.num_queries)
        F = int(flow_num_queries) if flow_num_queries is not None else int(flow_query.shape[0])
        if flow_query is None:
            raise ValueError("[LatentVLAModel] flow_query is None.")
        vlm_out = self._vlm_forward_with_queries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            act_placeholder_mask=act_placeholder_mask,
            act_query=self.act_query,
            act_num_queries=Q,
            flow_placeholder_mask=flow_placeholder_mask,
            flow_query=flow_query,
            flow_num_queries=F,
        )
        hidden = vlm_out.hidden_states[-1]
        pred_latent = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=act_placeholder_mask,
            num_queries=Q,
        )
        return {
            "h_vlm": hidden,
            "pred_latent": pred_latent,
            "vlm_out": vlm_out,
        }

    # -------------------------
    # Projection / loss helpers
    # -------------------------
    def _project_action_hidden_to_lam(
        self,
        *,
        hidden: torch.Tensor,
        act_placeholder_mask: torch.BoolTensor,
        num_queries: int,
    ) -> Optional[torch.Tensor]:
        """Project action/query hidden states to LAM code space: returns [B,1,D_lam] or None."""
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
        return self.vlm_to_lam(h_act)  # [B, 1, D_lam]

    def _compute_latent_loss(
        self,
        *,
        pred_latent: Optional[torch.Tensor],
        teacher_latent: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute latent regression/distill loss (cosine or mse)."""
        if pred_latent is None:
            return None
        if pred_latent.shape != teacher_latent.shape:
            raise ValueError(
                f"[LatentVLAModel] latent shape mismatch: pred={tuple(pred_latent.shape)} "
                f"teacher={tuple(teacher_latent.shape)}. "
                "This usually means `vlm_to_lam` out_features was set to code_dim instead of input_dim."
            )
        if self.latent_loss_type == "mse":
            return F.mse_loss(pred_latent, teacher_latent)
        return 1 - F.cosine_similarity(pred_latent, teacher_latent, dim=-1).mean()

    def _compute_decoder_perceptual_loss(
        self,
        *,
        lam_out: Dict[str, torch.Tensor],
        pred_latent: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute perceptual loss via LAM decoder recon_pred vs recon_target (from LAM encoder).
        
        Returns:
            Tuple of (loss, delta_student):
            - loss: MSE loss between delta_student and delta_teacher, or None if disabled
            - delta_student: recon_pred - dec_in_f, used for identity shortcut metric, or None if disabled
        """
        if not (self.enable_lam_decoder_perceptual and self.lam_decoder_perceptual_weight > 0):
            return None, None
        if pred_latent is None:
            raise ValueError(
                "[LatentVLAModel] enable_lam_decoder_perceptual requires pred_latent."
            )
        dec_in = lam_out.get("dec_in", None)
        if dec_in is None:
            raise ValueError("[LatentVLAModel] lam_out['dec_in'] is None; cannot compute decoder perceptual loss.")
        # 目标类型：teacher 预测或真实 future 特征
        target_kind = self.lam_decoder_target
        if target_kind == "gt":
            recon_target = lam_out.get("tgt", None)
            if recon_target is None:
                raise ValueError("[LatentVLAModel] lam_out['tgt'] is None; cannot use ground_truth target for perceptual loss.")
        else:
            recon_target = lam_out.get("recon", None)
            if recon_target is None:
                raise ValueError(
                    "[LatentVLAModel] lam_out['recon'] is None; ensure LAM encoder was called with predict_future_frame=True."
                )
        # LAM encoder runs under torch.inference_mode(): returned tensors are "inference tensors" and cannot be
        # saved for backward in autograd ops. Clone them to normal tensors (still no grad) before using in loss/decoder.
        try:
            dec_in = dec_in.detach().clone()
        except Exception:
            pass
        try:
            recon_target = recon_target.detach().clone()
        except Exception:
            pass

        dec_in_f = dec_in.to(device=pred_latent.device, dtype=pred_latent.dtype)

        recon_pred = self.lam.decoder(
            features=dec_in_f,
            actions=pred_latent,
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

    def _compute_identity_shortcut(self, delta_student: Optional[torch.Tensor]) -> Optional[float]:
        if delta_student is None:
            return None
        with torch.no_grad():
            return delta_student.abs().mean().item()

    def _sum_losses(
        self,
        *,
        base_device: torch.device,
        base_dtype: torch.dtype,
        loss_main: Optional[torch.Tensor],
        loss_distill: Optional[torch.Tensor],
        loss_perceptual: Optional[torch.Tensor],
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
        # DDP safety: 通常不使用 logits/lm_head 来计算 loss，
        # 会导致 lm_head（及其 bias）等参数在某些配置下被判定为 unused。
        # 加一个 0 * logits.sum()，使其进入 autograd 图但不改变数值。
        if logits is not None:
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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        推理：通过可学习 query 直接回归连续 embedding，返回
          pred_latent: VLM 回归的动作 embedding，形状 [B, Q, D_lam]
          （若传入 lam_videos/lam_states，则额外返回 lam_latent_idx/lam_latent 供对比/评估，
           注意：这些 indices 仅用于参考，不参与训练或推理）
        Q 由训练时配置 self.num_queries 决定；若缺失则报错。
        """
        placeholder_mask = (
            act_placeholder_mask.to(dtype=torch.bool, device=input_ids.device)
            if act_placeholder_mask is not None
            else input_ids == self.placeholder_token_id
        )
        if self.num_queries is None:
            raise ValueError("[LatentVLAModel] num_queries is None; ensure it is configured.")
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
        pred_latent = self._project_action_hidden_to_lam(
            hidden=hidden,
            act_placeholder_mask=placeholder_mask,
            num_queries=Q,
        )

        out: Dict[str, torch.Tensor] = {
            "pred_latent": pred_latent,
            "logits": vlm_out.logits,
        }

        # 如提供 LAM 输入，返回 teacher embedding/indices 便于比较或评估
        if lam_videos is not None and lam_states is not None:
            lam_out = self._lam_encode_teacher(
                lam_videos=lam_videos,
                lam_states=lam_states,
                lam_dec_videos=lam_dec_videos,
                lam_dataset_ids=lam_dataset_ids,
            )
            out["lam_latent_idx"] = lam_out["indices"].detach()
            out["lam_latent"] = lam_out["quantized"].detach()

        return out
