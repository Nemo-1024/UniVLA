import enum
from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model
from .flowmatching_expert import ConditionalFlowMatchingConfig, ConditionalFlowMatchingHead
from .simple_action_expert import SimpleActionHead, SimpleActionConfig
from prismatic.vla.latent_vla_model import LatentVLAModel


class FutureFeatureMode(str, enum.Enum):
    """控制传入Flow Matching的未来表征来源（CFG为固定机制，不再作为模式）"""

    LAM_FROM_VLM = "lam_from_vlm"  # 使用VLM输出的z_a，经LAM decoder得到 h_{t+1}^hat
    LAM_FROM_GT = "lam_from_gt"  # 使用GT的z_a索引，经LAM decoder得到 h_{t+1}^hat
    VJEPA_GT = "vjepa_gt"  # 直接使用VJEPA对 I_{t+1} 的编码 h_{t+1}


HOME_PATH = "/mnt/mnt/public/jlchen"


def _build_lam_teacher_inputs_for_distill(
    *,
    lam: nn.Module,
    lam_videos: torch.Tensor,
    lam_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align temporal length of distill teacher inputs to LAM encoder expectation.

    World-model finetune batches often carry full windows (e.g. T=10), while LAM encoder
    is configured with a fixed `num_frames` (commonly 2 in latent distill setup).
    """
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

    # Evenly sample frames to match expected T; for expected_t=2 this becomes [first, last].
    idx = torch.linspace(0, cur_t - 1, steps=expected_t, device=lam_videos.device).round().long()
    return lam_videos.index_select(1, idx), lam_states.index_select(1, idx.to(lam_states.device))


def _compute_distill_loss_via_latent_vla(
    *,
    latent_vla: LatentVLAModel,
    pred_latent: torch.Tensor,
    lam_videos: torch.Tensor,
    lam_states: torch.Tensor,
) -> torch.Tensor:
    """Reuse LatentVLAModel internals for teacher encode + latent distill loss."""
    lam_videos_t, lam_states_t = _build_lam_teacher_inputs_for_distill(
        lam=latent_vla.lam,
        lam_videos=lam_videos,
        lam_states=lam_states,
    )
    _, teacher_latent = latent_vla._run_lam_teacher(
        lam_videos=lam_videos_t,
        lam_states=lam_states_t,
        lam_dec_videos=None,
        lam_dataset_ids=None,
    )
    teacher_latent = teacher_latent.to(device=pred_latent.device, dtype=pred_latent.dtype)
    loss_distill = latent_vla._compute_latent_loss(pred_latent=pred_latent, teacher_latent=teacher_latent)
    if loss_distill is None:
        raise ValueError("[LatentWorldVLA] pred_latent is None; cannot compute distill loss.")
    return loss_distill


def _extract_lam_vision_features(lam: nn.Module, videos: torch.Tensor) -> torch.Tensor:
    return lam.extract_vision_features(videos)


@dataclass
class LatentWorldVLAConfig:
    """LatentWorldVLA 结构配置。只涉及模型与前向路径，不包含训练/数据集逻辑"""

    # CFG 超参：训练掉落概率（在Flow头部内部使用），推理guidance scale
    cfg_guidance_scale: float = 1.5
    num_inference_steps: int = 50  # Flow Matching 推理步数
    # Flow Matching Head 超参（ConditionalFlowMatchingConfig）
    # 通过 finetune YAML: `model_cfg.flow_cfg.*` 覆盖
    flow_cfg: ConditionalFlowMatchingConfig = field(default_factory=ConditionalFlowMatchingConfig)

    # ===== 新增：模型加载相关参数（由原 FinetuneConfig 迁移而来）=====
    # Base VLM & LAM
    model_id: str = HOME_PATH + "/code/UniVLA/vla_scripts/vla_log/0107_103411+vla_emb_unfreeze_llm4_libero_90/checkpoints/checkpoint-10000"
    hf_cache_dir: Optional[Union[str, Path]] = None
    # lam_path用于加载latent_vla中的lam权重
    lam_ckpt_path: str = HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bridge/version_0/checkpoints/epoch=39.ckpt"
    lam_yaml_path: str = HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bridge/version_0/dino_base_ae.yaml"
    
    # 训练配置 yaml（用于恢复训练时的配置）
    yaml_path: Optional[Union[str, Path]] = None

    # VLM 精度
    vlm_dtype: torch.dtype = torch.bfloat16
    
    # 冻结策略（细粒度控制）
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False
    unfreeze_llm_last_n_layers: Optional[int] = None
    unfreeze_lam_decoder: bool = False 
    # 占位符 token
    latent_action_placeholder_token: str = "<ACT_PH>"
    perceptual_weight: float = 0.1
    # 可选 distill 损失：默认关闭，开启后约束 VLM 预测 latent 对齐 LAM teacher latent
    enable_loss_distill: bool = True
    latent_loss_type: str = "mse"  # 可选: "cosine" / "mse"
    lam_encoder_distill_weight: float = 1.0

    future_prediction: bool = False # 是否预测未来特征
    repeated_diffusion_steps: int = 4  # 重复每个样本的次数
    enable_wrist_view: bool = False  # 是否启用腕部视角（仅提取 h_t_w 注入 flow，不影响 h_t1_star）
    # Flow-only 梯度模式：仅允许 flow placeholder 位置的 VLM hidden 接收 flow loss 梯度
    flow_only_mode: bool = False
    # ===== SimpleLatentWorldVLA 专用配置 =====
    use_simple_action_head: bool = False  # 是否使用 SimpleActionHead 替代 FlowMatching
    simple_action_dim: int = 7  # SimpleActionHead 动作维度
    simple_window_size: int = 10  # SimpleActionHead 动作序列长度
    simple_vlm_dim: int = 2048  # SimpleActionHead VLM 输出维度
    simple_hidden_dim: int = 768  # SimpleActionHead 中间层维度
    simple_qformer_layers: int = 8  # SimpleActionHead QFormer 层数
    simple_qformer_heads: int = 8  # SimpleActionHead QFormer 注意力头数
    simple_dropout: float = 0.1  # SimpleActionHead Dropout

class LatentWorldVLA(nn.Module):
    """
    隐式世界模型VLA：
    - 冻结 VLM 与 LAM（含 VJEPA Encoder 与 LAM Decoder）
    - 仅训练 FlowMatching 头与各路输入的可学习编码器
    - 未来表征 h_{t+1} 可选择：
        * LAM_FROM_VLM: 由VLM输出的 latent action 通过 LAM Decoder 得到
        * LAM_FROM_GT:  使用GT latent action indices 通过 LAM Decoder 得到
        * VJEPA_GT:     直接VJEPA对 I_{t+1} 的编码
        * CFG_MASK:     置零（或按概率drop）
    条件向量 = Eh_t || h_{t+1}^* || h_vlm || q_t
    送入 ConditionalFlowMatchingHead 预测速度场
    """

    def __init__(
        self,
        model_cfg: LatentWorldVLAConfig,
    ) -> None:
        super().__init__()

        self.model_cfg = model_cfg

        # 1) 加载完整的 LatentVLAModel（包含 VLM、LAM、act_query、vlm_to_lam、tokenizer 配置）
        self.latent_vla, self.processor = LatentVLAModel.from_config(self.model_cfg)
        self.vlm = self.latent_vla.vlm
        self.tokenizer = self.latent_vla.tokenizer
        self.lam = self.latent_vla.lam  # LAM 已在 LatentVLAModel.from_config 中加载，直接引用

        # 1.5) 应用细粒度冻结策略
        from prismatic.models import freeze_qwen3vl, _unfreeze_last_n_llm_layers, _resolve_llm_module
        freeze_qwen3vl(
            self.vlm,
            freeze_vision_backbone=self.model_cfg.freeze_vision_backbone,
            freeze_llm_backbone=self.model_cfg.freeze_llm_backbone,
            freeze_last_llm_layer=self.model_cfg.freeze_last_llm_layer,
            freeze_embedding=self.model_cfg.freeze_embedding,
            unfreeze_vision_merger=self.model_cfg.unfreeze_vision_merger,
        )

        # 解冻最后 N 层（如果配置）
        if self.model_cfg.freeze_llm_backbone and self.model_cfg.unfreeze_llm_last_n_layers:
            llm_module = _resolve_llm_module(self.vlm)
            if llm_module:
                _unfreeze_last_n_llm_layers(llm_module, self.model_cfg.unfreeze_llm_last_n_layers)

        # 2) 获取 LAM 信息
        self.code_book_size = self.lam.codebook_size
        self.placeholder_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.model_cfg.latent_action_placeholder_token)
        )

        # 2.5) 自动对齐 Flow 视觉配置到当前 LAM（视觉维度直接读取 lam.input_dim）
        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_size = int(getattr(self.lam.encoder, "grid_size", ) or 0)
        lam_num_tokens = int(lam_grid_size * lam_grid_size) if lam_grid_size > 0 else int(
            self.model_cfg.flow_cfg.num_vision_tokens
        )
        if int(self.model_cfg.flow_cfg.vision_dim) != lam_vision_dim:
            print(
                f"[LatentWorldVLA] flow_cfg.vision_dim={self.model_cfg.flow_cfg.vision_dim} "
                f"与 LAM 输出维度 {lam_vision_dim} 不一致，自动对齐。"
            )
        self.model_cfg.flow_cfg.vision_dim = lam_vision_dim
        if int(self.model_cfg.flow_cfg.num_vision_tokens) != lam_num_tokens:
            print(
                f"[LatentWorldVLA] flow_cfg.num_vision_tokens={self.model_cfg.flow_cfg.num_vision_tokens} "
                f"与 LAM token 数 {lam_num_tokens} 不一致，自动对齐。"
            )
            self.model_cfg.flow_cfg.num_vision_tokens = lam_num_tokens

        # 3) Flow Matching 头（可由 model_cfg.flow_cfg 配置）
        self.flow = ConditionalFlowMatchingHead(config=self.model_cfg.flow_cfg)

    @classmethod
    def from_config(
        cls,
        cfg: LatentWorldVLAConfig,
        *,
        preserve_checkpoint_model_id: bool = True,
    ) -> Tuple["LatentWorldVLA", Any]:
        """
        装配模型与 Processor，返回 (model, processor)。
        
        Args:
            cfg: LatentWorldVLAConfig 配置对象（包含所有路径配置，包括可选的 yaml_path）
            preserve_checkpoint_model_id: 若为 True，则当 cfg.model_id 指向一个本地目录时，
                       不使用 YAML 中的 model_id 覆盖它，以避免把 checkpoint 路径改回 base 模型路径。
        
        完整加载流程：
        1. 从 yaml 覆盖配置（如果 cfg.yaml_path 提供）
        2. 加载 VLM（通过 LatentVLAModel.from_config）
        3. 加载 LAM（通过 load_latent_action_model）
        4. 初始化 Flow Matching 头
        5. 尝试从 model_id 目录加载 Flow 权重（如果存在 flow.pt）
           - 场景1：model_id 是 LatentVLAModel checkpoint（无 flow.pt）→ 用于开始训练
           - 场景2：model_id 是 LatentWorldVLA checkpoint（有 flow.pt）→ 用于继续训练/推理
        
        注意：数据集统计信息（dataset_statistics）由 LatentVLAProcessor 负责加载和使用
        """
        import os
        import json
        import yaml
        from pathlib import Path
        
        # --- Optional: override cfg from training YAML ---
        if cfg.yaml_path is not None:
            yaml_path = Path(cfg.yaml_path)
            try:
                print(f"[LatentWorldVLA] Loading config from yaml: {yaml_path}")
                with open(str(yaml_path), "r", encoding="utf-8") as f:
                    ycfg = yaml.safe_load(f) or {}
                
                # Support either flat yaml or a nested {"train": {...}} / {"config": {...}} style
                if isinstance(ycfg, dict):
                    for nest_key in ("train", "config", "cfg"):
                        if nest_key in ycfg and isinstance(ycfg[nest_key], dict):
                            ycfg = ycfg[nest_key]
                            break
                
                # Preserve checkpoint dir model_id if requested
                keep_model_id = False
                try:
                    model_id_val = str(cfg.model_id or "")
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
                    # Heuristic: some dirs are Paths but may be None in cfg
                    if cur is None and key in ("hf_cache_dir", "data_root_dir", "run_root_dir", "yaml_path") and isinstance(val, str):
                        return Path(val) if val else None
                    return val
                
                if isinstance(ycfg, dict):
                    for k, v in ycfg.items():
                        try:
                            cur = getattr(cfg, k, None)
                            setattr(cfg, k, _cast_like(str(k), cur, v))
                        except Exception:
                            # Best-effort: do not break loading if a key cannot be applied
                            pass
                    print(f"[LatentWorldVLA] Applied yaml overrides from `{yaml_path}` (preserve_model_id={keep_model_id})")
            except Exception as e:
                print(f"[LatentWorldVLA] Yaml override skipped/failed: {e}")
        
        # 实例化模型（内部完成 VLM + LAM + Flow 初始化）
        model = cls(cfg)
        
        # 尝试从 model_id 目录加载 Flow 权重（可选）
        try:
            flow_path = Path(cfg.model_id) / "flow.pt"
            if flow_path.exists():
                print(f"[LatentWorldVLA] Loading flow weights from: {flow_path}")
                state_dict = torch.load(str(flow_path), map_location='cpu')
                model.flow.load_state_dict(state_dict, strict=True)
                print(f"[LatentWorldVLA] Flow weights loaded successfully")
            else:
                print(f"[LatentWorldVLA] No flow.pt found in {cfg.model_id}, using randomly initialized Flow head")
        except Exception as e:
            print(f"[LatentWorldVLA] Flow weights loading skipped or failed: {e}")
            print(f"[LatentWorldVLA] Using randomly initialized Flow head (expected when starting training)")
        
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
        """
        从保存的 checkpoint 目录加载完整的 LatentWorldVLA 模型。
        
        Args:
            checkpoint_dir: VLM checkpoint 目录（flow.pt 如果存在也在此目录中）
            lam_ckpt_path: LAM checkpoint 路径
            lam_yaml_path: LAM 配置文件路径
            device: 模型加载到的设备（默认自动选择）
            dtype: 模型精度（默认 torch.float32）
            
        Returns:
            Tuple[LatentWorldVLA, processor]: 加载完成的模型与 processor
            
        注意：
        - Flow 权重会自动从 checkpoint_dir/flow.pt 加载（如果存在）
        - 支持两种场景：
          * LatentVLAModel checkpoint（无 flow.pt）
          * LatentWorldVLA checkpoint（有 flow.pt）
        - 数据集统计信息（dataset_statistics）应由 LatentVLAProcessor 单独加载
            
        示例:
            >>> vla, processor = LatentWorldVLA.from_pretrained(
            ...     checkpoint_dir="/path/to/checkpoint",
            ...     lam_ckpt_path="/path/to/lam.ckpt",
            ...     lam_yaml_path="/path/to/lam.yaml",
            ... )
        """
        from pathlib import Path
        
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # 构造配置
        model_cfg = LatentWorldVLAConfig(
            model_id=str(checkpoint_dir),
            lam_ckpt_path=lam_ckpt_path,
            lam_yaml_path=lam_yaml_path,
        )
        
        # 使用 from_config 加载（内部会自动尝试加载 flow.pt）
        print(f"[LatentWorldVLA] Loading from checkpoint: {checkpoint_dir}")
        model, processor = cls.from_config(model_cfg)
        
        # 移到指定设备和精度
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32
        
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # 冻结所有参数（推理模式）
        for p in model.parameters():
            p.requires_grad = False
        
        print(f"[LatentWorldVLA] Model loaded successfully on {device} with dtype {dtype}")
        
        return model, processor

    def train(self, mode: bool = True):
        """
        细粒度训练模式控制：
        - 委托给 LatentVLAModel 处理 VLM + LAM 的训练模式
        - 额外处理 LatentWorldVLA 新增的组件（Flow Matching Head）
        """
        super().train(mode)
        
        # LatentVLAModel 会处理 VLM 和 LAM 的细粒度训练模式控制
        self.latent_vla.train(mode)
        
        # 只需额外处理 Flow Matching Head
        if mode:
            self._set_module_train_mode_by_params(self.flow)
        
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


    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Save a complete LatentWorldVLA checkpoint:
        - Delegates to LatentVLAModel.save_pretrained() to save VLM + latent_vla_extra.pt
          (act_query/vlm_to_lam/optional lam_decoder).
        - Saves FlowMatching head weights as `flow.pt` in the same directory.
        """
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = None
        try:
            if getattr(self, "latent_vla", None) is not None and hasattr(self.latent_vla, "save_pretrained"):
                out = self.latent_vla.save_pretrained(str(save_dir), **kwargs)
        except Exception:
            out = None

        try:
            torch.save(self.flow.state_dict(), save_dir / "flow.pt")
            print(f"[LatentWorldVLA] Flow weights saved to: {save_dir / 'flow.pt'}")
        except Exception as e:
            print(f"[LatentWorldVLA] WARNING: Failed to save flow.pt: {e}")

        return out

    # ------------------
    # Helper: placeholder masks
    # ------------------
    def _infer_act_flow_masks(
        self,
        *,
        input_ids: torch.Tensor,
        act_placeholder_mask: Optional[torch.Tensor],
        flow_placeholder_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        act_q = int(getattr(self.latent_vla, "num_queries", 0))
        flow_query = getattr(self.flow, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldVLA] flow_action_query is None; cannot infer flow queries.")
        flow_q = int(flow_query.shape[0])
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

        placeholder_mask = (input_ids == self.placeholder_token_id).to(dtype=torch.bool, device=device)
        B = int(placeholder_mask.shape[0])
        expected_total = act_q + flow_q
        act_mask = torch.zeros_like(placeholder_mask, dtype=torch.bool)
        flow_mask = torch.zeros_like(placeholder_mask, dtype=torch.bool)
        for b in range(B):
            pos = torch.nonzero(placeholder_mask[b], as_tuple=False).flatten()
            if int(pos.numel()) != int(expected_total):
                raise ValueError(
                    f"[LatentWorldVLA] placeholder count mismatch for sample {b}: "
                    f"got={int(pos.numel())}, expected={int(expected_total)}"
                )
            act_mask[b, pos[:act_q]] = True
            flow_mask[b, pos[act_q:expected_total]] = True

        if act_placeholder_mask is not None:
            act_mask_in = act_placeholder_mask.to(device=device, dtype=torch.bool)
            if not torch.equal(act_mask_in, act_mask):
                raise ValueError("[LatentWorldVLA] act_placeholder_mask mismatch with inferred positions.")
        if flow_placeholder_mask is not None:
            flow_mask_in = flow_placeholder_mask.to(device=device, dtype=torch.bool)
            if not torch.equal(flow_mask_in, flow_mask):
                raise ValueError("[LatentWorldVLA] flow_placeholder_mask mismatch with inferred positions.")
        return act_mask, flow_mask

    def _apply_flow_only_grad_to_h_vlm(
        self,
        *,
        h_vlm: torch.Tensor,
        flow_placeholder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Keep full VLM context values for flow, but restrict gradient to flow placeholder positions.
        """
        if not bool(getattr(self.model_cfg, "flow_only_mode", False)):
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


    # ------------------
    # 前向接口
    # ------------------
    def forward(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        attention_mask: torch.Tensor,    # [B, L]
        act_placeholder_mask: torch.Tensor,  # [B, L], True at <ACT_PH>
        flow_placeholder_mask: Optional[torch.Tensor] = None,  # [B, L], True at flow <ACT_PH>
        lam_videos: torch.Tensor,        # [B, T, 3, H, W] 供 LAM 编码
        lam_states: torch.Tensor,        # [B, T, Dq]
        actions: torch.Tensor,           # [B, T, Da]
        proprio: torch.Tensor,           # [B, Dq]
        image_grid_thw: Optional[torch.Tensor] = None,  # [B*num_imgs, 3] 可选
        wrist_videos: Optional[torch.Tensor] = None,  # [B, 1, 3, H, W] 腕部视角（可选）
        labels: Optional[torch.Tensor] = None,  # 占位，不使用
    ) -> Dict[str, torch.Tensor]:
        # 1) LAM 视觉编码：获取 h_t / h_t1_gt（不经过离散化）
        with torch.no_grad():
            features = _extract_lam_vision_features(self.lam, lam_videos)
        if features is None:
            raise ValueError("[LatentWorldVLA] lam visual feature extraction returned None; check LAM config.")
        h_t = features[:, 0, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)
        h_t1_gt = features[:, -1, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)

        # 2) VLM 前向：注入 act_query + flow_query（两组占位符）
        act_placeholder_mask, flow_placeholder_mask = self._infer_act_flow_masks(
            input_ids=input_ids,
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
        )
        flow_query = getattr(self.flow, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldVLA] flow_action_query is None; check ConditionalFlowMatchingHead.")
        vlm_out_dict = self.latent_vla.forward_vlm_queries_supervise_latent_with_flow(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
            flow_query=flow_query,
        )
        h_vlm = vlm_out_dict["h_vlm"]
        h_vlm_for_flow = self._apply_flow_only_grad_to_h_vlm(
            h_vlm=h_vlm,
            flow_placeholder_mask=flow_placeholder_mask,
        )
        pred_action_emb = vlm_out_dict["pred_latent"]
        loss_distill = torch.tensor(0.0, device=pixel_values.device, dtype=pred_action_emb.dtype)
        if bool(self.model_cfg.enable_loss_distill):
            loss_distill = _compute_distill_loss_via_latent_vla(
                latent_vla=self.latent_vla,
                pred_latent=pred_action_emb,
                lam_videos=lam_videos,
                lam_states=lam_states,
            )
        if self.model_cfg.future_prediction:
            # 3) 预测动作 embedding 并通过 LAM decoder 想象未来特征
            h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
            if isinstance(h_t1_pred, tuple):
                h_t1_pred = h_t1_pred[0]
            # LAM decoder 可能返回 [B, 1, K, D]（或 [B, T, K, D]）；Flow 需要 [B, K, D]
            if h_t1_pred.dim() == 4:
                # 常见情况：预测单帧 -> squeeze 时间维
                if h_t1_pred.shape[1] == 1:
                    h_t1_pred = h_t1_pred[:, 0, :, :]
                else:
                    # 多帧时取最后一帧作为未来表征
                    h_t1_pred = h_t1_pred[:, -1, :, :]
            # 对齐 dtype 以避免 MSE/梯度数值问题（h_t1_gt 已按 pixel_values.dtype）
            h_t1_pred = h_t1_pred.to(dtype=h_t1_gt.dtype)
            
            # 5) 感知损失：预测未来特征 vs 真实未来特征
            loss_perceptual = F.mse_loss(h_t1_pred, h_t1_gt)
        else:
            h_t1_pred = h_t
            loss_perceptual = torch.tensor(0.0, device=pixel_values.device, dtype=h_t1_gt.dtype)
        
        # 6) 腕部视角编码（如果启用）：仅提取当前帧特征 h_t_w
        if self.model_cfg.enable_wrist_view and wrist_videos is not None:
            with torch.no_grad():
                # 只取当前帧（第 0 帧）以减少计算
                features_w = _extract_lam_vision_features(self.lam, wrist_videos)
            if features_w is None:
                raise ValueError("[LatentWorldVLA] wrist visual feature extraction returned None.")
            h_t_w = features_w[:, 0, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)
            # 拼接到 h_t（primary tokens + wrist tokens）
            h_t = torch.cat([h_t_w, h_t], dim=1)  # [B, 256+256, D] -> [B, 512, D]

        with torch.autocast("cuda", dtype=torch.float32):
        # 4) Flow Matching（未来特征/当前特征在 flow 内部做 CFG/drop）
            # 参照 CogACT/M1 的加速技巧：重复每个样本多次以加速训练
            repeated_diffusion_steps = self.model_cfg.repeated_diffusion_steps  # 默认重复 4 次
            
            # 重复所有输入张量：[B, ...] -> [repeated_diffusion_steps*B, ...]
            # 使用通用的重复方式，第一维重复 repeated_diffusion_steps 次，其余维度保持不变
            h_t_repeated = h_t.repeat(repeated_diffusion_steps, *([1] * (h_t.ndim - 1)))
            h_t1_pred_repeated = h_t1_pred.repeat(repeated_diffusion_steps, *([1] * (h_t1_pred.ndim - 1)))
            h_vlm_repeated = h_vlm_for_flow.repeat(repeated_diffusion_steps, *([1] * (h_vlm_for_flow.ndim - 1)))
            proprio_repeated = proprio.repeat(repeated_diffusion_steps, *([1] * (proprio.ndim - 1)))
            actions_repeated = actions.repeat(repeated_diffusion_steps, *([1] * (actions.ndim - 1)))
            attention_mask_repeated = attention_mask.repeat(repeated_diffusion_steps, *([1] * (attention_mask.ndim - 1)))
            
            loss_flow = self.flow(
                h_t=h_t_repeated,
                h_t1_star=h_t1_pred_repeated.detach().clone(),
                h_vlm=h_vlm_repeated,
                proprio=proprio_repeated,
                actions=actions_repeated,
                attention_mask=attention_mask_repeated==1,
            )

        loss_total = (
            loss_flow
            + self.model_cfg.perceptual_weight * loss_perceptual
            + self.model_cfg.lam_encoder_distill_weight * loss_distill
        )

        # 无 CE 损失，占位返回 0
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
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用) 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lam_videos: Optional[torch.Tensor] = None,
        lam_states: Optional[torch.Tensor] = None,
        act_placeholder_mask: Optional[torch.Tensor] = None,
        flow_placeholder_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        wrist_videos: Optional[torch.Tensor] = None,  # 腕部视角（可选）
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        debug: bool = False,  # 新增：诊断模式
        return_intermediates: bool = False,  # 新增：是否返回中间特征（h_t, h_t1_pred）
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        推理路径：从当前观察预测动作序列。
        
        Args:
            pixel_values: [B, 3, 448, 448] VLM 图像输入
            input_ids: [B, L] 文本指令 token IDs
            attention_mask: [B, L] 可选的注意力掩码
            lam_videos: [B, T, 3, H, W] LAM 视频输入（支持单帧 T=1 或视频序列）
            lam_states: [B, T, Dq] LAM 状态输入（当前未使用，保留接口兼容性）
            act_placeholder_mask: [B, L] 标记 latent <ACT_PH> 位置的布尔掩码
            flow_placeholder_mask: [B, L] 标记 flow <ACT_PH> 位置的布尔掩码
            proprio: [B, Dq] 本体感受信息
            image_grid_thw: [B*num_imgs, 3] 可选的图像网格参数
            guidance_scale: CFG 引导强度（None 时使用配置默认值）
            num_inference_steps: Flow Matching 采样步数（None 时使用配置默认值）
            return_intermediates: 是否返回中间特征（默认 False）
            
        Returns:
            如果 return_intermediates=False:
                actions: [B, window_size, action_dim] 预测的动作序列
            如果 return_intermediates=True:
                (actions, intermediates): 其中 intermediates 包含:
                    - h_t: [B, K, D] 当前帧 DINO 特征 tokens
                    - h_t1_pred: [B, K, D] 预测未来特征 tokens
                    - vision_tokens_hw: (H, W) 特征网格尺寸
        """
        # ===== 1. 输入验证与预处理 =====
        device = pixel_values.device
        dtype = pixel_values.dtype
        # 从 input_ids 获取批次大小，而不是 pixel_values（pixel_values 可能已被 flatten）
        batch_size = input_ids.shape[0]
        
        # 处理 lam_videos：支持单帧输入自动扩展为 [B, 1, 3, H, W]
        if lam_videos is None:
            raise ValueError("predict_action 需要 lam_videos 输入以提取视觉特征。")
        
        if lam_videos.dim() == 4:  # [B, 3, H, W] -> [B, 1, 3, H, W]
            lam_videos = lam_videos.unsqueeze(1)
        
        if lam_videos.shape[0] != batch_size:
            raise ValueError(
                f"lam_videos batch size mismatch: {lam_videos.shape[0]} vs pixel_values {batch_size}"
            )
        
        # 处理 act_placeholder_mask：如果未提供，从 input_ids 中自动检测
        if act_placeholder_mask is not None:
            act_placeholder_mask = act_placeholder_mask.to(dtype=torch.bool, device=device)
        if flow_placeholder_mask is not None:
            flow_placeholder_mask = flow_placeholder_mask.to(dtype=torch.bool, device=device)
        
        # 处理 attention_mask：如果未提供，默认全部有效
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # 处理 proprio：如果未提供，使用零向量
        if proprio is None:
            # 从 flow 配置中获取 proprio_dim（默认为 8）
            proprio_dim = getattr(self.flow.config, "proprio_dim", 8)
            proprio = torch.zeros(batch_size, proprio_dim, dtype=dtype, device=device)
        
        # 处理推理超参（仅在未显式传入时使用配置默认值）
        if guidance_scale is None:
            guidance_scale = self.model_cfg.cfg_guidance_scale
        if num_inference_steps is None:
            num_inference_steps = self.model_cfg.num_inference_steps
        
        # ===== 2. 提取当前帧视觉特征 h_t =====
        features = _extract_lam_vision_features(self.lam, lam_videos)
        if features is None:
            raise ValueError("[predict_action] lam visual feature extraction returned None; check LAM config.")
        h_t_original = features[:, 0, :, :].to(device=device, dtype=dtype)  # [B, num_vision_tokens, vision_dim]
        h_t = h_t_original  # 保留原始 h_t 用于 intermediates 返回
        
        # ===== 3. VLM 前向推理：注入 act_query + flow_query =====
        act_placeholder_mask, flow_placeholder_mask = self._infer_act_flow_masks(
            input_ids=input_ids,
            act_placeholder_mask=act_placeholder_mask,
            flow_placeholder_mask=flow_placeholder_mask,
        )
        flow_query = getattr(self.flow, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldVLA] flow_action_query is None; check ConditionalFlowMatchingHead.")
        vlm_out_dict = self.latent_vla.forward_vlm_queries_supervise_latent_with_flow(
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
            # ===== 4. 预测未来视觉特征 h_t1_pred =====
            h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
            
            # 处理 decoder 返回的维度（可能是 tuple 或多维张量）
            if isinstance(h_t1_pred, tuple):
                h_t1_pred = h_t1_pred[0]
            
            # LAM decoder 可能返回 [B, 1, K, D] 或 [B, T, K, D]；Flow 需要 [B, K, D]
            if h_t1_pred.dim() == 4:
                if h_t1_pred.shape[1] == 1:
                    # 单帧预测：squeeze 时间维
                    h_t1_pred = h_t1_pred[:, 0, :, :]
                else:
                    # 多帧预测：取最后一帧作为未来表征
                    h_t1_pred = h_t1_pred[:, -1, :, :]
            
            # 对齐 dtype
            h_t1_pred = h_t1_pred.to(dtype=dtype)
        else:
            h_t1_pred = h_t
        
        # 5) 腕部视角编码（如果启用）：仅提取当前帧特征 h_t_w
        if self.model_cfg.enable_wrist_view and wrist_videos is not None:
            # 只取当前帧（第 0 帧）以减少计算
            features_w = _extract_lam_vision_features(self.lam, wrist_videos)
            if features_w is None:
                raise ValueError("[predict_action] wrist visual feature extraction returned None.")
            h_t_w = features_w[:, 0, :, :].to(device=device, dtype=dtype)
            # 拼接到 h_t（primary tokens + wrist tokens）
            h_t = torch.cat([h_t_w, h_t], dim=1)  # [B, 256+256, D] -> [B, 512, D]
            # 注意：h_t_original 保持为原始单视角特征（用于 intermediates）

        with torch.autocast("cuda", dtype=torch.float32):
            # ===== 5. Flow Matching 采样动作序列 =====
            actions = self.flow.sample_actions_cfg(
                h_t=h_t,
                h_t1_star=h_t1_pred,
                h_vlm=h_vlm,
                proprio=proprio,
                cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                attention_mask=attention_mask==1,
            )
        
        
        # 动作应处于训练时的归一化区间 [-1, 1]，避免采样漂移导致的异常幅度
        # actions = torch.clamp(actions, min=-1.0, max=1.0)
        
        # ===== 6. 返回结果 =====
        if not return_intermediates:
            return actions  # [B, window_size, action_dim]
        
        # 返回中间特征：h_t_original（不含腕部拼接）和 h_t1_pred
        # 计算特征网格尺寸：K = H*W，通常 K=256 -> H=W=16
        num_tokens = h_t_original.shape[1]
        h_w = int(num_tokens ** 0.5)
        if h_w * h_w != num_tokens:
            # 如果不是完全平方数，尝试从 LAM 配置读取
            print(f"[WARNING] num_tokens={num_tokens} 不是完全平方数，尝试使用默认 16x16")
            h_w = 16
        
        intermediates = {
            "h_t": h_t_original.detach().cpu(),  # [B, K, D]
            "h_t1_pred": h_t1_pred.detach().cpu(),  # [B, K, D]
            "vision_tokens_hw": (h_w, h_w),
        }
        
        return actions, intermediates


# ============================================================================
# SimpleLatentWorldVLA: 简化版本，仅使用 VLM + SimpleActionHead
# ============================================================================

class SimpleLatentWorldVLA(nn.Module):
    """
    简化的世界模型 VLA：
    - 仅使用 VLM 输出的 h_vlm
    - 通过 SimpleActionHead (QFormer + MLP) 直接预测动作序列
    - 移除 LAM 视觉编码、Flow Matching 等复杂组件
    - 用于快速验证 libero 管线和作为简单 baseline
    """

    def __init__(
        self,
        model_cfg: LatentWorldVLAConfig,
    ) -> None:
        super().__init__()

        self.model_cfg = model_cfg

        # 1) 加载完整的 LatentVLAModel（包含 VLM、LAM、act_query、vlm_to_lam、tokenizer 配置）
        self.latent_vla, self.processor = LatentVLAModel.from_config(self.model_cfg)
        self.vlm = self.latent_vla.vlm
        self.tokenizer = self.latent_vla.tokenizer
        self.lam = self.latent_vla.lam  # 保留但不使用（为了接口兼容）

        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_size = int(getattr(self.lam.encoder, "grid_size", 0) or 0)
        lam_num_tokens = int(lam_grid_size * lam_grid_size) if lam_grid_size > 0 else 256

        # 从 model_cfg 构造 SimpleActionConfig（自动对齐到当前 LAM 视觉维度）
        simple_action_cfg = SimpleActionConfig(
            action_dim=model_cfg.simple_action_dim,
            window_size=model_cfg.simple_window_size,
            vlm_dim=model_cfg.simple_vlm_dim,
            vision_dim=lam_vision_dim,
            num_vision_tokens=lam_num_tokens,
            hidden_dim=model_cfg.simple_hidden_dim,
            qformer_layers=model_cfg.simple_qformer_layers,
            qformer_heads=model_cfg.simple_qformer_heads,
            dropout=model_cfg.simple_dropout,
        )

        # 1.5) 应用细粒度冻结策略
        from prismatic.models import freeze_qwen3vl, _unfreeze_last_n_llm_layers, _resolve_llm_module
        freeze_qwen3vl(
            self.vlm,
            freeze_vision_backbone=self.model_cfg.freeze_vision_backbone,
            freeze_llm_backbone=self.model_cfg.freeze_llm_backbone,
            freeze_last_llm_layer=self.model_cfg.freeze_last_llm_layer,
            freeze_embedding=self.model_cfg.freeze_embedding,
            unfreeze_vision_merger=self.model_cfg.unfreeze_vision_merger,
        )

        # 解冻最后 N 层（如果配置）
        if self.model_cfg.freeze_llm_backbone and self.model_cfg.unfreeze_llm_last_n_layers:
            llm_module = _resolve_llm_module(self.vlm)
            if llm_module:
                _unfreeze_last_n_llm_layers(llm_module, self.model_cfg.unfreeze_llm_last_n_layers)

        # 2) 获取 placeholder token ID
        self.placeholder_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.model_cfg.latent_action_placeholder_token)
        )

        # 3) Simple Action Head（替代 Flow Matching）
        self.simple_action_head = SimpleActionHead(simple_action_cfg)

    @classmethod
    def from_config(
        cls,
        cfg: LatentWorldVLAConfig,
        *,
        preserve_checkpoint_model_id: bool = True,
    ) -> Tuple["SimpleLatentWorldVLA", Any]:
        """
        装配模型与 Processor，返回 (model, processor)。
        
        Args:
            cfg: LatentWorldVLAConfig 配置对象
            simple_action_cfg: SimpleActionConfig 配置对象（可选）
            preserve_checkpoint_model_id: 若为 True，则保留 checkpoint 的 model_id
        
        完整加载流程：
        1. 从 yaml 覆盖配置（如果 cfg.yaml_path 提供）
        2. 加载 VLM（通过 LatentVLAModel.from_config）
        3. 初始化 SimpleActionHead
        4. 尝试从 model_id 目录加载 SimpleActionHead 权重（如果存在 simple_action_head.pt）
        """
        import os
        import yaml
        from pathlib import Path
        
        # --- Optional: override cfg from training YAML ---
        if cfg.yaml_path is not None:
            yaml_path = Path(cfg.yaml_path)
            try:
                print(f"[SimpleLatentWorldVLA] Loading config from yaml: {yaml_path}")
                with open(str(yaml_path), "r", encoding="utf-8") as f:
                    ycfg = yaml.safe_load(f) or {}
                
                # Support either flat yaml or a nested {"train": {...}} / {"config": {...}} style
                if isinstance(ycfg, dict):
                    for nest_key in ("train", "config", "cfg"):
                        if nest_key in ycfg and isinstance(ycfg[nest_key], dict):
                            ycfg = ycfg[nest_key]
                            break
                
                # Preserve checkpoint dir model_id if requested
                keep_model_id = False
                try:
                    model_id_val = str(cfg.model_id or "")
                    if preserve_checkpoint_model_id and model_id_val and os.path.isdir(model_id_val):
                        keep_model_id = True
                except Exception:
                    keep_model_id = False
                
                def _cast_like(key: str, cur, val):
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
                    if cur is None and key in ("hf_cache_dir", "data_root_dir", "run_root_dir", "yaml_path") and isinstance(val, str):
                        return Path(val) if val else None
                    return val
                
                if isinstance(ycfg, dict):
                    for k, v in ycfg.items():
                        try:
                            cur = getattr(cfg, k, None)
                            setattr(cfg, k, _cast_like(str(k), cur, v))
                        except Exception:
                            pass
                    print(f"[SimpleLatentWorldVLA] Applied yaml overrides from `{yaml_path}` (preserve_model_id={keep_model_id})")
            except Exception as e:
                print(f"[SimpleLatentWorldVLA] Yaml override skipped/failed: {e}")
        
        # 实例化模型（内部完成 VLM + SimpleActionHead 初始化）
        model = cls(cfg)
        
        # 尝试从 model_id 目录加载 SimpleActionHead 权重（可选）
        try:
            action_head_path = Path(cfg.model_id) / "simple_action_head.pt"
            if action_head_path.exists():
                print(f"[SimpleLatentWorldVLA] Loading simple_action_head weights from: {action_head_path}")
                state_dict = torch.load(str(action_head_path), map_location='cpu')
                model.simple_action_head.load_state_dict(state_dict, strict=True)
                print(f"[SimpleLatentWorldVLA] SimpleActionHead weights loaded successfully")
            else:
                print(f"[SimpleLatentWorldVLA] No simple_action_head.pt found in {cfg.model_id}, using randomly initialized head")
        except Exception as e:
            print(f"[SimpleLatentWorldVLA] SimpleActionHead weights loading skipped or failed: {e}")
            print(f"[SimpleLatentWorldVLA] Using randomly initialized SimpleActionHead (expected when starting training)")
        
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
    ) -> Tuple["SimpleLatentWorldVLA", Any]:
        """
        从保存的 checkpoint 目录加载完整的 SimpleLatentWorldVLA 模型。
        
        Args:
            checkpoint_dir: VLM checkpoint 目录（simple_action_head.pt 如果存在也在此目录中）
            lam_ckpt_path: LAM checkpoint 路径（保留接口兼容性，实际不使用）
            lam_yaml_path: LAM 配置文件路径（保留接口兼容性，实际不使用）
            device: 模型加载到的设备（默认自动选择）
            dtype: 模型精度（默认 torch.float32）
            
        Returns:
            Tuple[SimpleLatentWorldVLA, processor]: 加载完成的模型与 processor
            
        示例:
            >>> vla, processor = SimpleLatentWorldVLA.from_pretrained(
            ...     checkpoint_dir="/path/to/checkpoint",
            ...     lam_ckpt_path="/path/to/lam.ckpt",
            ...     lam_yaml_path="/path/to/lam.yaml",
            ... )
        """
        from pathlib import Path
        
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # 构造配置
        model_cfg = LatentWorldVLAConfig(
            model_id=str(checkpoint_dir),
            lam_ckpt_path=lam_ckpt_path,
            lam_yaml_path=lam_yaml_path,
        )
        
        # 使用 from_config 加载（内部会自动尝试加载 simple_action_head.pt）
        print(f"[SimpleLatentWorldVLA] Loading from checkpoint: {checkpoint_dir}")
        model, processor = cls.from_config(model_cfg)
        
        # 移到指定设备和精度
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32
        
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # 冻结所有参数（推理模式）
        for p in model.parameters():
            p.requires_grad = False
        
        print(f"[SimpleLatentWorldVLA] Model loaded successfully on {device} with dtype {dtype}")
        
        return model, processor

    def train(self, mode: bool = True):
        """
        细粒度训练模式控制：
        - 委托给 LatentVLAModel 处理 VLM + LAM 的训练模式
        - 额外处理 SimpleLatentWorldVLA 新增的组件（SimpleActionHead）
        """
        super().train(mode)
        
        # LatentVLAModel 会处理 VLM 和 LAM 的细粒度训练模式控制
        self.latent_vla.train(mode)
        
        # 只需额外处理 SimpleActionHead
        if mode:
            self._set_module_train_mode_by_params(self.simple_action_head)
        
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

    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Save a complete SimpleLatentWorldVLA checkpoint:
        - Delegates to LatentVLAModel.save_pretrained() to save VLM + latent_vla_extra.pt
        - Saves SimpleActionHead weights as `simple_action_head.pt` in the same directory.
        """
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = None
        try:
            if getattr(self, "latent_vla", None) is not None and hasattr(self.latent_vla, "save_pretrained"):
                out = self.latent_vla.save_pretrained(str(save_dir), **kwargs)
        except Exception:
            out = None

        try:
            torch.save(self.simple_action_head.state_dict(), save_dir / "simple_action_head.pt")
        except Exception:
            pass

        return out

    # ------------------
    # 前向接口
    # ------------------
    def forward(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        attention_mask: torch.Tensor,    # [B, L]
        act_placeholder_mask: torch.Tensor,  # [B, L], True at <ACT_PH>
        actions: torch.Tensor,           # [B, T, Da]
        lam_videos: torch.Tensor,        # [B, T, 3, H, W] 供 LAM.extract_vision_features
        # 以下参数保留接口兼容性
        lam_states: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        wrist_videos: Optional[torch.Tensor] = None,  # [B, 1, 3, H, W] 腕部视角（可选，保留接口兼容性）
    ) -> Dict[str, torch.Tensor]:
        """
        简化的训练前向路径：使用 VLM 输出的 h_vlm + LAM 预测的未来特征预测动作
        
        Args:
            pixel_values: [B, 3, 448, 448] VLM 图像输入
            input_ids: [B, L] 文本指令 token IDs
            attention_mask: [B, L] 注意力掩码
            act_placeholder_mask: [B, L] 标记 <ACT_PH> 位置
            actions: [B, T, Da] Ground truth actions
            lam_videos: [B, T, 3, H, W] LAM 视频输入
            
        Returns:
            Dict with loss_total, loss_action, loss_perceptual, and loss_distill
        """
        # 1) LAM 视觉编码：获取 h_t / h_t1_gt
        with torch.no_grad():
            features = _extract_lam_vision_features(self.lam, lam_videos)
        if features is None:
            raise ValueError("[SimpleLatentWorldVLA] lam visual feature extraction returned None; check LAM config.")
        h_t = features[:, 0, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)
        h_t1_gt = features[:, -1, :, :].to(device=pixel_values.device, dtype=pixel_values.dtype)

        # 2) VLM 前向：获取 h_vlm 和 pred_action_emb
        vlm_out_dict = self.latent_vla.forward_vlm_queries_supervise_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
        )
        h_vlm = vlm_out_dict["h_vlm"]  # [B, vlm_dim]
        pred_action_emb = vlm_out_dict["pred_latent"]  # [B, Q, D_lam]
        loss_distill = torch.tensor(0.0, device=pixel_values.device, dtype=pred_action_emb.dtype)
        if bool(self.model_cfg.enable_loss_distill):
            if lam_states is None:
                raise ValueError(
                    "[SimpleLatentWorldVLA] loss_distill requires `lam_states`, but got None. "
                    "Please provide lam_states in batch/collator when enable_loss_distill=True."
                )
            loss_distill = _compute_distill_loss_via_latent_vla(
                latent_vla=self.latent_vla,
                pred_latent=pred_action_emb,
                lam_videos=lam_videos,
                lam_states=lam_states,
            )

        # 3) 预测动作 embedding 并通过 LAM decoder 想象未来特征
        h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
        if isinstance(h_t1_pred, tuple):
            h_t1_pred = h_t1_pred[0]
        # LAM decoder 可能返回 [B, 1, K, D] 或 [B, T, K, D]；需要 [B, K, D]
        if h_t1_pred.dim() == 4:
            if h_t1_pred.shape[1] == 1:
                h_t1_pred = h_t1_pred[:, 0, :, :]
            else:
                h_t1_pred = h_t1_pred[:, -1, :, :]
        # 对齐 dtype
        h_t1_pred = h_t1_pred.to(dtype=h_t1_gt.dtype)

        # 4) Simple Action Head：拼接 h_t, h_t1_pred, h_vlm 后预测动作
        loss_action = self.simple_action_head(
            h_vlm=h_vlm,
            actions=actions,
            h_t=h_t,
            h_t1_pred=h_t1_pred.detach(),
        )

        # 5) 感知损失：预测未来特征 vs 真实未来特征
        loss_perceptual = F.mse_loss(h_t1_pred, h_t1_gt)
        loss_total = (
            loss_action
            + self.model_cfg.perceptual_weight * loss_perceptual
            + self.model_cfg.lam_encoder_distill_weight * loss_distill
        )

        return {
            "loss_action": loss_action,
            "loss_flow": loss_action,  # 兼容性：训练脚本读取 loss_flow
            "loss_perceptual": loss_perceptual,
            "loss_distill": loss_distill,
            "loss_total": loss_total,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用) 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        act_placeholder_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        lam_videos: Optional[torch.Tensor] = None,  # [B, T, 3, H, W] LAM 视频输入
        # 以下参数保留接口兼容性
        lam_states: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        return_intermediates: bool = False,  # 新增：是否返回中间特征
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        简化的推理路径：从 VLM hidden states 和视觉特征预测动作序列
        
        Args:
            pixel_values: [B, 3, 448, 448] VLM 图像输入
            input_ids: [B, L] 文本指令 token IDs
            attention_mask: [B, L] 可选的注意力掩码
            act_placeholder_mask: [B, L] 标记 <ACT_PH> 位置的布尔掩码
            image_grid_thw: [B*num_imgs, 3] 可选的图像网格参数
            lam_videos: [B, T, 3, H, W] LAM 视频输入（可选，用于提取视觉特征）
            return_intermediates: 是否返回中间特征（默认 False）
            
        Returns:
            如果 return_intermediates=False:
                actions: [B, window_size, action_dim] 预测的动作序列
            如果 return_intermediates=True:
                (actions, intermediates): 包含 h_t, h_t1_pred, vision_tokens_hw
        """
        # 处理输入
        device = pixel_values.device
        dtype = pixel_values.dtype
        batch_size = input_ids.shape[0]
        
        # 处理 act_placeholder_mask
        if act_placeholder_mask is None:
            act_placeholder_mask = (input_ids == self.placeholder_token_id)
        else:
            act_placeholder_mask = act_placeholder_mask.to(dtype=torch.bool, device=device)
        
        # 处理 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # 1) 提取视觉特征（如果提供了 lam_videos）
        h_t = None
        h_t1_pred = None
        h_t_original = None  # 保留原始 h_t 用于 intermediates
        if lam_videos is not None:
            # 处理 lam_videos：支持单帧输入自动扩展为 [B, 1, 3, H, W]
            if lam_videos.dim() == 4:  # [B, 3, H, W] -> [B, 1, 3, H, W]
                lam_videos = lam_videos.unsqueeze(1)
            
            # 提取当前帧视觉特征
            features = _extract_lam_vision_features(self.lam, lam_videos)
            if features is not None:
                h_t = features[:, 0, :, :].to(device=device, dtype=dtype)  # [B, K, D]
                h_t_original = h_t  # 保留副本
        
        # 2) VLM 前向：获取 h_vlm 和 pred_action_emb
        vlm_out_dict = self.latent_vla.forward_vlm_queries_supervise_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            act_placeholder_mask=act_placeholder_mask,
        )
        h_vlm = vlm_out_dict["h_vlm"]  # [B, vlm_dim]
        
        # 3) 预测未来视觉特征（如果 h_t 可用）
        if h_t is not None:
            pred_action_emb = vlm_out_dict["pred_latent"]  # [B, Q, D_lam]
            h_t1_pred = self.lam.decoder(h_t, pred_action_emb)
            
            if isinstance(h_t1_pred, tuple):
                h_t1_pred = h_t1_pred[0]
            
            # 处理维度：[B, 1, K, D] -> [B, K, D]
            if h_t1_pred.dim() == 4:
                if h_t1_pred.shape[1] == 1:
                    h_t1_pred = h_t1_pred[:, 0, :, :]
                else:
                    h_t1_pred = h_t1_pred[:, -1, :, :]
            
            h_t1_pred = h_t1_pred.to(dtype=dtype)
        
        # 4) Simple Action Head：预测动作（使用 h_t, h_t1_pred, h_vlm）
        actions = self.simple_action_head.predict(
            h_vlm=h_vlm,
            h_t=h_t,
            h_t1_pred=h_t1_pred,
        )
        
        if not return_intermediates:
            return actions  # [B, window_size, action_dim]
        
        # 返回中间特征
        if h_t_original is not None:
            num_tokens = h_t_original.shape[1]
            h_w = int(num_tokens ** 0.5)
            if h_w * h_w != num_tokens:
                print(f"[WARNING] num_tokens={num_tokens} 不是完全平方数，使用默认 16x16")
                h_w = 16
            
            intermediates = {
                "h_t": h_t_original.detach().cpu(),
                "h_t1_pred": h_t1_pred.detach().cpu() if h_t1_pred is not None else h_t_original.detach().cpu(),
                "vision_tokens_hw": (h_w, h_w),
            }
        else:
            # 如果没有 lam_videos，返回空字典
            intermediates = {
                "h_t": None,
                "h_t1_pred": None,
                "vision_tokens_hw": (16, 16),
            }
        
        return actions, intermediates
