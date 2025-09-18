import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import LatentLAMModel
from latent_action_model.core.vq import NSVQ
from prismatic.models.policy.flowmatching_expert import (
    ConditionalFlowMatchingHead,
)


class FutureFeatureMode(str, enum.Enum):
    """控制传入Flow Matching的未来表征来源（CFG为固定机制，不再作为模式）"""

    LAM_FROM_VLM = "lam_from_vlm"  # 使用VLM输出的z_a，经LAM decoder得到 h_{t+1}^hat
    LAM_FROM_GT = "lam_from_gt"  # 使用GT的z_a索引，经LAM decoder得到 h_{t+1}^hat
    VJEPA_GT = "vjepa_gt"  # 直接使用VJEPA对 I_{t+1} 的编码 h_{t+1}







 





@dataclass
class LatentWorldVLAConfig:
    """LatentWorldVLA 结构配置。只涉及模型与前向路径，不包含训练/数据集逻辑"""

    # 维度
    vjepa_feat_dim: int = 1024  # 来自 VJEPAEncoder 的通道维度 D_v
    proprio_dim: int = 8
    vlm_hidden_dim: int = 1024  # VLM多模态隐状态的维度（需与实际一致）

    # codebook 与 LAM 超参（用于检查与构造）
    num_queries: int = 4

    # flow matching 头部参数
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_num_steps: int = 50

    # 三源混合概率（总和应为1.0）
    p_lam_from_vlm: float = 0.34
    p_lam_from_gt: float = 0.33
    p_vjepa_gt: float = 0.33

    # 动作token起始ID，用于 VLM 生成token -> 码本索引的映射
    action_token_begin_id: int = 151679

    # CFG 超参：训练掉落概率（在Flow头部内部使用），推理guidance scale
    cfg_drop_prob: float = 0.1
    cfg_guidance_scale: float = 1.5


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
    条件向量 = Enc(h_t) || Enc(h_{t+1}^*) || Enc(h_vlm) || Enc(q_t)
    送入 ConditionalFlowMatchingHead 预测速度场
    """

    def __init__(
        self,
        lam: LatentLAMModel,
        model_cfg: LatentWorldVLAConfig,
        vlm: nn.Module,
    ) -> None:
        super().__init__()

        self.lam = lam.eval()
        for p in self.lam.parameters():
            p.requires_grad_(False)
        self.vlm = vlm.eval()
        for p in self.vlm.parameters():
            p.requires_grad_(False)

        self.model_cfg = model_cfg

        # 可学习编码器已迁移至 flowmatching_expert，由 Flow 头内部管理


        # Flow Matching 头（内部自配置）
        self.flow = ConditionalFlowMatchingHead()
        self.code_book_size = self.lam.vq.get_codebook_size()


    def set_cfg_drop_prob(self, p: float) -> None:
        self.model_cfg.cfg_drop_prob = float(max(0.0, min(1.0, p)))

    # ------------------
    # 内部功能
    # ------------------
    @torch.no_grad()
    def _encode_vjepa_two_frames(self, image_t: torch.Tensor, image_t1: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 构造两帧视频 [B, T=2, C, H, W]
        if image_t1 is None:
            video = torch.stack([image_t, image_t], dim=1)
        else:
            video = torch.stack([image_t, image_t1], dim=1)
        feats = self.lam.vision_encoder.encode_video_frames(video)  # [B, 2, K, D]
        return feats[:, 0], feats[:, 1]



    def _decode_lam_next(self, h_t: torch.Tensor, code_indices: torch.Tensor) -> torch.Tensor:
        z_q = F.embedding(code_indices, self.lam.vq.codebooks)
        return self.lam.decoder(h_t, z_q)


    @torch.no_grad()
    def extract_action_idx_and_hidden_states(self, hidden: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """直接用一次 forward 获取 hidden_states 与 index，提取两项：
        - h_vlm: 最后Q个token隐状态的均值 [B, Dvlm]
        - vlm_action_idx: 在 <ACT_*> 子词表上的 argmax 索引 [B, Q] (0..K-1)
        """

        action_hidden, action_logits = [], []
        for b in range(hidden.size(0)):
            positions = torch.nonzero(hidden[b] >= self.model_cfg.action_token_begin_id, as_tuple=False).squeeze(1)
            assert positions.numel() == 4, f"Batch {b} action token数量不为 {4}"
            action_hidden.append(hidden[b, positions, :])
            action_logits.append(logits[b, positions, :])
        action_hidden, action_logits = torch.stack(action_logits), torch.stack(action_hidden)

        # 在 <ACT_*> 子词表上取 argmax
        act_ids = torch.arange(
            self.model_cfg.action_token_begin_id,
            self.model_cfg.action_token_begin_id + self.code_book_size,
            device=action_logits.device,
            dtype=torch.long,
        )
        action_logits = action_logits.index_select(dim=-1, index=act_ids)  # [B,Q,K]
        action_idx = torch.argmax(action_logits, dim=-1)  # [B,Q]
        return action_hidden, action_idx

    def _ht1_sampling(self, h_t1_lam_from_vlm: torch.Tensor, h_t1_lam_from_gt: torch.Tensor, h_t1_vjepa: torch.Tensor) -> torch.Tensor:
        B = h_t1_vjepa.shape[0]
        # 三源采样概率
        probs = torch.tensor([
            self.model_cfg.p_lam_from_vlm,
            self.model_cfg.p_lam_from_gt,
            self.model_cfg.p_vjepa_gt,
        ], device=h_t1_vjepa.device, dtype=torch.float32)
        probs = probs / probs.sum()  # 归一化

        # Categorical 分布采样
        modes = torch.distributions.Categorical(probs).sample(torch.Size([B]))  # [B], 值为 0/1/2

        # 将三种来源堆叠
        h_sources = torch.stack([h_t1_lam_from_vlm, h_t1_lam_from_gt, h_t1_vjepa], dim=0)  # [3, B, ...]
        # 矢量化索引选择
        h_t1_star = h_sources[modes, torch.arange(B)]  # [B, ...]
        return h_t1_star

    # ------------------
    # 前向接口
    # ------------------
    def forward(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        actions: torch.Tensor,           # [B, A]
        latent_action_idx: torch.Tensor, # [B, Q]        (GT)
        proprio: torch.Tensor,           # [B, Dq]
        image_features : torch.Tensor,   # [B, 2, K, Dv]
        attention_mask: torch.Tensor,    # [B, L]
        labels: torch.Tensor,            # [B, L]
    ) -> Dict[str, torch.Tensor]:
    
        h_t, h_t1 = image_features[:, 0, :, :], image_features[:, 1, :, :]
        # 1) 无梯度：三路未来特征
        with torch.no_grad():
            out_dict = self.vlm(input_ids=input_ids, pixel_values=pixel_values, output_hidden_states=True,attention_mask=attention_mask,max_new_tokens=4)
            hidden, logits = out_dict.hidden_states[-1], out_dict.logits  # [B, L, H], [B, L, V]
            # VLM 路
            h_vlm, vlm_idx = self.extract_action_idx_and_hidden_states(hidden, logits)
            h_t1_lam_from_vlm = self._decode_lam_next(h_t, vlm_idx)
            # GT-LAM 路
            h_t1_lam_from_gt = self._decode_lam_next(h_t, latent_action_idx)

        # VJEPA-GT 路
        h_t1_vjepa = h_t1

        # 三源采样概率
        h_t1_star = self._ht1_sampling(h_t1_lam_from_vlm, h_t1_lam_from_gt, h_t1_vjepa)  # [B, ...]



        # 2) Flow Matching（训练期内部执行CFG-drop；噪声/时间由flow内部采样）
        losses = self.flow(h_t=h_t.detach(), h_t1_star=h_t1_star.detach(), h_vlm=h_vlm.detach(), proprio=proprio, actions=actions)

        return {
            "loss_flow_per_sample": losses,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        proprio: torch.Tensor,           # [B, Dq]
        image_features: torch.Tensor,    # [B, 2, K, Dv]
        attention_mask: torch.Tensor,    # [B, L]
        labels: torch.Tensor,            # [B, L]
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        # 与 forward 对齐：构造三路未来特征并形成条件向量
        h_t, h_t1 = image_features[:, 0, :, :], image_features[:, 1, :, :]

        with torch.no_grad():
            # 先前向一次 VLM，提取 hidden/logits，再得到 h_vlm 与索引
            out_dict = self.vlm(input_ids=input_ids, pixel_values=pixel_values, output_hidden_states=True, attention_mask=attention_mask, max_new_tokens=4)
            hidden, logits = out_dict.hidden_states[-1], out_dict.logits
            h_vlm, vlm_idx = self.extract_action_idx_and_hidden_states(hidden, logits)
            # 通过 LAM decoder 得到未来表征
            h_t1_lam_from_vlm = self._decode_lam_next(h_t, vlm_idx)
            # 推理期无 GT，可复用 VLM 路径作为占位
            h_t1_lam_from_gt = h_t1_lam_from_vlm
            # VJEPA-GT 路
            h_t1_vjepa = h_t1
            # 三源采样融合
            h_t1_star = self._ht1_sampling(h_t1_lam_from_vlm, h_t1_lam_from_gt, h_t1_vjepa)

        scale = self.model_cfg.cfg_guidance_scale if guidance_scale is None else float(guidance_scale)
        return self.flow.sample_actions_cfg(h_t=h_t.detach(), h_t1_star=h_t1_star.detach(), h_vlm=h_vlm.detach(), proprio=proprio, guidance_scale=scale)


