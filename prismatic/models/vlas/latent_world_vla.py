import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model
from .flowmatching_expert import ConditionalFlowMatchingHead
from prismatic.models.load import load_InternVL, freeze_internvl


class FutureFeatureMode(str, enum.Enum):
    """控制传入Flow Matching的未来表征来源（CFG为固定机制，不再作为模式）"""

    LAM_FROM_VLM = "lam_from_vlm"  # 使用VLM输出的z_a，经LAM decoder得到 h_{t+1}^hat
    LAM_FROM_GT = "lam_from_gt"  # 使用GT的z_a索引，经LAM decoder得到 h_{t+1}^hat
    VJEPA_GT = "vjepa_gt"  # 直接使用VJEPA对 I_{t+1} 的编码 h_{t+1}


HOME_PATH = "/mnt/mnt/public/jlchen"


@dataclass
class LatentWorldVLAConfig:
    """LatentWorldVLA 结构配置。只涉及模型与前向路径，不包含训练/数据集逻辑"""

    # 维度
    vjepa_feat_dim: int = 1024  # 来自 VJEPAEncoder 的通道维度 D_v
    proprio_dim: int = 8
    vlm_hidden_dim: int = 1024  # VLM多模态隐状态的维度（需与实际一致）

    # codebook 与 LAM 超参（用于检查与构造）
    num_queries: int = 4
    # 三源混合概率（总和应为1.0）
    p_lam_from_vlm: float = 0.2
    p_lam_from_gt: float = 0.3
    p_vjepa_gt: float = 0.5

    # 动作token起始ID，用于 VLM 生成token -> 码本索引的映射
    action_token_begin_id: int = 151679

    # CFG 超参：训练掉落概率（在Flow头部内部使用），推理guidance scale
    cfg_drop_prob: float = 0.1
    cfg_guidance_scale: float = 1.5

    # ===== 新增：模型加载相关参数（由原 FinetuneConfig 迁移而来）=====
    # Base VLM & LAM
    model_id: str = HOME_PATH + "/code/UniVLA/vla_scripts/vla_log/1027_122446+nsvq_bridgensvq_bridge/checkpoints/checkpoint-4000"
    lam_ckpt_path: str = HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_bridge_bi_cls/version_1/checkpoints/epoch=17.ckpt"
    lam_yaml_path: str = HOME_PATH + "/code/UniVLA/latent_action_model/logs/dino_bridge_bi_cls/version_1/dino_bridge.yaml"

    # LAM/codebook
    codebook_size: int = 16


    # VLM 精度
    vlm_dtype: torch.dtype = torch.bfloat16


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

        # 1) 加载基础 InternVL 模型与 Processor
        self.vlm, self.processor = load_InternVL(
            self.model_cfg.model_id,
            dtype=self.model_cfg.vlm_dtype,
        )
        # 便捷引用 tokenizer
        self.tokenizer = self.processor.tokenizer

        # 2) 向 tokenizer 注入离散动作 tokens，并根据 tokenizer 计算 action_token_begin_id
        special_tokens_dict = {"additional_special_tokens": [f"<ACT_{i}>" for i in range(self.model_cfg.codebook_size)]}
        try:
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
        except Exception:
            num_added_toks = 0
        if num_added_toks > 0 and hasattr(self.vlm, "resize_token_embeddings"):
            self.vlm.resize_token_embeddings(len(self.tokenizer))

        # 根据 tokenizer 中 ACT_ token 的起始 id 校准 action_token_begin_id（若可用）
        act_tokens = [f"<ACT_{i}>" for i in range(self.model_cfg.codebook_size)]
        act_ids = self.tokenizer.convert_tokens_to_ids(act_tokens)
        if isinstance(act_ids, list) and len(act_ids) > 0 and min(act_ids) != -1:
            self.model_cfg.action_token_begin_id = min(act_ids)

        # 3) 冻结策略：根据配置冻结/解冻 VLM 各部分
        if self.model_cfg.freeze_vlm:
            freeze_internvl(self.vlm, True, True, True, True)
        else:
            freeze_internvl(self.vlm, False, False, False, False)

        # 4) 加载 LAM（含 VJEPA Encoder 与 Decoder），默认 eval 与冻结参数
        lam_model = load_latent_action_model(self.model_cfg.lam_ckpt_path, self.model_cfg.lam_yaml_path)
        self.lam = lam_model.eval()
        # 可学习编码器已迁移至 flowmatching_expert，由 Flow 头内部管理


        # Flow Matching 头（内部自配置）
        self.flow = ConditionalFlowMatchingHead()
        self.code_book_size = self.lam.codebook_size
        self.norm_stats = None

    def set_cfg_drop_prob(self, p: float) -> None:
        self.model_cfg.cfg_drop_prob = float(max(0.0, min(1.0, p)))

    def world_imagine_next(self, h_t: torch.Tensor, code_indices: torch.Tensor) -> torch.Tensor:
        z_q = F.embedding(code_indices, self.lam.vq.codebooks)
        return self.lam.decoder(h_t, z_q).to(h_t.dtype)


    @torch.no_grad()
    def extract_action_idx_and_hidden_states(
        self,
        hidden: torch.Tensor,   # [B, L, D], 模型最后一层 hidden states
        logits: torch.Tensor,   # [B, L, V], 模型输出的 logits
        labels: Optional[torch.Tensor] = None,  # [B, L], 若提供则按 labels 精确定位动作 token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于 labels 精确提取 Q(=num_queries) 个 latent action tokens 对应的 hidden 与预测 index。
        - 若 labels 提供：使用 (labels != -100) 作为掩码定位动作 token，按时间顺序提取。
        - 若 labels 缺失：回退为取序列最后 Q 个位置（仅用于推理/生成场景）。
        返回：
          vlm_hidden: [B, L, D]
          action_idx:    [B, Q] （码本内索引 0..code_book_size-1）
        """

        B, L, D = hidden.shape
        act_begin = self.model_cfg.action_token_begin_id
        act_end = act_begin + self.code_book_size
        Q = int(self.model_cfg.num_queries)

        # 基于 (labels != -100) 的掩码定位动作 token，保证每个样本恰有 Q 个
        valid_mask = (labels != -100)  # [B, L]
        # 将无效位置赋值为 -1，有效位置为其时间索引，随后取 topk 得到每样本 Q 个位置
        time_indices = torch.arange(L, device=labels.device).unsqueeze(0).expand(B, L)
        scores = torch.where(valid_mask, time_indices, torch.full_like(time_indices, -1))  # [B, L]
        positions = torch.topk(scores, k=Q, dim=1).indices  # [B, Q]（按时间索引降序）
        positions, _ = torch.sort(positions, dim=1)  # 升序保证时间顺序

        # 矢量化 gather 取出对应 hidden / logits
        # vlm_hidden = hidden.gather(dim=1, index=positions.unsqueeze(-1).expand(B, Q, D))  # [B, Q, D]
        action_logits_full = logits.gather(dim=1, index=positions.unsqueeze(-1).expand(B, Q, logits.size(-1)))  # [B, Q, V]

        # 限制到 <ACT_*> 子词表并求 argmax，得到码本内索引
        act_ids = torch.arange(act_begin, act_end, device=hidden.device)
        act_logits = action_logits_full.index_select(dim=-1, index=act_ids)  # [B, Q, code_book_size]
        action_idx = torch.argmax(act_logits, dim=-1)  # [B, Q]
        vlm_hidden = hidden[:,261:]         # [B, Q, D]   当前从language emb的第一个token选起
        return vlm_hidden, action_idx
    # ------------------
    # 前向接口
    # ------------------
    def forward(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        labels: torch.Tensor,            # [B, L]
        attention_mask: torch.Tensor,    # [B, L]
        actions: torch.Tensor,           # [B, T, Da]
        latent_action_idx: torch.Tensor, # [B, Q]        (GT)
        proprio: torch.Tensor,           # [B, Dq]
        image_features : torch.Tensor,   # [B, 2, K, D]
    ) -> Dict[str, torch.Tensor]:
    
        h_t, h_t1 = image_features[:, 0, :, :], image_features[:, -1, :, :]
        # 1) 计算 VLM 前向（保留梯度用于 LoRA 训练），但在后续 Flow 头中使用 detach 特征
        # 只要对VLM传入labels，就一定会返回loss键  odict_keys(['loss', 'logits', 'past_key_values', 'hidden_states'])
        out_dict = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )
        hidden, logits = out_dict.hidden_states[-1], out_dict.logits  # [B, L, H], [B, L, V]
        # VLM 路：基于 labels 精确提取动作 token 的 hidden 与索引
        h_vlm, vlm_idx = self.extract_action_idx_and_hidden_states(hidden, logits, labels=labels)
        h_t1_lam_from_vlm = self.world_imagine_next(h_t, vlm_idx)
        # GT-LAM 路
        h_t1_lam_from_gt = self.world_imagine_next(h_t, latent_action_idx)

        # VJEPA-GT 路
        h_t1_vjepa = h_t1

        # 三源采样概率
        h_t1_star = self._ht1_sampling(h_t1_lam_from_vlm, h_t1_lam_from_gt, h_t1_vjepa)  # [B, ...]


        # loss_vlm = nn.CrossEntropyLoss(logits.view(-1, logits.size(-1)), labels.view(-1))
        # 2) Flow Matching（训练期内部执行CFG-drop；噪声/时间由flow内部采样）
        losses = self.flow(
            h_t=h_t.detach(),
            h_t1_star=h_t1_star.detach(),
            h_vlm=h_vlm,
            proprio=proprio,
            actions=actions,
        )

        # 返回 VLM 的 CE 损失（若不可用则置零张量，避免分支判断）
        vlm_loss = getattr(out_dict, "loss", None)
        if vlm_loss is None:
            print("VLM 损失不可用，已置零！！！！！")
            vlm_loss = torch.tensor(0.0, device=pixel_values.device, dtype=hidden.dtype)

        # 计算动作 token 的准确率（对被监督的 label 位置，即 labels != -100）
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            valid_mask = (labels != -100)
            denom = valid_mask.sum()
            if denom.item() > 0:
                correct = ((preds == labels) & valid_mask).sum()
                action_accuracy = (correct.float() / denom.float())
            else:
                action_accuracy = torch.tensor(0.0, device=pixel_values.device, dtype=hidden.dtype)

        return {
            "loss_flow": losses,
            "loss_vlm": vlm_loss,
            "vlm_action_accuracy": action_accuracy,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        pixel_values: torch.Tensor,      # [B, 3, H, W]  (VLM用)  448*448
        input_ids: torch.Tensor,         # [B, L]        (VLM用)
        attention_mask: Optional[torch.Tensor] = None,    # [B, L]
        image_4_jepa: Optional[torch.Tensor] = None,      # [B, 3, H, W]  (VJEPA用) 256*256
        proprio: torch.Tensor,           # [B, Dq]        (Flow用)
        guidance_scale: Optional[float] = 1.5,
        window_size: int = 10,
        image_feat_4_lam: Optional[torch.Tensor] = None,  #直接提供jepa特征，无需image_4_jepa，用于eval
        **kwargs  # 捕获额外参数
    ) -> torch.Tensor:

        # 先前向一次 VLM，提取 hidden/logits，再得到 h_vlm 与索引
        out_dict = self.vlm.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask if attention_mask is not None else None,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
            min_new_tokens=self.model_cfg.num_queries,
            max_new_tokens=self.model_cfg.num_queries,
            do_sample=False
        )

        # 在 codebook 范围内用每步 logits 选取索引
        act_begin = int(self.model_cfg.action_token_begin_id)
        act_end = act_begin + int(self.code_book_size)
        act_ids = torch.arange(act_begin, act_end, device=pixel_values.device)
        # scores 为长度为 Q 的列表，每个元素形状 [B, V]
        latent_action_idx = torch.stack(
            [torch.argmax(s.index_select(-1, act_ids), dim=-1) for s in out_dict.scores],
            dim=1
        )

        # 从每步生成返回的 hidden_states 中取出该步新 token 的最后一层隐状态并堆叠
        last_hidden_per_step = [hs[-1][:, -1, :] for hs in out_dict.hidden_states]  # Q 个 [B, D]
        h_vlm = torch.stack(last_hidden_per_step, dim=1)  # [B, Q, D]
        # 通过 LAM decoder 得到未来表征
        if image_4_jepa is not None:
            # 统一使用 encode 接口，传入 [B, C, H, W]，返回 [B, 1, K, D]
            feats_bt_k_d = self.lam.vision_encoder.encode(image_4_jepa)
            h_t = feats_bt_k_d[:, 0, :, :].to(dtype=h_vlm.dtype, device=pixel_values.device)
        else:
            h_t = image_feat_4_lam[:, 0, :, :].to(dtype=h_vlm.dtype, device=pixel_values.device)
        h_t1_star = self.world_imagine_next(h_t, latent_action_idx)

        scale = self.model_cfg.cfg_guidance_scale if guidance_scale is None else float(guidance_scale)
        actions = self.flow.sample_actions_cfg(h_t=h_t, h_t1_star=h_t1_star, h_vlm=h_vlm, proprio=proprio, cfg_scale=scale,action_horizon=window_size)
        return actions


