#!/usr/bin/env python
"""
Simple Action Expert: 直接从 VLM hidden states 预测动作序列
使用 QFormer + MLP 架构，用于快速验证 libero 管线
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


@dataclass
class SimpleActionConfig:
    """Simple Action Expert 配置"""
    # 动作维度与序列长度
    action_dim: int = 7
    window_size: int = 10
    
    # VLM 特征维度
    vlm_dim: int = 2048
    
    # 视觉特征维度（h_t 和 h_t1_pred）
    vision_dim: int = 768  # DINO 特征维度
    num_vision_tokens: int = 256  # 16x16 tokens
    
    # 中间隐层维度
    hidden_dim: int = 768
    
    # QFormer 配置
    qformer_layers: int = 6
    qformer_heads: int = 8
    qformer_mlp_ratio: float = 4.0
    
    # Dropout
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class MAPAttention(nn.Module):
    """Multi-Headed Attention Pooling (from Set Transformers)"""
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Q from queries, KV from context
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, C] 可学习的 query 向量
            context: [B, N, C] VLM 上下文特征
        Returns:
            output: [B, Q, C] 聚合后的特征
        """
        B_q, Q, C_q = queries.shape
        B_c, N, C_c = context.shape
        assert C_q == C_c, "Queries and context must have same embed_dim!"

        # Project queries to Q, context to KV
        q = self.q(queries).reshape(B_q, Q, self.n_heads, C_q // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B_c, N, 2, self.n_heads, C_c // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention
        scores = q @ (k.transpose(-2, -1) * self.scale)
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_q, Q, C_q)

        return self.proj(vals)


class QFormerBlock(nn.Module):
    """Single QFormer Transformer Block with cross-attention"""
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        # Cross-attention (queries attend to context)
        self.cross_attn_norm = RMSNorm(embed_dim)
        self.cross_attn = MAPAttention(embed_dim, n_heads)
        
        # Self-attention (queries attend to queries)
        self.self_attn_norm = RMSNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        self.mlp_norm = RMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, D]
            context: [B, N, D]
        Returns:
            queries: [B, Q, D]
        """
        # Cross-attention: queries attend to context
        queries = queries + self.cross_attn(self.cross_attn_norm(queries), context)
        
        # Self-attention: queries attend to themselves
        normed = self.self_attn_norm(queries)
        attn_out, _ = self.self_attn(normed, normed, normed)
        queries = queries + attn_out
        
        # MLP
        queries = queries + self.mlp(self.mlp_norm(queries))
        
        return queries


class SimpleActionHead(nn.Module):
    """
    简单的 Action Expert：从 VLM hidden states 和视觉特征预测动作序列
    
    架构：[h_t || h_t1_pred || h_vlm] -> Projection -> QFormer -> Action MLP -> actions
    """
    def __init__(self, config: Optional[SimpleActionConfig] = None):
        super().__init__()
        self.config = config or SimpleActionConfig()
        
        # VLM 特征投影到 hidden_dim
        self.vlm_proj = nn.Sequential(
            nn.LayerNorm(self.config.vlm_dim),
            nn.Linear(self.config.vlm_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        
        # 视觉特征投影（h_t 和 h_t1_pred）
        # 输入：[B, K, D_vision]，输出：[B, K, hidden_dim]
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.config.vision_dim),
            nn.Linear(self.config.vision_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        
        # 可学习的 queries（用于从特征中提取动作信息）
        self.queries = nn.Parameter(
            torch.zeros(self.config.window_size, self.config.hidden_dim),
            requires_grad=True
        )
        nn.init.normal_(self.queries, std=0.02)
        
        # QFormer: 从拼接的特征中提取动作表征
        self.qformer_blocks = nn.ModuleList([
            QFormerBlock(
                embed_dim=self.config.hidden_dim,
                n_heads=self.config.qformer_heads,
                mlp_ratio=self.config.qformer_mlp_ratio,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.qformer_layers)
        ])
        
        # Action decoder: 将 QFormer 输出映射到动作空间
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.action_dim),
        )

    def forward(
        self,
        h_vlm: torch.Tensor,
        actions: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        h_t1_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        训练前向路径：预测动作并计算损失
        
        Args:
            h_vlm: [B, vlm_dim] VLM hidden states
            actions: [B, T, action_dim] Ground truth actions
            h_t: [B, K, D_vision] 当前帧视觉特征（可选）
            h_t1_pred: [B, K, D_vision] 预测的未来帧视觉特征（可选）
            
        Returns:
            loss: L1 loss between predicted and ground truth actions
        """
        pred_actions = self._predict_actions(h_vlm, h_t, h_t1_pred)
        
        # 确保序列长度匹配
        if pred_actions.shape[1] != actions.shape[1]:
            # 如果不匹配，截断或填充
            min_len = min(pred_actions.shape[1], actions.shape[1])
            pred_actions = pred_actions[:, :min_len, :]
            actions = actions[:, :min_len, :]
        
        # 计算 L1 loss
        loss = F.l1_loss(pred_actions, actions)
        return loss

    def predict(
        self,
        h_vlm: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        h_t1_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        推理路径：直接预测动作序列
        
        Args:
            h_vlm: [B, vlm_dim] VLM hidden states
            h_t: [B, K, D_vision] 当前帧视觉特征（可选）
            h_t1_pred: [B, K, D_vision] 预测的未来帧视觉特征（可选）
            
        Returns:
            actions: [B, window_size, action_dim] Predicted actions
        """
        return self._predict_actions(h_vlm, h_t, h_t1_pred)

    def _predict_actions(
        self,
        h_vlm: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        h_t1_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        内部方法：从 h_vlm 和视觉特征预测动作序列
        
        Args:
            h_vlm: [B, vlm_dim] or [B, 1, vlm_dim]
            h_t: [B, K, D_vision] 当前帧视觉特征（可选）
            h_t1_pred: [B, K, D_vision] 预测的未来帧视觉特征（可选）
            
        Returns:
            actions: [B, window_size, action_dim]
        """
        # 处理输入维度
        if h_vlm.dim() == 2:
            h_vlm = h_vlm.unsqueeze(1)  # [B, vlm_dim] -> [B, 1, vlm_dim]
        
        batch_size = h_vlm.shape[0]
        
        # 1. 投影 VLM 特征
        vlm_context = self.vlm_proj(h_vlm)  # [B, 1, hidden_dim]
        
        # 2. 拼接所有上下文特征
        context_list = [vlm_context]
        
        if h_t is not None:
            # h_t: [B, K, D_vision] -> [B, K, hidden_dim]
            h_t_proj = self.vision_proj(h_t)
            context_list.append(h_t_proj)
        
        if h_t1_pred is not None:
            # h_t1_pred: [B, K, D_vision] -> [B, K, hidden_dim]
            h_t1_proj = self.vision_proj(h_t1_pred)
            context_list.append(h_t1_proj)
        
        # 拼接所有上下文：[B, 1 + K + K, hidden_dim] 或 [B, 1, hidden_dim]
        context = torch.cat(context_list, dim=1)
        
        # 3. 准备 queries
        queries = repeat(self.queries, "w d -> b w d", b=batch_size)  # [B, W, D]
        
        # 4. QFormer: 从 context 中提取动作信息
        for block in self.qformer_blocks:
            queries = block(queries, context)  # [B, w, D]
        
        # 5. 解码为动作
        actions = self.action_head(queries)  # [B, w, action_dim]
        
        return actions
