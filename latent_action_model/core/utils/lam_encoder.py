import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torchvision import transforms

from einops import repeat
from .pos_embs import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, Fixed3DPositionalEncoding, PositionalEncoding
from .dpt_head import DPTHead
from .ac_predictor import VisionTransformerPredictorAC
from .modules import QFormer, QFormer_att


class LAMEncoder(nn.Module):
    def __init__(self, context_dim: int, input_dim: int=1024, ar_query: bool = False, 
                 num_layers: int=4, num_heads: int=16, ffn_expansion_factor=4,
                 dropout: float = 0.0,  num_frames: int=5, num_queries: int=1, grid_size: int=16, patch_size: int = 16, add_state: bool = False, modal_mask: bool = False, code_dim: Optional[int] = None):
        super().__init__()
        self.num_frames = num_frames
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.context_dim = context_dim

        if input_dim != context_dim:
            self.project_in = nn.Linear(input_dim, context_dim)
        else:
            self.project_in = nn.Identity()
        if code_dim != context_dim:
            self.out_proj = nn.Linear(context_dim, code_dim)
        else:
            self.out_proj = nn.Identity()

        self.pos_embed = Fixed3DPositionalEncoding(context_dim, num_frames, grid_size, grid_size)
        if add_state:
            self.pos_state_embed = PositionalEncoding(context_dim)
            self.state_project = nn.Sequential(nn.Linear(8, context_dim//4), nn.GELU(), nn.Linear(context_dim//4, context_dim))
        else:
            self.pos_state_embed = None
            self.state_project = None
        self.add_state = add_state
        # 3) QFormer：从 AC-Predictor 输出的上下文中提取 latent actions
        self.QFormer = QFormer_att(
            query_dim=context_dim,
            context_dim=context_dim,
            num_frames=num_frames,
            num_queries=num_queries,
            grid_size=grid_size,
            add_tokens=1 if add_state else 0,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout=dropout,
            ar_query=ar_query,
            use_mask=modal_mask   #启用mask
        )
        
    def forward(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        输入:
          - features: DINO 隐状态序列（Tensor）
              支持形状 [B, T, K, D]（K=grid_size^2）
          - states:   [B, T, 1, 8] 或 [B, T, 8]
        输出:
          - latents:  [B, num_queries, context_dim]
        """
        # 统一 states 形状到 [B, T, 8]，以匹配 AC predictor 的输入
        if states.dim() == 4 and states.size(-2) == 1 and states.size(-1) == 8:
            states = states.squeeze(-2)
        elif states.dim() == 3 and states.size(-1) == 8:
            pass
        else:
            raise ValueError(f"states 期望为 [B,T,8] 或 [B,T,1,8]，得到: {tuple(states.shape)}")

        B, T = states.shape[0], states.shape[1]
        x_ctx = self.project_in(features)
        # print(x_ctx.shape)
        x_ctx = self.pos_embed(x_ctx)  #[B, T, hw, D]
        # breakpoint()
        if self.add_state:
            states = self.pos_state_embed(self.state_project(states))
        # 约定后置拼接：每帧为 [image..., state]
            x_ctx = torch.cat([x_ctx, states.unsqueeze(-2)], dim=-2) #[B, T, (hw+1), D]
        else:
            x_ctx = x_ctx
        # 3) QFormer: 从上下文中提取 latent actions
        latents = self.QFormer(x_ctx)       # [B, num_queries, context_dim]

        return self.out_proj(latents)

