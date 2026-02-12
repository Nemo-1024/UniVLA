# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            ),
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = None,
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        # NOTE: eval/加载旧权重时 config 里可能没有 compute_dtype，做兼容
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=compute_dtype)

        all_blocks = []
        for idx in range(self.config.num_layers):
            # 修复：明确区分自注意力和交叉注意力层的初始化
            # 当 interleave_self_attention=True 时：
            # - 奇数层（idx % 2 == 1）：自注意力，cross_attention_dim=None
            # - 偶数层（idx % 2 == 0）：交叉注意力，cross_attention_dim=cross_attention_dim
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            
            # 验证：确保自注意力层的 cross_attention_dim 为 None
            if use_self_attn and curr_cross_attention_dim is not None:
                raise ValueError(
                    f"Layer {idx}: interleave_self_attention=True but cross_attention_dim={curr_cross_attention_dim} "
                    f"(expected None for self-attention layers)"
                )

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Sequential(
                                nn.Linear(self.inner_dim, self.config.output_dim*4),
                                nn.ReLU(),
                                nn.Linear(self.config.output_dim*4, self.config.output_dim)
                    )
        # print(
        #     "Total number of DiT parameters: ",
        #     sum(p.numel() for p in self.parameters() if p.requires_grad),
        # )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        # NOTE: UniVLA 的 VLM hidden states 包含 padding，需要使用 encoder_attention_mask
        # 修复：确保 interleave_self_attention 模式下，奇数层的自注意力与初始化时一致
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                # 奇数层：自注意力模式（cross_attention_dim=None 在初始化时已设置）
                # 确保 encoder_hidden_states=None 以触发自注意力
                # 注意：即使 cross_attention_dim=None，仍需要显式传入 None 以确保一致性
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,  # 强制自注意力模式
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 偶数层：交叉注意力模式（cross_attention_dim 在初始化时已设置）
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class AlternateVLDiT(DiT):
    """
    交替视觉-语言 DiT，在交叉注意力时分别关注视觉特征和VLM特征
    参照 NVIDIA Isaac-GR00T 实现
    
    注意：AlternateVLDiT 强制启用 interleave_self_attention，无需在配置中指定
    """
    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        # 强制启用 interleave_self_attention，AlternateVLDiT 必须使用交替自注意力模式
        kwargs['interleave_self_attention'] = True
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,  # padding mask: [B, S], True=有效
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,  # [B, S], True=视觉tokens
        vlm_mask: Optional[torch.Tensor] = None,    # [B, S], True=VLM tokens
    ):
        """
        AlternateVLDiT forward: 交替关注视觉特征和VLM特征
        参照 NVIDIA Isaac-GR00T 实现
        """
        assert image_mask is not None and vlm_mask is not None, \
            "AlternateVLDiT requires image_mask and vlm_mask"
        
        # 编码时间步
        temb = self.timestep_encoder(timestep)
        
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        
        # 预先计算 attention masks（与 NVIDIA 保持一致）
        # encoder_attention_mask 作为 backbone_attention_mask 使用
        if encoder_attention_mask is not None:
            backbone_attention_mask = encoder_attention_mask.bool()
        else:
            # 如果没有提供 padding mask，默认全部有效
            backbone_attention_mask = torch.ones(
                hidden_states.shape[0], encoder_hidden_states.shape[1], 
                dtype=torch.bool, device=hidden_states.device
            )
        
        # 参照 NVIDIA: 预先计算 image 和 non-image (VLM) 的 attention masks
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = vlm_mask & backbone_attention_mask  # 等价于 (~image_mask) & backbone_attention_mask
        
        # DEBUG: 验证 mask 的统计信息（可在调试后删除）
        # print(f"[AlternateVLDiT] image_mask: {image_mask.sum(dim=1).tolist()}, vlm_mask: {vlm_mask.sum(dim=1).tolist()}")
        # print(f"[AlternateVLDiT] backbone_mask: {backbone_attention_mask.sum(dim=1).tolist()}")
        # print(f"[AlternateVLDiT] image_attn_mask: {image_attention_mask.sum(dim=1).tolist()}, non_image: {non_image_attention_mask.sum(dim=1).tolist()}")
        
        all_hidden_states = [hidden_states]
        
        # 遍历 transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # 自注意力块（因为 interleave_self_attention=True 已断言）
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 交叉注意力块：交替关注视觉和VLM特征
                # 参照 NVIDIA: idx=0 关注 non-image (VLM)，然后交替
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # 关注 VLM/text 特征
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    # 关注视觉特征
                    curr_encoder_attention_mask = image_attention_mask
                
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)
        
        # 输出处理
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)
