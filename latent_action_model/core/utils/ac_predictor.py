# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from .modules import ACBlock as Block
from .modules import build_decoder_block_attention_mask
from .utils import trunc_normal_
from .pos_embs import get_3d_sincos_pos_embed

class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=5,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=8,   
        state_embed_dim=8, # actually use the state 3+3+pad+gripper
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(state_embed_dim, predictor_embed_dim, bias=True)
        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power
        # self.pos_embed = get_3d_sincos_pos_embed(predictor_embed_dim, grid_size=self.grid_height, grid_depth=self.num_frames)
        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,    
                )                           #never use is_causal=True, because we use custom block mask
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # 统一使用 Decoder 专用掩码（add_tokens 前置；query=action 在索引 0）
        # 返回的是“可见(True)”的掩码，且若 is_frame_causal=True 则已合并因果约束
        self.attn_mask = build_decoder_block_attention_mask(
            self.num_frames, self.grid_height, self.grid_width, add_tokens=2, query_index=0, is_causal=self.is_frame_causal
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, states, actions):
        """
        :param x: context tokens T-1
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = states.shape[1]

        # Interleave action tokens（前置顺序：[action(query), state, image...]）
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        x = x.view(B, T, -1, D)  # [B, T, H*W, D]
        expected_tokens = self.grid_height * self.grid_width
        assert x.size(2) == expected_tokens, f"Token grid mismatch: expected {expected_tokens}, got {x.size(2)}"
        x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]
        cond_tokens = 2
        assert x.size(1) == self.attn_mask.size(0), "x.size(1) != self.attn_mask.size(0)"
        # move mask once outside the block loop to avoid repeated .to(...)
        attn_mask = self.attn_mask.to(x.device, non_blocking=True) if self.attn_mask is not None else None
        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )
        # Split out action and frame tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
