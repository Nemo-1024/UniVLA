#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from argparse import Action
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .cross_attention_dit import DiT



class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


def normalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """归一化到[0,1]范围"""
    return (x - min_val) / (max_val - min_val)


def unnormalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """反归一化"""
    return x * (max_val - min_val) + min_val


def safe_arcsin(value: torch.Tensor) -> torch.Tensor:
    """安全的arcsin函数，确保输入在[-1,1]范围内"""
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value: torch.Tensor) -> torch.Tensor:
    """将ALOHA夹爪位置转换为角度空间 - 保持原有实现"""
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value: torch.Tensor) -> torch.Tensor:
    """从角度空间转换为ALOHA夹爪位置 - 保持原有实现"""
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value: torch.Tensor) -> torch.Tensor:
    """aloha_gripper_from_angular的逆函数 - 保持原有实现"""
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)
    


def swish(x):
    return x * torch.sigmoid(x)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = nn.Linear(action_dim, hidden_size)  # (d -> w)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = nn.Linear(hidden_size, hidden_size)  # (w -> w)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x)

        return x

class VectorMLP(nn.Module):
    """简单向量投影器，用于 h_vlm / proprio 等向量输入到 hidden_dim"""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            x = x.unsqueeze(1)
        return self.net(x)



        
"""
Conditional Flow Matching head for unified conditioning (CFG inside the head).
This module follows FlowMatchingHead's time/noise schedule and linear path.
"""

from dataclasses import dataclass as _dataclass


@_dataclass
class ConditionalFlowMatchingConfig:
    # 动作维度（输出）
    action_dim: int = 7
    window_size: int = 10
    # Flow 维度与步数
    hidden_dim: int = 512
    num_layers: int = 8  #DiT层数
    num_steps: int = 50
    cfg_drop_prob: float = 0.25
    cfg_scale: float = 0.8
    interleave_self_attention: bool = True
    num_timestep_buckets: int = 1000

    # 可学习编码器（内部构造 cond）所需配置
    vlm_dim: int = 1024
    vision_dim: int = 1024
    num_vision_tokens: int = 256
    proprio_dim: int = 8
    num_vision_queries: int = 64
    qformer_layers: int = 2
    enc_num_heads: int = 4
    enc_hidden_dim: int = 512  # Enc(h_*) 输出维度；cond_dim = enc_hidden_dim * 4


class ConditionalFlowMatchingHead(nn.Module):
    def __init__(self, config: Optional[ConditionalFlowMatchingConfig] = None):
        super().__init__()
        self.config = config or ConditionalFlowMatchingConfig()

        # 内部可学习编码器：将 (h_t, h_t1*, h_vlm, proprio) -> cond
               
        # self.enc_h_t_t1 = LAMEncoder(
        #     context_dim=self.config.vision_dim,
        #     query_dim=self.config.hidden_dim,
        #     num_queries=self.config.num_vision_queries,
        #     num_layers=self.config.qformer_layers,
        # )
        self.enc_vlm = VectorMLP(in_dim=self.config.vlm_dim, hidden_dim=self.config.hidden_dim)
        self.enc_a_p_to_a = VectorMLP(in_dim=2 * self.config.hidden_dim, hidden_dim=self.config.hidden_dim)
        self.enc_prop = VectorMLP(in_dim=self.config.proprio_dim, hidden_dim=self.config.hidden_dim)
        
        self.action_encoder = ActionEncoder(action_dim=self.config.action_dim, hidden_size=self.config.hidden_dim)
        self.position_embedding = nn.Embedding(512, self.config.hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.DiT = DiT(
            num_attention_heads=8,
            attention_head_dim=int(self.config.hidden_dim//8),
            output_dim=self.config.action_dim,
            num_layers=self.config.num_layers,
            interleave_self_attention=self.config.interleave_self_attention,
            cross_attention_dim=self.config.vision_dim, # default None 修改是为了确保cond被交叉注意力关注到
        )
        self.velocity_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        self.cfg_embeddings = nn.Parameter(torch.randn(1, self.config.num_vision_tokens, self.config.vision_dim))

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device,dtype: torch.dtype) -> torch.Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample(torch.Size([bsize])).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001
    def _add_state(self, x_t: torch.Tensor, cond_prop: torch.Tensor) -> torch.Tensor:
        x_t = torch.cat([x_t, cond_prop], dim=-1)
        x_t = self.enc_prop(x_t)
        return x_t

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor, # [B, D]
        actions: torch.Tensor, # [B, T, K]
    ) -> torch.Tensor:
        assert actions.shape[1] == self.config.window_size, "actions.shape[1] must be equal to window_size"
        device = actions.device
        noise = self.sample_noise(actions.shape, device, actions.dtype)
        time = self.sample_time(actions.shape[0], device)
        time = time[:,None,None]

        x_t = time * noise + (1 - time) * actions
        u_t = noise - actions
        # Convert (continuous) t -> discrete if needed
        t_discretized = (time[:, 0, 0] * self.config.num_timestep_buckets).long()
        x_t = self.action_encoder(x_t, t_discretized)
        pos_ids = torch.arange(x_t.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        x_t = x_t + pos_embs


        cond_prop = self.enc_prop(proprio)
        # x_t = torch.cat([x_t, cond_prop], dim=-1)
        # x_t = self.enc_a_p_to_a(x_t)
        # 只能使用当前时刻的state，而不是整个chunk的state，否则会出现训练推理不一致的问题

        # training-time CFG drop
        if self.training and self.config.cfg_drop_prob > 0.0:
            bsz = h_t.shape[0]
            mask = (torch.rand(bsz, device=device) < self.config.cfg_drop_prob).view(bsz, 1,1)
            cond_future = torch.where(mask, self.cfg_embeddings.expand(bsz, -1, -1), h_t1_star)
        else:
            cond_future = h_t1_star

        cond_vision = torch.cat((h_t, cond_future), dim=1)
        cond_vlm = self.enc_vlm(h_vlm)  # h_vlm 已经是 bfloat16，无需再次转换
        

        action_horizon = x_t.shape[1]  # 记录动作序列长度，确保与推理时一致
        sa_embs = torch.cat((cond_vlm, cond_prop, x_t), dim=1)
        dit_output = self.DiT(hidden_states=sa_embs, encoder_hidden_states=cond_vision, timestep=t_discretized)
        v_t = dit_output[:, -action_horizon:, :]
        
        losses = F.mse_loss(u_t, v_t)
        return losses

    @torch.inference_mode()
    def sample_actions_cfg(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: int = 14,
        cfg_scale: Optional[float] = None,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """
        参照 forward 的流匹配定义进行推理，从 t=1 的噪声积分到 t=0 的数据；
        通过 cfg_scale 控制是否启用 CFG：cfg_scale <= 1 视为关闭（仅条件分支），>1 启用。
        
        Args:
            h_t: 当前视觉特征 [B, num_vision_tokens, vision_dim]
            h_t1_star: 目标视觉特征 [B, num_vision_tokens, vision_dim] 
            h_vlm: VLM 特征 [B, vlm_dim]
            proprio: 本体感受特征 [B, proprio_dim]
            action_horizon: 动作序列长度
            cfg_scale: CFG 引导强度
            num_inference_steps: 推理步数
            
        Returns:
            actions: 采样的动作序列 [B, action_horizon, action_dim]
        """
        device = h_t.device
        batch_size = h_t.shape[0]
        
        # 初始化为 t=1 的噪声（对应 forward 中 x_t = t*noise + (1-t)*actions 的噪声端）
        x = torch.randn(
            size=(batch_size, action_horizon, self.config.action_dim),
            dtype=h_t.dtype,
            device=device,
        )

        # 设置推理步数与时间步长（从 t=1 -> t=0，负向时间积分等价为 x = x - dt * v）
        num_steps = num_inference_steps
        dt = 1.0 / float(num_steps)

        # 条件编码（与 forward 一致，不含训练时的随机 drop）
        cond_vision = torch.cat((h_t, h_t1_star), dim=1)
        cond_vlm = self.enc_vlm(h_vlm)
        cond_prop = self.enc_prop(proprio)  # 本体感受编码（在循环中不变，提前计算）

        # 是否启用 CFG（仅当 cfg_scale > 1 才计算无条件分支以节省算力）
        # 注意：训练时CFG drop只作用在h_t1_star上，因此推理时无条件分支只替换视觉特征
        use_cfg = cfg_scale is not None
        
        if use_cfg:
            # 无条件分支：用cfg_embeddings替换h_t1_star（未来视觉特征）
            uncond_vision = torch.cat((
                h_t, self.cfg_embeddings.expand(batch_size, -1, -1)
            ), dim=1)
            # VLM特征在训练时未做drop，推理时保持一致（与条件分支相同）
            uncond_vlm = cond_vlm  # 复用条件分支的VLM编码，避免重复计算

        # 反向时间积分：从 t=1, ..., 1/num_steps 到 0
        for step in range(num_steps, 0, -1):
            t_cont = step / float(num_steps)  # (0, 1]
            
            # 离散化时间步，与训练时保持一致（使用.long()而非int()）
            # 训练时：t_discretized = (time[:, 0, 0] * num_timestep_buckets).long()
            # 这里需要确保相同的离散化逻辑
            t_discretized = int((t_cont * self.config.num_timestep_buckets))
            # 边界裁剪：确保 t_discretized 在 [0, num_buckets-1] 范围内
            t_discretized = min(self.config.num_timestep_buckets - 1, max(0, t_discretized))

            # 编码当前 x_t
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
            )
            action_features = self.action_encoder(x, timesteps_tensor)
            
            # 添加位置编码（与训练时保持一致）
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

            # 条件路径（与训练时保持一致：直接拼接 cond_vlm, cond_prop, action_features）
            sa_embs_cond = torch.cat((cond_vlm, cond_prop, action_features), dim=1)
            model_output_cond = self.DiT(
                hidden_states=sa_embs_cond,
                encoder_hidden_states=cond_vision,
                timestep=timesteps_tensor,
            )
            pred_cond = model_output_cond[:, -action_horizon:, :]

            if use_cfg:
                # 无条件路径
                sa_embs_uncond = torch.cat((uncond_vlm, cond_prop, action_features), dim=1)
                model_output_uncond = self.DiT(
                    hidden_states=sa_embs_uncond,
                    encoder_hidden_states=uncond_vision,
                    timestep=timesteps_tensor,
                )
                pred_uncond = model_output_uncond[:, -action_horizon:, :]
                pred_velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_velocity = pred_cond

            # 反向欧拉积分（从噪声端走向数据端）：x_{t-dt} = x_t - dt * v
            x = x - dt * pred_velocity

        return x


