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
from latent_action_model.core.lam_model import QFormer


@dataclass
class SmolVLAConfig:
    """简化的SmolVLA配置类，保持与原版兼容但简化架构参数"""
    
    # 基础维度参数
    image_dim: int = 512
    state_dim: int = 14
    action_dim: int = 14
    hidden_dim: int = 512
    
    # Flow Matching 核心参数 - 保持原有逻辑
    num_steps: int = 100
    min_period: float = 0.001
    max_period: float = 1.0
    chunk_size: int = 1
    max_action_dim: int = 14
    max_state_dim: int = 14
    
    # 简化的架构参数
    num_layers: int = 4
    dropout: float = 0.1


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
    
def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device: str
) -> Tensor:
    """生成正弦/余弦位置编码，兼容 time 形状为 [B] 或 [B, 1, 1]。"""
    if time.ndim != 1:
        # 适配 [B,1,1] 或其它可展平到批次维的形状
        time = time.reshape(time.shape[0])

    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    dtype = time.dtype
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

class SimpleMLPEncoder(nn.Module):
    """简化的多层感知机编码器：支持 [..., input_dim] -> [..., output_dim]。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

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


        
"""
Conditional Flow Matching head for unified conditioning (CFG inside the head).
This module follows FlowMatchingHead's time/noise schedule and linear path.
"""

from dataclasses import dataclass as _dataclass


@_dataclass
class ConditionalFlowMatchingConfig:
    # 动作维度（输出）
    action_dim: int = 14

    # Flow 头部 MLP 维度与步数
    hidden_dim: int = 512
    num_layers: int = 4
    num_steps: int = 50
    min_period: float = 0.001
    max_period: float = 1.0
    cfg_drop_prob: float = 0.1

    # 可学习编码器（内部构造 cond）所需配置
    vlm_hidden_dim: int = 1024
    proprio_dim: int = 8
    num_queries: int = 4
    enc_num_layers: int = 2
    enc_num_heads: int = 4
    enc_hidden_dim: int = 512  # Enc(h_*) 输出维度；cond_dim = enc_hidden_dim * 4


class ConditionalFlowMatchingHead(nn.Module):
    def __init__(self, config: Optional[ConditionalFlowMatchingConfig] = None):
        super().__init__()
        self.config = config or ConditionalFlowMatchingConfig()

        # 内部可学习编码器：将 (h_t, h_t1*, h_vlm, proprio) -> cond
        enc_hidden_dim = self.config.enc_hidden_dim
        self.cond_enc = CondEncoders(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            proprio_dim=self.config.proprio_dim,
            hidden_dim=enc_hidden_dim,
            num_queries=self.config.num_queries,
            num_layers=self.config.enc_num_layers,
            num_heads=self.config.enc_num_heads,
        )

        # cond 编码到 flow 隐空间
        self.cond_encoder = SimpleMLPEncoder(
            input_dim=enc_hidden_dim * 4,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        )
        
        self.action_encoder = ActionEncoder(action_dim=self.config.action_dim, hidden_size=self.config.hidden_dim)

        self.time_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.fusion = SimpleMLPEncoder(
            input_dim=self.config.hidden_dim * 3,
            hidden_dim=self.config.hidden_dim * 2,
            output_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        )
        self.velocity_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample(torch.Size([bsize])).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def _encode(self, cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        device = cond.device
        h_c = self.cond_encoder(cond)
        h_a = self.action_encoder(x_t)
        t_emb = create_sinusoidal_pos_embedding(t, self.config.hidden_dim, self.config.min_period, self.config.max_period, str(device))
        h_t = self.time_encoder(t_emb)
        fused = torch.cat([h_c, h_a, h_t], dim=-1)
        return self.fusion(fused)

    def _make_uncond(self, cond: torch.Tensor) -> torch.Tensor:
        # 假设 cond = [h_t | h_t1 | h_vlm | h_prop] 等长拼接
        dim = cond.shape[-1]
        part = dim // 4
        h_t, h_t1, h_vlm, h_prop = torch.split(cond, [part, part, part, dim - 3 * part], dim=-1)
        zeros = torch.zeros_like(h_t1)
        return torch.cat([h_t, zeros, h_vlm, h_prop], dim=-1)

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor, # [B, T, K]
    ) -> torch.Tensor:
        device = actions.device
        noise = self.sample_noise(actions.shape, device)
        time = self.sample_time(actions.shape[0], device)
        time = time[:,None,None]

        x_t = time * noise + (1 - time) * actions
        u_t = noise - actions

        # 内部构造 cond
        with torch.no_grad():
            cond_raw = self.cond_enc(h_t.detach(), h_t1_star.detach(), h_vlm.detach(), proprio)

        # training-time CFG drop
        if self.training and self.config.cfg_drop_prob > 0.0:
            bsz = cond_raw.shape[0]
            mask = (torch.rand(bsz, device=device) < self.config.cfg_drop_prob).view(bsz, 1)
            cond_uncond = self._make_uncond(cond_raw)
            cond_in = torch.where(mask, cond_uncond, cond_raw)
        else:
            cond_in = cond_raw

        feats = self._encode(cond_in, x_t, time)
        v_t = self.velocity_head(feats)
        losses = F.mse_loss(u_t, v_t, reduction="none").mean(dim=-1)
        return losses

    @torch.no_grad()
    def sample_actions_cfg(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        # 内部构造 cond
        with torch.no_grad():
            cond = self.cond_enc(h_t.detach(), h_t1_star.detach(), h_vlm.detach(), proprio)

        device = cond.device
        bsz = cond.shape[0]
        noise = self.sample_noise((bsz, self.config.action_dim), device)

        x_t = noise
        dt = -1.0 / float(self.config.num_steps)
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        while t >= -dt / 2:
            t_expanded = t.expand(bsz)
            cond_uncond = self._make_uncond(cond)
            f_u = self._encode(cond_uncond, x_t, t_expanded)
            v_u = self.velocity_head(f_u)
            f_c = self._encode(cond, x_t, t_expanded)
            v_c = self.velocity_head(f_c)
            v = v_u + guidance_scale * (v_c - v_u)
            x_t = x_t + dt * v
            t = t + dt
        return x_t


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


class CondEncoders(nn.Module):
    """将 VLA 条件的可学习编码器打包：Enc(h_t), Enc(h_{t+1}^*), Enc(h_vlm), Enc(q_t)。

    - 使用 QFormer 处理 token 化的图像特征 [B, K, D] -> [B, hidden]
    - 使用 VectorMLP 处理向量特征
    - forward 返回按 dim=1 拼接后的 cond 向量
    """

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        num_queries: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.enc_h_t_t1 = QFormer(
            query_dim=hidden_dim,
            context_dim=vlm_hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.enc_vlm = VectorMLP(in_dim=vlm_hidden_dim, hidden_dim=hidden_dim)
        self.enc_prop = VectorMLP(in_dim=proprio_dim, hidden_dim=hidden_dim)

    def encode_parts(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_h_t = self.enc_h_t_t1(h_t)
        cond_h_t1 = self.enc_h_t_t1(h_t1_star)
        cond_vlm = self.enc_vlm(h_vlm)
        cond_prop = self.enc_prop(proprio)
        return cond_h_t, cond_h_t1, cond_vlm, cond_prop

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        cond_h_t, cond_h_t1, cond_vlm, cond_prop = self.encode_parts(h_t, h_t1_star, h_vlm, proprio)
        return torch.cat([cond_h_t, cond_h_t1, cond_vlm, cond_prop], dim=1)