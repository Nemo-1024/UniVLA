#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .cross_attention_dit import DiT, AlternateVLDiT



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
    hidden_dim: int = 768
    num_layers: int = 12  #DiT层数
    num_steps: int = 50
    cfg_drop_prob: float = 0.25
    cfg_scale: float = 1.0
    interleave_self_attention: bool = False  # 让所有层都关注 VLM 特征
    num_timestep_buckets: int = 1000

    # 可学习编码器（内部构造 cond）所需配置
    vlm_dim: int = 2048
    vision_dim: int = 768
    num_vision_tokens: int = 256
    proprio_dim: int = 8
    # num_vision_queries: int = 64
    # qformer_layers: int = 2
    # enc_num_heads: int = 4
    # enc_hidden_dim: int = 512  # Enc(h_*) 输出维度；cond_dim = enc_hidden_dim * 4
    
    # AlternateVLDiT 相关配置
    use_alternate_vldit: bool = False  # 是否使用交替注意力模式
    attend_text_every_n_blocks: int = 2  # 每多少个块关注一次VLM特征
    
    # 噪声采样配置（与 GR00T 对齐）
    noise_beta_alpha: float = 1.5  # Beta 分布的 alpha 参数
    noise_beta_beta: float = 1.0   # Beta 分布的 beta 参数
    noise_s: float = 0.999         # 时间变换的缩放因子


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
        self.enc_vlm = VectorMLP(in_dim=self.config.vlm_dim, hidden_dim=self.config.vision_dim)
        # self.enc_a_p_to_a = VectorMLP(in_dim=2 * self.config.hidden_dim, hidden_dim=self.config.hidden_dim)
        self.enc_prop = VectorMLP(in_dim=self.config.proprio_dim, hidden_dim=self.config.hidden_dim)
        self.action_encoder = ActionEncoder(action_dim=self.config.action_dim, hidden_size=self.config.hidden_dim)
        self.position_embedding = nn.Embedding(512, self.config.hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # 根据配置选择 DiT 类型
        DiTClass = AlternateVLDiT if self.config.use_alternate_vldit else DiT
        
        dit_kwargs = {
            "num_attention_heads": 16,
            "attention_head_dim": int(self.config.hidden_dim // 16),
            "output_dim": self.config.action_dim,
            "num_layers": self.config.num_layers,
            "interleave_self_attention": self.config.interleave_self_attention,
            "cross_attention_dim": self.config.vision_dim,  # default None 修改是为了确保cond被交叉注意力关注到
        }
        
        if self.config.use_alternate_vldit:
            dit_kwargs["attend_text_every_n_blocks"] = self.config.attend_text_every_n_blocks
        
        self.DiT = DiTClass(**dit_kwargs)
        self.velocity_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        self.cfg_embeddings = nn.Parameter(torch.randn(1, self.config.num_vision_tokens, self.config.vision_dim))
        
        # 初始化 Beta 分布（与 GR00T 一致）
        self.beta_dist = torch.distributions.Beta(
            concentration1=self.config.noise_beta_alpha,
            concentration0=self.config.noise_beta_beta
        )
        
        
    def sample_noise(
        self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # 与 GR00T 对齐：直接使用 randn；同时避免部分 backend 对 torch.normal+bfloat16 的限制
        return torch.randn(size=shape, dtype=dtype, device=device)

    def sample_time(self, bsize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        采样时间步，遵循 GR00T 的实现
        使用 Beta 分布并应用 (noise_s - sample) / noise_s 变换
        
        关键修复：原实现使用 (1 - sample) * noise_s，导致时间范围为 [0, 0.999]
        正确实现使用 (noise_s - sample) / noise_s，时间范围为 [~0, 1.0]
        这确保训练时能学习到 t=1 附近的速度场，这对推理至关重要
        """
        # Beta 采样在 float32 上更稳定；再 cast 回目标 dtype，避免 (bf16/half) 隐式升精度
        sample = self.beta_dist.sample([bsize]).to(device=device, dtype=torch.float32)
        sample = (self.config.noise_s - sample) / self.config.noise_s
        return sample.to(dtype=dtype)

    
    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor, # [B, D]
        actions: torch.Tensor, # [B, T, K]
        attention_mask: Optional[torch.Tensor] = None,  # [B, vlm_seq_len] VLM 的 attention_mask
    ) -> torch.Tensor:
        assert actions.shape[1] == self.config.window_size, "actions.shape[1] must be equal to window_size"
        device = actions.device
        batch_size = h_t.shape[0]
        
        # 采样噪声和时间
        noise = self.sample_noise(actions.shape, device, actions.dtype)
        time = self.sample_time(actions.shape[0], device, actions.dtype)
        time = time[:, None, None]

        # 流匹配插值（与 GR00T 一致：t=0 是噪声，t=1 是数据）
        noisy_trajectory = (1 - time) * noise + time * actions
        velocity = actions - noise
        # 离散化时间步，并确保在有效范围内 [0, num_timestep_buckets-1]
        t_discretized = (time[:, 0, 0] * self.config.num_timestep_buckets).long()
        t_discretized = torch.clamp(t_discretized, 0, self.config.num_timestep_buckets - 1)
        
        # 编码动作特征
        noisy_trajectory = self.action_encoder(noisy_trajectory, t_discretized)
        pos_ids = torch.arange(noisy_trajectory.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        noisy_trajectory = noisy_trajectory + pos_embs

        # 编码条件特征
        cond_prop = self.enc_prop(proprio)
        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]

        # CFG drop（仅作用于 h_t1_star）
        if self.training and self.config.cfg_drop_prob > 0.0:
            bsz = h_t.shape[0]
            mask = (torch.rand(bsz, device=device) < self.config.cfg_drop_prob).view(bsz, 1, 1)
            cond_future = torch.where(mask, self.cfg_embeddings.expand(bsz, -1, -1), h_t1_star)
        else:
            cond_future = h_t1_star

        # 统一数据流：VLM 特征合并到 encoder_hidden_states
        encoder_hidden_states = torch.cat((h_t, cond_future, cond_vlm), dim=1)
        hidden_states = torch.cat((cond_prop, noisy_trajectory), dim=1)
        
        action_horizon = noisy_trajectory.shape[1]  # 记录动作序列长度，确保与推理时一致
        
        # 构造 encoder_attention_mask：视觉部分(h_t+h_t1)全关注 + VLM部分使用原始 attention_mask
        num_vision = h_t.shape[1] + cond_future.shape[1]  # 256 + 256 = 512
        num_vlm = cond_vlm.shape[1]
        if attention_mask is not None:
            # diffusers/SDPA 要求 mask dtype 为 bool 或 float（或与 query dtype 一致）
            # 这里统一用 bool mask：True=有效/可见，False=padding/不可见
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            # 无 mask 时全部关注
            encoder_attention_mask = None
        
        # 根据模式选择调用方式
        if self.config.use_alternate_vldit:
            # 构建 attention masks
            num_h_t = h_t.shape[1]
            num_h_t1 = cond_future.shape[1]
            num_vlm = cond_vlm.shape[1]
            
            # image_mask: 视觉部分为 True
            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            # vlm_mask: VLM 部分为 True
            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_discretized,
                image_mask=image_mask,
                vlm_mask=vlm_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            # 标准 DiT
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_discretized,
                encoder_attention_mask=encoder_attention_mask,
            )
        
        v_t = dit_output[:, -action_horizon:, :]
        # Flow Matching loss: 预测速度场 vs 真实速度场
        # 遵循 PyTorch 约定：loss(prediction, target)
        losses = F.mse_loss(v_t, velocity)
        return losses

    @torch.inference_mode()
    def sample_actions_cfg(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        proprio: torch.Tensor,
        cfg_scale: Optional[float] = None,
        num_inference_steps: int = 50,
        attention_mask: Optional[torch.Tensor] = None,  # [B, vlm_seq_len] VLM 的 attention_mask
    ) -> torch.Tensor:
        """
        参照 forward 的流匹配定义进行推理，从 t=0 的噪声积分到 t=1 的数据；
        通过 cfg_scale 控制是否启用 CFG：cfg_scale = 1.0 或 None 时关闭（仅条件分支），!= 1.0 时启用。
        
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
        action_horizon = self.config.window_size
        # 初始化为纯噪声（t=0 的起点）
        actions = torch.randn(
            size=(batch_size, action_horizon, self.config.action_dim),
            dtype=h_t.dtype,
            device=device,
        )

        dt = 1.0 / float(num_inference_steps)

        # 编码条件特征（循环外，只需计算一次）
        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]
        cond_prop = self.enc_prop(proprio)

        # 统一数据流：构建 encoder_hidden_states
        cond_encoder_hidden = torch.cat((h_t, h_t1_star, cond_vlm), dim=1)

        # 构造 encoder_attention_mask：视觉部分(h_t+h_t1)全关注 + VLM部分使用原始 attention_mask
        num_vision = h_t.shape[1] + h_t1_star.shape[1]  # 256 + 256 = 512
        if attention_mask is not None:
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            encoder_attention_mask = None

        # 修正CFG判断：只有当 cfg_scale 存在且 != 1.0 时才启用CFG
        # cfg_scale=1.0 时，CFG公式退化为纯条件预测，应避免计算无条件分支
        use_cfg = cfg_scale is not None and cfg_scale != 1.0
        if use_cfg:
            # 无条件分支：仅替换 h_t1_star
            uncond_encoder_hidden = torch.cat((
                h_t,
                self.cfg_embeddings.expand(batch_size, -1, -1),
                cond_vlm
            ), dim=1)

        # 如果使用 AlternateVLDiT，预先构建 masks
        if self.config.use_alternate_vldit:
            num_h_t = h_t.shape[1]
            num_h_t1 = h_t1_star.shape[1]
            num_vlm = cond_vlm.shape[1]

            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

        # 降噪循环：从 t=0 正向积分到 t=1（噪声→数据）
        for step in range(num_inference_steps):
            t_cont = step / float(num_inference_steps)  # 从 0 到接近 1
            # 离散化时间步，与训练时保持一致的方式
            t_discretized = int(t_cont * self.config.num_timestep_buckets)
            t_discretized = min(self.config.num_timestep_buckets - 1, max(0, t_discretized))

            # 编码当前动作
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
            )
            action_features = self.action_encoder(actions, timesteps_tensor)

            # 添加位置编码
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

            # 构建 hidden_states
            hidden_states = torch.cat((cond_prop, action_features), dim=1)

            # 根据模式调用 DiT
            if self.config.use_alternate_vldit:
                # 条件预测
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    image_mask=image_mask,
                    vlm_mask=vlm_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                pred_cond = model_output_cond[:, -action_horizon:, :]

                if use_cfg:
                    # 无条件预测
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        image_mask=image_mask,
                        vlm_mask=vlm_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    pred_uncond = model_output_uncond[:, -action_horizon:, :]
                    pred_velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                else:
                    pred_velocity = pred_cond
            else:
                # 标准 DiT
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    encoder_attention_mask=encoder_attention_mask,
                )
                pred_cond = model_output_cond[:, -action_horizon:, :]

                if use_cfg:
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    pred_uncond = model_output_uncond[:, -action_horizon:, :]
                    pred_velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                else:
                    pred_velocity = pred_cond

            # 正向欧拉积分（从噪声走向数据）
            actions = actions + dt * pred_velocity

        return actions


