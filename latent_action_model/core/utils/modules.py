# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    N_T = add_tokens + (H * W)
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


def build_modal_block_attention_mask(T, H, W, add_tokens=1, ar_query: bool = True, num_queries: int = 1):
    """
    基于“后置”约定构造模态掩码（统一将附加 token 放在末尾）：
    - 约定每帧 token 排布为：
      ar_query=True:  [H*W 图像 patch, add_tokens..., query(num_queries)]
      ar_query=False: [H*W 图像 patch, add_tokens...]，并在序列尾部追加 num_queries 个全局 query
    - 规则：
      1) image_feature 只能看到 image_feature
      2) state 只能看到 state
      3) query 能看到所有模态（包括 query/state/image）
    - 返回：allowed 掩码（True 表示允许注意），形状 [N, N]
      其中：
        ar_query=True  -> N = T * (H*W + add_tokens + num_queries)
        ar_query=False -> N = T * (H*W + add_tokens) + num_queries
    """
    assert add_tokens >= 0, "add_tokens 表示每帧额外的非 query 帧级 token 数（如 state），可为 0 或更大"
    if ar_query:
        # 每帧排列：[image..., add_tokens..., query]
        N_T = (H * W) + add_tokens + num_queries
        frame_modality_ids = torch.full((N_T,), 1, dtype=torch.long)  # 默认 image=1
        if add_tokens > 0:
            frame_modality_ids[H * W : H * W + add_tokens] = 0  # state-like
        frame_modality_ids[-num_queries:] = 2  # query
        modality_ids = frame_modality_ids.repeat(T)  # [T*N_T]
    else:
        # 帧内排列：[image..., add_tokens...]，全局在序列尾部追加 1 个 query
        N_T = (H * W) + add_tokens
        frame_modality_ids = torch.full((N_T,), 1, dtype=torch.long)  # 默认 image=1
        if add_tokens > 0:
            frame_modality_ids[H * W :] = 0  # state-like
        modality_ids = frame_modality_ids.repeat(T)  # [T*N_T]
        modality_ids = torch.cat([modality_ids, torch.tensor([2] * num_queries, dtype=torch.long)], dim=0)  # 追加全局 query
    N = modality_ids.numel()
    row = modality_ids.unsqueeze(1)  # [N,1]
    col = modality_ids.unsqueeze(0)  # [1,N]
    same_modality = row == col
    # query 行可见所有列
    row_is_query = (row == 2).expand(-1, N)
    allowed = same_modality | row_is_query
    return allowed


def build_decoder_block_attention_mask(T, H, W, add_tokens=2, query_index=0, is_causal=True):
    """
    LAMDecoder 用的“前置”掩码（ACRoPEAttention 要求 add_tokens 前置）：
    - 每帧排列：[add_tokens..., H*W image]
      其中 add_tokens 中包含 1 个 query（用 action 表示），其余为其它帧级 token（如 state）
    - 规则：
      1) image 仅能看 image
      2) state-like(其它 add_tokens) 仅能看 state-like
      3) 所有 row 皆可看 query（latent）
      4) latent(query) 仅能看 query
    - 若 is_causal=True，则再与帧级 block 因果掩码做“与”合并（保留过去与当前帧）
    - 返回：allowed 掩码（True=允许注意），形状 [N, N]，N = T * (add_tokens + H*W)
    """
    assert 0 <= query_index < add_tokens, "query_index 需在 [0, add_tokens) 范围内"
    N_T = add_tokens + (H * W)
    # 类型编码：0=state-like, 1=image, 2=query
    frame_types = torch.full((N_T,), 1, dtype=torch.long)  # image=1
    if add_tokens > 0:
        frame_types[:add_tokens] = 0  # state-like
    frame_types[query_index] = 2  # query(action)
    types = frame_types.repeat(T)  # [N]
    row = types.unsqueeze(1)  # [N,1]
    col = types.unsqueeze(0)  # [1,N]
    # modal 可见性：同模态可见 + query 全可见；其中 query 行仅能看 query
    same_type = row == col
    col_is_query = col == 2
    allowed_modal = same_type | col_is_query
    if is_causal:
        allowed_time = build_action_block_causal_attention_mask(T, H, W, add_tokens=add_tokens)
        return allowed_modal & allowed_time
    else:
        return allowed_modal


def rotate_queries_or_keys(x, pos, omega: torch.Tensor = None):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position (allow passing precomputed omega to avoid recompute)
    if omega is None:
        omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
        omega /= D / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)
    else:
        # ensure dtype/device alignment
        omega = omega.to(dtype=x.dtype, device=x.device, non_blocking=True)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by commenting
    # -- out the two lines below, and uncommenting the following two lines.
    # -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    # emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    # emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        # Precompute omega frequencies for d/h/w sub-dimensions to avoid per-forward construction
        def _build_omega(dim_half: int) -> torch.Tensor:
            base = torch.arange(dim_half, dtype=torch.float32)
            base /= float(dim_half)
            return 1.0 / (10000.0 ** base)  # (D/2,)
        self.register_buffer("omega_d_base", _build_omega(self.d_dim // 2), persistent=False)
        self.register_buffer("omega_h_base", _build_omega(self.h_dim // 2), persistent=False)
        self.register_buffer("omega_w_base", _build_omega(self.w_dim // 2), persistent=False)

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        # -- snap spatial positions to grid size
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        # -- split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
                # --
                pos_T = torch.arange(T, device=x.device, dtype=torch.float32)
                qd = rotate_queries_or_keys(
                    q[..., : self.d_dim], pos=pos_T, omega=self.omega_d_base
                )
                kd = rotate_queries_or_keys(
                    k[..., : self.d_dim], pos=pos_T, omega=self.omega_d_base
                )
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        # Precompute omega frequencies for d/h/w sub-dimensions to avoid per-forward construction
        def _build_omega(dim_half: int) -> torch.Tensor:
            base = torch.arange(dim_half, dtype=torch.float32)
            base /= float(dim_half)
            return 1.0 / (10000.0 ** base)  # (D/2,)
        self.register_buffer("omega_d_base", _build_omega(self.d_dim // 2), persistent=False)
        self.register_buffer("omega_h_base", _build_omega(self.h_dim // 2), persistent=False)
        self.register_buffer("omega_w_base", _build_omega(self.w_dim // 2), persistent=False)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class QFormer(nn.Module):
    """
    Q-Former 模型。
    通过堆叠多个 QFormerBlock，使用一组可学习的查询向量从给定的上下文中提取特征。
    """
    def __init__(self, query_dim, context_dim, num_queries=4, num_layers=6, num_heads=16, ffn_expansion_factor=2, dropout=0.1):
        """
        初始化 QFormer 模型。
        
        参数:
            num_queries (int): 可学习的查询向量数量 (n)。
            query_dim (int): 查询向量和最终输出的维度 (d)。
            context_dim (int): 输入上下文特征的维度 (D)。
            num_layers (int): QFormerBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
            ffn_expansion_factor (int): FFN 的扩展因子。
            dropout (float): Dropout 比率。
        """
        super().__init__()

        # 可学习的查询向量，形状为 [1, n, d]，可以广播到整个 batch
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))

        # 直接使用 MultiheadAttention 实现 Q 与 K/V 不同维度的跨注意力
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=query_dim,    # Q 的维度
                kdim=context_dim,       # K 的维度
                vdim=context_dim,       # V 的维度
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        # 规范化与前馈网络
        self.norm_qs = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(num_layers)])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(context_dim) for _ in range(num_layers)])
        hidden_dim = int(query_dim * ffn_expansion_factor)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, query_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(query_dim)

    def forward(self, context):
        """
        前向传播。
        
        参数:
            context (torch.Tensor): 来自时空主干的输出特征，
                                    形状应为 [B, N, D]。
        返回:
            torch.Tensor: 经过 Q-Former 提取和处理后的特征，
                          形状为 [B, n, d]，可以直接用于 VQ 量化。
        """

        B, N, Dc = context.shape
        # 将可学习的查询广播到当前 batch 的大小
        queries = self.queries.expand(B, -1, -1)  # [B, num_queries, query_dim]
        # 逐层：Query 与 Context 的跨注意力 + FFN
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(norm_q(queries))
        queries = self.final_norm(queries)

        return queries

class QFormer_att(nn.Module):
    """
    Q-Former 模型。
    通过堆叠多个 QFormerBlock，使用一组可学习的查询向量从给定的上下文中提取特征。
    """
    def __init__(self, query_dim, context_dim, num_frames, num_queries, grid_size, add_tokens=1, num_layers=6, num_heads=16, ffn_expansion_factor=2, dropout=0.1, ar_query: bool = False, use_mask: bool = False):
        """
        初始化 QFormer 模型。
        
        参数:
            num_queries (int): 可学习的查询向量数量 (n)。
            query_dim (int): 查询向量和最终输出的维度 (d)。
            context_dim (int): 输入上下文特征的维度 (D)。
            num_layers (int): QFormerBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
            ffn_expansion_factor (int): FFN 的扩展因子。
            dropout (float): Dropout 比率。
        """
        super().__init__()

        self.ar_query = ar_query
        self.query_dim = query_dim
        if self.ar_query:
            # 将查询向量直接以最终形状注册为 Parameter，确保随模块一起迁移设备
            self.queries = nn.Parameter(torch.randn(1, num_frames, 1, context_dim))  # [1, T, 1, context_dim]
        else:
            self.queries = nn.Parameter(torch.randn(1, num_queries, context_dim))  #[1, 1, context_dim]
        self.q_cross_attn = CrossAttentionBlock(context_dim, num_heads)
        self.num_queries = num_queries

        # 堆叠多个 QFormerBlock
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=context_dim, nhead=num_heads, dim_feedforward=int(context_dim*ffn_expansion_factor), dropout=dropout, batch_first=True, norm_first=True) for _ in range(num_layers)
        ])

        # 预构建“模态自注意 + query全可见”的掩码，并与（若有）因果掩码合并
        # 形状对齐规则：
        #  - 输入 context 为 [B, T, (hw+1), D]
        #  - 采用“后置”约定：
        #       ar_query=True  -> 每帧重排为 [image(hw), state(1), query(1)]，全序列长度 L = T*(hw+2)
        #       ar_query=False -> 每帧重排为 [image(hw), state(1)]，并在序列末尾追加 1 个全局 query，长度 L = T*(hw+1)+1
        if use_mask:
            if self.ar_query:
                # 使用“后置”约定：每帧 [image..., state, query]（add_tokens=1 表示 state）
                modal_allowed = build_modal_block_attention_mask(num_frames, grid_size, grid_size, add_tokens=add_tokens, ar_query=True)
                modal_mask = ~modal_allowed
                # 合并因果掩码（帧级 block 因果），注意每帧额外 token 数为 2（state + query）
                causal_disallow = ~build_action_block_causal_attention_mask(num_frames, grid_size, grid_size, add_tokens=add_tokens+1)
                combined_mask = modal_mask | causal_disallow
                self.register_buffer("src_mask", combined_mask, persistent=False)
            else:
                # 非自回归：帧内为 [image..., state]（add_tokens=1），序列末尾追加全局 query
                modal_allowed = build_modal_block_attention_mask(num_frames, grid_size, grid_size, add_tokens=add_tokens, ar_query=False, num_queries=num_queries)
                modality_mask = ~modal_allowed
                self.register_buffer("src_mask", modality_mask, persistent=False)
        else:
            self.src_mask = None
    def forward(self, context):
        """
        前向传播。
        
        参数:
            context (torch.Tensor): 来自时空主干的输出特征，
                                    形状应为 [B, T, (hw+1), D]。
        返回:
            torch.Tensor: 经过 Q-Former 提取和处理后的特征，
                          形状为 [B, n, d]，可以直接用于 VQ 量化。
        """

        B, T, _, D = context.shape

        # queries = self.proj_in(self.queries)

        # 将 queries 扩展到 batch 维度，便于与 context 拼接
        if self.ar_query:
            queries = self.queries.expand(B, -1, -1, -1)  # [B, T, 1, D]
            # 直接在每帧 [image..., state] 后追加 query，得到 [image..., state, query]
            ctx = torch.cat([context, queries], dim=-2).reshape(B, -1, D)  # [B, T*(hw+2), D]
        else:
            queries = self.queries.expand(B, -1, -1)  # [B, n, D]
            # 输入已为 [image..., state]，直接展平后追加全局 query
            ctx = context.reshape(B, -1, D)
            queries = self.q_cross_attn(queries, ctx)
            ctx = torch.cat([ctx, queries], dim=1)
        # 依次通过每个 QFormerBlock
        for layer in self.layers:
            ctx = layer(ctx, src_mask=self.src_mask)
        if self.ar_query:
            return ctx.reshape(B, T, -1, self.query_dim)[:,1:,-1,:] #B,T-1,D    
        else:
            return ctx[:, -self.num_queries:,:]   #B,n,D
    
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class PatchEmbed(nn.Module):
    """
    可学习的 Patch Embedding 层：
    - 输入支持 [B,T,C,H,W] 或 [B*T,C,H,W]
    - 输出 [B,T,K,D]，K=(H/patch_size)*(W/patch_size)，D=embed_dim
    - 默认包含 ImageNet 标准化
    """
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def _encode_btchw(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images).flatten(2).transpose(1, 2)  # [B*T, K, D]
        return tokens

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T = 1
        tokens = self._encode_btchw(images)
        return tokens.reshape(B, T, tokens.shape[-2], tokens.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
      
class Attn_Crossn_Block(nn.Module):
    """
    LAMDecoder 的核心构建块。
    它将“动作”信息 (z_q) 融合到“状态”特征 (f_t) 中。
    """
    def __init__(self, feature_dim, num_heads=8, ffn_expansion_factor=4, dropout=0.1):
        """
        初始化 DecoderBlock。
        
        参数:
            feature_dim (int): 状态特征 f_t 的维度 (D_feat)。
            node_dim (int): 动作特征 z_q 的维度 (d)。
            num_heads (int): 多头注意力的头数。
            ffn_expansion_factor (int): FFN 中间层的扩展因子。
            dropout (float): Dropout 比率。
        """
        super().__init__()

        # 1. 自注意力 (在状态 f_t 上)
        self.norm_sa = nn.LayerNorm(feature_dim)

        self.attn_sa = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 2. 交叉注意力 (从动作 z_q 到状态 f_t)
        self.norm_ca_q = nn.LayerNorm(feature_dim)
        self.norm_ca_kv = nn.LayerNorm(feature_dim)

        self.attn_ca = nn.MultiheadAttention(
            embed_dim=feature_dim,   # Query (和输出) 的维度
            kdim=feature_dim,       # Key 的维度 (投影后的维度)
            vdim=feature_dim,       # Value 的维度 (投影后的维度)
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 前馈网络
        self.norm_ffn = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim * ffn_expansion_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(feature_dim * ffn_expansion_factor), feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sa_features, ca_features):
        """
        前向传播。
        
        参数:
            sa_features (torch.Tensor): f_t，形状为 [B, K, D_feat]。
            ca_features (torch.Tensor): z_q，形状为 [B, 4, d]。
        返回:
            torch.Tensor: 更新后的自注意力特征和交叉注意力特征，形状为 [B, K, D_feat]。
        """
        # print("state_features shape:", state_features.shape)
        # print("LAM_features shape:", LAM_features.shape)
        # 自注意力 + 残差连接 (在 state_features 上)
        sa_output, _ = self.attn_sa(self.norm_sa(sa_features), self.norm_sa(sa_features), self.norm_sa(sa_features))
        sa_features = sa_features + sa_output
        
        # 交叉注意力 + 残差连接
        # Query 来自 state，Key 和 Value 来自 LAM
        ca_output, _ = self.attn_ca(query=self.norm_ca_q(sa_features), key=self.norm_ca_kv(ca_features), value=self.norm_ca_kv(ca_features)) 
        sa_features = sa_features + ca_output

        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm_ffn(sa_features))
        sa_features = sa_features + ffn_output

        return sa_features

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim),
                                nn.GELU(),
                                nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        kv = self.norm1(kv)
        attn_out, _ = self.attn(q, kv, kv)
        q = q + attn_out
        q = self.norm2(q)
        q = q + self.ff(q)
        return q