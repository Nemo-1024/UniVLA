import torch.nn as nn   
import torch
import math
from typing import Optional
from .ac_predictor import VisionTransformerPredictorAC
from .modules import Attn_Crossn_Block
from .pos_embs import Fixed2DPositionalEncoding
class LAMDecoder(nn.Module):
    """
    使用 VisionTransformerPredictorAC 作为 backbone 的自回归解码器。
    
    - 支持按时间维度自回归（frame-causal），即第 t 帧仅可看见 <= t 的信息
    - 要求 z_q 与 f_0 在时间维度对齐：z_q.shape = [B, T, context_dim]，f_0.shape = [B, T, K, input_dim] 或 [B, 1, K, input_dim]
    """
    def __init__(
        self,
        context_dim,
        input_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        dropout: float = 0.1,
        frame_to_pre: int = 4,
        train_in_latent: bool = True,
        ffn_expansion_factor: int = 2,
        img_size = (256, 256),
        patch_size: int = 16,
        dataset_vocab_size: int = 16,
        code_dim: Optional[int] = None,
    ):
        """
        参数:
            context_dim: predictor 的中间通道（predictor_embed_dim），也是动作/状态条件维度
            input_dim: patch 特征维度（embed_dim）
            num_layers: backbone 深度（Transformer blocks 数量）
            num_heads: 注意力头数
            dropout: dropout 与 attn_drop
            train_in_latent: True 表示在视觉 latent 空间训练（返回 token 特征），False 可投影回像素
            img_size, patch_size: 用于推断每帧 token 网格大小（H=W=sqrt(K) 时与 K 对齐）
            is_frame_causal: 是否启用时间自回归掩码
            use_rope: 是否启用旋转位置编码
        """
        super().__init__()
        self.feature_dim = context_dim
        self.input_dim = input_dim
        self.train_in_latent = train_in_latent
        self.img_size = img_size
        self.patch_size = patch_size

        # 采用 ac_predictor 作为自回归 backbone
        self.backbone = VisionTransformerPredictorAC(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=frame_to_pre,  
            embed_dim=input_dim,
            predictor_embed_dim=context_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            uniform_power=True,
            use_silu=False,
            wide_silu=True,
            is_frame_causal=True,
            use_activation_checkpointing=False,
            use_rope=True,
            action_embed_dim=context_dim,
        )
        if code_dim is not None and code_dim != context_dim:
            self.action_in_proj = nn.Linear(code_dim, context_dim)
        else:
            self.action_in_proj = nn.Identity()

        # 若需要从 latent 还原到像素（通常不在预训练视觉特征上使用）
        if not train_in_latent:
            self.to_pixel = nn.ConvTranspose2d(input_dim, 3, kernel_size=patch_size, stride=patch_size)

        self.state_predictor = StatePredictor(
            context_dim,
            dropout=dropout,
            num_datasets=dataset_vocab_size,
            code_dim=code_dim,
        )

    def forward(self, features, actions, states, dataset_id):
        """
        前向传播（自回归）。
        
        参数:
            features: [B, T, K, input_dim] 或 [B, 1, K, input_dim] 或 [B, K, input_dim]
            actions: [B, T, context_dim]（默认与 features 在时间维度上对齐）
            states: [B, T, 8]
            dataset_id: [B] 或 [B, 1]，用于选择对应数据集的可学习嵌入
        返回:
            torch.Tensor:
                - 若 train_in_latent=True: [B, T, K, input_dim]
                - 若 train_in_latent=False: [B, T, 3, H*patch, W*patch]
        """
        # 规范化输入形状
        if features.dim() == 3:
            # [B, K, D] -> [B, 1, K, D]
            features = features.unsqueeze(1)
        elif features.dim() != 4:
            raise ValueError(f"features 期望为 [B, T, K, D] 或 [B, K, D]，但获得 {features.shape}")

        B, T, K, D = features.shape
        actions = self.action_in_proj(actions)

        # backbone 期望的输入为 [B, T*H*W, D]
        x = features.reshape(B, T * K, D)

        # 自回归预测
        y = self.backbone(x, actions=actions, states=states)  # [B, T, 2+H*W, D]
        f_pre = y[:,:,2:,:]
        s_pre = self.state_predictor(y[:,:,0,:], states, dataset_id) # [B, T, 8]
        return f_pre, s_pre

        # # 如需还原到像素空间：对每帧分别映射
        # BT = B * T
        # y_img = y.reshape(BT, H * W, D).transpose(1, 2).reshape(BT, D, H, W)
        # y_img = self.to_pixel(y_img)  # [B*T, 3, H*patch, W*patch]
        # y_img = y_img.view(B, T, *y_img.shape[1:])
        # return y_img

class LAMDecoder_v2(nn.Module):
    """
    通过堆叠多个 DecoderBlock，将动作应用到状态上，以重建下一帧的特征。
    """
    def __init__(self, context_dim, input_dim: int=1024, num_queries: int=1, num_layers=6, num_heads=16, dropout=0.1, grid_size: int=16, train_in_latent: bool = True, ffn_expansion_factor=2, dataset_vocab_size: int = 16, code_dim: Optional[int] = None):
        """
        初始化 LAMDecoder。
        
        参数:
            context_dim (int): 状态/动作特征维度 (D_feat)。
            input_dim (int): 输入特征的维度 (D_feat)。
            num_layers (int): DecoderBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
        """
        super().__init__()
        self.feature_dim = context_dim
        self.input_dim = input_dim
        self.train_in_latent = train_in_latent
            
        # 解码层
        # self.dec_layers = nn.ModuleList([
        #     CrossAttentionBlock(feature_dim, num_heads=num_heads, ffn_ratio=4, dropout=dropout) 
        #     for _ in range(num_layers)
        # ])
        self.num_queries = num_queries
        # if self.num_queries == 1:
        self.dec_layers = nn.ModuleList([nn.TransformerEncoderLayer(context_dim, num_heads, dim_feedforward=int(context_dim*ffn_expansion_factor), dropout=dropout, batch_first=True, norm_first=True) for _ in range(num_layers)])
        # else:
            # self.dec_layers = nn.ModuleList([Attn_Crossn_Block(feature_dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, dropout=dropout) for _ in range(num_layers)])
        self.pos_embed = Fixed2DPositionalEncoding(context_dim, grid_size, grid_size)
        # # Query self-attention增强
        # self.query_attn = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        
        # 简化输入输出投影，避免冗余
        if input_dim == context_dim:
            self.project_input = nn.Identity()
            self.project_output = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, context_dim)
            self.project_output = nn.Linear(context_dim, input_dim)
        if code_dim is not None and code_dim != context_dim:
            self.action_in_proj = nn.Linear(code_dim, context_dim)
        else:
            self.action_in_proj = nn.Identity()
        if not train_in_latent:
            self.to_pixel = nn.ConvTranspose2d(input_dim, 3, kernel_size=16, stride=16)
        
    def forward(self, features, actions):
        """
        前向传播。
        
        参数:
            features (torch.Tensor): 初始帧的特征，形状 [B, 1, K, input_dim]。
            actions (torch.Tensor): VQ量化后的动作code，形状 [B, 1, node_dim]。
            states (torch.Tensor): 初始状态，形状 [B, 1, 8]。
            dataset_id (torch.Tensor): [B] 或 [B,1]，选择数据集特定的嵌入。
            
        返回:
            torch.Tensor: 重建的最后一帧特征 f_hat_T，形状 [B, 1, K, input_dim]。
        """
        # 投影query并使用self-attention增强
        actions_tokens = self.action_in_proj(actions)  # [B, 1, feature_dim]
        
        # 投影输入特征（只调用一次，避免冗余）
        features_tokens = self.project_input(features)  # [B, 1, K, feature_dim] 或 [B, K, feature_dim]

        if features_tokens.dim() == 4:
            # 将单帧时间维压缩，Transformer 期望 3D: [B, S, E]
            features_tokens = features_tokens.squeeze(1)  # [B, K, feature_dim]
        features_tokens = self.pos_embed(features_tokens)
        if actions_tokens.shape[1] == 1:
            # 将动作条件加到每个空间token上，自动在K维广播
            x = features_tokens + actions_tokens  # [B, K, feature_dim]
            # x = torch.cat([features_tokens, actions_tokens], dim=1)
            # 通过解码层堆叠
            for layer in self.dec_layers:
                x = layer(x)
        else:
            x = torch.cat([features_tokens, actions_tokens], dim=1)
            # x = features_tokens
            for layer in self.dec_layers:
                x = layer(x)
        # 输出投影
        reconstructed_features = self.project_output(x[:, :features_tokens.shape[1]])  # [B, K, input_dim]
        # 统一返回形状为 [B, 1, K, *]
        if not self.train_in_latent:
            B, K, D = reconstructed_features.shape
            h = w = int(K ** 0.5)
            rec_img = self.to_pixel(reconstructed_features.transpose(1, 2).reshape(B, D, h, w))  # [B, 3, H, W]
            return rec_img.unsqueeze(1)  # [B, 1, 3, H, W]
        else:
            return reconstructed_features.unsqueeze(1)  # [B, 1, K, input_dim]


class StatePredictor(nn.Module):
    """
    物理接地末端执行器状态预测器 (Physical Grounding State Predictor)

    对潜动作序列在 query 维度做均值池化并映射，与 state 的最后一维映射值
    以及数据集 token 拼接后，经 MLP 预测下一时刻的 EEF 状态。

    输入:
        z_t: [B, num_queries, latent_dim]
        state_0: [B, 1, 8]
    输出:
        s_pred: [B, 8]  (与目标状态对应)
    """

    def __init__(self, latent_dim: int, dropout: float = 0.1, num_datasets: int = 16, num_queries: int = 1, state_dim: int = 8, code_dim: Optional[int] = None):
        super().__init__()
        self.dataset_embed = nn.Embedding(num_datasets, latent_dim)
        # 对 query 进行均值池化后的线性映射
        z_input_dim = code_dim if code_dim is not None else latent_dim
        self.z_proj = nn.Linear(z_input_dim, latent_dim)
        # 对完整 state 做线性映射
        self.state_proj = nn.Linear(state_dim, latent_dim)
        # MLP 预测最终状态
        self.mlp = nn.Sequential(
            # nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 8),
        )

    def forward(self, z_t: torch.Tensor, state_0: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z_t: [B, num_queries, latent_dim]
            state_0: [B, 1, 8]
            dataset_id: [B] 或 [B,1]，用于选择对应数据集的嵌入
        Returns:
            s_pred: [B, 8]
        """
        if dataset_id is None:
            raise ValueError("dataset_id 不能为空")
        B = state_0.size(0)
        dataset_id = dataset_id.view(B).long()
        if state_0.dim() == 3:
            state_0 = state_0.squeeze(1)
        # 1️⃣ 对 query 在第二维求均值并线性映射
        z_mean = z_t.mean(dim=1)
        z_embed = self.z_proj(z_mean)  # [B, latent_dim]
        # 2️⃣ 对完整 state 做线性映射
        state_full = state_0.contiguous()
        state_embed = self.state_proj(state_full)  # [B, latent_dim]
        # 3️⃣ 拼接 dataset token 并送入 MLP
        ds_token = self.dataset_embed(dataset_id)  # [B, latent_dim]
        fused = torch.cat([z_embed+ ds_token, state_embed], dim=-1)   # [B, latent_dim*2]
        s_pred = self.mlp(fused)  # [B, 8]
        # 4️⃣ 对连续维度做范围约束
        s_pred = torch.tanh(s_pred)

        return s_pred