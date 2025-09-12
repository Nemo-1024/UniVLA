import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from .vq import NSVQ
from .vjepa_encoder import VJEPAEncoder

import torch
import torch.nn as nn
import math

class QFormerBlock(nn.Module):
    """
    一个完整的 Q-Former 构建块。
    它包含：
    1. 在查询向量上的自注意力 (Self-Attention)
    2. 从上下文到查询的交叉注意力 (Cross-Attention)
    3. 一个前馈网络 (Feed-Forward Network)
    
    使用了前置层归一化（Pre-LayerNorm）和残差连接，以获得更好的训练稳定性。
    """
    def __init__(self, query_dim, context_dim=None, num_heads=8, ffn_expansion_factor=4, dropout=0.1):
        """
        初始化 QFormerBlock。
        
        参数:
            query_dim (int): 查询向量和输出的维度 (d)。
            context_dim (int, optional): 上下文特征的维度 (D)。如果为 None，则默认为 query_dim。
            num_heads (int): 多头注意力的头数。
            ffn_expansion_factor (int): FFN 中间隐藏层的扩展因子。
            dropout (float): Dropout 的比率。
        """
        super().__init__()
        
        # 如果没有提供 context_dim，则假设它与 query_dim 相同（用于自注意力场景）
        if context_dim is None:
            context_dim = query_dim

        # 1. 自注意力部分
        self.norm_sa = nn.LayerNorm(query_dim)
        self.attn_sa = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 2. 交叉注意力部分
        self.norm_ca = nn.LayerNorm(query_dim)
        self.attn_ca = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=context_dim,  # Key 的维度来自上下文
            vdim=context_dim,  # Value 的维度来自上下文
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. 前馈网络部分
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * ffn_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * ffn_expansion_factor, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, queries, context):
        """
        前向传播。
        
        参数:
            queries (torch.Tensor): 查询张量，形状为 [B, n, d]。
            context (torch.Tensor): 上下文张量，形状为 [B, N, D]。
                                    (其中 N = 2*K)
        返回:
            torch.Tensor: 处理后的查询张量，形状为 [B, n, d]。
        """
        # 1. 自注意力 + 残差连接
        sa_output, _ = self.attn_sa(self.norm_sa(queries), self.norm_sa(queries), self.norm_sa(queries))
        queries = queries + sa_output

        # 2. 交叉注意力 + 残差连接
        ca_output, _ = self.attn_ca(query=self.norm_ca(queries), key=context, value=context)
        queries = queries + ca_output

        # 3. 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm_ffn(queries))
        queries = queries + ffn_output
        
        return queries


class QFormer(nn.Module):
    """
    Q-Former 模型。
    通过堆叠多个 QFormerBlock，使用一组可学习的查询向量从给定的上下文中提取特征。
    """
    def __init__(self, query_dim, context_dim,num_queries=4, num_layers=6, num_heads=8, ffn_expansion_factor=4, dropout=0.1):
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
        
        # 堆叠多个 QFormerBlock
        self.layers = nn.ModuleList([
            QFormerBlock(
                query_dim=query_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
    def forward(self, context):
        """
        前向传播。
        
        参数:
            context (torch.Tensor): 来自时空主干的输出特征，
                                    形状应为 [B, N, D]，其中 N = 2*K。
        返回:
            torch.Tensor: 经过 Q-Former 提取和处理后的特征，
                          形状为 [B, n, d]，可以直接用于 VQ 量化。
        """
        batch_size = context.shape[0]
        
        # 将可学习的查询广播到当前 batch 的大小
        queries = self.queries.expand(batch_size, -1, -1)
        
        # 依次通过每个 QFormerBlock
        for layer in self.layers:
            queries = layer(queries, context)
            
        return queries


class SpatioTemporalBlock(nn.Module):
    """
    一个空间-时间交替的Transformer Block。
    """
    def __init__(self, dim: int, space_heads: int=12, time_heads: int=12, dropout: float = 0.0):
        super().__init__()
        self.spatio_transformer_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=space_heads, batch_first=True,activation="gelu",dropout=dropout)
        self.temporal_transformer_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=time_heads, batch_first=True,activation="gelu",dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_space = x.reshape(B * T, N, D)
        x_space = self.spatio_transformer_layer(x_space)
        x_space = x_space.reshape(B, T, N, D)
        x_time = x_space.permute(0, 2, 1, 3)
        x_time = x_time.reshape(B * N, T, D)
        x_time = self.temporal_transformer_layer(x_time, is_causal=True)
        x_time = x_time.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x_time

class LAMEncoder(nn.Module):
    """
    LAM编码器：多层空间-时间交替Transformer。
    """
    def __init__(self, context_dim: int, query_dim: int, num_queries: int=4, num_layers: int=4, dropout: float = 0.0):
        super().__init__()
        # self.blocks = nn.ModuleList([
        #     SpatioTemporalBlock(dim, dropout=dropout) for _ in range(num_layers)
        # ])
        self.QFormer = QFormer(query_dim=query_dim, context_dim=context_dim, num_queries=num_queries, num_layers=num_layers, dropout=dropout)
        # 0 代表 t, 1 代表 t+1
        self.temporal_embeddings = nn.Embedding(2, context_dim)
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        B, T, K, D = feature.shape
        f_t = feature[:,0]
        f_t1 = feature[:,1]
        # 给所有 t 时刻的 K 个 token 加上 t 时刻的嵌入
        time_ids_t = torch.zeros(B, K, dtype=torch.long, device=f_t.device)
        f_t_enhanced = f_t + self.temporal_embeddings(time_ids_t)

        # 给所有 t+1 时刻的 K 个 token 加上 t+1 时刻的嵌入
        time_ids_t1 = torch.ones(B, K, dtype=torch.long, device=f_t1.device)
        f_t1_enhanced = f_t1 + self.temporal_embeddings(time_ids_t1)
        context = torch.cat([f_t_enhanced, f_t1_enhanced], dim=1) # Shape: [B, 2*K, dim]
        latents=self.QFormer(context)
    
        return latents

class DecoderBlock(nn.Module):
    """
    LAMDecoder 的核心构建块。
    它将“动作”信息 (z_q) 融合到“状态”特征 (f_t) 中。
    """
    def __init__(self, feature_dim, node_dim, num_heads=8, ffn_expansion_factor=4, dropout=0.1):
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
        self.norm_ca = nn.LayerNorm(feature_dim)

        self.attn_ca = nn.MultiheadAttention(
            embed_dim=feature_dim,   # Query (和输出) 的维度
            kdim=node_dim,       # Key 的维度
            vdim=node_dim,       # Value 的维度
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 前馈网络
        self.norm_ffn = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * ffn_expansion_factor, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, state_features, LAM_features):
        """
        前向传播。
        
        参数:
            state_features (torch.Tensor): f_t，形状为 [B, K, D_feat]。
            LAM_features (torch.Tensor): z_q，形状为 [B, 4, d]。
        返回:
            torch.Tensor: 更新后的状态特征，形状为 [B, K, D_feat]。
        """
        # print("state_features shape:", state_features.shape)
        # print("LAM_features shape:", LAM_features.shape)
        # 自注意力 + 残差连接 (在 state_features 上)
        sa_output, _ = self.attn_sa(self.norm_sa(state_features), self.norm_sa(state_features), self.norm_sa(state_features))
        state_features = state_features + sa_output
        
        # 交叉注意力 + 残差连接
        # Query 来自 state，Key 和 Value 来自 LAM
        ca_output, _ = self.attn_ca(query=self.norm_ca(state_features), key=LAM_features, value=LAM_features)
        state_features = state_features + ca_output

        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm_ffn(state_features))
        state_features = state_features + ffn_output

        return state_features


class LAMDecoder(nn.Module):
    """
    通过堆叠多个 DecoderBlock，将动作应用到状态上，以重建下一帧的特征。
    """
    def __init__(self, feature_dim, node_dim, num_layers=6, num_heads=8, dropout=0.1):
        """
        初始化 LAMDecoder。
        
        参数:
            feature_dim (int): 状态特征 f_t 的维度 (D_feat)。
            node_dim (int): 动作特征 z_q 的维度 (d)。
            num_layers (int): DecoderBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(
                feature_dim=feature_dim,
                node_dim=node_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 注意：输入的 f_t 通常已经包含了 ViT 的位置编码，这里无需再添加。

    def forward(self, f_t, z_q):
        """
        前向传播。
        
        参数:
            f_t (torch.Tensor): 前一帧的特征，形状 [B, K, D_feat]。
            z_q (torch.Tensor): VQ量化后的4个动作code，形状 [B, 4, d]。
            
        返回:
            torch.Tensor: 重建的后一帧特征 f_hat_t+1，形状 [B, K, D_feat]。
        """
        # 将 f_t 作为可更新的状态，依次通过所有解码器层
        reconstructed_features = f_t
        for layer in self.layers:
            reconstructed_features = layer(reconstructed_features, z_q)
            
        return reconstructed_features



class StateDeltaPredictor(nn.Module):
    """
    物理接地状态差解码器 (Physical Grounding State-Delta Decoder)
    
    将潜动作向量 z_t 通过内置自注意力机制处理后映射到预测的末端执行器状态差 delta_s_xyz
    输入形状固定为 [B, num_queries, latent_dim]
    """
    def __init__(
        self, 
        latent_dim: int, 
        dropout: float = 0.1,
    ):
        """
        初始化状态差预测器
        
        Args:
            latent_dim (int): 潜动作向量的维度 (与 z_t 的维度相同)
            dropout (float): Dropout 比率
        """
        super().__init__()
        
        self.attention_dim = latent_dim * 2
        
        # 自注意力投影层
        self.query_proj = nn.Linear(latent_dim, self.attention_dim)
        self.key_proj = nn.Linear(latent_dim, self.attention_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        # 注意力层归一化
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # 全局聚合层：将多个query聚合为单个表示
        self.global_aggregator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, 3)  # 修复：输入维度应该是latent_dim // 4
        )
        
        
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        内置自注意力机制
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, num_queries, latent_dim]
            
        Returns:
            torch.Tensor: 注意力处理后的张量，形状为 [B, num_queries, latent_dim]
        """
        # 计算 Query, Key, Value
        Q = self.query_proj(x)  # [B, num_queries, attention_dim]
        K = self.key_proj(x)    # [B, num_queries, attention_dim]
        V = self.value_proj(x)  # [B, num_queries, latent_dim]

        attended_values = F.scaled_dot_product_attention(
        Q, K, V, 
        dropout_p=0.1 if self.training else 0.0
        )
        
        # 输出投影
        output = self.out_proj(attended_values)  # [B, num_queries, latent_dim]
        
        # 残差连接和Layer Normalization
        output = self.layer_norm(x + output)
        
        return output
        
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z_t (torch.Tensor): 潜动作向量，形状为 [B, num_queries, latent_dim]
            
        Returns:
            torch.Tensor: 预测的状态差，形状为 [B, 3]
        """
        # 自注意力处理query间的交互
        z_t_attended = self._self_attention(z_t)  # [B, num_queries, latent_dim]
        
        z_t_aggregated = torch.mean(z_t_attended, dim=1)  # [B, latent_dim]

        # 通过全局聚合器进一步处理
        delta_s_pred = self.global_aggregator(z_t_aggregated)  # [B, 3]
        
        return delta_s_pred


class PhysicalGroundingLoss(nn.Module):
    """
    物理接地混合损失函数
    
    包含：
    1. 方向损失 (Direction Loss) - 余弦相似度
    2. 幅度正则化 (Magnitude Regularizer) - Huber损失
    3. 运动权重 (Movement Weighting) - Sigmoid激活
    """
    def __init__(
        self, 
        lambda_dir: float = 1.0,
        lambda_mag_reg: float = 0.1,
        motion_threshold_beta: float = 0.01,  # 1cm
        motion_scale_alpha: float = 100.0,
        huber_delta: float = 1.0
    ):
        """
        初始化物理接地损失函数
        
        Args:
            lambda_dir (float): 方向损失权重（主要）
            lambda_mag_reg (float): 幅度正则化权重（次要）
            motion_threshold_beta (float): 运动激活阈值
            motion_scale_alpha (float): Sigmoid斜率参数
            huber_delta (float): Huber损失的delta参数
        """
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_mag_reg = lambda_mag_reg
        self.motion_threshold_beta = motion_threshold_beta
        self.motion_scale_alpha = motion_scale_alpha
        self.huber_delta = huber_delta
        
    def forward(self, delta_s_pred: torch.Tensor, delta_s_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算物理接地损失
        
        Args:
            delta_s_pred (torch.Tensor): 预测的状态差，形状 [B, 3]
            delta_s_gt (torch.Tensor): 真实的状态差，形状 [B, 3]
            
        Returns:
            Dict[str, torch.Tensor]: 包含各种损失项的字典
        """
        # 1. 方向损失 (Direction Loss) - 余弦相似度
        # 避免零向量导致的数值不稳定
        eps = 1e-8
        cosine_sim = F.cosine_similarity(delta_s_pred, delta_s_gt, dim=-1, eps=eps)
        direction_loss = 1.0 - cosine_sim  # [B]
        
        # 2. 幅度正则化 (Magnitude Regularizer) - Huber损失
        pred_magnitude = torch.norm(delta_s_pred, p=2, dim=-1)  # [B]
        gt_magnitude = torch.norm(delta_s_gt, p=2, dim=-1)      # [B]
        magnitude_loss = F.huber_loss(pred_magnitude, gt_magnitude, reduction='none', delta=self.huber_delta)  # [B]
        
        # 3. 运动权重 (Movement Weighting) - Sigmoid激活
        with torch.no_grad():
            # 只有当真实运动幅度超过阈值时，损失才会被显著计入
            motion_weight = torch.sigmoid(
                self.motion_scale_alpha * (gt_magnitude - self.motion_threshold_beta)
            )  # [B]
        
        # 4. 组合损失
        combined_loss_per_sample = motion_weight * (
            self.lambda_dir * direction_loss + 
            self.lambda_mag_reg * magnitude_loss
        )  # [B]
        
        # 5. 平均损失
        total_loss = combined_loss_per_sample.mean()
        
        # 返回详细的损失信息用于日志记录
        return {
            'total_loss': total_loss,
            'magnitude_loss': magnitude_loss.mean(),
            'motion_weight': motion_weight.mean(),
            'cosine_similarity': cosine_sim.mean()
        }


class LatentLAMModel(nn.Module):
    """
    LAM主模型：自动实例化Encoder/Decoder/NSVQ，QFormer实现稀疏离散化。
    现在包含物理接地状态差解码器。
    """
    def __init__(
        self,
        dim: int=1024,
        enc_layers: int = 4,
        codebook_size: int = 16,
        code_dim: int = 128,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 4,
        dec_self_heads: int = 4,
        dec_cross_heads: int = 4,
        dropout: float = 0.1,
        num_queries: int = 4,
        # 新增：状态差预测器参数
        enable_state_delta_prediction: bool = True,

    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 集成视觉编码器：负责将 videos 编码为 [B, T, N, D] 特征
        self.vision_encoder = VJEPAEncoder().to(self.device)
        self.encoder = LAMEncoder(context_dim=dim, query_dim=code_dim, num_queries=num_queries, num_layers=enc_layers, dropout=dropout).to(self.device)
        self.decoder = LAMDecoder(dim, code_dim, dec_layers,  dropout=dropout).to(self.device)
        vq_kwargs = vq_kwargs or {}
        self.vq = NSVQ(
            code_seq_len=num_queries,
            codebook_size=codebook_size,
            code_dim=code_dim,
            **vq_kwargs
        ).to(self.device)

        self.num_queries = num_queries
        
        # 新增：状态差预测器
        self.enable_state_delta_prediction = enable_state_delta_prediction
        if enable_state_delta_prediction:
            self.state_delta_predictor = StateDeltaPredictor(
                latent_dim=code_dim, 
                dropout=dropout
            ).to(self.device)

    def forward(self, videos: torch.Tensor, state_pair: Optional[torch.Tensor] = None):
        """
        Args:
            videos: 视频帧张量，形状取决于 VJEPAEncoder 的实现，例如 [B, T, C, H, W]
            state_pair: [B, T, state_dim] # 可选的状态信息，用于状态差预测
        Returns:
            tuple: (recon, perplexity, indices, delta_s_pred, features)
                recon: [B, N, D] 重建的下一帧 patch 特征
                perplexity: 标量 VQ困惑度
                indices: [B, num_queries] VQ索引
                delta_s_pred: [B, 3] 预测的状态差（如果启用）或 None
                features: [B, T, N, D] 由视觉编码器得到的特征
        """
        return self._run(videos=videos, vq_training=True)


    
    def _run(
        self,
        videos: torch.Tensor,
        user_specific: Optional[int] = None,
        vq_training: bool = True,
    ):
        """统一的执行路径，仅在 VQ 调用上区分训练/推理。
        Args:
            videos: 原始视频帧张量
            user_specific: 指定 codebook（仅推理时生效）
            vq_training: True 使用 self.vq(...)，False 使用 self.vq.inference(...)
        Returns:
            (recon, perplexity, indices, delta_s_pred, features)
        """
        # 冻结视觉编码器参数，与原 Lightning 行为保持一致
        with torch.no_grad():
            features = self.vision_encoder.encode_video_frames(videos)

        nodes = self.encoder(features)  # [B, num_queries, code_dim]
        if vq_training:
            quantized, perplexity, indices = self.vq(nodes)
        else:
            quantized, perplexity, indices = self.vq.inference(nodes, user_specific=user_specific)

        recon = self.decoder(features[:, 0], quantized)

        delta_s_pred = None
        if self.enable_state_delta_prediction:
            delta_s_pred = self.state_delta_predictor(quantized)

        return recon, perplexity, indices, delta_s_pred, features


    def inference(self, videos: torch.Tensor, user_specific=None):
        return self._run(videos=videos, user_specific=user_specific, vq_training=False)

    @torch.no_grad()
    def vq_encode(self, videos: torch.Tensor, user_specific=None):
        """
        推理流程：videos -> 视觉编码 -> 编码 -> VQ.inference(user_specific) -> 解码
        Args:
            videos: 输入视频帧张量
            user_specific: int or list, 指定VQ codebook索引
            return_indices: 保留参数（无效，保持兼容）
        Returns:
            tuple: (recon, perplexity, indices, delta_s_pred)
                recon: [B, N, D]
                perplexity: 标量Tensor（与训练定义一致，基于索引统计得到）
                indices: [B, num_queries]
                delta_s_pred: [B, 3] 或 None（若未启用）
        """
        
        recon, perplexity, indices, delta_s_pred, features =  self._run(
            videos=videos,
            user_specific=user_specific,
            vq_training=False,
        )
        return {
            'recon': recon,
            'perplexity': perplexity,
            'indices': indices,
            'delta_s_pred': delta_s_pred,
            'features': features,
        }




