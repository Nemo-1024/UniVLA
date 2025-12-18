import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms

from einops import repeat
class MotionAwareMultiCodePooling(nn.Module):
    def __init__(self, context_dim: int, query_dim: int, num_codes: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_codes = num_codes
        self.context_dim = context_dim
        self.motion_dim = context_dim // 2  # 从//4改为//2，保留更多信息

        # 运动特征提取器 - 添加多层结构和残差
        self.motion_extractor = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, self.motion_dim)
        )
        
        # 运动特征归一化
        self.motion_norm = nn.LayerNorm(self.motion_dim)

        # 运动感知查询生成器
        self.query_generator = nn.Linear(context_dim + self.motion_dim, num_codes * context_dim)

        # 空间注意力
        self.motion_attention = nn.MultiheadAttention(embed_dim=context_dim, num_heads=num_heads, batch_first=True)

        # code 投影 - 添加残差路径
        self.code_projection = nn.Sequential(
            nn.Linear(context_dim + self.motion_dim, query_dim),
            nn.LayerNorm(query_dim)
        )
        
        # 如果维度匹配，添加skip connection
        self.skip_connection = nn.Linear(context_dim, query_dim) if context_dim != query_dim else nn.Identity()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B, T, N, D = context.shape  # B, T, N, D

        # -------------------------------
        # 1. 运动特征（帧差 + 空间加权平均）
        # -------------------------------
        temporal_diff = context[:, 1:] - context[:, :-1]            # B, T-1, N, D
        motion_intensity = torch.norm(temporal_diff, dim=-1, keepdim=True)
        motion_intensity = motion_intensity / (motion_intensity.sum(dim=2, keepdim=True) + 1e-6)
        motion_features_raw = (temporal_diff * motion_intensity).sum(dim=2).mean(dim=1)  # B, D
        
        # 提取并归一化运动特征
        motion_features = self.motion_extractor(motion_features_raw)  # B, motion_dim
        motion_features = self.motion_norm(motion_features)

        # -------------------------------
        # 2. 生成运动感知查询
        # -------------------------------
        global_context = context.mean(dim=(1, 2))                                # B, D
        motion_enhanced = torch.cat([global_context, motion_features], dim=-1)  # B, D + motion_dim
        queries = self.query_generator(motion_enhanced).view(B, self.num_codes, D)  # B, num_codes, D

        # -------------------------------
        # 3. 空间注意力
        # -------------------------------
        spatial_context = context.reshape(B * T, N, D)                          # B*T, N, D
        spatial_queries = queries.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, self.num_codes, D)  # B*T, num_codes, D
        spatial_attended, _ = self.motion_attention(spatial_queries, spatial_context, spatial_context)
        temporal_context = spatial_attended.view(B, T, self.num_codes, D).mean(dim=1)  # B, num_codes, D

        # -------------------------------
        # 4. 生成 codes（添加skip connection）
        # -------------------------------
        combined = torch.cat([temporal_context, motion_features.unsqueeze(1).expand(-1, self.num_codes, -1)], dim=-1)  # B, num_codes, D+motion_dim
        codes = self.code_projection(combined)  # B, num_codes, query_dim
        
        # 添加来自原始context的skip connection
        context_skip = self.skip_connection(temporal_context)  # B, num_codes, query_dim
        codes = codes + context_skip

        return codes
class PositionalEncoding2D(nn.Module):
    """
    2D 位置编码：时间维度 + 空间维度
    输入 x 形状: [batch, time, space, feature_dim]
    输出 x + positional_encoding
    """
    def __init__(self, feature_dim: int, max_time: int = 5000, max_space: int = 5000):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 生成时间位置编码
        time_pos = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim))
        pe_time = torch.zeros(max_time, feature_dim)
        pe_time[:, 0::2] = torch.sin(time_pos * div_term)
        pe_time[:, 1::2] = torch.cos(time_pos * div_term)
        self.register_buffer('pe_time', pe_time)  # [T, feature_dim]

        # 生成空间位置编码
        space_pos = torch.arange(0, max_space, dtype=torch.float).unsqueeze(1)  # [S,1]
        pe_space = torch.zeros(max_space, feature_dim)
        pe_space[:, 0::2] = torch.sin(space_pos * div_term)
        pe_space[:, 1::2] = torch.cos(space_pos * div_term)
        self.register_buffer('pe_space', pe_space)  # [S, feature_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, time, space, feature_dim]
        """
        B, T, S, D = x.shape
        assert D == self.feature_dim, "Feature dimension mismatch."

        # 获取时间位置编码 [T,D] -> [1,T,1,D]
        pe_t = self.pe_time[:T, :].unsqueeze(0).unsqueeze(2)  # [1,T,1,D]
        # 获取空间位置编码 [S,D] -> [1,1,S,D]
        pe_s = self.pe_space[:S, :].unsqueeze(0).unsqueeze(1)  # [1,1,S,D]

        # 相加 broadcasting
        pe = pe_t + pe_s  # [1, T, S, D]
        return x + pe.to(x.device)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # [1, max_len, model_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, model_dim]
        return x + self.pos_enc[:, :x.size(1), :].to(x.device)

class QFormerBlock(nn.Module):
    """
    标准化的 Q-Former Transformer Block
    包含：
        1. Self-Attention（仅在 Query 上）
        2. Cross-Attention（Query 从 Context 中提取信息）
        3. Feed-Forward Network (FFN)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = query_dim

        # --- Self-Attention ---
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Cross-Attention ---
        self.norm_cross_q = nn.LayerNorm(query_dim)
        self.norm_cross_kv = nn.LayerNorm(context_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Feed-Forward Network ---
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, int(query_dim * ffn_expansion_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(query_dim * ffn_expansion_factor), query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Nq, Dq] learnable query tokens
            context: [B, Nc, Dc] encoded visual/text features
            self_attn_mask: optional mask for self-attention
            cross_attn_mask: optional mask for cross-attention
        Returns:
            Updated queries [B, Nq, Dq]
        """

        # --- 1. Self-Attention (Query-Query) ---
        q = self.norm_self(queries)
        self_attn_out, _ = self.self_attention(q, q, q, attn_mask=self_attn_mask)
        queries = queries + self_attn_out  # 残差连接

        # --- 2. Cross-Attention (Query-Context) ---
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(context)
        cross_attn_out, _ = self.cross_attention(
            q, kv, kv, attn_mask=cross_attn_mask
        )
        queries = queries + cross_attn_out  # 残差连接

        # --- 3. Feed-Forward Network ---
        ffn_out = self.ffn(self.norm_ffn(queries))
        queries = queries + ffn_out  # 残差连接

        return queries



class QFormer(nn.Module):
    """
    Q-Former 模型。
    通过堆叠多个 QFormerBlock，使用一组可学习的查询向量从给定的上下文中提取特征。
    """
    def __init__(self, query_dim, context_dim,num_queries=4, num_layers=6, num_heads=16, ffn_expansion_factor=4, dropout=0.1):
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
        
    def forward(self, context, self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None):
        """
        前向传播。
        
        参数:
            context (torch.Tensor): 来自时空主干的输出特征，
                                    形状应为 [B, N, D]。
        返回:
            torch.Tensor: 经过 Q-Former 提取和处理后的特征，
                          形状为 [B, n, d]，可以直接用于 VQ 量化。
        """
        batch_size = context.shape[0]
        
        # 将可学习的查询广播到当前 batch 的大小
        queries = self.queries.expand(batch_size, -1, -1)
        
        # 依次通过每个 QFormerBlock
        for layer in self.layers:
            queries = layer(queries, context, self_attn_mask, cross_attn_mask)
            
        return queries
class QFormer4JEPA(nn.Module):
    """
    Q-Former 模型。
    通过堆叠多个 QFormerBlock，使用一组可学习的查询向量从给定的上下文中提取特征。
    """
    def __init__(self, query_dim, context_dim,num_queries=4, num_layers=6, num_heads=16, ffn_expansion_factor=2, dropout=0.1):
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
        self.proj_in = nn.Linear(query_dim, context_dim)
        self.proj_out = nn.Linear(context_dim, query_dim)
        # 堆叠多个 QFormerBlock
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=context_dim, nhead=num_heads, dim_feedforward=int(context_dim*ffn_expansion_factor), dropout=dropout, batch_first=True, norm_first=True) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(context_dim)
        self.rms_norm = nn.RMSNorm(query_dim)
    def forward(self, context, self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None):
        """
        前向传播。
        
        参数:
            context (torch.Tensor): 来自时空主干的输出特征，
                                    形状应为 [B, N, D]。
        返回:
            torch.Tensor: 经过 Q-Former 提取和处理后的特征，
                          形状为 [B, n, d]，可以直接用于 VQ 量化。
        """

        B, N, D = context.shape
        # 将可学习的查询广播到当前 batch 的大小
        queries = self.queries.expand(B, -1, -1)
        queries = self.proj_in(queries)
        queries = torch.cat([queries, context], dim=1)
        # 依次通过每个 QFormerBlock
        for layer in self.layers:
            queries = layer(queries)
        queries = self.layer_norm(queries[:,:-N,:])
        queries = self.rms_norm(self.proj_out(queries))

        return queries

class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim: int, space_heads: int = 16, time_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        # 空间注意力
        self.norm_sa = nn.LayerNorm(dim)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=space_heads,
            dropout=dropout,
            batch_first=True
        )

        # 时间注意力
        self.norm_time = nn.LayerNorm(dim)
        self.temporal_transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=time_heads,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [B, T, N, D]
        输出: x_time [B, T, N, D]
        """
        B, T, N, D = x.shape

        # ----- 空间注意 -----
        x_space = x.reshape(B * T, N, D)
        x_norm = self.norm_sa(x_space)
        x_out, _ = self.spatial_attention(x_norm, x_norm, x_norm)
        x_space = x_space + x_out
        x_space = x_space.reshape(B, T, N, D)

        # ----- 时间注意 (因果) -----
        x_time = x_space.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_time = self.norm_time(x_time)
        
        # 创建因果注意力掩码 (上三角矩阵，对角线以上为True)
        # causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        # 半因果注意力掩码（允许访问 t+1）
        # 含义：t 时刻可以看到 [0, 1, ..., t, t+1]，但不能看到 t+2, t+3, ...
        # semi_causal_mask = torch.triu(
        #     torch.ones(T, T, device=x.device, dtype=torch.bool),
        #     diagonal=2  # ⬅️ 改成2而不是1
        # )
        x_time = self.temporal_transformer_layer(x_time, src_mask=None)  # 因果注意
        x_time = x_time.reshape(B, N, T, D).permute(0, 2, 1, 3)

        return x_time

class LAMEncoder(nn.Module):
    def __init__(self, context_dim: int, query_dim: int, input_dim: int=1024, num_queries: int=4,
                 num_layers: int=4, num_heads: int=16, ffn_expansion_factor=2,
                 dropout: float = 0.0, skip_connection: bool = False):
        super().__init__()
        self.num_queries = num_queries
        # self.QFormer = QFormer(
        #     query_dim=query_dim,
        #     context_dim=context_dim,
        #     num_queries=num_queries,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     dropout=dropout
        # )
        self.QFormer = QFormer4JEPA(
            query_dim=query_dim,
            context_dim=context_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout=dropout
        )

        
        # 简化输入投影：直接映射，避免冗余的膨胀-压缩
        if input_dim == context_dim:
            self.project_input = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, context_dim)
        self.state_project = nn.Sequential(nn.Linear(8, context_dim//4), nn.GELU(), nn.Linear(context_dim//4, context_dim), nn.LayerNorm(context_dim))
        # 添加skip connection路径（从输入直接到输出）
        self.input_skip = nn.Linear(input_dim, query_dim) if input_dim != query_dim  and skip_connection else None
        
        
    def forward(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        输入: features [B, T, K, input_dim], states [B, T, 1, 8]
        输出: latents [B, T-1, query_dim] - 每个latent代表相邻时间片之间的action
        """
        assert states.shape[1] == features.shape[1], f" time length mismatch: states.shape[1]: {states.shape[1]} != features.shape[1]: {features.shape[1]}"
        states = self.state_project(states)
        # 投影到context维度
        features = self.project_input(features)
        features = torch.cat([states, features], dim=-2)
        # 保存原始输入用于skip connection
        original_features = features
        B, T, K, D = features.shape

        # causal_mask = torch.triu(torch.ones(self.num_queries, self.num_queries, device=features.device, dtype=torch.bool), diagonal=1)
        latents = self.QFormer(features.reshape(B, -1, D))  # [B, T-1, query_dim]

        # 添加skip connection：将原始特征的时间差值信息融入latents
        if self.input_skip is not None:
            # 计算原始特征的时间差值并池化
            original_diffs = original_features[:, 1:] - original_features[:, :-1]  # [B, T-1, K, input_dim]
            skip_features = original_diffs.mean(dim=2)  # [B, T-1, input_dim]
            skip_latents = self.input_skip(skip_features)  # [B, T-1, query_dim]
            latents = latents + skip_latents
    
        return latents



class TemporalDiffQFormer(nn.Module):
    """
    基于时间差值生成latent action queries的QFormer（因果版本，向量化）
    利用时序差分特征和因果上下文挖掘潜在动作
    query数量自动等于时间片数减一 (T-1)
    每个action只能看到当前及之前的时间步，保证因果性
    使用向量化操作和注意力掩码，无需for循环
    """
    def __init__(self, query_dim, context_dim, num_queries: int=5, num_layers=6, num_heads=16, dropout=0.1):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        
        # 多尺度差值特征提取
        self.diff_feature_extractor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim)
        )
        
        # 因果上下文感知查询初始化
        self.context_aware_init = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, query_dim),
            nn.LayerNorm(query_dim)
        )
        
        # 可学习的查询模板（作为基础prior）
        self.query_prior = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)
        
        # 使用标准cross-attention替代QFormerBlock，支持attention mask
        self.refine_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=query_dim,
                kdim=context_dim,
                vdim=context_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.refine_norms = nn.ModuleList([
            nn.LayerNorm(query_dim) for _ in range(num_layers)
        ])
        
        self.refine_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_dim, query_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(query_dim * 4, query_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 因果时间一致性注意力（只看过去的actions）
        self.temporal_consistency = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
    def forward(self, temporal_diffs, context):
        """
        参数:
            temporal_diffs: 时间差值特征 [B, T-1, K, D]
            context: 时空上下文特征 [B, T, K, D] （因果处理后的）
        返回:
            queries: [B, T-1, query_dim] - 每个query代表一个latent action
        """
        B, T_minus_1, K, D = temporal_diffs.shape
        T = T_minus_1 + 1
        
        # === 第一阶段：从时间差值中提取初始action表示（向量化） ===
        
        # 1. 空间维度直接使用mean pooling
        diff_pooled = temporal_diffs.mean(dim=2)  # [B, T-1, D]
        
        # 2. 提取差值特征
        diff_features = self.diff_feature_extractor(diff_pooled)  # [B, T-1, D]
        
        # 3. 使用cumsum向量化计算因果上下文
        # 先对空间维度池化 [B, T, D]
        context_pooled = context.mean(dim=2)  # [B, T, D]
        
        # 使用cumsum计算累积和，然后计算累积平均
        # cumsum: [B, T, D]
        context_cumsum = torch.cumsum(context_pooled, dim=1)
        
        # 计算每个位置的累积平均（位置t对应时间0到t的平均）
        # [B, T-1, D] - 取前T-1个位置作为action_0到action_{T-2}的上下文
        divisor = torch.arange(1, T, device=context.device).view(1, -1, 1)  # [1, T-1, 1]
        causal_context_summary = context_cumsum[:, :-1, :] / divisor  # [B, T-1, D]
        
        # 融合差值和因果上下文
        combined_features = torch.cat([diff_features, causal_context_summary], dim=-1)  # [B, T-1, 2D]
        
        # 4. 生成初始queries
        queries = self.context_aware_init(combined_features)  # [B, T-1, query_dim]
        
        # 添加可学习的prior
        queries = queries + self.query_prior.expand(B, -1, -1)
        
        # === 第二阶段：通过attention精炼queries（使用因果掩码） ===
        # 将context reshape为 [B, T*K, D] 用于cross-attention
        context_flat = context.reshape(B, T * K, D)  # [B, T*K, D]
        
        # 创建因果注意力掩码 [T-1, T*K]（向量化）
        # action_t (位置t) 只能attend到时间 [0, t] 的所有spatial tokens
        # 使用broadcasting创建掩码
        query_time_idx = torch.arange(T_minus_1, device=queries.device).view(-1, 1)  # [T-1, 1]
        kv_time_idx = torch.arange(T, device=queries.device).repeat_interleave(K)  # [T*K]
        # action_t (时间t) 可以看到时间<=t的所有tokens
        causal_attn_mask = kv_time_idx.view(1, -1) > query_time_idx  # [T-1, T*K]
        
        # 通过refinement layers
        for attn, norm, ffn in zip(self.refine_layers, self.refine_norms, self.refine_ffns):
            # Cross-attention with causal mask
            q_norm = norm(queries)
            attn_out, _ = attn(
                q_norm, context_flat, context_flat,
                attn_mask=causal_attn_mask
            )  # [B, T-1, query_dim]
            queries = queries + attn_out
            
            # FFN
            queries = queries + ffn(norm(queries))
        
        # === 第三阶段：因果时间一致性建模 ===
        # 创建因果掩码，让每个action只能看到之前的actions
        temporal_causal_mask = torch.triu(
            torch.ones(T_minus_1, T_minus_1, device=queries.device, dtype=torch.bool), 
            diagonal=1
        )
        queries = self.temporal_consistency(queries, src_mask=temporal_causal_mask)
        
        return queries


class AttentionPooling(nn.Module):
    def __init__(
        self,
        context_dim: int,
        query_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rmsnorm: bool = True,
    ):
        """
        Attention-based pooling layer without learnable queries.
        Args:
            context_dim (int): Dimension of input embeddings.
            query_dim (int): Output dimension of pooled latent action.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Expansion ratio in MLP.
            dropout (float): Dropout rate in attention and MLP.
            use_rmsnorm (bool): Whether to use RMSNorm instead of LayerNorm.
        """
        super().__init__()
        self.context_dim = context_dim
        self.num_heads = num_heads

        # Self-attention
        self.attn = nn.MultiheadAttention(context_dim, num_heads, dropout=dropout, batch_first=True)

        # Norms
        self.norm1 = nn.RMSNorm(context_dim) if use_rmsnorm else nn.LayerNorm(context_dim)
        self.norm2 = nn.RMSNorm(context_dim) if use_rmsnorm else nn.LayerNorm(context_dim)

        # Feed-forward MLP
        hidden_dim = int(context_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

        # Projection to output latent dimension
        self.final_proj = nn.Linear(context_dim, query_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, D], N = T*K flattened time+space
        Returns:
            pooled: Tensor of shape [B, query_dim], single latent action
        """
        B, N, D = x.shape

        # Use mean of x as query for attention pooling
        # Q shape: [B, 1, D], K/V shape: [B, N, D]
        queries = x.mean(dim=1, keepdim=True)

        attn_out, _ = self.attn(queries, x, x)  # Q=queries, K=V=x
        x_pooled = self.norm1(queries + attn_out)

        # Feed-forward refinement
        mlp_out = self.mlp(x_pooled)
        x_pooled = self.norm2(x_pooled + mlp_out)

        # Project to final latent action
        return self.final_proj(x_pooled)  # [B, 1, query_dim]


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
    def __init__(self, d_model, num_heads=16, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # Cross-attention: x as query, context as key/value
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attn(x_norm, context, context)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))
        return x

class LAMDecoder_v2(nn.Module):
    """
    通过堆叠多个 DecoderBlock，将动作应用到状态上，以重建下一帧的特征。
    """
    def __init__(self, feature_dim, node_dim, input_dim: int=1024, num_layers=6, num_heads=16, dropout=0.1, skip_connection: bool = False, train_in_latent: bool = True, ffn_expansion_factor=2):
        """
        初始化 LAMDecoder。
        
        参数:
            feature_dim (int): 状态特征 f_t 的维度 (D_feat)。
            node_dim (int): 动作特征 z_q 的维度 (d)。
            input_dim (int): 输入特征的维度 (D_feat)。
            num_layers (int): DecoderBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.train_in_latent = train_in_latent
        # 解码层
        # self.dec_layers = nn.ModuleList([
        #     CrossAttentionBlock(feature_dim, num_heads=num_heads, ffn_ratio=4, dropout=dropout) 
        #     for _ in range(num_layers)
        # ])
        self.dec_layers = nn.ModuleList([nn.TransformerEncoderLayer(feature_dim, num_heads, dim_feedforward=int(feature_dim*ffn_expansion_factor), dropout=dropout, batch_first=True, norm_first=True) for _ in range(num_layers)])
        # 简化query投影：去除冗余的第二层
        self.project_query = nn.Linear(node_dim, feature_dim)
        
        # # Query self-attention增强
        # self.query_attn = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        
        # 简化输入输出投影，避免冗余
        if input_dim == feature_dim:
            self.project_input = nn.Identity()
            self.project_output = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, feature_dim)
            self.project_output = nn.Linear(feature_dim, input_dim)
        if not train_in_latent:
            self.to_pixel = nn.ConvTranspose2d(input_dim, 3, kernel_size=16, stride=16)
        
        # 添加skip connection
        self.input_skip = nn.Linear(input_dim, input_dim) if skip_connection else None
    def forward(self, f_0, z_q):
        """
        前向传播。
        
        参数:
            f_0 (torch.Tensor): 初始帧的特征，形状 [B, K, input_dim]。
            z_q (torch.Tensor): VQ量化后的动作code，形状 [B, 1, node_dim]。
            
        返回:
            torch.Tensor: 重建的最后一帧特征 f_hat_T，形状 [B, K, input_dim]。
        """
        # 保存原始输入用于skip connection
        f_0_original = f_0
        
        # 投影query并使用self-attention增强
        z_q_tokens = self.project_query(z_q)  # [B, 1, feature_dim]
        
        # 投影输入特征（只调用一次，避免冗余）
        f_0_tokens = self.project_input(f_0)  # [B, K, feature_dim]
        x = z_q_tokens + f_0_tokens
        # 通过解码层堆叠
        for layer in self.dec_layers:
            x = layer(x)
        
        # 输出投影
        reconstructed_features = self.project_output(x)
        
        # 添加原始输入的skip connection
        if self.input_skip is not None:
            reconstructed_features = reconstructed_features + self.input_skip(f_0_original)
        if not self.train_in_latent:
            B, K, D = reconstructed_features.shape
            h = w = int(K ** 0.5)
            reconstructed_features = self.to_pixel(reconstructed_features.transpose(1, 2).reshape(B, D, h, w))
        return reconstructed_features
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
        self.attn_layers = nn.ModuleList([
            Attn_Crossn_Block(
                feature_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        # 投影query到feature_dim
        self.project_query = nn.Sequential(
            nn.Linear(node_dim, 4*node_dim), 
            nn.GELU(), 
            nn.Linear(4*node_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, f_t, z_q):
        """
        前向传播。
        
        参数:
            f_t (torch.Tensor): 前一帧的特征，形状 [B, K, D_feat]。
            z_q (torch.Tensor): VQ量化后的4个动作code，形状 [B, 4, d]。
            
        返回:
            torch.Tensor: 重建的后一帧特征 f_hat_t+1，形状 [B, K, D_feat]。
        """
        # 投影动作码到feature维度
        z_q = self.project_query(z_q)
        
        # 使用f_t作为初始状态
        reconstructed_features = f_t
        
        # 通过注意力层堆叠，逐步融合动作信息
        for layer in self.attn_layers:
            reconstructed_features = reconstructed_features + layer(reconstructed_features, z_q)
        
        return reconstructed_features



# 导出所有主要的类，以便 Hydra 可以正确解析
__all__ = [
    'LAMEncoder',
    'LAMEncoder_v2', 
    'LAMDecoder_v2',
    'LAMDecoder',
]

# ---------------------------
# PatchEmbed 及其 Encoder（非预训练视觉编码器，作为回退选项）
# ---------------------------

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

# 导出 PatchEmbed
__all__.append('PatchEmbed')


def angle_difference(a_T, a_0):
    """数值稳定、方向保留的欧拉角差"""
    sin_diff = torch.sin(a_T) * torch.cos(a_0) - torch.cos(a_T) * torch.sin(a_0)
    cos_diff = torch.cos(a_T) * torch.cos(a_0) + torch.sin(a_T) * torch.sin(a_0)
    return torch.atan2(sin_diff, cos_diff)  # [-π, π]




def eef_reconstruction_loss(
    state: torch.Tensor,
    s_pred: torch.Tensor,
    pos_loss_type: str = "l1",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算末帧 EEF 状态的重建损失（位置 + 朝向 + 抓手开合）。
    
    Args:
        state: 实际状态 (..., T, 8)
        s_pred: 预测状态 (..., 8)
        loss_type: 'l1' 或 'l2'
        reduction: 'mean' | 'sum' | 'none'
    """
    # --- 提取末帧 ---
    state_T = state[..., -1, :]       # (..., 8)
    # 兼容输入形状：允许 s_pred 为 (..., 8) 或 (..., 1, 8)
    if s_pred.dim() == state_T.dim() + 1 and s_pred.shape[-2] == 1:
        s_pred = s_pred[..., 0, :]
    elif s_pred.dim() == state_T.dim():
        pass  # 已经是 (..., 8)
    else:
        raise ValueError(
            f"Unsupported s_pred shape: {tuple(s_pred.shape)}. Expected (..., 8) or (..., 1, 8)"
        )

    # --- 拆分状态分量 ---
    pos_ori_target = state_T[..., :-1]   # 位置 + 朝向 (7维)
    gripper_target = (state_T[..., -1:] > 0.0).float()  # 抓手状态 (1维)

    pos_ori_pred = s_pred[..., :-1]
    gripper_pred = s_pred[..., -1:]

    # --- 计算位置 + 朝向损失 ---
    if pos_loss_type == "l1":
        pos_ori_loss = torch.abs(pos_ori_pred - pos_ori_target)
    elif pos_loss_type == "l2":
        pos_ori_loss = (pos_ori_pred - pos_ori_target) ** 2
    else:
        raise ValueError(f"Unsupported pos_loss_type: {pos_loss_type}")

    # --- 计算抓手二分类损失 ---
    gripper_loss = F.binary_cross_entropy_with_logits(
        gripper_pred, gripper_target, reduction='none'
    )

    # --- 聚合 ---
    total_loss = pos_ori_loss.mean(dim=-1) + gripper_loss.squeeze(-1)

    if reduction == "mean":
        total_loss = total_loss.mean()
    elif reduction == "sum":
        total_loss = total_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return total_loss



class StatePredictor(nn.Module):
    """
    物理接地末端执行器状态预测器 (Physical Grounding State Predictor)
    
    将潜动作向量 z_t 与初始状态 state_0 融合，通过注意力机制预测下一时刻的 EEF 状态。
    输出直接对应于 eef_reconstruction_loss 所需的 s_pred（绝对状态）。
    
    输入:
        z_t: [B, num_queries, latent_dim]
        state_0: [B, 8]
    输出:
        s_pred: [B, 8]  (与目标状态对应)
    """

    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()

        # 归一化层
        self.norm = nn.LayerNorm(latent_dim)

        # 注意力层
        self.attn = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dropout=dropout,
            batch_first=True,
        )

        # 将 state_0 投射到相同潜空间
        self.proj_state = nn.Linear(1, latent_dim)

        # 聚合潜动作特征与状态
        self.global_aggregator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),  # 输出与状态维度对齐
        )

    def forward(self, z_t: torch.Tensor, state_0: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z_t: [B, num_queries, latent_dim]
            state_0: [B, 8]
        Returns:
            s_pred: [B, 8]
        """
        B = state_0.size(0)

        # 1️⃣ 将 state_0 投射并加入上下文
        state_embed = self.proj_state(state_0.unsqueeze(-1))  # [B, 8, latent_dim]

        # 2️⃣ 拼接潜动作 + 状态
        z_cat = torch.cat([z_t, state_embed], dim=1)  # [B, num_queries + 8, latent_dim]
        z_cat = self.norm(z_cat)

        # 3️⃣ 自注意力聚合潜动作
        z_cat = self.attn(z_cat)

        # 4️⃣ 使用状态token的输出进行预测
        state_token = z_cat[:, z_t.shape[1]:, :]  # [B, 8, latent_dim]

        # 5️⃣ 输出预测状态
        s_pred = self.global_aggregator(state_token).squeeze(-1)  # [B, 8]

        # 6️⃣ 对输出范围做约束（仅作用于前 7 个连续维度）
        s_pred = torch.cat([torch.tanh(s_pred[..., :-1]), s_pred[..., -1:]], dim=-1)

        # gripper 最后一维保持原始 logit，方便 BCEWithLogits

        return s_pred



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
        lambda_mag_reg: float = 0.5,
        motion_threshold_beta: float = 0.1,  # 1cm
        motion_scale_alpha: float = 15.0,
        huber_delta: float = 0.1
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
        
    def forward(self, delta_s_pred: torch.Tensor, delta_s_gt: torch.Tensor):
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

