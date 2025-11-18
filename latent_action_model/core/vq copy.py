import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class VQ(nn.Module):
    """
    NSVQ - Soft assignment with temperature annealing + EMA codebook update + entropy regularization.

    Features:
      - Soft assignment via temperature annealing.
      - Commitment loss (encoder-only MSE to quantized outputs).
      - Entropy regularization to avoid codebook collapse.
      - Codebook stability via EMA.
    """

    def __init__(
        self,
        codebook_size: int = 16,
        code_dim: int = 128,
        discarding_threshold: float = 0.01,
        initialization: str = "uniform",
        # data-dependent init
        data_dependent_init: bool = False,
        # temperature schedule
        tau_start: float = 10.0,
        tau_end: float = 0.9,
        anneal_steps: int = 50000,
        # EMA parameters
        ema_decay: float = 0.99,
        # 【修正1】: 重新引入 ema_eps，专门用于 EMA 更新的数值稳定
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.discarding_threshold = float(discarding_threshold)
        self.data_dependent_init = bool(data_dependent_init)

        # ---- Codebook initialization ----
        if initialization == "normal":
            codebooks_data = torch.randn(self.codebook_size, self.code_dim)
        elif initialization == "uniform":
            codebooks_data = torch.empty(self.codebook_size, self.code_dim)
            nn.init.uniform_(codebooks_data, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            raise ValueError("initialization must be 'normal' or 'uniform'")

        self.codebooks = nn.Parameter(codebooks_data, requires_grad=False)  # EMA 更新，无梯度
        self.register_buffer("node_count", torch.zeros(self.codebook_size, dtype=torch.long))
        # 首批数据驱动初始化标记
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

        # ---- Temperature scheduling ----
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_steps = int(anneal_steps)
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("tau", torch.tensor(self.tau_start, dtype=torch.float32))

        # ---- EMA buffers ----
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps) # 【修正1】
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_codebook", self.codebooks.data.clone())

    @torch.no_grad()
    def _kmeans_init(self, data: Tensor, num_clusters: int, max_iters: int = 25) -> Tensor:
        """
        使用 K-Means 聚类在首个 batch 上进行码本初始化。

        - 仅用于初始化阶段（no_grad）。
        - 对空簇采用回退到随机数据点的策略以保证稳定性。

        Args:
            data: [N, D]
            num_clusters: 目标簇数 K
            max_iters: 最大迭代次数

        Returns:
            Tensor: [num_clusters, D] 的聚类中心
        """
        if data.numel() == 0:
            return self.codebooks.data

        num_points, feature_dim = data.shape
        device = data.device

        effective_k = min(num_clusters, num_points)

        # 初始化中心（随机不重复采样）
        perm = torch.randperm(num_points, device=device)
        centers = data[perm[:effective_k]].clone()  # [k, D]

        for _ in range(max_iters):
            # 分配：按平方欧氏距离最近原则
            data_norm = (data * data).sum(dim=1, keepdim=True)                    # [N, 1]
            center_norm = (centers * centers).sum(dim=1, keepdim=True).t()        # [1, k]
            distances = data_norm + center_norm - 2.0 * data @ centers.t()        # [N, k]
            distances = torch.clamp(distances, min=0.0)
            assignments = torch.argmin(distances, dim=-1)                         # [N]

            # 重新计算每个簇的均值
            new_centers = centers.clone()
            for cluster_id in range(effective_k):
                mask = (assignments == cluster_id)
                if mask.any():
                    new_centers[cluster_id] = data[mask].mean(dim=0)
                else:
                    random_idx = torch.randint(0, num_points, (1,), device=device)
                    new_centers[cluster_id] = data[random_idx]

            center_shift = (new_centers - centers).pow(2).sum().sqrt()
            centers = new_centers
            if center_shift < 1e-6:
                break

        if effective_k < num_clusters:
            repeats = (num_clusters + effective_k - 1) // effective_k
            centers = centers.repeat(repeats, 1)[:num_clusters]

        return centers.detach()

    def entropy_loss(self, affinity: Tensor, temperature: float = 0.01) -> Tensor:
        """
        平衡样本熵与平均熵，防止码本坍塌。
        affinity: 相似度矩阵（例如负的平方距离），形状 [N, K]
        """
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        probs = F.softmax(flat_affinity / temperature, dim=-1)
        log_probs = F.log_softmax(flat_affinity / temperature + 1e-5, dim=-1)
        target_probs = probs
        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        loss = sample_entropy - avg_entropy
        return loss

    # ------------------ temperature annealing ------------------
    def _update_temperature(self):
        """Linear annealing of tau from tau_start to tau_end over anneal_steps."""
        if self.anneal_steps <= 0:
            self.tau.fill_(self.tau_end)
            return
        step = float(self.global_step.item())
        progress = min(1.0, step / float(self.anneal_steps))
        new_tau = self.tau_start * (1.0 - progress) + self.tau_end * progress
        self.tau.fill_(new_tau)


    # ------------------ EMA codebook update ------------------
    @torch.no_grad()
    def _ema_update_codebooks(self, probs_flat: Tensor, nodes_flat: Tensor):
        """
        Update codebooks using soft assignments (EMA version).
        probs_flat: [N, K]  soft assignment probabilities
        nodes_flat: [N, D]  encoder outputs
        """
        decay = self.ema_decay
        n_k = probs_flat.sum(dim=0)                      # [K]
        m_k = probs_flat.t() @ nodes_flat                # [K, D]

        # EMA updates
        self.ema_cluster_size.mul_(decay).add_(n_k * (1 - decay))
        self.ema_codebook.mul_(decay).add_(m_k * (1 - decay))

        # Normalize
        n_k_smooth = self.ema_cluster_size + self.ema_eps
        new_codebooks = self.ema_codebook / n_k_smooth.unsqueeze(1)

        self.codebooks.data.copy_(new_codebooks)

    def forward(self, nodes: Tensor, update_counts: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            nodes (Tensor): [B, S, D]
        Returns:
            quantized (Tensor): [B, S, D]
            perplexity (Tensor): scalar
            indices (Tensor): [B, S] (long)
            slot_diversity_loss (Tensor): scalar
            commitment_loss (Tensor): scalar
        """
        if nodes.dim() != 3:
            raise ValueError("nodes must be [B, S, D]")

        B, S, D = nodes.shape
        assert D == self.code_dim, f"nodes last dim {D} != code_dim {self.code_dim}"

        nodes_flat = nodes.reshape(-1, D)  # [B*S, D]

        # 首批数据驱动的 K-Means 初始化（含 DDP 广播）
        if (
            self.training
            and self.data_dependent_init
            and int(self.initialized.item()) == 0
        ):
            ddp_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
            with torch.no_grad():
                if ddp_initialized:
                    rank = torch.distributed.get_rank()
                    if rank == 0:
                        init_vectors = self._kmeans_init(nodes_flat, self.codebook_size)
                        self.codebooks.data.copy_(init_vectors)
                        # 重置 EMA 缓冲，确保与新码本一致
                        self.ema_cluster_size.fill_(1.0)
                        self.ema_codebook.data.copy_(self.codebooks.data)
                        self.initialized.fill_(1)
                    torch.distributed.broadcast(self.codebooks.data, src=0)
                    torch.distributed.broadcast(self.ema_cluster_size, src=0)
                    torch.distributed.broadcast(self.ema_codebook, src=0)
                    torch.distributed.broadcast(self.initialized, src=0)
                    torch.distributed.barrier()
                else:
                    init_vectors = self._kmeans_init(nodes_flat, self.codebook_size)
                    self.codebooks.data.copy_(init_vectors)
                    self.ema_cluster_size.fill_(1.0)
                    self.ema_codebook.data.copy_(self.codebooks.data)
                    self.initialized.fill_(1)
        distances = torch.cdist(nodes_flat, self.codebooks)  # [B*S, K]
        logits = -distances  # closer -> larger logit
        # ensure tau up-to-date (如果用 global_step 退火的话）
        if self.training:
            self._update_temperature()
            self.global_step.add_(1)

        tau_val = float(self.tau.item())

        # soft probabilities (用于 perplexity / diversity loss / 可导路径)
        probs_flat = F.softmax(logits / (tau_val + 1e-12), dim=-1)  # [B*S, K]
        probs = probs_flat.reshape(B, S, self.codebook_size)        # [B, S, K]

        # gumbel_softmax 输出在 hard=True 时是 one-hot，但梯度由内部分布近似传回
        gumbel = F.gumbel_softmax(logits, tau=tau_val, hard=True, dim=-1)  # [B*S, K]
        # sampled discrete quantized via sampled one-hot
        quantized_flat = gumbel @ self.codebooks  # [B*S, D]
        # sampled indices (用于统计/替换/日志)
        sampled_indices = gumbel.argmax(dim=-1)  # [B*S]
        indices = sampled_indices.reshape(B, S)

        quantized = quantized_flat.reshape(B, S, D)

        # 1. perplexity 基于最后实际选取的索引计算（硬分配的边际分布）
        indices_flat_for_ppx = indices.reshape(-1)  # [B*S]
        counts = torch.bincount(indices_flat_for_ppx, minlength=self.codebook_size).float()
        total = counts.sum().clamp(min=1.0)
        p_usage = counts / total
        # 熵 H = -sum_k p_k * log p_k（忽略 p_k = 0 项）
        nonzero = p_usage > 0
        entropy_all = -(p_usage[nonzero] * p_usage[nonzero].log()).sum()
        perplexity = torch.exp(entropy_all)

        # 2. 使用基于平方距离的 entropy_loss 替代 slot_diversity_loss
        nodes_norm2 = (nodes_flat * nodes_flat).sum(dim=1, keepdim=True)               # [N, 1]
        code_norm2 = (self.codebooks * self.codebooks).sum(dim=1)                      # [K]
        distances_sq = torch.clamp(nodes_norm2 + code_norm2.unsqueeze(0) - 2.0 * nodes_flat @ self.codebooks.t(), min=0.0)
        entropy_loss = self.entropy_loss(-distances_sq)

        # 3. commitment loss（只推动 encoder 输出靠近量化向量；码本用 EMA 更新）
        commitment_loss = F.mse_loss(nodes, quantized.detach())

        # EMA update (only in training)
        if self.training:
            with torch.no_grad():
                self._ema_update_codebooks(probs_flat, nodes_flat)

        # 更新统计计数（训练时使用 sampled indices，这样 tau 生效；eval 时用 argmax）
        if update_counts:
            with torch.no_grad():
                flat_indices = indices.reshape(-1).to(self.node_count.device, dtype=torch.long)
                self.node_count.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=self.node_count.dtype))


        return quantized, perplexity, indices, entropy_loss, commitment_loss


    @torch.no_grad()
    def replace_unused_codebooks(self) -> int:
        """
        【修正版】
        基于 EMA cluster size 替换未使用的码本，并正确重置其 EMA 状态。
        建议在每个 epoch 结束时调用此方法。
        """
        # 1. 识别未使用码本
        denom = self.ema_cluster_size.sum().clamp(min=1.0)
        usage_ratio = self.ema_cluster_size / denom
        unused_mask = usage_ratio < self.discarding_threshold
        used_mask = ~unused_mask

        unused_indices = torch.where(unused_mask)[0]
        used_indices = torch.where(used_mask)[0]
        num_unused = len(unused_indices)

        if num_unused == 0:
            return 0
        
        # 2. 处理替换逻辑
        if used_indices.numel() == 0:
            # 极端情况：所有码本都未使用，全部随机重置
            new_codebooks_data = torch.empty_like(self.codebooks.data)
            nn.init.uniform_(new_codebooks_data, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
            self.codebooks.data.copy_(new_codebooks_data)
            
            # 【关键】重置所有 EMA 状态
            self.ema_cluster_size.fill_(1.0) # 给每个码本一个公平的起始计数
            self.ema_codebook.data.copy_(self.codebooks.data)
            return self.codebook_size

        # 从使用过的码本中随机抽样作为替换源
        used_codebooks = self.codebooks.data[used_indices]
        num_repeats = (num_unused + len(used_codebooks) - 1) // len(used_codebooks)
        replacements = used_codebooks.repeat(num_repeats, 1)
        replacements = replacements[torch.randperm(len(replacements))][:num_unused]

        # 添加少量噪声
        codebook_std = self.codebooks.data[used_indices].std()
        noise = (torch.randn_like(replacements) * codebook_std * 0.01)
        new_vectors = replacements + noise
        
        # 3. 更新码本并重置对应的 EMA 状态
        self.codebooks.data[unused_indices] = new_vectors
        
        # 【关键】计算存活码本的平均“使用量”
        avg_cluster_size = self.ema_cluster_size[used_indices].mean()
        # 确保平均值不为0，以防万一
        avg_cluster_size = avg_cluster_size.clamp(min=1.0) 

        # 将新码本的 EMA 状态设置为一个公平的起点
        self.ema_cluster_size[unused_indices] = avg_cluster_size
        self.ema_codebook.data[unused_indices] = new_vectors * avg_cluster_size

        return num_unused


    def reset_node_count(self) -> None:
        """Reset usage counters (call e.g. after replacement or each epoch)."""
        self.node_count.zero_()

    @torch.no_grad()
    def inference(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        高效的推理函数，执行硬量化。

        该函数找到每个输入向量最近的码本向量，并返回量化后的向量和对应的索引。
        它不计算 softmax 概率或 perplexity，以实现最大性能。

        Args:
            nodes (Tensor): 输入张量，形状为 [B, S, D]。

        Returns:
            Tuple[Tensor, Tensor]:
            - quantized (Tensor): 量化后的输出张量，形状为 [B, S, D]。
            - indices (Tensor): 选中的码本索引，形状为 [B, S]。
        """
        # 1. 检查输入维度
        if nodes.dim() != 3:
            raise ValueError("输入张量 `nodes` 的维度必须是 [B, S, D]")
        
        B, S, D = nodes.shape
        
        # 2. 将输入展平以便进行矩阵运算
        nodes_flat = nodes.reshape(-1, D)
        
        # 3. 计算输入向量与所有码本向量之间的距离
        distances = torch.cdist(nodes_flat, self.codebooks)  # 形状: [B*S, K]
        
        # 4. 找到最近的码本向量的索引 (argmin on distances === argmax on -distances)
        indices_flat = torch.argmin(distances, dim=-1)  # 形状: [B*S]
        
        # 5. 使用索引直接从码本中提取量化后的向量
        quantized_flat = self.codebooks[indices_flat]    # 形状: [B*S, D]
        
        # 6. 将输出和索引的形状恢复为 [B, S, ...]
        quantized = quantized_flat.reshape(B, S, D)
        indices = indices_flat.reshape(B, S)
        
        return quantized, indices
