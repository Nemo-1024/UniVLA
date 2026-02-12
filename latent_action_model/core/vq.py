import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, Tuple, Union, List


class VAEQuantizer(nn.Module):
    """
    Continuous alternative to VQ: a lightweight VAE bottleneck.

    Interface-compatible with VQ/NSVQ/EMAVQ used by `LatentLAMModel`:
      - forward(...) returns (quantized, perplexity, indices, entropy_loss, vq_loss)
      - inference(...) returns (quantized, indices[, distances/logits/probs when requested])

    Notes:
      - `vq_loss` is the KL divergence loss (optionally weighted by `beta`)
      - `perplexity`, `indices`, `entropy_loss` are not applicable and returned as zeros / None
      - Accepts and ignores extra kwargs so existing `vq_kwargs` configs won't break.
      - Now operates directly in code_dim space (projection layers moved to encoder/decoder).
    """

    def __init__(
        self,
        code_dim: int = 128,
        beta: float = 1.0,
        clamp_logvar: Optional[float] = 10.0,
        layer_norm: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.clamp_logvar = float(clamp_logvar) if clamp_logvar is not None else None

        self.pre_norm = nn.LayerNorm(self.code_dim) if layer_norm else nn.Identity()
        self.mu = nn.Linear(self.code_dim, self.code_dim)
        self.logvar = nn.Linear(self.code_dim, self.code_dim)

        # for logging parity with VQ modules
        self.last_kl_loss: Optional[Tensor] = None
        # mutual information estimate (see forward for details)
        self.last_mutual_info: Optional[Tensor] = None
        # KL(q(z) || p(z)) term used in MI computation
        self.last_qz_kl: Optional[Tensor] = None

    def _encode(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        # Expect nodes: [B, Q, D] or [B, D]; average over query dim to a single latent
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(1)  # [B, 1, D]
        nodes_pooled = nodes.mean(dim=1, keepdim=True)  # [B, 1, D]
        h = self.pre_norm(nodes_pooled)
        mu = self.mu(h)
        logvar = self.logvar(h)
        if self.clamp_logvar is not None:
            logvar = torch.clamp(logvar, min=-self.clamp_logvar, max=self.clamp_logvar)
        return mu, logvar

    @staticmethod
    def _kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        # KL(q(z|x) || N(0, I)) = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.sum(dim=-1)  # [...], sum over latent dim

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self._encode(nodes)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterized sample
        quantized = z

        # mean KL across batch/query positions (standard VAE objective)
        kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)  # [...]
        kl_loss = kl_per_sample.mean()
        self.last_kl_loss = kl_loss.detach()

        perplexity = nodes.new_tensor(0.0)
        indices = None
        entropy_loss = nodes.new_tensor(0.0)
        vq_loss = kl_loss * self.beta
        return quantized, perplexity, indices, entropy_loss, vq_loss, mu, logvar

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific=None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: float = 1.0,
        sample: bool = False,
        return_stats: bool = False,
    ):
        # Deterministic by default: use mean. Optional sampling for analysis.
        mu, logvar = self._encode(nodes)
        if sample:
            std = (0.5 * logvar).exp() * float(temperature)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        quantized = z
        indices = None
        if return_distance or return_logits or return_probs:
            return quantized, indices, None, None, None
        if return_stats:
            return quantized, indices, mu, logvar
        return quantized, indices


class AEQuantizer(nn.Module):
    """
    Simple linear bottleneck used when vq_type='ae'.
    - Keeps interface compatible with VQ/VAE modules.
    - Now operates directly in code_dim space (projection layers moved to encoder/decoder).
    """

    def __init__(
        self,
        code_dim: int = 128,
        layer_norm: bool = False,
        codebook_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        # Align with VQ API: expose codebook_size for downstream components
        self.codebook_size = int(codebook_size) if codebook_size is not None else int(code_dim)
        self.pre_norm = nn.LayerNorm(self.code_dim) if layer_norm else nn.Identity()
        self.last_nodes_norm: Optional[Tensor] = None
        # Keep parity with VQ modules that expose this attribute
        self.nodes_norm: Optional[Tensor] = None

    def forward(self, nodes: Tensor):
        # nodes: [B, Q, D] or [B, D]
        nodes_proj = self.pre_norm(nodes)
        with torch.no_grad():
            norm_val = torch.norm(nodes_proj, p=2, dim=-1).mean()
            self.last_nodes_norm = norm_val
            self.nodes_norm = norm_val
        quantized = nodes_proj
        batch = nodes.shape[0]
        num_queries = nodes.shape[1] if nodes.dim() > 1 else 1
        indices = torch.zeros((batch, num_queries), device=nodes.device, dtype=torch.long)
        zero_scalar = nodes.new_tensor(0.0)
        perplexity = zero_scalar
        entropy_loss = zero_scalar
        vq_loss = zero_scalar
        return quantized, perplexity, indices, entropy_loss, vq_loss

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific=None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        nodes_proj = self.pre_norm(nodes)
        quantized = nodes_proj
        norm_val = torch.norm(nodes_proj, p=2, dim=-1).mean()
        self.last_nodes_norm = norm_val
        self.nodes_norm = norm_val
        batch = nodes.shape[0]
        num_queries = nodes.shape[1] if nodes.dim() > 1 else 1
        indices = torch.zeros((batch, num_queries), device=nodes.device, dtype=torch.long)
        if return_distance or return_logits or return_probs:
            return quantized, indices, None, None, None
        return quantized, indices

class VQ(nn.Module):

    def __init__(
        self,
        codebook_size: int = 1024,
        code_dim: int = 128,
        discarding_threshold: float = 0.01,
        initialization: str = 'uniform',
        data_dependent_init: bool = True,
        kmeans_iters: int = 10,
        kmeans_max_samples: int = 100000,
        beta: float = 0.25,
        orthogonal_loss_weight: float = 0.0,
        lambda_sample_entropy: float = 0.0,
        lambda_codebook_entropy: float = 0.0,
        max_code_replaced_per_step: int = 2,
        use_cosine_sim: bool=False,
        layer_norm: bool=False,
        scale: float=1.0,
        use_soft_assignment: bool = False,
        use_temperature_schedule: bool = False,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        temperature_decay_steps: int = 10000,
        *args,
        **kwargs,
    ):
        """
        初始化 VQ 模块。

        参数:
            codebook_size (int): 码本中的向量（码字）数量。
            code_dim (int): 每个码字的维度。
            discarding_threshold (float): 用于判断码字是否“未使用”的阈值。
            initialization (str): 码本的初始化方法，可选 'normal' 或 'uniform'。
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.data_dependent_init = bool(data_dependent_init)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_max_samples = int(kmeans_max_samples)
        self.beta = beta
        self.orthogonal_loss_weight = float(orthogonal_loss_weight)
        self.lambda_sample_entropy = float(lambda_sample_entropy)
        self.lambda_codebook_entropy = float(lambda_codebook_entropy)
        self.max_code_replaced_per_step = int(max_code_replaced_per_step)
        self.use_soft_assignment = bool(use_soft_assignment)
        self.use_temperature_schedule = bool(use_temperature_schedule)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature_decay_steps = int(temperature_decay_steps)
        # 基于离散码索引的 slot 重复率指标
        # - last_slot_inter_redundancy:  slot 间重复率：同一样本内，不同 slot/query 是否落在同一 code 上的平均概率
        # - last_slot_inner_redundancy:  slot 内部重复率：不同样本间，同一 slot/query 是否落在同一 code 上的平均概率
        self.last_slot_inter_redundancy = None
        self.last_slot_inner_redundancy = None
        self.scale = scale
        if self.use_temperature_schedule:
            self.register_buffer("temperature_step", torch.tensor(0, dtype=torch.long))
            self.register_buffer("last_temperature", torch.tensor(float(self.temperature_start)))
        else:
            self.temperature_step = None
            self.last_temperature = torch.tensor(float(self.temperature_start))
        # 初始化码本参数
        if initialization == 'normal':
            codebooks_data = torch.randn(self.codebook_size, self.code_dim)
        elif initialization == 'uniform':
            codebooks_data = torch.empty(self.codebook_size, self.code_dim)
            nn.init.uniform_(codebooks_data, -1 / self.codebook_size, 1 / self.codebook_size)
        else:
            raise ValueError("初始化方法应为 'normal' 或 'uniform' 之一")
        
        self.codebooks = nn.Parameter(codebooks_data)
        self.use_cosine_sim = use_cosine_sim
        # 仅保留 in_norm 用于可选的层归一化（投影层已移至 encoder/decoder）
        self.in_norm = nn.LayerNorm(self.code_dim, elementwise_affine=not self.use_cosine_sim) if (layer_norm or self.use_cosine_sim) else nn.Identity()
        # 注册码字使用计数器作为缓冲区
        # 这使得它成为模块状态的一部分，并能随模块移动到不同设备
        self.register_buffer('node_count', torch.zeros(self.codebook_size, dtype=torch.long))
        # KMeans 初始化相关的状态
        # 标记是否已经完成过 KMeans 初始化
        # self.register_buffer('kmeans_initialized', torch.tensor(0, dtype=torch.bool))
        # self.initialized = torch.tensor(0, dtype=torch.bool)
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
        # 若未启用数据依赖初始化，则视为“已初始化”，允许后续维护逻辑运行
        if not self.data_dependent_init:
            self.initialized.fill_(True)

    def _norm_nodes(self, nodes: Tensor) -> Tensor:
        """归一化输入节点。"""
        return F.normalize(nodes, p=2, dim=-1, eps=self.eps)

    def get_nodes_proj(self, nodes: Tensor) -> Tensor:
        """
        返回经过 in_norm 处理后的 nodes（与 forward 中一致）。
        不修改内部状态，仅作纯函数计算。
        """
        return self.in_norm(nodes)

    def _get_indices(self, nodes: Tensor) -> Tensor:
        """计算输入节点与码本之间的最近索引（平方欧氏距离或负余弦相似度）。"""
        distances = self._compute_distance(nodes, self.codebooks)
        return torch.argmin(distances, dim=-1)

    def _compute_distance(
        self,
        inputs: Tensor,
        codebook: Tensor,
        return_normed: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        计算 inputs 与 codebook 之间的距离/相似度矩阵：
        - use_cosine_sim=True: 返回负余弦相似度（可与 argmin 配合）；可返回归一化后的 inputs/codebook
        - 否则：返回欧氏距离平方
        """
        if self.use_cosine_sim:
            inputs_normed = F.normalize(inputs, dim=-1, eps=self.eps)
            code_normed = F.normalize(codebook, dim=-1, eps=self.eps)
            sim = inputs_normed @ code_normed.t()
            dist = -sim
            return (dist, inputs_normed, code_normed) if return_normed else dist
        inputs_norm = (inputs * inputs).sum(dim=-1, keepdim=True)
        code_norm = (codebook * codebook).sum(dim=-1)
        dist = inputs_norm + code_norm - 2.0 * inputs @ codebook.t()
        dist = torch.clamp(dist, min=0.0)
        return (dist, inputs, codebook) if return_normed else dist

    @torch.no_grad()
    def _update_slot_redundancy_metrics(self, min_indices: Tensor) -> None:
        """
        基于离散码索引，更新 slot 相关重复率指标（完全矩阵运算）：
        - last_slot_inner_redundancy:  Slot 内重复率（跨样本）:
              对每个 slot s：1 - (#unique code across batch) / B，再在 s 维求均值
        - last_slot_inter_redundancy:  Slot 间重复率（单样本内部）:
              对每个样本 b：1 - (#unique code across slots) / Q，再在 b 维求均值
        """
        # 兼容 1D/2D/3D 输入：视倒数第二维为 slot/query 维
        if min_indices.dim() == 1:
            B, Q = min_indices.shape[0], 1
            idx_codes = min_indices.view(B, 1)
        else:
            B, Q = min_indices.shape[0], min_indices.shape[1]
            idx_codes = min_indices

        if B == 0 or Q == 0:
            self.last_slot_inter_redundancy = None
            self.last_slot_inner_redundancy = None
            return

        K = self.codebook_size
        # one-hot 编码: [B, Q, K]
        one_hot = F.one_hot(idx_codes, num_classes=K).to(torch.float32)

        # 1) Slot 内重复率（跨样本）
        #    对每个 slot s，看该列在 batch 上用了多少个不同 code
        #    unique_per_slot: [Q]
        slot_code_counts = one_hot.sum(dim=0)                 # [Q, K]  
        unique_per_slot = (slot_code_counts > 0).sum(dim=-1)  # [Q]
        dup_per_slot = 1.0 - unique_per_slot.to(torch.float32) / float(B)  # [Q]
        self.last_slot_inner_redundancy = dup_per_slot.mean()

        # 2) Slot 间重复率（单样本内部）
        #    对每个样本 b，看该行上用了多少个不同 code
        #    unique_per_sample: [B]
        sample_code_counts = one_hot.sum(dim=1)                   # [B, K]
        unique_per_sample = (sample_code_counts > 0).sum(dim=-1)  # [B]
        dup_per_sample = 1.0 - unique_per_sample.to(torch.float32) / float(Q)  # [B]
        self.last_slot_inter_redundancy = dup_per_sample.mean()

    @torch.no_grad()
    def kmeans_init(
        self,
        nodes: Tensor,
        num_iters: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        使用 KMeans 对码本进行数据依赖的初始化。

        在函数内部会先进行跨节点样本聚合（若处于分布式环境中），随后运行 KMeans，
        最后将得到的码本参数广播到各个节点。

        参数:
            nodes (Tensor): 当前 batch 的特征，形状 [N, D] 或 [B, *, D]，内部自动 reshape。
            num_iters (int, 可选): KMeans 迭代次数，默认使用初始化时的 kmeans_iters。
            max_samples (int, 可选): 用于 KMeans 的最大样本数，默认使用 kmeans_max_samples。
        """

        # 将输入展平为 [N, D]
        nodes = nodes.reshape(-1, self.code_dim)
        # 一些上游操作（如 permute/view 等）可能产生非连续张量
        # DDP 的 all_gather 要求参与通信的张量必须是 contiguous
        nodes = nodes.contiguous()
        if nodes.numel() == 0:
            return

        # 使用配置中的默认值
        num_iters = int(num_iters) if num_iters is not None else int(self.kmeans_iters)
        max_samples = int(max_samples) if max_samples is not None else int(self.kmeans_max_samples)

        # 跨进程聚合样本
        ddp_available = dist.is_available() and dist.is_initialized()
        if ddp_available:
            world_size = dist.get_world_size()
            # all_gather 需要每个 rank 拥有相同 shape 的 tensor
            # 这里假设各 rank batch size 接近，可直接 all_gather
            gathered = [torch.zeros_like(nodes) for _ in range(world_size)]
            dist.all_gather(gathered, nodes)
            all_nodes = torch.cat(gathered, dim=0)
            rank = dist.get_rank()
        else:
            all_nodes = nodes
            rank = 0

        # 随机子采样，避免样本过多导致 KMeans 过慢
        if all_nodes.shape[0] > max_samples:
            perm = torch.randperm(all_nodes.shape[0], device=all_nodes.device)
            all_nodes = all_nodes[perm[:max_samples]]

        K = self.codebook_size
        N = all_nodes.shape[0]

        if N == 0:
            return

        # 若样本数小于码字数，重复采样以凑够 K 个中心
        if N < K:
            extra_indices = torch.randint(0, N, (K - N,), device=all_nodes.device)
            all_nodes = torch.cat([all_nodes, all_nodes[extra_indices]], dim=0)
            N = all_nodes.shape[0]

        # 仅在 rank 0 上执行 KMeans，随后广播
        if rank == 0:
            print("=" * 50, "\n")
            print("Starting VQ process...")
            print("KMeans initializing codebooks... \n")
            print("=" * 50, "\n")
            # 随机选取 K 个样本作为初始中心
            init_perm = torch.randperm(N, device=all_nodes.device)
            centers = all_nodes[init_perm[:K]]  # [K, D]

            for _ in range(num_iters):
                # 计算每个样本到每个中心的距离并分配簇
                nodes_norm = (all_nodes * all_nodes).sum(dim=1, keepdim=True)          # [N, 1]
                centers_norm = (centers * centers).sum(dim=1)                          # [K]
                dist2 = nodes_norm + centers_norm.unsqueeze(0) - 2.0 * all_nodes @ centers.t()
                dist2 = torch.clamp(dist2, min=0.0)
                assignment = torch.argmin(dist2, dim=1)                                 # [N]

                # one-hot 编码后用矩阵乘实现分组求均值
                one_hot = F.one_hot(assignment, num_classes=K).to(all_nodes.dtype)      # [N, K]
                counts = one_hot.sum(dim=0).clamp(min=1.0).unsqueeze(-1)                # [K, 1]
                centers = (one_hot.t() @ all_nodes) / counts                            # [K, D]

        # 将 KMeans 得到的中心写入码本（仅 rank0）
        if rank == 0:
            centers_converted = centers.to(device=self.codebooks.data.device, dtype=self.codebooks.data.dtype)
            if self.use_cosine_sim:
                centers_converted = F.normalize(centers_converted, dim=-1, eps=self.eps)
            self.codebooks.data.copy_(centers_converted)

        # 将 rank0 上的码本广播到所有 rank
        if ddp_available:
            dist.broadcast(self.codebooks.data, src=0)

        # 标记已完成 KMeans 初始化
        self.initialized.fill_(True)
        
        # 重置 node_count，因为 codebook 已被重新初始化，旧的统计不再有效
        # 这确保后续的 codebook replacement 基于新的 codebook 使用情况
        self.reset_node_count()

    @torch.no_grad()
    def _maybe_data_dependent_init(self, nodes_code: Tensor) -> None:
        """
        在训练开始时触发一次基于数据的 KMeans 初始化（若启用，且仅一次）。
        """
        if not self.data_dependent_init:
            return
        if bool(self.initialized.item()):
            return
        # 使用当前 batch 的特征进行一次 KMeans 初始化（仅一次）
        self.kmeans_init(nodes_code)

    def entropy_loss(
        self,
        affinity: Tensor,
        temperature: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """
        输入:
            affinity: [B, T, K]  相似度（负距离）
        功能:
            - 鼓励 sample-wise 分布变尖锐 (低 entropy)
            - 鼓励 batch-wise 码本使用均匀 (KL(avg_probs || uniform))
        """
        assert affinity.dim() == 3, "affinity must be [B, T, K]"

        # 1. 防止超低温导致数值爆炸
        if temperature is None:
            temperature = 0.1 if self.use_cosine_sim else 1.0
        logits = affinity / temperature

        # 2. softmax 概率
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # ----------------------------------------------------
        # Part 1: Per-sample entropy（鼓励每个样本的整体分布更“自信”）
        # ----------------------------------------------------
        # 对时间维做平均，得到每个样本整体的 code 使用分布 [B, K]
        per_sample_probs = probs.mean(dim=1).clamp(min=self.eps, max=1.0 - self.eps)  # [B, K]
        per_sample_log_probs = torch.log(per_sample_probs)
        # 每个样本的熵，然后在 batch 上取均值
        per_sample_entropy = -torch.mean(torch.sum(per_sample_probs * per_sample_log_probs, dim=-1))

        # ----------------------------------------------------
        # Part 2: Codebook entropy（鼓励在 batch 内均匀使用所有 code）
        # ----------------------------------------------------
        # 先在 batch 维上取平均，得到本进程上的平均使用分布 [K]
        avg_prob = per_sample_probs.mean(dim=0)  # [K]
        avg_prob = avg_prob.clamp(min=self.eps, max=1.0 - self.eps)
        codebook_entropy = -torch.sum(avg_prob * torch.log(avg_prob))

        # ----------------------------------------------------
        # Final loss（类似 DiVAE / DiVQ 的多样性正则）
        # 1. per_sample_entropy 被压低（每个样本输出更“自信”）
        # 2. codebook_entropy 被抬高（在 batch 内更均匀地使用所有 code）
        #    实现方式：在 loss 中减去一个带权重的 codebook_entropy
        # ----------------------------------------------------
        self.last_sample_entropy = per_sample_entropy.detach()
        self.last_codebook_entropy = codebook_entropy.detach()

        entropy_aux_loss = (
            self.lambda_sample_entropy * per_sample_entropy - self.lambda_codebook_entropy * codebook_entropy
        )
        return entropy_aux_loss





    @torch.no_grad()
    def _compute_batch_perplexity(self, min_indices: Tensor) -> Tensor:
        """
        使用当前 batch 的 code 使用分布计算即刻 perplexity（无滑动窗口）。
        """
        if min_indices.numel() == 0:
            return torch.tensor(0.0, device=self.codebooks.device)
        counts = torch.bincount(min_indices.reshape(-1), minlength=self.codebook_size).to(self.codebooks.device)
        total = counts.sum()
        if total.item() == 0:
            return torch.tensor(0.0, device=self.codebooks.device)
        probs = (counts.float() / total.float()).clamp(min=self.eps, max=1.0)
        return torch.exp(-torch.sum(probs * torch.log(probs)))

    @torch.no_grad()
    def _compute_avg_unique_codes(self, min_indices: Tensor) -> Tensor:
        """
        统计每个样本在一个 batch 中所使用到的不同 code 的个数，
        并返回当前 batch 的平均唯一 code 数。

        参数:
            min_indices: 量化得到的 code 索引，形状为 [B, ...]（例如 [B, T]）。
        """
        if min_indices.numel() == 0:
            avg_unique = torch.tensor(0.0, device=self.codebooks.device)
            self.last_avg_unique_codes = avg_unique
            return avg_unique

        batch_size = min_indices.shape[0]
        # 展平除 batch 维以外的所有维度，从而统计每个样本在整个序列中用到多少个不同的 code
        flat_indices = min_indices.view(batch_size, -1)  # [B, L]

        # one-hot 到 code 维度，然后在时间维上做 any，最后统计每个样本被激活的 code 种类数
        # 注意：由于处于 no_grad 环境中，这个计算不会参与反向传播
        one_hot = F.one_hot(flat_indices, num_classes=self.codebook_size)  # [B, L, K]
        used_mask = one_hot.any(dim=1)  # [B, K]，每个样本哪些 code 被使用过
        unique_counts_tensor = used_mask.sum(dim=1).float()  # [B]
        avg_unique = unique_counts_tensor.mean()  # 标量

        # 作为最近一次 forward 的统计量保存下来，便于外部日志记录
        self.last_avg_unique_codes = avg_unique
        return avg_unique

    def orthogonality_loss(self) -> Tensor:
        """鼓励码本向量正交，提升多样性。"""
        if self.codebook_size <= 1:
            return self.codebooks.new_zeros(())
        normalized_codebooks = F.normalize(self.codebooks, dim=1, p=2, eps=self.eps)
        gram = normalized_codebooks @ normalized_codebooks.t()
        identity = torch.eye(self.codebook_size, device=gram.device, dtype=gram.dtype)
        off_diag = gram - identity
        loss = (off_diag.pow(2).sum() - torch.diagonal(off_diag).pow(2).sum()) / (self.codebook_size * (self.codebook_size - 1))
        return loss.clamp_min(0.0)


    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练阶段的前向传播。
        """
        nodes_proj = self.in_norm(nodes)
        with torch.no_grad():
            # 记录用于距离计算的表征范数（余弦模式下为 1，欧氏模式下为投影空间范数）
            self.nodes_norm = torch.norm(nodes_proj, p=2, dim=-1).mean()
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        # 在训练早期，根据数据进行一次可选的 KMeans 初始化（余弦模式使用归一化后的特征）
        self._maybe_data_dependent_init(nodes_for_init)
        # 计算距离/相似度用于后续指标/损失
        distances, nodes_used, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)
        # temperature scheduling only when enabled
        base_temperature = torch.tensor(self.temperature_start, device=self.codebooks.device, dtype=self.codebooks.dtype)
        temperature = self._get_temperature() if self.use_temperature_schedule else base_temperature
        if self.use_soft_assignment:
            # soft assignment over codebooks
            logits = (-distances) / temperature
            soft_weights = F.softmax(logits, dim=-1)
            soft_weights = soft_weights.clamp(min=self.eps, max=1.0)
            quantized_soft = torch.matmul(soft_weights.view(-1, self.codebook_size), code_used).view(*distances.shape[:-1], self.code_dim)
        else:
            quantized_soft = None

        min_indices = torch.argmin(distances, dim=-1)
        hard_quantized = F.embedding(min_indices, code_used)
        # 基于离散索引计算 slot 重复率指标（不参与梯度）
        self._update_slot_redundancy_metrics(min_indices)

        # 统计每个样本在当前 batch 中使用到的唯一 code 数量，并记录 batch 内的平均值
        self._compute_avg_unique_codes(min_indices)

        # 2. VQ 损失 (含 Straight-Through)
        # 使用投影前后的空间做损失：codebook_loss 对投影后的表征，commitment 对原空间
        codebook_loss = F.mse_loss(hard_quantized, nodes_used.detach())
        commitment_loss = F.mse_loss(hard_quantized.detach(), nodes_used)
        self.last_commitment_loss = commitment_loss.detach()
        quantized_st = nodes_used + (hard_quantized - nodes_used).detach()
        if self.use_soft_assignment and quantized_soft is not None:
            # forward uses soft assignment, backward follows straight-through hard path
            quantized = quantized_st + (quantized_soft - quantized_st).detach()
        else:
            quantized = quantized_st
        if self.orthogonal_loss_weight > 0:
            orth_loss = self.orthogonality_loss()
            self.last_orthogonal_loss = orth_loss.detach()
            vq_loss = codebook_loss + self.beta * commitment_loss + self.orthogonal_loss_weight * orth_loss
        else:
            vq_loss = codebook_loss + self.beta * commitment_loss
            self.last_orthogonal_loss = None

        # 3. 计算困惑度 (Perplexity) —— 仅基于当前 batch
        perplexity = self._compute_batch_perplexity(min_indices)

        # 5. 计算 entropy_loss
        entropy_temp = temperature if (self.use_temperature_schedule or self.use_soft_assignment) else None
        entropy_loss = self.entropy_loss(-distances, temperature=entropy_temp)
        self.node_count.index_add_(0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype))
        # 码本间距统计
        self.compute_inter_code_stats()

        quantized_out = quantized

        return (
            quantized_out,
            perplexity,
            min_indices,
            entropy_loss,
            vq_loss
        )

    @torch.no_grad()
    def _get_temperature(self) -> Tensor:
        """
        Exponentially decayed temperature: start -> end over decay steps.
        """
        if self.temperature_decay_steps <= 0:
            current = self.temperature_end
        else:
            step = min(int(self.temperature_step.item()), self.temperature_decay_steps)
            ratio = step / float(self.temperature_decay_steps)
            decay = (self.temperature_end / self.temperature_start) ** ratio
            current = self.temperature_start * decay
            current = max(self.temperature_end, current)
        temp_tensor = torch.tensor(current, device=self.codebooks.device, dtype=self.codebooks.dtype)
        if self.training and self.use_temperature_schedule:
            self.temperature_step.add_(1)
        self.last_temperature = temp_tensor
        return temp_tensor

    @torch.no_grad()
    def compute_inter_code_stats(self) -> Tuple[Tensor, Tensor]:
        """
        计算码本两两距离/角度的最小非零值与平均值。
        - use_cosine_sim=True：使用余弦相似度 -> 角度 = arccos(sim)
        - use_cosine_sim=False：使用欧氏距离平方根（与 torch.cdist 一致）
        返回: (min_value, avg_value)
        """
        if self.codebook_size <= 1:
            zero = torch.tensor(0.0, device=self.codebooks.device)
            return zero, zero
        if self.use_cosine_sim:
            code_normed = F.normalize(self.codebooks, dim=-1, eps=self.eps)
            sim = code_normed @ code_normed.t()
            eye = torch.eye(self.codebook_size, device=sim.device, dtype=torch.bool)
            sim_no_diag = sim.masked_fill(eye, -1.1)  # remove self
            max_sim = sim_no_diag.max()  # 最小角度对应最大相似度
            # 角度（弧度），裁剪避免数值问题
            angles = torch.arccos(torch.clamp(sim_no_diag, -1.0 + 1e-6, 1.0 - 1e-6))
            min_angle = torch.arccos(torch.clamp(max_sim, -1.0 + 1e-6, 1.0 - 1e-6))
            avg_angle = angles.mean()
            self.last_min_inter_code_dist = min_angle
            self.last_avg_inter_code_dist = avg_angle
            return min_angle, avg_angle
        else:
            dists = torch.cdist(self.codebooks, self.codebooks)
            # mask self-distance
            masked = dists.where(dists > 1e-6, torch.tensor(float("inf"), device=dists.device, dtype=dists.dtype))
            min_dist = masked.min()
            # exclude diagonal for mean
            off_diag = dists[~torch.eye(self.codebook_size, dtype=torch.bool, device=dists.device)]
            avg_dist = off_diag.mean()
            self.last_min_inter_code_dist = min_dist
            self.last_avg_inter_code_dist = avg_dist
            return min_dist, avg_dist

    @torch.no_grad()
    def replace_unused_codebooks(self) -> Tuple[int, Tensor]:
        """
        替换在最近的训练迭代中长时间未被使用的码本条目。
        这有助于防止码本坍缩（部分码字永远得不到训练）。

        逻辑:
            1. 先通过 discarding_threshold 找出“未使用”的码字；
            2. 再根据一个（可选的）上限控制 **本次最多重置多少个码字**。

        若在实例上设置属性 `self.max_codebooks_replaced_per_step` 且为正数，
        则本次实际重置的数量为 `min(未使用码字数, max_codebooks_replaced_per_step)`；
        若未设置该属性，则默认不额外限制。

        返回:
            int: 本次实际被替换的码字数量。
        """
        # 确保只在 kmeans 初始化完成后才执行 replacement
        # 避免在初始化前使用无效的 node_count 统计
        if not bool(self.initialized.item()):
            return 0, torch.empty(0, device=self.codebooks.device, dtype=torch.long)

        # std = self.codebooks.data.std()
        # eps_noise = 0.01 * torch.clamp(std, min=1e-5)
        eps_noise = 1e-10  # 自适应噪声，保证不为0

        # 1) 汇总全局 node_count（DDP 下跨设备累加）
        ddp_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        if ddp_initialized:
            global_node_count = self.node_count.clone()
            torch.distributed.all_reduce(global_node_count, op=torch.distributed.ReduceOp.SUM)
            rank = torch.distributed.get_rank()
        else:
            global_node_count = self.node_count
            rank = 0

        total_count = global_node_count.sum()
        # 避免除 0：total=0 时将分母视为 1，使比率为 0
        denom = total_count.clamp(min=1).float()

        # 2) 基于全局使用频率判断未使用码字
        usage_ratio = global_node_count.float() / denom
        unused_mask = usage_ratio < self.discarding_threshold
        used_mask = ~unused_mask

        unused_indices = torch.where(unused_mask)[0]
        used_indices = torch.where(used_mask)[0]

        # rank0 负责修改与日志，随后广播参数与返回值
        if ddp_initialized:
            num_replaced_tensor = torch.zeros(1, device=self.codebooks.device, dtype=torch.long)
            replaced_buf = torch.full((self.codebook_size,), -1, device=self.codebooks.device, dtype=torch.long)

        num_unused = int(unused_indices.numel())
        num_replaced = 0  # 实际被替换的数量（受上限控制）

        replaced_indices_out = torch.full((self.codebook_size,), -1, device=self.codebooks.device, dtype=torch.long)

        if rank == 0:
            if num_unused == 0:
                print("=" * 50, "\n")
                print("No unused codebooks to replace. global_node_count: ", global_node_count)
            else:
                print("=" * 50, "\n")
                print("global_node_count", global_node_count)

                # 2.1 计算本次实际要替换的数量（加入上限）
                max_per_step = self.max_code_replaced_per_step
                if max_per_step is not None and int(max_per_step) > 0:
                    max_per_step = int(max_per_step)
                    num_replaced = min(num_unused, max_per_step)
                else:
                    num_replaced = num_unused

                if num_replaced == 0:
                    # 有未使用码字，但上限为 0 或其他原因导致本次不替换
                    print(
                        f"Found {num_unused} unused codebooks, "
                        f"but max_codebooks_replaced_per_step == 0, skip replacement."
                    )
                elif used_indices.numel() == 0:
                    # 所有码字都未被使用：添加少量噪声以重新激活
                    self.codebooks.data += eps_noise * torch.randn_like(self.codebooks.data)
                    print("All codebooks are unused, adding noise to reactivate")
                else:
                    # 根据计数占比进行重要性采样来替换未使用的码本
                    used_counts = global_node_count[used_indices]  # 使用过的码字的计数
                    # 计算概率分布（基于计数占比）
                    used_counts_sum = used_counts.sum().float()
                    if used_counts_sum > 0:
                        probs = used_counts.float() / used_counts_sum
                    else:
                        # 如果所有计数都为0，则使用均匀分布
                        probs = torch.ones_like(used_counts, dtype=torch.float) / used_indices.numel()

                    # 在未使用码字中，优先选择 **使用次数最少** 的那几个进行重置
                    # 根据 global_node_count 对 unused_indices 排序（从小到大）
                    unused_counts = global_node_count[unused_indices]
                    _, sort_idx = torch.sort(unused_counts, descending=False)
                    unused_indices = unused_indices[sort_idx[:num_replaced]]

                    # 使用重要性采样（允许重复采样）
                    sampled_indices = torch.multinomial(probs, num_replaced, replacement=True)
                    # 将采样索引映射回原始码本索引
                    sampled_used_indices = used_indices[sampled_indices]
                    # 获取采样得到的码字
                    replacements = self.codebooks.data[sampled_used_indices]
                    # 添加少量噪声以增加多样性
                    noise = eps_noise * torch.randn_like(replacements)
                    self.codebooks.data[unused_indices] = replacements + noise
                    print("=" * 50)
                    print(
                        "Replaced {} unused codebooks (found {}, cap: {}) using importance sampling based on node_count".format(
                            num_replaced, num_unused, max_per_step if max_per_step is not None else -1
                        )
                    )
                    replaced_indices_out[:num_replaced] = unused_indices

            if ddp_initialized:
                num_replaced_tensor.fill_(num_replaced)

        # 3) DDP：将新的 codebooks 广播到所有进程，并同步返回值
        if ddp_initialized:
            torch.distributed.broadcast(self.codebooks.data, src=0)
            torch.distributed.barrier()
            torch.distributed.broadcast(num_replaced_tensor, src=0)
            torch.distributed.broadcast(replaced_indices_out, src=0)
            return int(num_replaced_tensor.item()), replaced_indices_out[replaced_indices_out >= 0]
        else:
            return num_replaced, replaced_indices_out[replaced_indices_out >= 0]

    def reset_node_count(self) -> None:
        """重置码字使用计数器。通常在一个 epoch 或一轮替换操作后调用。"""

        # print("Resetting node count")
        # print("=" * 50)
        self.node_count.zero_()

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific: Optional[Union[int, List[int]]] = None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        推理（或评估）阶段的量化。
        使用与训练一致的投影与距离定义，返回最近的码字，并在需要时返回距离/softmax。
        
        参数:
            nodes (Tensor): 输入张量，形状为 [B, ..., D]。
            user_specific (Optional[Union[int, List[int]]]): 
                如果提供，将忽略最近邻搜索，强制使用指定的索引。
                - int: 所有输入都使用这一个索引。
                - List[int]: 为批次中的每个输入使用全部索引。
            return_distance (bool): 若为 True，返回 nodes 与 codebook 的距离矩阵。
            return_logits (bool): 若为 True，返回基于距离的 logits（-dist / temperature）。
            return_probs (bool): 若为 True，返回 logits 的 softmax 概率。
            temperature (Optional[float]): logits 的温度；若 None 则使用 1.0。
        
        返回:
            Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
            - quantized (Tensor): 量化后的向量（即最近的码字）。
            - indices (Tensor): 对应的码字索引。
            - distances (Tensor | None): 若 return_distance，则为距离/负相似度矩阵。
            - logits (Tensor | None): 若 return_logits，则为距离转换的 logits。
            - probs (Tensor | None): 若 return_probs，则为 logits 的 softmax。
        """
        batch_size = nodes.shape[0]
        nodes_proj = self.in_norm(nodes)

        distances: Optional[Tensor] = None
        code_used: Optional[Tensor] = None

        if user_specific is not None:
            if isinstance(user_specific, list):
                base_indices = torch.tensor(user_specific, device=nodes.device, dtype=torch.long)
                if base_indices.dim() != 1:
                    raise ValueError("user_specific as list must be a 1D list of indices.")
                min_indices = base_indices.unsqueeze(0).expand(batch_size, -1)
            else:
                min_indices = torch.full((batch_size,), user_specific, device=nodes.device, dtype=torch.long)
            code_used = F.normalize(self.codebooks, dim=-1, eps=self.eps) if self.use_cosine_sim else self.codebooks
        else:
            distances, nodes_used, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)
            min_indices = torch.argmin(distances, dim=-1)

        if (return_distance or return_logits or return_probs) and distances is None:
            # 当 user_specific 不为 None 时，需要额外计算距离以供 soft 输出使用
            distances, _, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)

        quantized = F.embedding(min_indices, code_used)
        quantized_out = quantized

        logits_out: Optional[Tensor] = None
        probs_out: Optional[Tensor] = None

        if distances is not None and (return_distance or return_logits or return_probs):
            temp = float(temperature) if temperature is not None else 1.0
            logits = -distances / temp
            logits_out = logits if return_logits else None
            if return_probs:
                probs_out = F.softmax(logits, dim=-1)

        if return_distance or return_logits or return_probs:
            return (
                quantized_out.reshape(batch_size, -1, self.code_dim),
                min_indices.view(batch_size, -1),
                distances,
                logits_out,
                probs_out,
            )

        return (
            quantized_out.reshape(batch_size, -1, self.code_dim),
            min_indices.view(batch_size, -1),
        )
    def codebook_reinit(self) -> None:
        """
        （辅助功能）完全重新初始化码本和计数器。
        """
        if isinstance(self.codebooks, nn.Parameter):
            nn.init.uniform_(self.codebooks.data, -1 / self.codebook_size, 1 / self.codebook_size)
        self.reset_node_count()

    def get_codebooks(self) -> Tensor:
        return self.codebooks
    
    def get_codebook_size(self) -> int:
        return self.codebook_size
    
class EMAVQ(VQ):
    def __init__(
        self,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(self.codebook_size, self.code_dim))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))
        # Codebook weights are maintained by EMA only
        self.codebooks.requires_grad_(False)

    @torch.no_grad()
    def _init_ema_from_codebooks(self) -> None:
        """Seed EMA statistics from current codebooks."""
        codebooks_data = self.codebooks.data
        if self.use_cosine_sim:
            codebooks_data = F.normalize(codebooks_data, dim=-1, eps=self.eps)
            self.codebooks.data.copy_(codebooks_data)
        self.ema_cluster_size.fill_(self.ema_eps)
        self.ema_embed_sum.copy_(codebooks_data)
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, min_indices: Tensor, nodes: Tensor) -> None:
        """EMA update of codebooks using current assignments."""
        if min_indices.numel() == 0:
            return
        flat_indices = min_indices.reshape(-1)
        flat_nodes = nodes.reshape(-1, self.code_dim)
        encodings = F.one_hot(flat_indices, num_classes=self.codebook_size).to(flat_nodes.dtype)
        # Local batch stats
        cluster_size = encodings.sum(dim=0)
        embed_sum = encodings.t() @ flat_nodes
        # All-reduce for DDP
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size)
            dist.all_reduce(embed_sum)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1.0 - self.ema_decay))
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum * (1.0 - self.ema_decay))
        denom = self.ema_cluster_size + self.ema_eps * float(self.codebook_size)
        updated_codebook = self.ema_embed_sum / denom.unsqueeze(1)
        if self.use_cosine_sim:
            updated_codebook = F.normalize(updated_codebook, dim=-1, eps=self.eps)
        self.codebooks.data.copy_(updated_codebook)

    @torch.no_grad()
    def kmeans_init(
        self,
        nodes: Tensor,
        num_iters: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().kmeans_init(nodes, num_iters=num_iters, max_samples=max_samples)
        self._init_ema_from_codebooks()

    @torch.no_grad()
    def codebook_reinit(self) -> None:
        super().codebook_reinit()
        self._init_ema_from_codebooks()

    @torch.no_grad()
    def replace_unused_codebooks(self) -> Tuple[int, Tensor]:
        """
        在基类替换后，同步刷新 EMA 缓冲区，避免旧统计在下一次 EMA 更新时覆盖新 codebook。
        """
        num_replaced, replaced_indices = super().replace_unused_codebooks()
        if num_replaced > 0 and replaced_indices.numel() > 0:
            # 仅刷新被替换的条目对应的 EMA 统计
            codebooks_slice = self.codebooks.data[replaced_indices]
            if self.use_cosine_sim:
                codebooks_slice = F.normalize(codebooks_slice, dim=-1, eps=self.eps)
                self.codebooks.data[replaced_indices] = codebooks_slice
            # 用正常权重播种，避免极小权重导致新码被旧统计拖回
            self.ema_cluster_size[replaced_indices] = 1.0
            self.ema_embed_sum[replaced_indices] = codebooks_slice
        return num_replaced, replaced_indices

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Data-dependent init (k-means) if needed
        nodes_proj = self.in_norm(nodes)
        with torch.no_grad():
            self.nodes_norm = torch.norm(nodes_proj, p=2, dim=-1).mean()
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        self._maybe_data_dependent_init(nodes_for_init)
        with torch.no_grad():
            if not bool(self.ema_initialized.item()):
                self._init_ema_from_codebooks()

        # Nearest neighbors (cosine 或欧氏)
        distances, nodes_used, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)
        base_temperature = torch.tensor(self.temperature_start, device=self.codebooks.device, dtype=self.codebooks.dtype)
        temperature = self._get_temperature() if self.use_temperature_schedule else base_temperature
        if self.use_soft_assignment:
            logits = (-distances) / temperature
            soft_weights = F.softmax(logits, dim=-1)
            soft_weights = soft_weights.clamp(min=self.eps, max=1.0)
            quantized_soft = torch.matmul(soft_weights.view(-1, self.codebook_size), code_used).view(*distances.shape[:-1], self.code_dim)
        else:
            quantized_soft = None

        min_indices = torch.argmin(distances, dim=-1)
        hard_quantized = F.embedding(min_indices, code_used)

        # Metrics
        self._update_slot_redundancy_metrics(min_indices)
        self._compute_avg_unique_codes(min_indices)

        # EMA update
        self._ema_update(min_indices, nodes_used)

        # Losses
        commitment_loss = F.mse_loss(hard_quantized.detach(), nodes_used)
        self.last_commitment_loss = commitment_loss.detach()
        quantized_st = nodes_used + (hard_quantized - nodes_used).detach()
        if self.use_soft_assignment and quantized_soft is not None:
            quantized = quantized_st + (quantized_soft - quantized_st).detach()
        else:
            quantized = quantized_st
        vq_loss = self.beta * commitment_loss

        perplexity = self._compute_batch_perplexity(min_indices)
        entropy_temp = temperature if (self.use_temperature_schedule or self.use_soft_assignment) else None
        entropy_loss = self.entropy_loss(-distances, temperature=entropy_temp)
        self.node_count.index_add_(0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype))
        # 码本间距统计
        self.compute_inter_code_stats()

        quantized_out = quantized

        return (
            quantized_out,
            perplexity,
            min_indices,
            entropy_loss,
            vq_loss,
        )

class NSVQ(VQ):
    def __init__(self, use_diveq: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_diveq = use_diveq

    def NSVQ_core(self, nodes: Tensor, hard_quantized: Tensor) -> Tensor:
        """
        NSVQ 的核心量化过程。
        """
        random_vector = torch.randn_like(nodes)
        norm_quantization_residual = torch.linalg.norm(nodes - hard_quantized, dim=-1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=-1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector 
        return nodes + vq_error 
                 
    def DiVeQ_core(self, nodes: Tensor, hard_quantized: Tensor, noise_variance = 1e-3):
        error_dir = hard_quantized - nodes
        error_dir_norm = error_dir.norm(dim = -1, keepdim = True)

        noised_dir = error_dir + torch.sqrt(torch.tensor(noise_variance, device=error_dir.device)) * torch.randn_like(error_dir)
        unit_noised_dir = F.normalize(noised_dir, dim=-1, p=2)

        return nodes + error_dir_norm * unit_noised_dir.detach()

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练阶段的前向传播。
        """
        nodes_proj = self.in_norm(nodes)
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        # 在训练早期，根据数据进行一次可选的 KMeans 初始化
        if self.training:
            self._maybe_data_dependent_init(nodes_for_init)
        # 1. 找到最近的码字（欧氏或余弦）
        distances, nodes_used, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)
        with torch.no_grad():
            self.nodes_norm = torch.norm(nodes_used, p=2, dim=-1).mean()
        min_indices = torch.argmin(distances, dim=-1)           # 形状通常为 [B, num_queries]
        hard_quantized = F.embedding(min_indices, code_used)
        self.last_commitment_loss = F.mse_loss(hard_quantized, nodes_used).detach()

        # 基于离散索引计算 slot 重复率指标（不参与梯度）
        self._update_slot_redundancy_metrics(min_indices)

        # 统计每个样本在当前 batch 中使用到的唯一 code 数量，并记录 batch 内的平均值
        self._compute_avg_unique_codes(min_indices)

        # 2. NSVQ 量化：使用随机向量按残差范数比例注入，作为可微近似
        if self.use_diveq:
            quantized = self.DiVeQ_core(nodes_used, hard_quantized)
        else:
            quantized = self.NSVQ_core(nodes_used, hard_quantized)

        # 3. 计算困惑度 (Perplexity) —— 仅基于当前 batch
        perplexity = self._compute_batch_perplexity(min_indices)
        self.node_count.index_add_(0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype))
        # 恢复 batch 结构以计算 per-sample diversity
        entropy_loss = self.entropy_loss(-distances)
        if self.orthogonal_loss_weight > 0:
            orth_loss = self.orthogonality_loss()
            vq_loss = self.orthogonal_loss_weight * orth_loss
        else:
            vq_loss = torch.zeros_like(entropy_loss)

        quantized_out = quantized

        return (
            quantized_out,
            perplexity,
            min_indices,
            entropy_loss,
            vq_loss
        )