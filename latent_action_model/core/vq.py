import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, Tuple, Union, List

class VQ(nn.Module):

    def __init__(
        self,
        codebook_size: int = 1024,
        code_dim: int = 128,
        discarding_threshold: float = 0.01,
        initialization: str = 'uniform',
        data_dependent_init: bool = True,
        kmeans_init_after: int = 5000,
        kmeans_iters: int = 10,
        kmeans_max_samples: int = 100000,
        beta: float = 0.25,
        perplexity_accum_steps: int = 100,
        orthogonal_loss_weight: float = 0.0,
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
        self.kmeans_init_after = int(kmeans_init_after)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_max_samples = int(kmeans_max_samples)
        self.beta = beta
        self.perplexity_accum_steps = int(perplexity_accum_steps)
        self.orthogonal_loss_weight = float(orthogonal_loss_weight)

        # 初始化码本参数
        if initialization == 'normal':
            codebooks_data = torch.randn(self.codebook_size, self.code_dim)
        elif initialization == 'uniform':
            codebooks_data = torch.empty(self.codebook_size, self.code_dim)
            nn.init.uniform_(codebooks_data, -1 / self.codebook_size, 1 / self.codebook_size)
        else:
            raise ValueError("初始化方法应为 'normal' 或 'uniform' 之一")
        
        self.codebooks = nn.Parameter(codebooks_data)

        # 注册码字使用计数器作为缓冲区
        # 这使得它成为模块状态的一部分，并能随模块移动到不同设备
        self.register_buffer('node_count', torch.zeros(self.codebook_size, dtype=torch.long))
        # 滑动窗口（环形缓冲）用于更稳健地估计 perplexity
        self.register_buffer('perplexity_window', torch.zeros(self.perplexity_accum_steps, self.codebook_size, dtype=torch.long))
        self.register_buffer('perplexity_cursor', torch.tensor(0, dtype=torch.long))
        self.register_buffer('perplexity_filled', torch.tensor(0, dtype=torch.long))
        self.register_buffer('perplexity_sum_counts', torch.zeros(self.codebook_size, dtype=torch.long))

        # KMeans 初始化相关的状态
        # 记录已经经历的迭代/step 数（通常按 batch 计数）
        self.register_buffer('kmeans_seen_steps', torch.tensor(0, dtype=torch.long))
        # self.kmeans_seen_steps = torch.tensor(0, dtype=torch.long)
        # 标记是否已经完成过 KMeans 初始化
        # self.register_buffer('kmeans_initialized', torch.tensor(0, dtype=torch.bool))
        # self.initialized = torch.tensor(0, dtype=torch.bool)
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    def _get_indices(self, nodes: Tensor) -> Tensor:
        """计算输入节点与码本之间的最近索引（平方欧氏距离）。"""
        # 使用平方欧氏距离，避免不必要的 sqrt，数值更稳定
        nodes_norm = (nodes * nodes).sum(dim=1, keepdim=True)              # [N, 1]
        code_norm = (self.codebooks * self.codebooks).sum(dim=1)           # [K]
        distances = nodes_norm + code_norm.unsqueeze(0) - 2.0 * nodes @ self.codebooks.t()
        distances = torch.clamp(distances, min=0.0)
        return torch.argmin(distances, dim=-1)

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
        if nodes is None:
            return

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

            # 将 KMeans 得到的中心写入码本
            centers_converted = centers.to(device=self.codebooks.data.device, dtype=self.codebooks.data.dtype)
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
    def _maybe_data_dependent_init(self, nodes: Tensor) -> None:
        """
        在满足条件时触发一次基于数据的 KMeans 初始化。

        条件:
            - data_dependent_init 为 True
            - 还未进行过 KMeans 初始化
            - kmeans_init_after > 0
            - 已累计的 step 数 >= kmeans_init_after
        """
        if not self.data_dependent_init:
            return
        if bool(self.initialized.item()):
            return
        if int(self.kmeans_init_after) <= 0:
            return

        # 累计 step 数（通常按 batch 计数）
        self.kmeans_seen_steps.add_(1)

        if int(self.kmeans_seen_steps.item()) >= int(self.kmeans_init_after):
            # 使用当前 batch 的特征进行一次 KMeans 初始化
            self.kmeans_init(nodes)

    def entropy_loss(self, affinity: Tensor, temperature: float = 0.01) -> Tensor:
        """
        通过平衡样本熵和平均熵来计算损失，以防止码本坍塌。
        affinity: 相似度矩阵，这里是负的平方距离。形状为 [N, K]。
        """
        # 数值稳定：避免对 0 取对数导致的 NaN
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        # 防止极低温度造成的数值爆炸
        safe_temp = max(float(temperature), 1e-6)
        logits = flat_affinity / safe_temp
        probs = F.softmax(logits, dim=-1)
        # clamp 概率，避免出现 0 或 1
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        log_probs = torch.log(probs)
        target_probs = probs
        avg_probs = torch.mean(target_probs, dim=0).clamp(min=self.eps, max=1.0 - self.eps)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        
        loss = sample_entropy - avg_entropy
        return loss




    @torch.no_grad()
    def _update_perplexity_stats(self, min_indices: Tensor, allow_eval_update: bool = False) -> Tensor:
        """
        更新困惑度滑动窗口并返回当前 perplexity。

        allow_eval_update: 在 eval 模式下是否也更新统计量（NSVQ 需要）。
        """
        should_update = self.training or allow_eval_update
        if not should_update:
            return self.get_running_perplexity()

        batch_counts = torch.bincount(min_indices, minlength=self.codebook_size).to(self.codebooks.device)
        window_size = int(self.perplexity_accum_steps)
        cursor = int(self.perplexity_cursor.item())
        filled = int(self.perplexity_filled.item())
        if filled >= window_size:
            old_counts = self.perplexity_window[cursor]
        else:
            old_counts = torch.zeros_like(batch_counts)

        self.perplexity_sum_counts.add_(batch_counts.to(self.perplexity_sum_counts.dtype))
        self.perplexity_sum_counts.sub_(old_counts.to(self.perplexity_sum_counts.dtype))
        self.perplexity_window[cursor].copy_(batch_counts.to(self.perplexity_window.dtype))

        cursor = (cursor + 1) % window_size
        self.perplexity_cursor.fill_(cursor)
        if filled < window_size:
            self.perplexity_filled.fill_(filled + 1)

        total = self.perplexity_sum_counts.sum()
        if total.item() > 0:
            probs_running = (self.perplexity_sum_counts.float() / total.float()).clamp(min=self.eps, max=1.0)
            perplexity = torch.exp(-torch.sum(probs_running * torch.log(probs_running)))
        else:
            avg_probs = (batch_counts.float() / max(1, int(batch_counts.sum().item()))).clamp(min=self.eps, max=1.0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs)))


        return perplexity

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
        batch_size = nodes.shape[0]
        # 在训练早期，根据数据进行一次可选的 KMeans 初始化
        self._maybe_data_dependent_init(nodes)
        

        if int(self.kmeans_seen_steps.item()) < int(self.kmeans_init_after):
            # 在提前返回之前，让 codebooks 参数参与计算图，避免被标记为 UNUSED
            # 通过计算 codebooks 的平方和并乘以0，确保它参与计算图但不影响训练
            # 将这个 dummy loss 包含在 vq_loss 中，确保梯度被计算
            dummy_vq_loss = self.codebooks.pow(2).sum() * 0.0
            return (
                nodes, 
                torch.tensor(0.0, device=nodes.device, dtype=nodes.dtype), 
                torch.zeros_like(nodes[:,:,0], dtype=nodes.dtype), 
                torch.tensor(0.0, device=nodes.device, dtype=nodes.dtype), 
                dummy_vq_loss
            )
        else:
            nodes = nodes.reshape(-1, self.code_dim)
            # 1. 找到最近的码字（平方欧氏距离）
            nodes_norm = (nodes * nodes).sum(dim=1, keepdim=True)              # [N, 1]
            code_norm = (self.codebooks * self.codebooks).sum(dim=1)           # [K]
            distances = nodes_norm + code_norm.unsqueeze(0) - 2.0 * nodes @ self.codebooks.t()
            distances = torch.clamp(distances, min=0.0)
            min_indices = torch.argmin(distances, dim=-1)
            hard_quantized = F.embedding(min_indices, self.codebooks)

            # 2. VQ 损失 (含 Straight-Through)
            codebook_loss = F.mse_loss(hard_quantized, nodes.detach())
            commitment_loss = F.mse_loss(hard_quantized.detach(), nodes)
            quantized = nodes + (hard_quantized - nodes).detach()
            orth_loss = self.orthogonality_loss()
            vq_loss = codebook_loss + self.beta * commitment_loss + self.orthogonal_loss_weight * orth_loss

            # 3. 计算困惑度 (Perplexity)
            perplexity = self._update_perplexity_stats(min_indices)

            # 5. 计算 entropy_loss
            entropy_loss = self.entropy_loss(-distances)
            self.node_count.index_add_(0, min_indices, torch.ones_like(min_indices, dtype=self.node_count.dtype))

        return (
            quantized.reshape(batch_size, -1, self.code_dim),
            perplexity,
            min_indices.reshape(batch_size, -1),
            entropy_loss,
            vq_loss
        )

    @torch.no_grad()
    def replace_unused_codebooks(self) -> int:
        """
        替换在最近的训练迭代中长时间未被使用的码本条目。
        这有助于防止码本坍缩（部分码字永远得不到训练）。

        参数:
            num_batches (int): 累计计数的总批次数。

        返回:
            int: 被替换的未使用码字的数量。
        """
        # 确保只在 kmeans 初始化完成后才执行 replacement
        # 避免在初始化前使用无效的 node_count 统计
        if not bool(self.initialized.item()):
            return 0
        
        codebook_std = self.codebooks.data.std()
        eps_noise = 1e-12  # 自适应噪声，保证不为0

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
            num_unused_tensor = torch.zeros(1, device=self.codebooks.device, dtype=torch.long)
        num_unused = int(unused_indices.numel())

        if rank == 0:
            if num_unused == 0:
                print("=" * 50, "\n")
                print("No unused codebooks to replace. global_node_count: ", global_node_count)
                
            else:
                print("=" * 50, "\n")
                print("global_node_count", global_node_count)

                if used_indices.numel() == 0:
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
                    
                    # 使用重要性采样（允许重复采样）
                    sampled_indices = torch.multinomial(probs, num_unused, replacement=True)
                    # 将采样索引映射回原始码本索引
                    sampled_used_indices = used_indices[sampled_indices]
                    # 获取采样得到的码字
                    replacements = self.codebooks.data[sampled_used_indices]
                    # 添加少量噪声以增加多样性
                    noise = eps_noise * torch.randn_like(replacements)
                    self.codebooks.data[unused_indices] = replacements + noise
                    print("=" * 50)
                    print("Replaced {} unused codebooks using importance sampling based on node_count".format(num_unused))

            if ddp_initialized:
                num_unused_tensor.fill_(num_unused)

        # 3) DDP：将新的 codebooks 广播到所有进程，并同步返回值
        if ddp_initialized:
            torch.distributed.broadcast(self.codebooks.data, src=0)
            torch.distributed.barrier()
            torch.distributed.broadcast(num_unused_tensor, src=0)
            return int(num_unused_tensor.item())
        else:
            return num_unused

    def reset_node_count(self) -> None:
        """重置码字使用计数器。通常在一个 epoch 或一轮替换操作后调用。"""

        # print("Resetting node count")
        # print("=" * 50)
        self.node_count.zero_()

    def reset_perplexity_stats(self) -> None:
        """重置跨 batch 困惑度的滑动窗口统计。"""
        if hasattr(self, 'perplexity_window'):
            self.perplexity_window.zero_()
        if hasattr(self, 'perplexity_sum_counts'):
            self.perplexity_sum_counts.zero_()
        if hasattr(self, 'perplexity_cursor'):
            self.perplexity_cursor.zero_()
        if hasattr(self, 'perplexity_filled'):
            self.perplexity_filled.zero_()

    @torch.no_grad()
    def get_running_perplexity(self) -> Tensor:
        """返回当前滑动窗口下的 perplexity（不改动内部状态）。"""
        total = self.perplexity_sum_counts.sum()
        if total.item() == 0:
            return torch.tensor(0.0, device=self.codebooks.device)
        probs_running = (self.perplexity_sum_counts.float() / total.float()).clamp(min=self.eps, max=1.0)
        return torch.exp(-torch.sum(probs_running * torch.log(probs_running)))

    @torch.no_grad()
    def inference(self, nodes: Tensor, user_specific: Optional[Union[int, List[int]]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        推理（或评估）阶段的量化。
        直接返回最近的码字，不使用噪声替换技巧。

        参数:
            nodes (Tensor): 输入张量，形状为 [B, D]。
            user_specific (Optional[Union[int, List[int]]]): 
                如果提供，将忽略最近邻搜索，强制使用指定的索引。
                - int: 所有输入都使用这一个索引。
                - List[int]: 为批次中的每个输入指定一个索引。

        返回:
            Tuple[Tensor, Tensor, Tensor]:
            - quantized (Tensor): 量化后的向量（即最近的码字）。
            - indices (Tensor): 对应的码字索引。
        """
        batch_size = nodes.shape[0]
        nodes = nodes.reshape(-1, self.code_dim)
        if user_specific is not None:
            if isinstance(user_specific, list):
                min_indices = torch.tensor(user_specific, device=nodes.device, dtype=torch.long)
            else: # int
                min_indices = torch.full((batch_size,), user_specific, device=nodes.device, dtype=torch.long)
        else:
            min_indices = self._get_indices(nodes)
        
        quantized = F.embedding(min_indices, self.codebooks)

        return (
            quantized.reshape(batch_size, -1, self.code_dim),
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
    

class NSVQ(VQ):
    def __init__(self, use_diveq: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_diveq = use_diveq

    def NSVQ_core(self, nodes: Tensor, hard_quantized: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        NSVQ 的核心量化过程。
        """
        random_vector = torch.randn_like(nodes)
        norm_quantization_residual = torch.linalg.norm(nodes - hard_quantized, dim=1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector 
        return nodes + vq_error 
                 
    def DiVeQ_core(self, nodes: Tensor, hard_quantized: Tensor, noise_variance = 1e-3):
        error_dir = hard_quantized - nodes
        error_dir_norm = error_dir.norm(dim = -1, keepdim = True)

        noised_dir = error_dir + torch.sqrt(torch.tensor(noise_variance, device=error_dir.device)) * torch.randn_like(error_dir)
        unit_noised_dir = F.normalize(noised_dir, dim=1, p=2)

        return nodes + error_dir_norm * unit_noised_dir.detach()

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练阶段的前向传播。
        """
        batch_size = nodes.shape[0]
        # 在训练早期，根据数据进行一次可选的 KMeans 初始化
        if self.training:
            self._maybe_data_dependent_init(nodes)

        if int(self.kmeans_seen_steps.item()) < int(self.kmeans_init_after):
            # 在提前返回之前，让 codebooks 参数参与计算图，避免被标记为 UNUSED
            # 通过计算 codebooks 的平方和并乘以0，确保它参与计算图但不影响训练
            # 将这个 dummy loss 包含在 vq_loss 中，确保梯度被计算
            dummy_vq_loss = self.codebooks.pow(2).sum() * 0.0
            return (
                nodes, 
                torch.tensor(0.0, device=nodes.device, dtype=nodes.dtype), 
                torch.zeros_like(nodes[:,:,0], dtype=torch.int64), 
                torch.tensor(0.0, device=nodes.device, dtype=nodes.dtype), 
                dummy_vq_loss
            )
        else:
            nodes = nodes.reshape(-1, self.code_dim)
            # 1. 找到最近的码字（平方欧氏距离）
            nodes_norm = (nodes * nodes).sum(dim=1, keepdim=True)              # [N, 1]
            code_norm = (self.codebooks * self.codebooks).sum(dim=1)           # [K]
            distances = nodes_norm + code_norm.unsqueeze(0) - 2.0 * nodes @ self.codebooks.t()
            distances = torch.clamp(distances, min=0.0)
            min_indices = torch.argmin(distances, dim=-1)
            hard_quantized = F.embedding(min_indices, self.codebooks)
            # codebook_loss = F.mse_loss(hard_quantized, nodes.detach())
            # commitment_loss = F.mse_loss(hard_quantized.detach(), nodes)
            # quantized = nodes + (hard_quantized - nodes).detach()
            # 2. NSVQ 量化：使用随机向量按残差范数比例注入，作为可微近似
            if self.use_diveq:
                quantized = self.DiVeQ_core(nodes, hard_quantized)
            else:
                quantized = self.NSVQ_core(nodes, hard_quantized)

            # 3. 计算困惑度 (Perplexity)
            perplexity = self._update_perplexity_stats(min_indices, allow_eval_update=True)
            self.node_count.index_add_(0, min_indices, torch.ones_like(min_indices, dtype=self.node_count.dtype))
            entropy_loss = self.entropy_loss(-distances)
            orth_loss = self.orthogonality_loss()
            vq_loss = self.orthogonal_loss_weight * orth_loss

            return (
                quantized.reshape(batch_size, -1, self.code_dim),
                perplexity,
                min_indices.reshape(batch_size, -1),
                entropy_loss,
                vq_loss
            )