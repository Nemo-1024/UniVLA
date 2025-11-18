import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # 数据依赖初始化标记：首个 batch 用数据来重设码本以匹配尺度
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def _get_indices(self, nodes: Tensor) -> Tensor:
        """计算输入节点与码本之间的最近索引（平方欧氏距离）。"""
        # 使用平方欧氏距离，避免不必要的 sqrt，数值更稳定
        nodes_norm = (nodes * nodes).sum(dim=1, keepdim=True)              # [N, 1]
        code_norm = (self.codebooks * self.codebooks).sum(dim=1)           # [K]
        distances = nodes_norm + code_norm.unsqueeze(0) - 2.0 * nodes @ self.codebooks.t()
        distances = torch.clamp(distances, min=0.0)
        return torch.argmin(distances, dim=-1)

    def entropy_loss(self, affinity: Tensor, temperature: float = 0.01) -> Tensor:
        """
        通过平衡样本熵和平均熵来计算损失，以防止码本坍塌。
        affinity: 相似度矩阵，这里是负的平方距离。形状为 [N, K]。
        """
        # affinity is the squared loss
        flat_affinity = affinity.reshape(
            -1, affinity.shape[-1]
        )
        probs = F.softmax(flat_affinity / temperature, dim=-1)
        log_probs = F.log_softmax(flat_affinity / temperature + 1e-5, dim=-1)
        target_probs = probs
        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        
        loss = sample_entropy - avg_entropy
        return loss



    @torch.no_grad()
    def _kmeans_init(self, data: Tensor, num_clusters: int, max_iters: int = 25) -> Tensor:
        """
        使用 K-Means 聚类在首个 batch 上进行码本初始化。

        - 仅用于初始化阶段（no_grad）。
        - 对空簇采用保留旧中心或随机重置为数据点的策略以保证稳定性。

        Args:
            data: [N, D]
            num_clusters: 目标簇数 K
            max_iters: 最大迭代次数

        Returns:
            Tensor: [num_clusters, D] 的聚类中心，用于作为初始码本
        """
        if data.numel() == 0:
            return self.codebooks.data

        num_points, feature_dim = data.shape
        device = data.device

        effective_k = min(num_clusters, num_points)

        # 1) 初始化中心（随机不重复采样）
        perm = torch.randperm(num_points, device=device)
        centers = data[perm[:effective_k]].clone()  # [k, D]

        # 2) 迭代更新
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
                    # 空簇：回退为随机数据点，避免 NaN
                    random_idx = torch.randint(0, num_points, (1,), device=device)
                    new_centers[cluster_id] = data[random_idx]

            # 收敛性检查（可选）：若变化极小则提前结束
            center_shift = (new_centers - centers).pow(2).sum().sqrt()
            centers = new_centers
            if center_shift < 1e-6:
                break

        # 若样本少于簇数，重复中心以填满
        if effective_k < num_clusters:
            repeats = (num_clusters + effective_k - 1) // effective_k
            centers = centers.repeat(repeats, 1)[:num_clusters]

        return centers.detach()


    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练阶段的前向传播。
        """
        batch_size = nodes.shape[0]
        nodes = nodes.reshape(-1, self.code_dim)

        # 首批数据驱动初始化（分布式安全）：仅在 rank0 初始化并广播到所有进程
        if self.training and self.data_dependent_init and int(self.initialized.item()) == 0:
            ddp_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
            with torch.no_grad():
                if ddp_initialized:
                    rank = torch.distributed.get_rank()
                    if rank == 0:
                        init_vectors = self._kmeans_init(nodes, self.codebook_size)
                        self.codebooks.data.copy_(init_vectors)
                        self.initialized.fill_(1)
                    # 广播参数与标记
                    torch.distributed.broadcast(self.codebooks.data, src=0)
                    torch.distributed.broadcast(self.initialized, src=0)
                    # 可选同步，确保所有进程看到一致状态
                    torch.distributed.barrier()
                else:
                    init_vectors = self._kmeans_init(nodes, self.codebook_size)
                    self.codebooks.data.copy_(init_vectors)
                    self.initialized.fill_(1)

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
        random_vector = torch.randn_like(nodes)
        norm_quantization_residual = torch.linalg.norm(nodes - hard_quantized, dim=1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector
        quantized = nodes + vq_error

        # 3. 计算困惑度 (Perplexity)
        encodings = F.one_hot(min_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        # 4. 更新码字使用计数
        with torch.no_grad():
            self.node_count.index_add_(0, min_indices, torch.ones_like(min_indices, dtype=self.node_count.dtype))

        entropy_loss = self.entropy_loss(-distances)
        codebook_loss = torch.tensor(0.0, device=nodes.device)
        commitment_loss = torch.tensor(0.0, device=nodes.device)

        return (
            quantized.reshape(batch_size, -1, self.code_dim),
            perplexity,
            min_indices.reshape(batch_size, -1),
            codebook_loss,
            entropy_loss,
            commitment_loss
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
        codebook_std = self.codebooks.data.std()
        eps_noise = max(1e-5, 0.01 * codebook_std)  # 自适应噪声，保证不为0
        # 判断标准：使用频率低于阈值
        unused_mask = (self.node_count.float() / self.node_count.sum()) < self.discarding_threshold
        used_mask = ~unused_mask
        
        unused_indices = torch.where(unused_mask)[0]
        used_indices = torch.where(used_mask)[0]
        
        num_unused = unused_indices.numel()
        if num_unused == 0:
            print("No unused codebooks to replace")
            return 0 # 没有需要替换的码字
        print("=" * 50,"\n")
        print("node_count",self.node_count)
        # 如果所有码字都未被使用，则添加少量噪声以重新激活
        if used_indices.numel() == 0:
            self.codebooks.data += eps_noise * torch.randn_like(self.codebooks.data)
            
            print("All codebooks are unused, adding noise to reactivate")
            
        else:
            # 从使用过的码本中随机抽取来替换未使用的码本
            used_codebooks = self.codebooks.data[used_indices]
            
            # 创建替换用的码字
            num_repeats = (num_unused + used_codebooks.size(0) - 1) // used_codebooks.size(0)
            replacements = used_codebooks.repeat(num_repeats, 1)
            replacements = replacements[torch.randperm(replacements.size(0))][:num_unused]

            # 添加少量噪声以增加多样性
            noise = eps_noise * torch.randn_like(replacements)
            self.codebooks.data[unused_indices] = replacements + noise
            print("=" * 50)
            print("Replaced {} unused codebooks with noise".format(num_unused))
            
        return num_unused

    def reset_node_count(self) -> None:
        """重置码字使用计数器。通常在一个 epoch 或一轮替换操作后调用。"""

        print("Resetting node count")
        print("=" * 50)
        self.node_count.zero_()

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
            - perplexity (Tensor): 基于选中索引计算得到的困惑度（与训练阶段定义一致）。
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
    

