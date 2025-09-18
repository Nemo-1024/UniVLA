import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union, List

class NSVQ(nn.Module):
    """
    NSVQ: Noise Substitution in Vector Quantization (重构版)

    适用于视觉、语音或动作等特征的量化。
    此版本优化了设备管理、状态处理和代码结构，同时保留了所有核心功能。
    """
    def __init__(
        self,
        codebook_size: int = 1024,
        code_dim: int = 128,
        discarding_threshold: float = 0.01,
        initialization: str = 'uniform',
        # 以下参数未在模块内部使用，但为了接口兼容性予以保留
        code_seq_len: Optional[int] = None,
        patch_size: Optional[int] = None,
        image_size: Optional[int] = None,
    ):
        """
        初始化 NSVQ 模块。

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

    def _get_indices(self, nodes: Tensor) -> Tensor:
        """计算输入节点与码本之间的最小距离索引。"""
        # 使用 torch.cdist 计算欧氏距离，更简洁高效
        distances = torch.cdist(nodes, self.codebooks)
        return torch.argmin(distances, dim=-1)

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练阶段的前向传播。

        参数:
            nodes (Tensor): 输入张量，形状为 [B, D]，其中 B 是批量大小，D 是特征维度。

        返回:
            Tuple[Tensor, Tensor, Tensor]:
            - quantized (Tensor): 经过噪声替换技巧后的量化向量。
            - perplexity (Tensor): 困惑度，衡量码本使用情况的指标。
            - indices (Tensor): 每个输入向量对应的码字索引。
        """
        batch_size = nodes.shape[0]
        nodes = nodes.reshape(-1, self.code_dim)
        # 1. 找到最近的码字
        min_indices = self._get_indices(nodes)
        hard_quantized = F.embedding(min_indices, self.codebooks)

        # 2. NSVQ 核心技巧：噪声替换
        # a. 生成与输入同形的随机噪声向量
        random_vector = torch.randn_like(nodes)
        # b. 计算量化误差（残差）的范数
        norm_residual = torch.norm(nodes - hard_quantized, dim=1, keepdim=True)
        # c. 计算随机向量的范数
        norm_random = torch.norm(random_vector, dim=1, keepdim=True)
        # d. 用归一化的误差缩放随机向量
        vq_error = (norm_residual / (norm_random + self.eps)) * random_vector
        # e. 将缩放后的噪声加回原始输入
        quantized = nodes + vq_error

        # 3. 计算困惑度 (Perplexity)
        # Perplexity 是衡量码本利用率的指标，值越高表示码本被利用得越均匀
        encodings = F.one_hot(min_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        # 4. 更新码字使用计数 (在 no_grad 上下文中)
        with torch.no_grad():
            # 使用 index_add_ 可以准确地为每个被选中的码字计数
            self.node_count.index_add_(0, min_indices, torch.ones_like(min_indices, dtype=self.node_count.dtype))

        return quantized.reshape(batch_size, -1, self.code_dim), perplexity, min_indices.reshape(batch_size, -1)

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
        # 判断标准：使用频率低于阈值
        unused_mask = (self.node_count.float() / self.node_count.sum()) < self.discarding_threshold
        used_mask = ~unused_mask
        
        unused_indices = torch.where(unused_mask)[0]
        used_indices = torch.where(used_mask)[0]
        
        num_unused = unused_indices.numel()
        if num_unused == 0:
            return 0 # 没有需要替换的码字
        print("=" * 50,"\n")
        print("node_count",self.node_count)
        # 如果所有码字都未被使用，则添加少量噪声以重新激活
        if used_indices.numel() == 0:
            self.codebooks.data += self.eps * torch.randn_like(self.codebooks.data)
            
            print("All codebooks are unused, adding noise to reactivate")
            
        else:
            # 从使用过的码本中随机抽取来替换未使用的码本
            used_codebooks = self.codebooks.data[used_indices]
            
            # 创建替换用的码字
            num_repeats = (num_unused + used_codebooks.size(0) - 1) // used_codebooks.size(0)
            replacements = used_codebooks.repeat(num_repeats, 1)
            replacements = replacements[torch.randperm(replacements.size(0))][:num_unused]

            # 添加少量噪声以增加多样性
            noise = self.eps * torch.randn_like(replacements)
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

        # 计算困惑度（与训练阶段一致）
        encodings = F.one_hot(min_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        return (
            quantized.reshape(batch_size, -1, self.code_dim),
            perplexity,
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
    

