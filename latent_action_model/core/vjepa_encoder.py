#!/usr/bin/env python3
"""
V-JEPA2 特征编码器
将V-JEPA2 3D Patch编码器改造为图片编码器，用于LAM模型的视觉特征提取

核心功能：
- 将[B,T,C,H,W]格式的视频数据转换为[B*T,C,2,H,W]以适配3D patch编码器
- 时间维度步长为2的3D patch编码通过数据复制来满足
- 输出特征恢复为[B,K,...]格式，其中K为每个时间步的空间特征数量
- 专注于连续特征提取，无需tokenization概念
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple
from pathlib import Path
from torchvision import transforms

warnings.filterwarnings('ignore')

# 使用 timm 的 ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VJEPAEncoder(nn.Module):
    """
    V-JEPA2视觉特征编码器
    
    将V-JEPA2的3D patch编码器改造为适用于图片序列的编码器：
    1. 输入[B,T,C,H,W] -> 重塑为[B*T,C,2,H,W] (复制帧满足时间步长=2)
    2. 通过V-JEPA2编码器提取空间-时间特征
    3. 输出[B*T,K,D] -> 重塑为[B,T,K,D]，其中K为空间特征数，D为特征维度
    
    特点：
    - 300M参数量，无需复杂内存管理
    - 输出连续特征表示，不是离散tokens
    - 专为LAM的latent action model训练设计
    """
    
    def __init__(
        self, 
        model_id: str = 'vjepa2_vit_large'
    ):
        """
        初始化V-JEPA2特征编码器
        
        Args:
            device: 计算设备 ('cuda', 'cpu', 或 None 自动检测)
            model_id: V-JEPA2模型ID
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        
        # 模型组件
        # self.encoder = None
        self.feature_dim = 1024  # V-JEPA2 ViT Large 特征维度
        
        #  标准化转换
        # self.ImageNet_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        
        # 加载模型
        self._load_model()
        
        # print(f"✅ V-JEPA2特征编码器初始化完成")
        # print(f"   - 设备: {self.device}")
        # print(f"   - 模型: {self.model_id}")
        # print(f"   - 特征维度: {self.feature_dim}")
        # print(f"   - 用途: LAM潜空间训练的视觉特征提取")
    
    def _load_model(self):
        """加载V-JEPA2模型编码器部分"""
        # print(f"🔄 加载V-JEPA2编码器...")
        
     
        # 加载V-JEPA2模型 (编码器+预测器的tuple)
        model= torch.hub.load('facebookresearch/vjepa2', self.model_id)
        # print(type(model), dir(model))
        encoder,_ = model
        
        # 将编码器注册为子模块（这样参数会被Lightning正确识别）
        self.encoder = encoder.to(self.device)

        self.encoder.eval()
        
        # 冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # print(f"✅ 编码器加载成功")
            

    
    def _prepare_temporal_input(self, videos: torch.Tensor) -> torch.Tensor:
        """
        将视频数据转换为3D patch编码器的输入格式，包含ImageNet标准化
        
        Args:
            videos: 输入视频张量 [B, T, C, H, W]
            
        Returns:
            prepared_input: [B*T, C, 2, H, W] 格式的标准化张量
        """
        B, T, C, H, W = videos.shape
        
        # 重塑为 [B*T, C, H, W]
        frames = videos.view(-1, C, H, W)  # [B*T, C, H, W]
        
        # 应用 ImageNet 标准化转换
        # frames = self.ImageNet_transform(frames)
        
        # 复制每一帧以满足时间维度步长=2的要求
        frames_duplicated = frames.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B*T, C, 2, H, W]
        
        return frames_duplicated
    
    def _restore_batch_format(self, features: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        将编码器输出恢复为批次格式
        
        Args:
            features: 编码器输出 [B, T, S, K]
            original_shape: 原始输入形状 (B, T, C, H, W)
            
        Returns:
            reshaped_features: [B, T, S, K] 格式的特征，S为空间特征数，K为特征维度
        """
        B, T = original_shape[:2]
        BT, S, K = features.shape #S=256,K=1024
        
        # 恢复为 [B, T, S, K]
        return features.view(B, T, S, K)
    
    def encode_video_frames(
        self, 
        videos: torch.Tensor, 
    ) -> torch.Tensor:
        """
        编码视频帧序列为特征表示
        
        Args:
            videos: 输入视频张量 [B, T, C, H, W]
            return_sequence: 是否返回完整序列特征，否则返回时间平均特征
            
        Returns:
            features: 特征张量
                - return_sequence=True: [B, T, K, D] K为空间特征数，D为特征维度
                - return_sequence=False: [B, K, D] 时间维度平均后的特征
        """
        if videos.dim() != 5:
            raise ValueError(f"期望5D张量 [B, T, C, H, W]，得到: {videos.shape}")
        
        original_shape = videos.shape
        B, T = original_shape[:2]
        
        # 转换数据格式以适配3D patch编码器（包含标准化）
        prepared_input = self._prepare_temporal_input(videos.to(self.device))
        
        # 通过编码器提取特征
        with torch.no_grad():
            encoded_features = self.encoder(prepared_input)  # [B*T, K, D]
        
        # 恢复批次格式
        batch_features = self._restore_batch_format(encoded_features, original_shape)  # [B, T, K, D]
        

        return batch_features  # [B, T, K, D]

    
    def get_model_info(self) -> dict:
        """获取编码器信息"""
        return {
            'model_id': self.model_id,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.encoder.parameters()),
            'encoder_type': type(self.encoder).__name__,
            'purpose': 'LAM潜空间训练的视觉特征提取'
        }
    
    def __repr__(self):
        return f"VJEPAEncoder(model={self.model_id}, device={self.device}, feature_dim={self.feature_dim})"


