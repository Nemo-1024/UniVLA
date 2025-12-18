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
from typing import Optional, Tuple, Union, Sequence
from pathlib import Path
from transformers import AutoModel
from .cosmos_tokenizer.image_lib import ImageTokenizer
warnings.filterwarnings('ignore')
from torchvision import transforms
import math
import torch.nn.functional as F
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
        model_id: str = "facebook/vjepa2-vitl-fpc64-256"
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
        

        # 加载模型
        self.model = AutoModel.from_pretrained(self.model_id,trust_remote_code=True,device_map=self.device, dtype=torch.bfloat16)    
        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
    @torch.no_grad()
    def encode(
        self, 
        images: torch.Tensor, 
        norm_latents: bool = False,
        n: int=-1
    ) -> torch.Tensor:
        """
        输入：[B*T, C, H, W]
        输出：[B*T, K, D]
        """
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, C, H, W)
        else:
            B, C, H, W = images.shape
            T=1
        assert images.dim() == 4, f"期望4D张量 [B*T, C, H, W]，得到: {images.shape}，图片维度不正确"
        video_like = images.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # [B*T, 2, C, H, W]
        # 通过编码器提取特征

        encoded_features = self.model.get_vision_features(video_like)  # [B*T, K, D]
        if norm_latents:
            encoded_features = F.normalize(encoded_features, dim=-1)
        # print(encoded_features.min(), encoded_features.max(), encoded_features.mean(), encoded_features.std())
        return encoded_features.reshape(B, T, encoded_features.shape[-2], encoded_features.shape[-1]).detach()  # [B, T, K, D]
        
    @torch.no_grad()
    def encode_video(
        self, 
        videos: torch.Tensor, 
        norm_latents: bool = False,
    ) -> torch.Tensor:
        """
        输入：[B, T, C, H, W]
        输出：[B, T//2, 256, D]
        """
        B, T, C, H, W = videos.shape
        with torch.no_grad():
            encoded_features = self.model.get_vision_features(videos)
        if norm_latents:
            encoded_features = F.normalize(encoded_features, dim=-1)
        return encoded_features.reshape(B, T//2, 256, self.feature_dim).detach()


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_latent_layers: int = 1,
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.num_latent_layers = max(int(num_latent_layers), 1)
        self.norm_layer_type = norm_layer_type
        self.enable_norm = enable_norm
        # 加载 DINOv3 模型
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, dtype=torch.float32)
        model.eval()
        self.model = model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # 记录特征维度
        hidden_size = getattr(self.model.config, 'hidden_size', None)
        self.feature_dim = int(hidden_size) if hidden_size is not None else 1024
        # 根据 norm 类型只初始化需要的归一化层，避免出现未使用的模块
        if self.norm_layer_type in ("bn", "ln"):
            # 关闭 affine，避免产生可训练参数从而触发“未使用参数”告警
            if self.norm_layer_type == "bn":
                norm_builder = lambda: nn.SyncBatchNorm(self.feature_dim, affine=False).to(self.device)
            else:
                norm_builder = lambda: nn.LayerNorm(self.feature_dim, elementwise_affine=False).to(self.device)
            self.latent_norms = nn.ModuleList([norm_builder() for _ in range(self.num_latent_layers)])
        else:
            self.latent_norms = None

    @torch.no_grad()
    def encode(self, images: torch.Tensor, remove_cls: bool = True, n: Union[int, Sequence] = [4, 11, 17, -1] ) -> torch.Tensor:
        """
        输入：[B, T, C, H, W]
        输出：[B, T, K, D]
        """
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T=1
        # 若仅需要最后一层，避免请求所有 hidden_states 以减少内存与时延
        need_all_layers = not (isinstance(n, int) and n == -1)
        outputs = self.model(pixel_values=images, output_hidden_states=need_all_layers)
        if not need_all_layers:
            last = outputs.last_hidden_state  # [B*T, 5+K, D]（含若干特殊token）
            if remove_cls:
                tokens = last[:, 5:, :]  # [B*T, K, D]
            else:
                tokens = last            # [B*T, 5+K, D]

            # 先在 token 维度上展平做归一化，然后再一次性 reshape 到 [B, T, K, D]
            if self.enable_norm:
                if self.norm_layer_type == "bn":
                    if self.latent_norms is None:
                        raise ValueError("当前 DINOv3Encoder 未初始化 BN 层，请将 norm_layer_type 设置为 'bn'。")
                    tokens_2d = tokens.reshape(-1, self.feature_dim)  # [B*T*K, D]
                    tokens_2d = self.latent_norms[0](tokens_2d)
                    tokens = tokens_2d.view(tokens.shape[0], tokens.shape[1], self.feature_dim)
                elif self.norm_layer_type == "ln":
                    if self.latent_norms is None:
                        raise ValueError("当前 DINOv3Encoder 未初始化 LN 层，请将 norm_layer_type 设置为 'ln'。")
                    tokens = self.latent_norms[0](tokens)
                elif self.norm_layer_type == "l2":
                    tokens = F.normalize(tokens, p=2, dim=-1)

            features = tokens.reshape(B, T, -1, self.feature_dim)
            return features.detach()
        else:
            hidden_states = outputs.hidden_states
            if isinstance(n, int):
                list_n = [n]
            else:
                list_n = n
            # 确保为每个待用层分配到对应的 BN
            assert len(list_n) <= self.num_latent_layers, (
                f"DINOv3Encoder 期望的 BN 层数为 {self.num_latent_layers}，"
                f"但传入的特征层数为 {len(list_n)}；请保证两者一致（通常为 len(latent_layer_to_use)）。"
            )
            features = []
            for idx, i in enumerate(list_n):
                layer_tokens = hidden_states[i]  # [B*T, 5+K, D]
                if remove_cls:
                    layer_tokens = layer_tokens[:, 5:, :]  # [B*T, K, D]

                if self.enable_norm:
                    if self.norm_layer_type == "bn":
                        if self.latent_norms is None:
                            raise ValueError("当前 DINOv3Encoder 未初始化 BN 层，请将 norm_layer_type 设置为 'bn'。")
                        lt_2d = layer_tokens.reshape(-1, self.feature_dim)  # [B*T*K, D]
                        lt_2d = self.latent_norms[idx](lt_2d)
                        layer_tokens = lt_2d.view(layer_tokens.shape[0], layer_tokens.shape[1], self.feature_dim)
                    elif self.norm_layer_type == "ln":
                        if self.latent_norms is None:
                            raise ValueError("当前 DINOv3Encoder 未初始化 LN 层，请将 norm_layer_type 设置为 'ln'。")
                        layer_tokens = self.latent_norms[idx](layer_tokens)
                    elif self.norm_layer_type == "l2":
                        layer_tokens = F.normalize(layer_tokens, p=2, dim=-1)

                features.append(layer_tokens.reshape(B, T, -1, self.feature_dim))

            return features[0].detach() if isinstance(n, int) else [f.detach() for f in features]

class CosmosAutoencoder(nn.Module):
    """
    Cosmos 图像自编码器（支持编码与解码）。

    包装 `cosmos_tokenizer.ImageTokenizer`，提供：
    - encode:  输入 [B*T,3,H,W]（ImageNet 标准化）→ 内部反标准化→[-1,1]→连续/离散潜变量
    - decode:  输入潜变量 → [B,3,H,W]（[-1,1] 范围）
    - autoencode: 编解码合一
    - encode_video_frames / decode_video_frames: 处理 [B,T,3,H,W]

    说明（简化策略）：
    - `model_id` 必须是一个本地目录路径，且目录中包含权重文件。
    - 优先加载 `autoencoder.jit`；否则使用 `encoder.jit` 与 `decoder.jit`。
    - 输入假定为 ImageNet 标准化（(x-mean)/std），类内会先反标准化到 [0,1]，再映射到 [-1,1] 后交由编码器。
    """

    def __init__(
        self,
        model_id: str = "cosmos:Cosmos-0.1-Tokenizer-CI16x16",
        encoder_ckpt: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        tokenizer_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_str = dtype

        # 严格场景：model_id 必须是目录；优先 autoencoder.jit，否则 encoder/decoder.jit
        model_path = Path(self.model_id)
        if not (model_path.exists() and model_path.is_dir()):
            raise ValueError("CosmosAutoencoder 期望 model_id 为本地目录路径，且包含权重文件。")

        ae = model_path / "autoencoder.jit"
        enc = model_path / "encoder.jit"
        dec = model_path / "decoder.jit"

        # 优先使用独立的 encoder/decoder 以支持 encode/decode 接口；若缺失再退回完整 autoencoder
        if enc.exists() and dec.exists():
            self.tokenizer = ImageTokenizer(
                checkpoint_enc=str(enc),
                checkpoint_dec=str(dec),
                tokenizer_config=tokenizer_config,
                device=self.device,
                dtype=self.dtype_str,
            )
        elif ae.exists():
            self.tokenizer = ImageTokenizer(
                checkpoint=str(ae),
                tokenizer_config=tokenizer_config,
                device=self.device,
                dtype=self.dtype_str,
            )
        else:
            raise FileNotFoundError(
                f"未找到有效权重：{ae} 或者成对的 {enc} 与 {dec}"
            )
                # 冻结视觉编码器参数，避免其参与训练图却未产生梯度被 DDP 标记为未使用
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        self.tokenizer.eval()
        # 预定义 ImageNet 反标准化参数（match torchvision/timm）
        self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _denormalize_imagenet_to_unit(self, images: torch.Tensor) -> torch.Tensor:
        """将 ImageNet 标准化张量还原到 [0,1]，输入 [B*T,3,H,W]。"""
        images = images.float() * self._imagenet_std + self._imagenet_mean
        return images

    @torch.no_grad()
    def _unit_to_negone_posone(self, images01: torch.Tensor) -> torch.Tensor:
        """将 [0,1] 映射到 [-1,1]。"""
        return images01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, norm_latents: bool = False):
        """编码图像为潜变量（输入为 ImageNet 标准化的 [B*T,3,H,W]）。
            连续 CI：Tuple(Tensor[B*T,h*w,16])
        """
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T=1
        images = self._denormalize_imagenet_to_unit(images)
        imgs11 = self._unit_to_negone_posone(images).to(getattr(torch, self.dtype_str))
        (token,) = self.tokenizer.encode(imgs11)
        BT, K, H, W = token.shape
        token = token.reshape(B, T, K, H*W).permute(0, 1, 3, 2).contiguous().detach()
        return token 

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """从潜变量解码图像。

        Args:
            latent: 连续 CI [B, h*w, K]
        Returns:
            Tensor[B,3,H,W]，范围[-1,1]
        """
        assert latent.shape[1] == 256, f"期望256个token，得到: {latent.shape[1]}"
        latent = latent.permute(0, 2, 1).contiguous()
        latent = latent.reshape(latent.shape[0], -1, 16, 16)
        return self.tokenizer.decode(latent).clamp(-1.0,1.0)

    @torch.no_grad()
    def autoencode(self, images: torch.Tensor) -> torch.Tensor:
        """图像自编码（输入为 ImageNet 标准化的 [B*T,3,H,W]，输出范围[-1,1]）。"""
        if images.dim() != 4:
            raise ValueError(f"期望 4D [B*T,3,H,W]，得到: {images.shape}")
        imgs01 = self._denormalize_imagenet_to_unit(images)
        imgs11 = self._unit_to_negone_posone(imgs01).to(getattr(torch, self.dtype_str))
        return self.tokenizer.autoencode(imgs11).clamp(-1.0,1.0)

 

def build_vision_encoder(model_id: str, num_latent_layers: int = 1, norm_layer_type: str = "l2", enable_norm: bool = False) -> nn.Module:
    """
    根据 model_id 中的关键词选择并构建视觉编码器实例。

    规则：
    - 包含 "dino" -> 返回 DINOv3Encoder
    - 包含 "vjepa" 或 "jepa" -> 返回 VJEPAEncoder
    - 包含 "cosmos" -> 返回 CosmosAutoencoder
    - 否则抛出异常
    """

    key = model_id.lower()
    if "dinov3-vitl16" in key:
        return DINOv3Encoder(
            model_id="/mnt/mnt/public/jlchen/weights/dinov3-vitl16-pretrain-lvd1689m",
            num_latent_layers=num_latent_layers,
            norm_layer_type=norm_layer_type,
            enable_norm=enable_norm,
        ), 1024
    elif "dinov3-vitb16" in key:
        return DINOv3Encoder(
            model_id="/mnt/mnt/public/jlchen/weights/dinov3-vitb16-pretrain-lvd1689m",
            num_latent_layers=num_latent_layers,
            norm_layer_type=norm_layer_type,
            enable_norm=enable_norm,
        ), 768
    elif "vjepa" in key or "jepa" in key:
        return VJEPAEncoder(model_id="/mnt/mnt/public/jlchen/weights/vjepa2-vitl-fpc64-256"), 1024
    elif "cosmos" in key:
        return CosmosAutoencoder(model_id="/mnt/mnt/public/jlchen/weights/Cosmos-0.1-Tokenizer-CI16x16"), 16

    else:
        print(f"未使用预训练模型，采用PatchEmbed编码器")
        return None, 0
