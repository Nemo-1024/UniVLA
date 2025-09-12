"""
vla.py (简化版)

最小化的 VLAConfig，用于控制训练超参与数据混合。去除注册表与多模型枚举，聚焦通用字段。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLAConfig:
    # 冻结策略（兼容现有训练分支判定）
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_projector: bool = False
    # 数据混合与缓冲
    data_mix: str = "oxe_magic_soup_plus"
    shuffle_buffer_size: int = 20_000
    image_resolution: int = 448

    # 训练超参
    epochs: int = 10
    max_steps: Optional[int] = None
    expected_world_size: int = 1
    global_batch_size: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    # 训练策略
    train_strategy: str = "fsdp-full-shard"

    # 训练加速
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True
