"""
models.py (HF-only)

简化后的模型配置：移除视觉/语言骨干与架构组合参数，仅保留 `model_id`（用于标识或记录HF模型）。
推荐直接在脚本中使用 Hugging Face model id，不依赖此注册表。
"""

from dataclasses import dataclass
from enum import Enum

from draccus.choice_types import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # 仅用作占位，方便已有代码通过 ChoiceRegistry 查找；推荐直接使用 HF model id
    model_id: str


class ModelRegistry(Enum):
    # 空注册表（兼容旧导入路径）；建议直接使用 HF model id
    pass

 

