"""
hf_vlm.py

轻量化的 `PrismaticVLM`：移除自研 vision/llm backbone 与 projector 依赖。
遵循 Hugging Face 风格：由 `from_pretrained` 负责 I/O 加载，构造器仅组装并透传 forward/generate。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast, List

import torch
import torch.nn as nn
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin


class PrismaticVLM(nn.Module):
    """通用 VLM 包装器：仅依赖 HF model id。

    - 构造器不做 I/O，只接收已构造好的 `model` 与可选的 `processor`。
    - 使用类方法 `from_pretrained(model_id_or_path, ...)` 完成加载与实例化，减少语义歧义。
    - 固定使用 InternVL 3.5：`InternVLForConditionalGeneration.from_pretrained(model_id)` 加载底层模型。
    - 使用 `AutoProcessor.from_pretrained(model_id)` 获取处理器（若仓库未提供处理器，此调用会抛错）。
    - 暴露 `.model` 与 `.processor/.tokenizer`，并将 `forward/generate` 透传给底层 HF 模型。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor,
    ) -> None:
        super().__init__()
            # 保存对原始、完整模型的引用
        self.model = model

        self.language_model = model.language_model
        self.lm_head = model.lm_head
        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector
        self.processor= processor
        self.tokenizer = processor.tokenizer
    # ---- 加载入口 ----
    @classmethod
    def from_pretrained(
        cls,
        hf_model_id_or_path: Union[str, Path],
        *,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "PrismaticVLM":
        """标准加载入口（HF 风格）。

        负责从 `hf_model_id_or_path` 拉取权重与处理器，并返回轻量实例：
        - 模型：InternVLForConditionalGeneration
        - 处理器：AutoProcessor
        """
        model: PreTrainedModel = InternVLForConditionalGeneration.from_pretrained(
            hf_model_id_or_path,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            device_map=device_map,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        processor = AutoProcessor.from_pretrained(
            hf_model_id_or_path,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            trust_remote_code=trust_remote_code,
        )

        return cls(model=model, processor=processor)
    
    # ---- 设备与 dtype 助手 ----
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ---- Module 接口转发 ----
    def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):  # noqa: D401 - 直接透传
        return self.model.generate(*args, **kwargs)


    def prepare_inputs(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Union[ProcessorMixin, PreTrainedTokenizerBase]]]:
        """预处理助手：若存在 processor 则返回其与处理后的 inputs。"""
        return kwargs, self.processor

    # ---- 冻结组件（vision encoder / LLM backbone / LLM head / projector）----
    def freeze_modules(
        self,
        *,
        vision_encoder: bool = False,
        llm_backbone: bool = False,
        projector: bool = False,
        llm_head: bool = False,
    ):
        """冻结底层 InternVL 组件（严格遵循官方命名，无额外检查）。

        严格命名约定（HF transformers v4.56 InternVL）：
        - 视觉编码器: self.model.vision_tower
        - 多模态投影器: self.model.multi_modal_projector
        - 语言模型骨干: self.model.language_model
        - 语言模型 head: self.lm_head

        参数:
            vision_encoder: 冻结视觉编码器。
            llm_backbone: 冻结语言模型骨干（不含 head）。
            llm_head: 冻结语言模型输出 head。
            projector: 冻结多模态投影器。

        参考:
            https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/models/internvl/modeling_internvl.py
        """

        if vision_encoder:
            self.vision_tower.requires_grad_(False)
        if projector:
            self.multi_modal_projector.requires_grad_(False)
        if llm_backbone:
            self.language_model.requires_grad_(False)
        if llm_head:
            self.lm_head.requires_grad_(False)



