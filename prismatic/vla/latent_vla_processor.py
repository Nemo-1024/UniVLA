"""
latent_vla_processor.py

Processor utilities for LatentWorldVLA evaluation.

This is a slimmed-down copy of the previous `LatentVLAProcessor` implementation that lived under
`prismatic/extern/hf/processing_prismatic.py`, but without any OpenVLA remote-code config/registration.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import TensorType


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _bounds_denorm(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (high - low) + low


def _bounds_norm(
    x: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    eps = 1e-8
    x_norm = 2.0 * (x - low) / (high - low + eps) - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)
    zeros_mask = (high - low).abs() <= eps
    x_norm = torch.where(zeros_mask, torch.zeros_like(x_norm), x_norm)
    if mask is not None:
        x_norm = torch.where(mask.bool(), x_norm, x)
    return x_norm


class LatentVLAProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        *,
        lam_image_resolution: int = 256,
        dataset_statistics_path: Optional[Union[str, os.PathLike]] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        base_processor: Optional[Any] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)
        self.lam_image_resolution = int(lam_image_resolution)
        self.norm_stats = norm_stats
        self.base_processor = base_processor
        self.lam_image_transform = T.Compose(
            [
                T.Resize((self.lam_image_resolution, self.lam_image_resolution)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    @classmethod
    def from_vlm_processor(
        cls,
        base_processor: Any,
        dataset_statistics_path: Union[str, os.PathLike],
    ) -> "LatentVLAProcessor":
        if not os.path.isfile(dataset_statistics_path):
            raise ValueError(f"Dataset statistics file {dataset_statistics_path} load failed.")
        with open(dataset_statistics_path, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        return cls(
            image_processor=getattr(base_processor, "image_processor", None),
            tokenizer=getattr(base_processor, "tokenizer", None),
            base_processor=base_processor,
            norm_stats=norm_stats,
        )

    def build_vla_features(
        self,
        *,
        messages: List[Dict[str, Any]],
        observation: Image.Image,
        wrist_image: Image.Image,
        proprio: Union[np.ndarray, torch.Tensor],
        unnorm_key: Optional[str],
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if self.base_processor is None or self.norm_stats is None:
            raise ValueError("LatentVLAProcessor.base_processor or norm_stats is not set.")

        has_assistant_reply = any(msg.get("role") == "assistant" for msg in messages)
        inputs = self.base_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=not has_assistant_reply,
            return_tensors=return_tensors,
            return_dict=True,
        )

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        image_grid_thw = getattr(inputs, "image_grid_thw", None)

        lam_image = self.lam_image_transform(observation.convert("RGB"))
        wrist_image_t = self.lam_image_transform(wrist_image.convert("RGB"))

        prop_t = torch.as_tensor(proprio)
        if prop_t.dim() == 2:
            prop_t = prop_t[-1]

        if self.norm_stats is not None and unnorm_key is not None and unnorm_key in self.norm_stats:
            pstats = self.norm_stats[unnorm_key].get("proprio", None)
            if pstats is not None:
                low = torch.as_tensor(pstats.get("q01", pstats.get("min")))
                high = torch.as_tensor(pstats.get("q99", pstats.get("max")))
                mask_np = pstats.get("mask", np.ones_like(pstats.get("min", []))) if "min" in pstats else None
                mask = torch.as_tensor(mask_np) if mask_np is not None else None
                prop_t = _bounds_norm(prop_t, low, high, mask)

        if prop_t.dim() == 1:
            prop_t = prop_t.unsqueeze(0)
        if lam_image.dim() == 3:
            lam_image = lam_image.unsqueeze(0)
        if wrist_image_t.dim() == 3:
            wrist_image_t = wrist_image_t.unsqueeze(0)

        data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lam_image": lam_image,
            "wrist_image": wrist_image_t,
            "proprio": prop_t,
        }
        if image_grid_thw is not None:
            data["image_grid_thw"] = image_grid_thw
        return BatchFeature(data=data, tensor_type=return_tensors)

    def postprocess_actions(
        self,
        actions_norm: torch.Tensor,
        *,
        unnorm_key: str,
        clip_to_min_max: bool = True,
        bounds_mode: str = "q99",
    ) -> torch.Tensor:
        if self.norm_stats is None or unnorm_key not in self.norm_stats:
            raise ValueError("self.norm_stats is None or unnorm_key not in self.norm_stats")
        astats = self.norm_stats[unnorm_key].get("action", None)
        if astats is None:
            raise ValueError(f"astats is None, unnorm_key: {unnorm_key}")

        if bounds_mode == "q99":
            low_key, high_key = "q01", "q99"
        elif bounds_mode == "min_max":
            low_key, high_key = "min", "max"
        else:
            raise ValueError(f"Invalid bounds_mode: {bounds_mode}. Use 'q99' or 'min_max'.")

        low = torch.as_tensor(
            astats.get(low_key, astats.get("min")),
            dtype=actions_norm.dtype,
            device=actions_norm.device,
        )
        high = torch.as_tensor(
            astats.get(high_key, astats.get("max")),
            dtype=actions_norm.dtype,
            device=actions_norm.device,
        )
        while low.dim() < actions_norm.dim():
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)

        denormed = _bounds_denorm(actions_norm, low, high)
        if clip_to_min_max:
            mn = torch.as_tensor(astats.get("min", low), dtype=actions_norm.dtype, device=actions_norm.device)
            mx = torch.as_tensor(astats.get("max", high), dtype=actions_norm.dtype, device=actions_norm.device)
            while mn.dim() < actions_norm.dim():
                mn = mn.unsqueeze(0)
                mx = mx.unsqueeze(0)
            denormed = torch.clamp(denormed, min=mn, max=mx)

        mask_np = astats.get("mask", None)
        if mask_np is not None:
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=actions_norm.device)
            while mask.dim() < actions_norm.dim():
                mask = mask.unsqueeze(0)
            actions = torch.where(mask, denormed, actions_norm)
        else:
            actions = denormed
        return actions

