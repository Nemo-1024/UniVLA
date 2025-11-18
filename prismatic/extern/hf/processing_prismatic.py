"""
processing_prismatic.py

HuggingFace-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. Default configuration
specifies `siglip-224px+7b`.
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union, Dict

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === Image Processing ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.
        @param use_fused_vision_backbone: Boolean indicating single or fused (dual) vision backbone
        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")
        @param mean: [TIMM :: `data_cfg`] Normalization mean as float tuple (or two-tuple if `fused_backbone`)
        @param std: [TIMM :: `data_cfg`] Normalization std as float tuple (or two-tuple if `fused_backbone`)
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,  # No image augmentations when loading the transform!
            )

            # [Validation] Ensure appropriate transform structure, expected sizes
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
            #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# === LatentVLAProcessor: wraps InternVL processor and adds JEPA/proprio for LWVLA ===
import os
import json
import numpy as np
import torchvision.transforms as T
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _bounds_q99_denorm(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (high - low) + low


def _bounds_q99_norm(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    eps = 1e-8
    x_norm = 2.0 * (x - low) / (high - low + eps) - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)
    # zeros_mask = (high - low).abs() <= eps
    # x_norm = torch.where(zeros_mask, torch.zeros_like(x_norm), x_norm)
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
        jepa_image_resolution: int = 256,
        dataset_statistics_path: Optional[Union[str, os.PathLike]] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        base_processor: Optional[Any] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)
        self.jepa_image_resolution = int(jepa_image_resolution)
        self.norm_stats = norm_stats
        # 保存 InternVL 的完整 Processor，用于 apply_chat_template 以正确处理图像与文本
        self.base_processor = base_processor
        self.jepa_transform = T.Compose(
            [
                T.Resize((self.jepa_image_resolution, self.jepa_image_resolution)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    @classmethod
    def from_internvl_processor(
        cls,
        base_processor: Any,
        dataset_statistics_path: Union[str, os.PathLike] = None,
    ) -> "LatentVLAProcessor":
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
        else:
            raise ValueError(f"Dataset statistics file {dataset_statistics_path} load failed.")
        return cls(
            image_processor=base_processor.image_processor,
            tokenizer=base_processor.tokenizer,
            base_processor=base_processor,
            norm_stats=norm_stats,
        )

    def _extract_first_image(self, messages: Any) -> Optional[Image.Image]:
        try:
            for turn in messages:
                if turn.get("role") == "user" and isinstance(turn.get("content"), list):
                    for c in turn["content"]:
                        if isinstance(c, dict) and c.get("type") == "image":
                            img = c.get("image")
                            return img.convert("RGB") if isinstance(img, Image.Image) else None
        except Exception:
            return None
        return None

    def build_vla_features(
        self,
        *,
        messages: List[Dict[str, Any]],
        proprio: Union[np.ndarray, torch.Tensor],
        unnorm_key: Optional[str],
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if self.base_processor is None or self.norm_stats is None:
            raise ValueError("LatentVLAProcessor.base_processor or norm_stats is not set. ")

        inputs = self.base_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=return_tensors,
            return_dict=True,
        )

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]

        img = self._extract_first_image(messages)
        if img is None:
            raise ValueError("messages must include at least one image in the user content.")
        image_4_jepa = self.jepa_transform(img).unsqueeze(0)

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
                prop_t = _bounds_q99_norm(prop_t, low, high, mask)

        # Ensure batch dimension alignment with tokenizer outputs
        # Determine batch size from input_ids (HF returns Tensor of shape [B, L])

        # Make sure proprio has shape [B, D]
        if prop_t.dim() == 1:
            prop_t = prop_t.unsqueeze(0)

        # Make sure image_4_jepa has shape [B, C, H, W]
        if image_4_jepa.dim() == 3:
            image_4_jepa = image_4_jepa.unsqueeze(0)


        data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_4_jepa": image_4_jepa,
            "proprio": prop_t,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def postprocess_actions(self, actions_norm: torch.Tensor, *, unnorm_key: str, clip_to_min_max: bool = False) -> torch.Tensor:
        if self.norm_stats is None or unnorm_key not in self.norm_stats:
            raise ValueError(f"self.norm_stats is None or unnorm_key not in self.norm_stats")
        astats = self.norm_stats[unnorm_key].get("action", None)
        if astats is None:
            raise ValueError(f"astats is None, unnorm_key: {unnorm_key}")
        low = torch.as_tensor(astats.get("q01", astats.get("min")), dtype=actions_norm.dtype, device=actions_norm.device)
        high = torch.as_tensor(astats.get("q99", astats.get("max")), dtype=actions_norm.dtype, device=actions_norm.device)
        while low.dim() < actions_norm.dim():
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        denormed = _bounds_q99_denorm(actions_norm, low, high)
        if clip_to_min_max:
            mn = torch.as_tensor(astats.get("min", low), dtype=actions_norm.dtype, device=actions_norm.device)
            mx = torch.as_tensor(astats.get("max", high), dtype=actions_norm.dtype, device=actions_norm.device)
            while mn.dim() < actions_norm.dim():
                mn = mn.unsqueeze(0)
                mx = mx.unsqueeze(0)
            denormed = torch.clamp(denormed, min=mn, max=mx)

        # Apply mask: only denormalize dimensions where mask==True
        mask_np = astats.get("mask", None)
        if mask_np is not None:
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=actions_norm.device)
            while mask.dim() < actions_norm.dim():
                mask = mask.unsqueeze(0)
            actions = torch.where(mask, denormed, actions_norm)
        else:
            actions = denormed
        return actions


# Register with AutoProcessor for OpenVLAConfig
from transformers import AutoProcessor as _AutoProcessor

_AutoProcessor.register(OpenVLAConfig, LatentVLAProcessor)
