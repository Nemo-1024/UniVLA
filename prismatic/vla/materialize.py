"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn as nn
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction,
    PaddedCollatorForLanguageModeling,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSBatchTransformLatentAction, RLDSDataset
import torchvision.transforms as transforms
from latent_action_model.core.lam_model import LatentLAMModel
import torch
# 使用 timm 的 ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: int,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForLanguageModeling]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        (3, default_image_resolution, default_image_resolution),
        padding_side=padding_side,
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=(default_image_resolution,default_image_resolution),
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator


def get_latent_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    latent_action_model: LatentLAMModel,
    tokenizer: PreTrainedTokenizerBase,
    default_image_resolution: int,
    padding_side: str = "right",
    predict_stop_token: bool = False,
    shuffle_buffer_size: int = 100_000,
    episodic: bool = False,
    image_aug: bool = False,
    data_transform_fn = RLDSBatchTransformLatentAction,
    collator_fn = PaddedCollatorForActionPrediction,

) -> Tuple[Dataset, PreTrainedTokenizerBase, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # action_tokenizer = ActionTokenizer(tokenizer)

    image_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    image_transform_lam =transforms.Compose([transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])


    data_transform = data_transform_fn(
        image_transform=image_transform,
        image_transform_lam=image_transform_lam,
    )
    #151667 原为"<think>"，但用不到，所以用int(151667)
    collator = collator_fn(
        tokenizer.model_max_length,
        int(151667),
        padding_side=padding_side,
        action_tokenizer=latent_action_model,
        base_tokenizer=tokenizer,
        predict_stop_token=False
    )


    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    train_dataset = cls(
        data_root_dir,
        data_mix,
        data_transform,
        resize_resolution=(default_image_resolution, default_image_resolution),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
        training_phase='pre-training',
    )
    val_dataset = cls(
        data_root_dir,
        data_mix,
        data_transform,
        resize_resolution=(default_image_resolution, default_image_resolution),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=False,
        training_phase='pre-training',
    )

    return train_dataset, val_dataset, tokenizer, collator