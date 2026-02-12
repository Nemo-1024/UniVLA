"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoProcessor
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction,
)
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransformLatentAction, RLDSDataset
import torchvision.transforms as transforms
from latent_action_model.core.lam_model import LatentLAMModel
import torch
# 使用 timm 的 ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_latent_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    processor: AutoProcessor,
    padding_side: str = "right",
    predict_stop_token: bool = False,
    shuffle_buffer_size: int = 10000,
    episodic: bool = False,
    image_aug: bool = False,
    training_phase: str = 'pre-training',
    data_transform_fn = RLDSBatchTransformLatentAction,
    collator_fn = PaddedCollatorForActionPrediction,
    latent_action_num_queries: int = None,
    debug_repeat_batch: bool = False,
    target_seq_len: int = 330,
    use_history_frame: bool = True,
    window_size: int = 20,
    load_camera_views: Tuple[str, ...] = ("primary",),
) -> Tuple[Dataset, PreTrainedTokenizerBase, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # action_tokenizer = ActionTokenizer(tokenizer)
    tokenizer = processor.tokenizer

    # image_transform = transforms.Compose([transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    # image_transform_lam =transforms.Compose([
    #     transforms.Resize((256,256)),
    #     # transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])


    data_transform = data_transform_fn()
    #tokenizer.pad_token_id: <|endoftext|> 151643
    assert latent_action_num_queries is not None, "latent_action_num_queries 不能为空（通常等于 LAM num_queries）"

    collator = collator_fn(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=padding_side,
        processor=processor,
        predict_stop_token=False,
        latent_action_num_queries=int(latent_action_num_queries),
        target_seq_len=target_seq_len,
        use_history_frame=use_history_frame,
    )


    # Normalize camera-view input: callers sometimes pass a bare string.
    if isinstance(load_camera_views, str):
        load_camera_views = (load_camera_views,)
    else:
        load_camera_views = tuple(load_camera_views)

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    train_dataset = cls(
        data_root_dir,
        data_mix,
        data_transform,
        resize_resolution=(256, 256),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
        training_phase=training_phase,
        debug_repeat_batch=debug_repeat_batch,
        use_history_frame=use_history_frame,
        window_size=window_size,
        load_camera_views=load_camera_views,
    )
    val_dataset = cls(
        data_root_dir,
        data_mix,
        data_transform,
        resize_resolution=(256,256),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=False,
        training_phase=training_phase,
        use_history_frame=use_history_frame,
        window_size=window_size,
        load_camera_views=load_camera_views,
    )

    return train_dataset, val_dataset, collator
