"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union
from torchvision import transforms
import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2, InterpolationMode
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
import threading
from queue import Queue, Full
import tensorflow as tf
from prismatic.util.data_utils import tree_map
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# From 3Hz to 5Hz control frequency
datasets_with_lower_frequency = ['fractal20220817_data', 'toto', 'berkeley_autolab_ur5', 
'nyu_franka_play_dataset_converted_externally_to_rlds', 
'ucsd_kitchen_dataset_converted_externally_to_rlds', 
'dlr_edan_shared_control_converted_externally_to_rlds', 'dobbe']

# From 15Hz to 30 Hz control frequency
datasets_with_higher_frequency = ['utaustin_mutex', 
'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 
'austin_sailor_dataset_converted_externally_to_rlds', 
'austin_sailor_dataset_converted_externally_to_rlds', 
'toto', 'viola', 'droid']


@dataclass
class RLDSBatchTransform:
    action_tokenizer: Any
    base_tokenizer: Any
    image_transform: Any
    prompt_builder_fn: Any
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # print(labels)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        out = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
        if "dataset_id" in rlds_batch:
            out["dataset_id"] = rlds_batch["dataset_id"]
        return out


@dataclass
class RLDSBatchTransformLIBERO_withHis:
    action_tokenizer: Any
    base_tokenizer: Any
    image_transform: Any
    image_transform_lam: Any
    prompt_builder_fn: Any
    predict_stop_token: bool = True
    window_size: int = 5

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        randomized_overlap = random.randint(0,1)
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size-1])

        input_img = Image.fromarray(rlds_batch["observation"]["image_primary"][randomized_overlap])
        pixel_values = self.image_transform(input_img)

        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(input_img)
            target_pixel_values= self.image_transform_lam(Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size - 1 + randomized_overlap]))

            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            latent_action_idx = self.action_tokenizer.get_latent_action(videos=video)['indices'].squeeze()

            if randomized_overlap > 0:
                initial_pixel_values = self.image_transform_lam(img)
                target_pixel_values= self.image_transform_lam(img_k)
                video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
                hist_action_idx = self.action_tokenizer.get_latent_action(videos=video)['indices'].squeeze()        

        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
        # print(action_vocab)
        action_tokens = ''
        for i, action in enumerate(action_vocab):
            action_tokens += action

        input_prompt = f"What action should the robot take to {lang}?"
        if randomized_overlap > 0:
            action_vocab = [f'<ACT_{i.item()}>' for i in hist_action_idx]   # [ACT_1, ACT_2, ... ACT_K]

            hist_action_tokens = ''
            for i, action in enumerate(action_vocab):
                hist_action_tokens += action

            input_prompt = f"What action should the robot take to {lang}? History action " + hist_action_tokens

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": input_prompt},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)


        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        out = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=rlds_batch["action"][randomized_overlap: self.window_size + randomized_overlap],
            latent_action_idx=latent_action_idx,
            dataset_name=dataset_name,
        )
        if "dataset_id" in rlds_batch:
            out["dataset_id"] = rlds_batch["dataset_id"]
        return out


@dataclass
class RLDSBatchTransformLIBERO:
    image_transform_lam: Optional[Any] = None
    vlm_resolution: int = 448
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        轻量化 Transform：仅提取必要的数据供 Collator 批量 LAM 编码。
        返回内容：
        - language_instruction: 原始字节串（不 decode）
        - pixel_values: 供 VLA 模型使用的图像张量
        - initial_pixel_values, target_pixel_values: 供 LAM 编码（两帧）
        - dataset_name: 可选，若存在则透传
        """
        # 原始语言（bytes，不解码）
        language_instruction = rlds_batch["task"]["language_instruction"]
        # ---- 获取视频帧 ----
        video = np.array(rlds_batch["observation"]["image_primary"])  # [T, H, W, C]（固定长度）
        total_frames = len(video)
        assert total_frames > 0, f"收到空视频帧序列: T={total_frames}"

        # ---- 视频采样与预处理 ----
        # 直接使用全部帧
        # 如果 image_transform 是 torch Transform，则逐帧应用（先将 numpy 转为 PIL 或 Tensor）


        # 获取 wrist 视角（如果存在）
        video_wrist = None
        if "image_wrist" in rlds_batch["observation"]:
            video_wrist = np.array(rlds_batch["observation"]["image_wrist"])
        
        out: Dict[str, Any] = {
            "language_instruction": language_instruction,
            "video": video,
            "proprio": np.array(rlds_batch["observation"]["proprio"]),
            "actions": np.array(rlds_batch["action"])
        }
        
        # 添加 wrist 视角（如果加载）
        if video_wrist is not None:
            out["video_wrist"] = video_wrist

        # 透传数据集标识（若存在）
        if "dataset_name" in rlds_batch:
            out["dataset_name"] = rlds_batch["dataset_name"]
        if "dataset_id" in rlds_batch:
            out["dataset_id"] = rlds_batch["dataset_id"]

        return out


@dataclass
class RLDSBatchTransformLatentAction:
    image_transform_lam: Optional[Any] = None
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        轻量化 Transform：仅提取必要的数据供 Collator 批量 LAM 编码。
        返回内容：
        - language_instruction: 原始字节串（不 decode）
        - pixel_values: 供 VLA 模型使用的图像张量
        - initial_pixel_values, target_pixel_values: 供 LAM 编码（两帧）
        - dataset_name: 可选，若存在则透传
        """
        # 原始语言（bytes，不解码）
        language_instruction = rlds_batch["task"]["language_instruction"]

        # ---- 获取视频帧 ----
        video = np.array(rlds_batch["observation"]["image_primary"])  # [T, H, W, C]（固定长度）
        total_frames = len(video)
        assert total_frames > 0, f"收到空视频帧序列: T={total_frames}"

        out: Dict[str, Any] = {
            "language_instruction": language_instruction,
            "video": video,
            "proprio": np.array(rlds_batch["observation"]["proprio"]),
        }

        # 透传数据集标识（若存在）
        if "dataset_name" in rlds_batch:
            out["dataset_name"] = rlds_batch["dataset_name"]
        if "dataset_id" in rlds_batch:
            out["dataset_id"] = rlds_batch["dataset_id"]

        return out

## 过去用于按等距采样视频帧的工具已不需要（上游已提供固定长度窗口）
@dataclass
class RLDSBatchTransformVideo:
    image_transform: Optional[Any] = None  # 可额外叠加的增强
    random_resized_crop: bool = True
    scale: tuple = (0.8, 1.0)
    # ratio: tuple = (0.5625, 1.0)
    ratio: tuple = (0.8, 1.0)    #不要低于1.0，否则会得到细长的图
    random_rotation: bool = True
    rotation_degrees: Tuple[float, float] = (-5, 5)



    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 RLDS 批次转换为训练所需格式（支持视频与独立首帧增强）。
        
        注意：保留完整的视频序列（包括历史帧），帧的选择将在 Collator 中进行。
        """
        action = np.array(rlds_batch["action"])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # ---- 获取完整视频帧序列（包括历史帧）----
        video_frames = np.array(rlds_batch["observation"]["image_primary"])  # [T, H, W, C]
        total_frames = len(video_frames)
        assert total_frames > 0, f"收到空视频帧序列: T={total_frames}"

        # ---- 转为 Tensor 并归一化到 [0,1] ----
        video = torch.as_tensor(video_frames).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)  # [T, C, H, W]
        # 直接从 video 取切片会产生 view；下游对 video 的就地标准化可能影响到这些 view。
        # 这里显式 clone，避免存储别名导致的偶发数值异常（例如可视化时某些样本颜色失真）。
        dec_video = video.clone()
        image_size = video.shape[2]
        # 🎯 1️⃣ 时序一致裁剪（视频）
        if self.random_resized_crop:
            # 手动采样一次参数，复用于所有帧
            # 直接使用 Tensor 采样参数，避免不必要的 PIL 转换
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                video[0], scale=self.scale, ratio=self.ratio
            )
            # ✅ 时序一致裁剪
            video = F.resized_crop(video, i, j, h, w, size=(image_size,image_size))

        # 🎯 2️⃣ 独立随机裁剪第一帧（Decoder 输入）
        if self.random_resized_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                dec_video[0], scale=self.scale, ratio=self.ratio
            )
            dec_video = F.resized_crop(dec_video, i, j, h, w, size=(image_size, image_size))

        # 🎯 3️⃣ 轻量旋转扰动（保持各自时序一致；decoder 视角仍独立采样一次）
        if self.random_rotation:
            angle_video = float(torch.empty(1).uniform_(self.rotation_degrees[0], self.rotation_degrees[1]).item())
            angle_dec = float(torch.empty(1).uniform_(self.rotation_degrees[0], self.rotation_degrees[1]).item())
            video = F.rotate(
                video,
                angle=angle_video,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            dec_video = F.rotate(
                dec_video,
                angle=angle_dec,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        
        # 保留完整的 proprio 序列（包括历史状态）
        proprio = np.array(rlds_batch["observation"]["proprio"])

        # 📦 输出
        result = {
            "video": video,      # [T, 3, H, W] - 完整序列，包括历史帧
            "dec_video": dec_video,    # [T, 3, H, W] - 完整序列，包括历史帧
            "task_instruction": lang,
            "action": action,
            "proprio": proprio,  # [T, state_dim] - 完整序列，包括历史状态
        }
        if "dataset_name" in rlds_batch:
            result["dataset_name"] = rlds_batch["dataset_name"]
        if "dataset_id" in rlds_batch:
            result["dataset_id"] = rlds_batch["dataset_id"]
        return result


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 20,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = 'lam',
        async_prefetch: bool = False,
        async_prefetch_size: int = 128,
        async_transform: bool = False,
        debug_repeat_batch: Union[bool, int] = False,
        use_history_frame: bool = True,
        load_camera_views: Tuple[str, ...] = ("primary",),
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.train: bool = train
        self.async_prefetch: bool = async_prefetch
        self.async_prefetch_size: int = int(async_prefetch_size)
        self.async_transform: bool = async_transform
        self.debug_repeat_batch: Union[bool, int] = debug_repeat_batch

        # Normalize camera-view input: a bare string would be treated as an iterable of characters.
        if isinstance(load_camera_views, str):
            load_camera_views = (load_camera_views,)
        else:
            load_camera_views = tuple(load_camera_views)
        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,  # 启用状态数据加载以支持物理接地损失
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            # action_proprio_normalization_type=NormalizationType.BOUNDS
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
                # Episode-level shuffle: shuffle full trajectories before chunking.
                # Only effective when train=True inside apply_trajectory_transforms.
                episode_shuffle_size=2048 if train else 0,
                use_history_frame=use_history_frame,                # 控制是否在观察窗口中包含历史帧
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=12,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            # 使用显式关键字参数形式，避免 dlimp/TF 在内部与 seed 关键字冲突（"Got multiple values for argument 'seed'"）
            # 注意：在 LAM 阶段不做 random_resized_crop，避免额外 CPU 开销与空间扰动。
            if training_phase == 'lam' or training_phase == 'lam_2f' or training_phase == 'post-training':
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    # 仅保留颜色类增强
                    random_brightness=dict(max_delta=0.3),
                    random_contrast=dict(lower=0.6, upper=1.4),
                    random_saturation=dict(lower=0.5, upper=1.5),
                    random_hue=dict(max_delta=0.08),
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )})
            else:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.75, 1.0]),
                    # TF: random_brightness(image, max_delta, seed)
                    random_brightness=dict(max_delta=0.3),
                    # TF: random_contrast(image, lower, upper, seed)
                    random_contrast=dict(lower=0.6, upper=1.4),
                    # TF: random_saturation(image, lower, upper, seed)
                    random_saturation=dict(lower=0.5, upper=1.5),
                    # TF: random_hue(image, max_delta, seed)
                    random_hue=dict(max_delta=0.08),
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )})
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        iterator = self.dataset.as_numpy_iterator()
        # === 🧠 调试模式：重复返回固定的 k 个样本（k 可配置，默认 1 个样本）===
        if self.debug_repeat_batch:
            repeat_k = int(self.debug_repeat_batch) if isinstance(self.debug_repeat_batch, int) else 1
            repeat_k = max(1, repeat_k)
            cached_samples = []
            for _ in range(repeat_k):
                cached_samples.append(self.batch_transform(next(iterator)))
            print(f"[RLDSDataset] Debug mode: Repeating {repeat_k} cached sample(s) indefinitely.")
            while True:
                for sample in cached_samples:
                    yield sample
            return
        # === 正常模式 ===
        if not self.async_prefetch:
            for rlds_batch in iterator:
                yield self.batch_transform(rlds_batch)
            return

        queue: "Queue[Any]" = Queue(maxsize=max(1, self.async_prefetch_size))
        stop_event = threading.Event()
        sentinel: object = object()
        producer_exception: Dict[str, BaseException] = {}

        def producer() -> None:
            try:
                for rlds_batch in iterator:
                    if stop_event.is_set():
                        break
                    item: Any
                    if self.async_transform:
                        item = self.batch_transform(rlds_batch)
                    else:
                        item = rlds_batch
                    while not stop_event.is_set():
                        try:
                            queue.put(item, timeout=0.1)
                            break
                        except Full:
                            if stop_event.is_set():
                                break
                            continue
            except BaseException as exc:  # noqa: BLE001
                producer_exception["exc"] = exc
            finally:
                # Signal end-of-stream
                try:
                    while True:
                        try:
                            queue.put(sentinel, timeout=0.1)
                            break
                        except Full:
                            if stop_event.is_set():
                                break
                            continue
                except Exception:
                    pass

        thread = threading.Thread(target=producer, name="RLDSDatasetPrefetch", daemon=True)
        thread.start()

        try:
            while True:
                item = queue.get()
                if item is sentinel:
                    # propagate background exception if any
                    if "exc" in producer_exception:
                        raise producer_exception["exc"]
                    break
                if not self.async_transform:
                    yield self.batch_transform(item)
                else:
                    yield item
                if "exc" in producer_exception:
                    raise producer_exception["exc"]
        finally:
            stop_event.set()

    def __len__(self) -> int:
        """
        返回预计算的数据集长度。
        - train 数据集：分布式下按 rank 分片，返回 dataset_length // world_size
        - eval 数据集：不分片，返回完整长度（因为只在 rank 0 评估完整验证集）
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # 只有 train 数据集需要除以 world_size，eval 数据集返回完整长度
            if self.train:
                world_size = torch.distributed.get_world_size()
                return int(self.dataset_length // world_size)
        return int(self.dataset_length)
    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
