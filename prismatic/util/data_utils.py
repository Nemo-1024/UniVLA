"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import re
import string
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Any, Optional, List

import numpy as np
from scipy.spatial.transform import Rotation as R
from tensorflow.python.ops.image_ops_impl import image_gradients
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
# 预编译标点清理正则，避免每个 batch 重新构造
PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_ids = [instance["dataset_id"] for instance in instances] if "dataset_id" in instances[0] else None

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        # Build attention mask from original sequence lengths to allow using <eos> as padding
        seq_lengths = [min(t.size(0), self.model_max_length) for t in input_ids_list]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Build `attention_mask` from sequence lengths (not value equality), so real <eos> tokens aren't masked
        attention_mask = pad_sequence(
            [torch.ones(l, dtype=torch.bool) for l in seq_lengths],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = attention_mask[:, : input_ids.size(1)]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids
        return output


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    processor: Any = None
    predict_stop_token: bool = False
    latent_action_num_queries: int = 4
    target_seq_len: int = 330
    use_history_frame: bool = True

    def __post_init__(self):
        assert self.processor is not None, "processor 不能为空"
        # 从 processor.tokenizer 中获取占位符 token id，避免外部传参不一致
        tok = self.processor.tokenizer
        placeholder_token_id = tok.convert_tokens_to_ids("<ACT_PH>")
        assert placeholder_token_id != tok.unk_token_id, "未找到 <ACT_PH> 占位符，请确保已注册"
        self.placeholder_token_id = int(placeholder_token_id)
        # 预先缓存 mean/std 张量（用于 LAM 输入归一化）
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean_5d = mean.view(1, 1, 3, 1, 1)
        self.std_5d = std.view(1, 1, 3, 1, 1)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 收集批次数据（来自轻量 Transform，collate 阶段统一处理）
        lang_instructions_list: List[Any] = [instance["language_instruction"] for instance in instances]
        
        # 根据 use_history_frame 确定 LAM 输入的起始索引
        # - use_history_frame=True: video[0]=历史帧, video[1]=初始帧, LAM 从 video[1] 开始
        # - use_history_frame=False: video[0]=初始帧, LAM 从 video[0] 开始
        lam_start_idx = 1 if self.use_history_frame else 0
        
        # VLM 输入：基于初始观察（窗口第一帧）预测动作
        if self.use_history_frame:
            img_pairs = [
                (
                    instance["video"][0],             # 历史帧 (t = -window_size)
                    instance["video"][lam_start_idx], # 初始帧 (t = -window_size+1)
                )
                for instance in instances
            ]
        else:
            img_pairs = [
                (
                    instance["video"][0], # 初始帧 (t = -window_size+1)
                )
                for instance in instances
            ]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        dataset_ids = [instance["dataset_id"] for instance in instances] if "dataset_id" in instances[0] else None

        # LAM 输入：从初始帧开始的视频序列与状态（保持原 shape，后续在模型侧归一化）
        lam_videos = torch.stack(
            [torch.as_tensor(instance["video"][lam_start_idx:]).permute(0, 3, 1, 2).float().div_(255.0) for instance in instances],
            dim=0,
        )  # [B, T, 3, H, W]
        # 归一化到与 LAM 训练一致
        lam_videos = (lam_videos - self.mean_5d) / self.std_5d
        lam_states = torch.stack(
            [torch.as_tensor(instance["proprio"][lam_start_idx:]).float() for instance in instances],
            dim=0,
        )  # [B, T, Dq]

        # 构造带占位符的 prompt 并调用 AutoProcessor
        pixel_values_list: List[torch.Tensor] = []
        image_grid_thw_list: List[torch.Tensor] = []
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        act_placeholder_mask_list: List[torch.Tensor] = []

        num_latents = int(self.latent_action_num_queries)
        placeholder_token_id = int(self.placeholder_token_id)
        # print(placeholder_token_id)
        placeholder_str = "".join(["<ACT_PH>" for _ in range(num_latents)])
        for b, lang_raw in enumerate(lang_instructions_list):
            lang: str = lang_raw.decode().lower()

            # 两帧图像分别作为独立 image 输入，避免被当作视频处理
            if self.use_history_frame:
                user_content = [
                    {"type": "image", "image": img_pairs[b][0]},
                    {"type": "image", "image": img_pairs[b][1]},
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ]
            else:
                user_content = [
                    {"type": "image", "image": img_pairs[b][0]},
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ]

            messages = [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": placeholder_str}
                    ],
                },
            ]

            prefix_ids = self.processor.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True
            )
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True
            )

            input_ids = inputs.input_ids.squeeze(0)
            pixel_values = inputs.pixel_values.squeeze(0)
            image_grid_thw = getattr(inputs, "image_grid_thw", None)

            # 构造 labels：仅对 assistant 动作部分监督，其余为 IGNORE_INDEX
            prefix_len = len(prefix_ids[0])
            labels = input_ids.clone()
            labels[:prefix_len] = IGNORE_INDEX
            # assistant 内容长度 = num_latents 个占位符 token
            keep_end = prefix_len + num_latents
            labels[keep_end:] = IGNORE_INDEX

            # 占位符 mask（后续模型中用来替换为真实 <ACT_i>）
            act_placeholder_mask = (input_ids == placeholder_token_id)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(pixel_values)
            act_placeholder_mask_list.append(act_placeholder_mask)
            if image_grid_thw is not None:
                image_grid_thw_list.append(image_grid_thw) # (N_img, 3)
        # padding_side check
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"

        target_len = int(self.target_seq_len)
        seq_lengths = [min(t.size(-1), target_len) for t in input_ids_list]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids[:, :target_len]
        if input_ids.size(1) < target_len:
            pad_amt = target_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad_amt), value=self.pad_token_id)

        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        labels = labels[:, :target_len]
        if labels.size(1) < target_len:
            pad_amt = target_len - labels.size(1)
            labels = F.pad(labels, (0, pad_amt), value=IGNORE_INDEX)

        act_placeholder_mask = pad_sequence(
            act_placeholder_mask_list, batch_first=True, padding_value=0
        )
        act_placeholder_mask = act_placeholder_mask[:, :target_len]
        if act_placeholder_mask.size(1) < target_len:
            pad_amt = target_len - act_placeholder_mask.size(1)
            act_placeholder_mask = F.pad(act_placeholder_mask, (0, pad_amt), value=0)
        act_placeholder_mask = act_placeholder_mask.bool()

        lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
        attention_mask = (torch.arange(target_len, dtype=torch.long).unsqueeze(0) < lengths_tensor.unsqueeze(1))

        if isinstance(pixel_values_list[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values_list)
        elif isinstance(pixel_values_list[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values_list[idx][k] for idx in range(len(input_ids))]) for k in pixel_values_list[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_list[0])}")

        # 保持 batch 维度：列表元素 shape 通常为 [num_imgs, 3]
        if image_grid_thw_list:
            image_grid_thw = torch.stack(image_grid_thw_list)  # [B, num_imgs, 3]
            image_grid_thw = image_grid_thw.view(-1, 3)  # 展平为 [B*num_imgs, 3] 供 Qwen fast_pos_embed_interpolate
        else:
            image_grid_thw = None

        # 仅在可用时返回 image_grid_thw，避免 Accelerate 拼接 None 值报错
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            act_placeholder_mask=act_placeholder_mask,
            lam_videos=lam_videos,
            lam_states=lam_states,
        )
        if image_grid_thw is not None:
            output["image_grid_thw"] = image_grid_thw
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids

        return output


@dataclass
class PaddedCollatorForLatentWorldVLA_LIBERO:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    processor: Any = None
    latent_action_num_queries: int = 4
    target_seq_len: int = 330
    use_history_frame: bool = True
    predict_stop_token: bool = False  # 兼容旧签名，当前不使用

    def __post_init__(self):
        assert self.processor is not None, "processor 不能为空"
        tok = self.processor.tokenizer
        placeholder_token_id = tok.convert_tokens_to_ids("<ACT_PH>")
        assert placeholder_token_id != tok.unk_token_id, "未找到 <ACT_PH> 占位符，请确保已注册"
        self.placeholder_token_id = int(placeholder_token_id)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean_5d = mean.view(1, 1, 3, 1, 1)
        self.std_5d = std.view(1, 1, 3, 1, 1)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        lang_instructions_list: List[Any] = [instance["language_instruction"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        dataset_ids = [instance["dataset_id"] for instance in instances] if "dataset_id" in instances[0] else None

        # LAM 输入：视频与状态
        # 根据 use_history_frame 决定起始索引：
        # - use_history_frame=True: video[1:] 跳过历史帧，从窗口第一帧开始
        # - use_history_frame=False: video[:] 从窗口第一帧开始（没有历史帧）
        lam_start_idx = 1 if self.use_history_frame else 0
        lam_videos = torch.stack(
            [torch.as_tensor(instance["video"][lam_start_idx:]).permute(0, 3, 1, 2).float().div_(255.0) for instance in instances],
            dim=0,
        )
        if self.mean_5d.device != lam_videos.device:
            self.mean_5d = self.mean_5d.to(device=lam_videos.device, dtype=lam_videos.dtype)
            self.std_5d = self.std_5d.to(device=lam_videos.device, dtype=lam_videos.dtype)
        lam_videos = (lam_videos - self.mean_5d) / self.std_5d
        lam_states = torch.stack(
            [torch.as_tensor(instance["proprio"][lam_start_idx:]).float() for instance in instances],
            dim=0,
        )

        # Flow/动作监督
        actions = torch.stack([torch.from_numpy(instance["actions"]) for instance in instances], dim=0)
        # Flow Matching 的初始状态：使用窗口第一帧的 proprio（不是最后一帧）
        # - use_history_frame=True: proprio[1] 对应 video[1]（窗口第一帧/初始帧）
        # - use_history_frame=False: proprio[0] 对应 video[0]（窗口第一帧/初始帧）
        # 训练和推理时，flow 都只能接触这一初始状态
        proprio = torch.stack([torch.from_numpy(instance["proprio"][lam_start_idx]).float() for instance in instances], dim=0)

        # 构造带占位符的 prompt
        pixel_values_list: List[torch.Tensor] = []
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        act_placeholder_mask_list: List[torch.Tensor] = []
        image_grid_thw_list: List[torch.Tensor] = []

        num_latents = int(self.latent_action_num_queries)
        placeholder_token_id = int(self.placeholder_token_id)
        placeholder_str = "".join(["<ACT_PH>" for _ in range(num_latents)])

        for b, lang_raw in enumerate(lang_instructions_list):
            lang: str = lang_raw.decode().lower()
            video = instances[b]["video"]
            # VLM 输入：基于初始观察（窗口第一帧）预测动作
            # - use_history_frame=True: [历史帧 video[0], 初始帧 video[1]]
            # - use_history_frame=False: [初始帧 video[0]]
            # 注意：初始帧对应 video[lam_start_idx]，与 proprio 和 LAM 输入对齐
            if self.use_history_frame:
                user_content = [
                    {"type": "image", "image": video[0]},             # 历史帧 (t = -window_size)
                    {"type": "image", "image": video[lam_start_idx]}, # 初始帧 (t = -window_size+1)
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ]
            else:
                user_content = [
                    {"type": "image", "image": video[lam_start_idx]}, # 初始帧 (t = -window_size+1)
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ]

            messages = [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": placeholder_str}
                    ],
                },
            ]

            prefix_ids = self.processor.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True
            )
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True
            )

            input_ids = inputs.input_ids.squeeze(0)
            pixel_values = inputs.pixel_values.squeeze(0)
            image_grid_thw = getattr(inputs, "image_grid_thw", None)

            # labels 不用于 supervise_quantized，直接全 IGNORE
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            act_placeholder_mask = (input_ids == placeholder_token_id)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            pixel_values_list.append(pixel_values)
            act_placeholder_mask_list.append(act_placeholder_mask)
            if image_grid_thw is not None:
                image_grid_thw_list.append(image_grid_thw)

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        target_len = int(self.target_seq_len)
        seq_lengths = [min(t.size(-1), target_len) for t in input_ids_list]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids[:, :target_len]
        if input_ids.size(1) < target_len:
            pad_amt = target_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad_amt), value=self.pad_token_id)

        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        attention_mask = attention_mask[:, :target_len]
        if attention_mask.size(1) < target_len:
            pad_amt = target_len - attention_mask.size(1)
            attention_mask = F.pad(attention_mask, (0, pad_amt), value=0)

        act_placeholder_mask = pad_sequence(
            act_placeholder_mask_list, batch_first=True, padding_value=0
        )
        act_placeholder_mask = act_placeholder_mask[:, :target_len]
        if act_placeholder_mask.size(1) < target_len:
            pad_amt = target_len - act_placeholder_mask.size(1)
            act_placeholder_mask = F.pad(act_placeholder_mask, (0, pad_amt), value=0)
        act_placeholder_mask = act_placeholder_mask.bool()

        if isinstance(pixel_values_list[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values_list)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_list[0])}")

        image_grid_thw = torch.stack(image_grid_thw_list).view(-1, 3) if image_grid_thw_list else None

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            act_placeholder_mask=act_placeholder_mask,
            lam_videos=lam_videos,
            lam_states=lam_states,
            actions=actions,
            proprio=proprio,
        )
        if image_grid_thw is not None:
            output["image_grid_thw"] = image_grid_thw
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids
        return output


@dataclass
class PaddedCollatorForActionPrediction_LIBERO:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    # 与 PaddedCollatorForActionPrediction 对齐的新增字段
    action_tokenizer: Any = None
    processor: Any = None
    predict_stop_token: bool = False
    def __post_init__(self):
        # 预先缓存 mean/std 张量（避免每次调用都创建）
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean_5d = mean.view(1, 1, 3, 1, 1)
        self.std_5d = std.view(1, 1, 3, 1, 1)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 与标准 Action Collator 对齐：需要 action_tokenizer 和 processor
        assert self.action_tokenizer is not None and self.processor is not None, (
            "action_tokenizer 和 processor 需要在 Collator 初始化时提供，用于批量 VQ 编码与模板生成"
        )
        device = self.action_tokenizer.device
        # 收集批次数据（来自轻量 Transform，collate 阶段统一处理）
        lang_instructions_list: List[Any] = [instance["language_instruction"] for instance in instances]
        img_list = [instance["img"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        # 低层策略训练所需：actions / proprio（来自 Transform，保留原始输出）
        actions = torch.stack([torch.from_numpy(instance["actions"]) for instance in instances], dim=0)
        proprio = torch.stack([torch.from_numpy(instance["proprio"]) for instance in instances], dim=0)
        proprio_lam = proprio.to(device, dtype=torch.float32)
         # 批量 VQ 编码
        video_batch = torch.stack([torch.from_numpy(instance["video"]) for instance in instances], dim=0).to(device=device, dtype=torch.float32)
        video_batch = video_batch.permute(0, 1, 4, 2, 3).div_(255.0)
        if self.mean_5d.device != device:
            self.mean_5d = self.mean_5d.to(device=device, dtype=video_batch.dtype)
            self.std_5d = self.std_5d.to(device=device, dtype=video_batch.dtype)
        
        # 按批次归一化 (in-place)
        video_batch.sub_(self.mean_5d).div_(self.std_5d)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vq_out = self.action_tokenizer.vq_encode(videos=video_batch, states=proprio_lam, dec_in=video_batch[:,0], tgt=video_batch[:,-1], predict_future_frame=False)
            # 确保返回到 CPU，便于 DataLoader pin_memory 与后续非阻塞拷贝
            latent_action_idx_batch = vq_out['indices'].detach().cpu()  # [B, Q]
            first_image_features = vq_out['features'][:,:1]
            last_image_features = vq_out['features'][:,-1:]
            image_features = torch.cat([first_image_features, last_image_features], dim=1).detach().cpu()          # [B, 2, K, D]

        proprio = proprio[:,0]

        # 基于 chat 模板构建 input_ids/labels
        pixel_values_list: List[torch.Tensor] = []
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        for b, lang_raw in enumerate(lang_instructions_list):
            lang: str = lang_raw.decode().lower()

            latent_action_idx = latent_action_idx_batch[b]
            action_tokens = ''.join([f'<ACT_{int(i.item())}>' for i in latent_action_idx])
            
            # 新写法：把 image 直接放到 content 列表中，让 processor 自动插入图像占位
            user_content = [
                {"type": "image", "image": img_list[b]},   # 支持 PIL.Image 或 Tensor（由 processor 决定）
                {"type": "text", "text": f"What action should the robot take to {lang}?"},
            ]

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a robot controller. Based on visual input and instructions, always output exactly 4 latent action tokens chosen from <ACT_0> ... <ACT_15>."}
                    ],
                },
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": action_tokens}
                    ],
                },
            ]

            prefix_ids = self.processor.apply_chat_template(
                messages[:2], tokenize=True, add_generation_prompt=True
            )
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt",return_dict=True
            )
            input_ids = inputs.input_ids.squeeze(0)
            # print(self.processor.tokenizer.decode(input_ids[34+5+256:],skip_special_tokens=False))
            input_ids_list.append(input_ids)
            pixel_values_list.append(inputs.pixel_values.squeeze(0))

            prefix_len = len(prefix_ids[0])
            labels = input_ids.clone()
            labels[:prefix_len] = IGNORE_INDEX
            act_ids = self.processor.tokenizer(action_tokens, add_special_tokens=False)["input_ids"]
            keep_end = prefix_len + len(act_ids)
            labels[keep_end:] = IGNORE_INDEX
            labels_list.append(labels)

        # padding_side check
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"

        # 目标长度与长度统计
        target_len = 320
        seq_lengths = [min(t.size(-1), target_len) for t in input_ids_list]

        # Pad input_ids
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids[:, :target_len]
        if input_ids.size(1) < target_len:
            pad_amt = target_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad_amt), value=self.pad_token_id)

        # Pad labels
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        labels = labels[:, :target_len]
        if labels.size(1) < target_len:
            pad_amt = target_len - labels.size(1)
            labels = F.pad(labels, (0, pad_amt), value=IGNORE_INDEX)

        # Attention mask
        lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
        attention_mask = (torch.arange(target_len, dtype=torch.long).unsqueeze(0) < lengths_tensor.unsqueeze(1))


        latent_action_idx = latent_action_idx_batch

        # pixel_values 组装
        if isinstance(pixel_values_list[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values_list)
        elif isinstance(pixel_values_list[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values_list[idx][k] for idx in range(len(input_ids))]) for k in pixel_values_list[0]
            }

        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_list[0])}")

        dataset_ids = [instance["dataset_id"] for instance in instances] if "dataset_id" in instances[0] else None
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            latent_action_idx=latent_action_idx,
            proprio=proprio,
            image_features=image_features,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids

        return output


@dataclass
class PaddedCollatorForActionPrediction_R2R:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        
        
        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(torch.stack(instance["initial_pixel_values_hist"]))
                target_pixel_values_hist.append(torch.stack(instance["target_pixel_values_hist"]))
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))   

        
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None
        if "dataset_id" in instances[0]:
            dataset_ids = [instance["dataset_id"] for instance in instances]
        else:
            dataset_ids = None

        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]

        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # labels=labels,
            actions=actions,
            # proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids
        return output
        
@dataclass
class PaddedCollatorForActionPrediction_CALVIN:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     



        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None
        dataset_ids = [instance["dataset_id"] for instance in instances] if "dataset_id" in instances[0] else None


        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # labels=labels,
            actions=actions,
            proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids
        return output


@dataclass
class CollatorForLatentAction:
    """
    纯 LAM 训练的 Collator。
    
    Args:
        pixel_values_dtype: 图像数据类型
        use_history_frame: 是否使用历史帧
            - True: 数据包含历史帧，需要跳过 video[0] 和 proprio[0]
            - False: 数据不包含历史帧，使用完整序列
    
    数据结构：
    - use_history_frame=True:
        - video[0] = 历史帧, video[1] = 初始帧, video[2:] = 未来帧
        - LAM 使用 video[1:]
    - use_history_frame=False:
        - video[0] = 初始帧, video[1:] = 未来帧
        - LAM 使用完整 video[:]
    """
    pixel_values_dtype: torch.dtype = torch.float32
    use_history_frame: bool = True

    def __post_init__(self):
        # 预先缓存 mean/std 张量（避免每次调用都创建）
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean_5d = mean.view(1, 1, 3, 1, 1)
        self.std_5d = std.view(1, 1, 3, 1, 1)

    @staticmethod
    def _relative_action_from_states(state_start: np.ndarray, state_end: np.ndarray) -> np.ndarray:
        """
        计算首末帧的相对状态增量（物理接地）。
        
        Args:
            state_start: 初始状态 (state_dim,) = [x, y, z, ori..., gripper]
            state_end: 最终状态 (state_dim,) = [x, y, z, ori..., gripper]
        
        Returns:
            相对增量 (state_dim,) = [dx, dy, dz, dori..., gripper_end]
        """
        p0 = state_start[:3]
        p1 = state_end[:3]
        delta_p = p1 - p0

        ori0 = state_start[3:-2]
        ori1 = state_end[3:-2]

        if ori0.shape[0] == 3:
            # 欧拉角 (roll, pitch, yaw)
            R0 = R.from_euler("xyz", ori0)
            R1 = R.from_euler("xyz", ori1)
            R_rel = R0.inv() * R1
            delta_r = R_rel.as_euler("xyz", degrees=False)
        elif ori0.shape[0] == 4:
            # 四元数 (x, y, z, w)
            R0 = R.from_quat(ori0)
            R1 = R.from_quat(ori1)
            R_rel = R0.inv() * R1
            delta_r = R_rel.as_quat()
        else:
            # 其他表示：直接差分
            delta_r = ori1 - ori0

        gripper = state_end[-2:]
        return np.concatenate([delta_p, delta_r, gripper], axis=-1)


    @torch.no_grad()
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将 batch 数据整理为 LAM 训练所需格式。
        
        输入数据结构（来自 RLDSBatchTransformVideo）：
        - video: [T, C, H, W] - 视频序列（根据 use_history_frame 可能包含历史帧）
        - dec_video: [T, C, H, W] - Decoder 输入（独立数据增强）
        - proprio: [T, state_dim] - 状态序列（根据 use_history_frame 可能包含历史状态）
        - action: [T, action_dim] - 动作序列
        
        帧选择（根据 use_history_frame）：
        - use_history_frame=True: 跳过 video[0] 和 proprio[0]（历史帧/状态）
        - use_history_frame=False: 使用完整序列
        
        输出：
        - videos: [B, T', C, H, W] - 归一化后的视频序列（从初始帧开始）
        - dec_videos: [B, T', C, H, W] - 归一化后的 decoder 输入（从初始帧开始）
        - action: [B, T, action_dim] - 动作序列
        - proprio: [B, T', state_dim] - 状态序列（从初始状态开始）
        - delta_proprio: [B, state_dim] - 初始状态到最终状态的相对增量（用于物理接地）
        """
        # 根据 use_history_frame 选择帧范围
        lam_start_idx = 1 if self.use_history_frame else 0
        
        videos = torch.stack([ins["video"][lam_start_idx:] for ins in instances], dim=0)  # [B, T', C, H, W]
        dec_videos = torch.stack([ins["dec_video"][lam_start_idx:] for ins in instances], dim=0)  # [B, T', C, H, W]

        # 提取 dtype/device
        device, dtype = videos.device, videos.dtype

        # 将 mean/std 移到相同 device（第一次时自动）
        if self.mean_5d.device != device:
            self.mean_5d = self.mean_5d.to(device=device, dtype=dtype)
            self.std_5d = self.std_5d.to(device=device, dtype=dtype)

        # 按批次归一化（in-place，ImageNet 标准化）
        videos.sub_(self.mean_5d).div_(self.std_5d)
        dec_videos.sub_(self.mean_5d).div_(self.std_5d)

        # 批量 action/proprio（根据 use_history_frame 选择起始索引）
        actions = torch.stack([torch.from_numpy(instance["action"]) for instance in instances], dim=0)  # [B, T, action_dim]
        proprios_np = [instance["proprio"][lam_start_idx:] for instance in instances]
        proprios = torch.stack([torch.from_numpy(p) for p in proprios_np], dim=0)  # [B, T', state_dim]

        # 计算相对状态增量（物理接地）- 批量处理
        # proprios_np[b][0] = 初始状态（窗口第一帧），proprios_np[b][-1] = 最终状态
        delta_states = torch.stack([
            torch.from_numpy(self._relative_action_from_states(p[0], p[-1])).float()
            for p in proprios_np
        ], dim=0)  # [B, state_dim]

        # 若有 dataset_name
        dataset_names = None
        dataset_ids = None
        if "dataset_name" in instances[0]:
            dataset_names = [ins["dataset_name"] for ins in instances]
        if "dataset_id" in instances[0]:
            dataset_ids = [ins["dataset_id"] for ins in instances]

        batch = {
            "videos": videos.contiguous(),
            "dec_videos": dec_videos.contiguous(),
            "action": actions,
            "proprio": proprios,
            "delta_proprio": delta_states,
        }

        if dataset_names is not None:
            batch["dataset_names"] = dataset_names
        if dataset_ids is not None:
            batch["dataset_ids"] = dataset_ids

        return batch


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None
        if "dataset_id" in instances[0]:
            dataset_ids = [instance["dataset_id"] for instance in instances]
        else:
            dataset_ids = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        initial_pixel_values_view2 = [instance["initial_pixel_values_view2"] for instance in instances]
        initial_pixel_values_view2 = torch.stack(initial_pixel_values_view2)
        
        target_pixel_values_view2 = [instance["target_pixel_values_view2"] for instance in instances]
        target_pixel_values_view2 = torch.stack(target_pixel_values_view2)
        pixel_values_view2 = torch.stack([initial_pixel_values_view2, target_pixel_values_view2], dim=1)
        


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            videos_view2=pixel_values_view2,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if dataset_ids is not None:
            output["dataset_ids"] = dataset_ids

        return output