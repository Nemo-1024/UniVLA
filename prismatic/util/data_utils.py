"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import re
import string
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Any, Optional, List

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

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    # 在 collate 阶段批处理 VQ 编码与文本模板（必需）
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
        # 批量 VQ 编码 + 文本模板生成（仅支持新路径，不保留向后兼容）
        assert self.action_tokenizer is not None and self.processor is not None, (
            "action_tokenizer 和 processor 需要在 Collator 初始化时提供，用于批量 VQ 编码与模板生成"
        )
        device = self.action_tokenizer.device
 
        # 收集批次数据（来自轻量 Transform，collate 阶段统一处理）
        lang_instructions_list: List[Any] = [instance["language_instruction"] for instance in instances]
        img_list = [instance["img"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        proprio = torch.as_tensor([instance["proprio"] for instance in instances], dtype=torch.float32, device=device)

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
                vq_out = self.action_tokenizer.vq_encode(videos=video_batch, states=proprio, predict_future_frame=False)
            # 确保返回到 CPU，便于 DataLoader pin_memory 与后续非阻塞拷贝
            latent_action_idx_batch = vq_out['indices'].detach().cpu()  # [B, Q]

        # 构建 input_ids/labels
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
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": "You are a robot controller. Based on visual input and instructions, always output exactly 4 latent action tokens chosen from <ACT_0> ... <ACT_15>."}
                #     ],
                # },
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": action_tokens}
                    ],
                },
            ]

            prefix_ids = self.processor.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True
            )
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt",return_dict=True
            )
            input_ids = inputs.input_ids.squeeze(0)
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

        # 目标长度
        target_len = 300
        seq_lengths = [min(t.size(-1), target_len) for t in input_ids_list]

        # Pad input_ids
        # input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)[:, :self.model_max_length]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids[:, :target_len]
        if input_ids.size(1) < target_len:
            pad_amt = target_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad_amt), value=self.pad_token_id)
        # labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)[:, :self.model_max_length]
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        labels = labels[:, :target_len]
        if labels.size(1) < target_len:
            pad_amt = target_len - labels.size(1)
            labels = F.pad(labels, (0, pad_amt), value=IGNORE_INDEX)

        # Attention mask
        lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
        attention_mask = (torch.arange(target_len, dtype=torch.long).unsqueeze(0) < lengths_tensor.unsqueeze(1))

        # Stack pixel_values
        if isinstance(pixel_values_list[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values_list)
        elif isinstance(pixel_values_list[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values_list[idx][k] for idx in range(len(input_ids))]) for k in pixel_values_list[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_list[0])}")

        # 输出
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if labels is not None:
            output["labels"] = labels
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

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
        target_len = 350
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
        return output


@dataclass
class CollatorForLatentAction:
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # 预先缓存 mean/std 张量（避免每次调用都创建）
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean_5d = mean.view(1, 1, 3, 1, 1)
        self.std_5d = std.view(1, 1, 3, 1, 1)


    @torch.no_grad()
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 直接堆叠 batch 张量
        videos = torch.stack([ins["video"] for ins in instances], dim=0)
        dec_videos = torch.stack([ins["dec_video"] for ins in instances], dim=0)


        # 仅一次性提取 dtype/device
        device, dtype = videos.device, videos.dtype

        # 将 mean/std 移到相同 device（第一次时自动）
        if self.mean_5d.device != device:
            self.mean_5d = self.mean_5d.to(device=device, dtype=dtype)
            self.std_5d = self.std_5d.to(device=device, dtype=dtype)

        # 按批次归一化 (in-place)
        videos.sub_(self.mean_5d).div_(self.std_5d)
        dec_videos.sub_(self.mean_5d).div_(self.std_5d)

        # 批量 action/proprio (直接 as_tensor)
        actions = torch.stack([torch.from_numpy(instance["action"]) for instance in instances], dim=0)
        proprios = torch.stack([torch.from_numpy(instance["proprio"]) for instance in instances], dim=0)

        # 若有 dataset_name
        dataset_names = None
        if "dataset_name" in instances[0]:
            dataset_names = [ins["dataset_name"] for ins in instances]

        batch = {
            "videos": videos.contiguous(),
            "dec_videos": dec_videos.contiguous(),
            "action": actions,
            "proprio": proprios,
        }

        if dataset_names is not None:
            batch["dataset_names"] = dataset_names

        return batch


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

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

        return output