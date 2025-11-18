import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.models import load_InternVL, freeze_internvl
from prismatic.vla import get_latent_vla_dataset_and_collator
from latent_action_model.core.lam_model import load_latent_action_model
from transformers import AutoProcessor
from torch.utils.data import DataLoader


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


home_path = "/mnt/public_zgc/home/jlchen"


@dataclass
class InferenceConfig:
    # Paths
    data_root_dir: Path = Path(home_path + "/datasets")
    run_root_dir: Path = Path(__file__).resolve().parent / "vla_log"
    model_id: str = home_path + "/weights/InternVL3_5-1B-Instruct-HF"
    hf_cache_dir: Optional[Path] = None
    lam_path: str = home_path + "/code/UniVLA/latent_action_model/logs/vq_div_01_vq_1/version_3/checkpoints/epoch=4_step=110000.ckpt"

    # Data
    data_mix: str = "droid_100"
    image_resolution: int = 448
    shuffle_buffer_size: int = 20_000
    image_aug: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Token/action config
    action_token_begin_id: int = 151679
    vision_model_id: str = home_path + "/weights/vjepa2-vitl-fpc64-256"
    codebook_size: int = 16

    # Generation
    per_device_eval_batch_size: int = 24
    max_input_length: int = 400
    max_new_tokens: int = 4
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50

    # Misc
    seed: int = 42
    run_id: Optional[str] = None
    run_id_note: Optional[str] = "infer_01"
    hf_token: Optional[str] = None


def _build_messages_for_sample(img: Any, instruction_bytes: bytes) -> List[Dict[str, Any]]:
    lang = instruction_bytes.decode().lower()
    user_content = [
        {"type": "image", "image": img},
        {"type": "text", "text": f"What action should the robot take to {lang}?"},
    ]
    messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": (
                "You are a robot controller. Based on visual input and instructions, "
                "always output exactly 4 latent action tokens chosen from <ACT_0> ... <ACT_15>."
            )}
        ],
    },
    {"role": "user", "content": user_content},
    ]
    return messages


def _inference_collate(instances: List[Dict[str, Any]], tokenizer, pad_token_id: int, max_input_length: int, lam_model) -> Dict[str, torch.Tensor]:
    input_ids_list: List[torch.Tensor] = []
    pixel_values_list: List[Any] = []
    seq_lengths: List[int] = []
    gt_action_ids_list: List[torch.Tensor] = []

    for inst in instances:
        messages = _build_messages_for_sample(inst["img"], inst["language_instruction"])  # system + user only
        # Use chat template with generation prompt
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        # `inputs` is expected to contain `input_ids` and `pixel_values`
        input_ids = inputs.input_ids  # [1, L]
        input_ids_list.append(input_ids)
        seq_lengths.append(int(input_ids.size(1)))
        pixel_values_list.append(inputs.pixel_values)

    # Build ground-truth latent action indices via LAM on the batched two-frame stack
    pair_stack = [torch.stack([inst["initial_pixel_values"], inst["target_pixel_values"]], dim=0) for inst in instances]
    video_batch = torch.stack(pair_stack, dim=0).to(lam_model.device)
    with torch.no_grad():
        vq_out = lam_model.vq_encode(video_batch)
        latent_action_idx_batch = vq_out['indices']  # [B, Q]
    gt_action_ids = latent_action_idx_batch.detach().cpu()

    # Determine target length (cap by max_input_length, or batch max)
    target_len = min(max(seq_lengths), max_input_length)

    # Pad input_ids to target_len (right padding)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids[:, :target_len]
    if input_ids.size(1) < target_len:
        pad_amt = target_len - input_ids.size(1)
        input_ids = F.pad(input_ids, (0, pad_amt), value=pad_token_id)

    # Attention mask from true lengths
    lengths_tensor = torch.tensor([min(l, target_len) for l in seq_lengths], dtype=torch.long)
    attention_mask = (torch.arange(target_len, dtype=torch.long).unsqueeze(0) < lengths_tensor.unsqueeze(1))

    # Stack pixel_values; support tensor or dict
    if isinstance(pixel_values_list[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values_list)
    elif isinstance(pixel_values_list[0], dict):
        pixel_values = {
            k: torch.stack([pixel_values_list[idx][k] for idx in range(len(pixel_values_list))])
            for k in pixel_values_list[0]
        }
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_list[0])}")

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        input_lengths=lengths_tensor,
        gt_action_ids=gt_action_ids,
    )


def _extract_action_tokens(text: str) -> Tuple[List[str], List[int]]:
    import re

    tokens = re.findall(r"<ACT_(\d+)>", text)
    ids = [int(t) for t in tokens]
    formatted = [f"<ACT_{i}>" for i in ids]
    return formatted, ids


def infer(cfg: InferenceConfig) -> None:
    overwatch.info("OpenVLA Inference :: Initializing")

    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Set seed
    _ = set_global_seed(cfg.seed, get_worker_init_fn=False)

    # Unique run directory
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    vla_tag = f"{cfg.model_id.split('/')[-1]}+{cfg.data_mix}"
    cfg.run_id = cfg.run_id or f"{vla_tag}+infer+seed{cfg.seed}"
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    run_dir = (cfg.run_root_dir / f"{timestamp}+{cfg.run_id}")

    os.makedirs(run_dir, exist_ok=True)
    try:
        import logging

        log_path = str(run_dir / "infer.log")
        root_logger = logging.getLogger()
        already_attached = False
        for h in list(root_logger.handlers):
            try:
                if hasattr(h, "baseFilename") and getattr(h, "baseFilename") == log_path:
                    already_attached = True
                    break
            except Exception:
                continue
        if not already_attached:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            formatter = logging.Formatter("| >> %(message)s", datefmt="%m/%d [%H:%M:%S]")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    except Exception:
        pass

    # Load model & processor
    overwatch.info(f"🔄 加载基础 InternVL `{cfg.model_id}`（HF from_pretrained）")
    vlm, processor = load_InternVL(cfg.model_id, cfg.hf_cache_dir, dtype=torch.bfloat16)
    tokenizer = processor.tokenizer
    vlm.generation_config.max_new_tokens = int(getattr(cfg, "max_new_tokens", 4))
    vlm.generation_config.pad_token_id = int(tokenizer.eos_token_id)
    vlm.config.use_cache = True

    # Optional frozen modules (for safety)
    freeze_internvl(vlm, True,True,True,True)

    # Ensure action tokens are recognized by tokenizer (no-op if already present)
    overwatch.info("🔧 注入/校验动作离散 token")
    special_tokens_dict = {"additional_special_tokens": [f"<ACT_{i}>" for i in range(cfg.codebook_size)]}
    try:
        _ = tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
    except Exception:
        pass
    act_tokens = [f"<ACT_{i}>" for i in range(cfg.codebook_size)]
    act_ids = tokenizer.convert_tokens_to_ids(act_tokens)
    expected_begin_id = min(act_ids)
    expected_end_id = max(act_ids)
    assert cfg.action_token_begin_id == expected_begin_id, (
        f"cfg.action_token_begin_id={cfg.action_token_begin_id} but tokenizer gives {expected_begin_id} "
        f"(range: {expected_begin_id}-{expected_end_id})"
    )

    # Load dataset (same as training), but we'll build our own inference collate
    overwatch.info(
        f"🔄 构建 RLDS 数据集（mixture=`{cfg.data_mix}`，image_res={cfg.image_resolution}）；推理模式"
    )
    latent_action_model = load_latent_action_model(cfg.lam_path, vision_model_id=cfg.vision_model_id)
    latent_action_model = latent_action_model.to(device_id).eval()

    train_dataset, val_dataset, _train_collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        latent_action_model,
        processor=processor,
        default_image_resolution=cfg.image_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # Inference DataLoader over validation split
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return _inference_collate(batch, tokenizer, int(tokenizer.eos_token_id), cfg.max_input_length, latent_action_model)

    dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=cfg.dataloader_pin_memory,
        collate_fn=collate_fn,
    )

    # Move model to device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    vlm = vlm.to(device)
    vlm.eval()

    # Configure generation params
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=cfg.max_new_tokens,
        num_beams=cfg.num_beams,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        pad_token_id=int(tokenizer.eos_token_id),
    )

    # Metrics accumulators
    n_examples = 0
    num_exact_match = 0
    token_correct = 0
    token_total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].tolist()
            gt_action_ids_batch = batch["gt_action_ids"]  # on CPU, shape [B, Q]

            # pixel_values can be Tensor or Dict[str, Tensor]
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(device, non_blocking=True) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(device, non_blocking=True)

            generated = vlm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **gen_kwargs,
            )

            # Evaluate predictions
            B = generated.size(0)
            Q = int(gt_action_ids_batch.size(1))
            for b in range(B):
                gen_seq = generated[b]
                start = int(input_lengths[b])
                new_tokens = gen_seq[start:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                _, pred_ids = _extract_action_tokens(text)

                gt_ids = gt_action_ids_batch[b].tolist()
                pred_ids = pred_ids[:Q]

                # Exact match
                if len(pred_ids) == Q and all(int(pred_ids[i]) == int(gt_ids[i]) for i in range(Q)):
                    num_exact_match += 1

                # Token-level accuracy (position-wise)
                for i in range(Q):
                    if i < len(pred_ids) and int(pred_ids[i]) == int(gt_ids[i]):
                        token_correct += 1
                    token_total += 1

            n_examples += B

    exact_match_acc = (num_exact_match / n_examples) if n_examples > 0 else 0.0
    token_acc = (token_correct / token_total) if token_total > 0 else 0.0
    overwatch.info(f"✅ 推理完成；样本数={n_examples} | 序列完全匹配精度={exact_match_acc:.4f} | 逐位精度={token_acc:.4f}")


if __name__ == "__main__":
    infer(InferenceConfig())


