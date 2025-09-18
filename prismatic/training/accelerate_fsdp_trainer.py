"""
accelerate_fsdp_trainer.py

使用 Hugging Face transformers.Trainer（内置 Accelerate/FSDP 集成）进行最小化训练循环，
专注 VLA 潜变量动作训练，并保留动作精度日志与 HF 兼容保存。

约定：
- 依赖新的 HF-only `PrismaticVLM`（InternVLForConditionalGeneration 封装）。
- 通过 Trainer 的 TrainingArguments 配置混合精度、分布式、FSDP、日志与检查点。
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Optional, Any, cast

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from dataclasses import asdict
from transformers import AutoModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM


class SaveProcessorCallback(TrainerCallback):
    """在保存 checkpoint 与训练结束时，额外保存 processor/tokenizer。"""

    def __init__(self, *, processor: Optional[Any], tokenizer: Optional[Any]):
        self.processor = processor
        self.tokenizer = tokenizer

    def on_save(self, args: TrainingArguments, state, control, **kwargs):  # type: ignore[override]
        try:
            ckpt_dir = os.path.join(cast(str, args.output_dir), f"checkpoint-{state.global_step}")
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(ckpt_dir)
            elif self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(ckpt_dir)
        except Exception:
            pass

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):  # type: ignore[override]
        try:
            out_dir = cast(str, args.output_dir)
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(out_dir)
            elif self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(out_dir)
        except Exception:
            pass


class ActionAccuracyTrainer(Trainer):
    """自定义 Trainer：在 compute_loss 中计算并记录 action_accuracy。"""

    def __init__(self, *args, action_token_begin_id: int = 32000, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_token_begin_id = int(action_token_begin_id)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):  # type: ignore[override]
        outputs: CausalLMOutputWithPast = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model must return loss in outputs for training.")

        # 动作精度（仅训练日志，不反传）
        try:
            with torch.no_grad():
                logits = cast(torch.Tensor, outputs.logits)
                labels = cast(torch.Tensor, inputs.get("labels"))
                if logits is not None and labels is not None:
                    pred_ids = logits.argmax(dim=2)
                    mask = (labels != -100) & (labels > self.action_token_begin_id)
                    num = ((pred_ids == labels) & mask).sum().float()
                    den = mask.sum().float().clamp_min(1)
                    acc = (num / den).detach()
                    # 仅在主进程记录；Trainer.log 会处理分布式可见性
                    self.log({"action_accuracy": float(acc.item())})
        except Exception:
            # 指标计算失败不应中断训练
            pass

        return (loss, outputs) if return_outputs else loss


def run_latent_action_training(
    *,
    cfg: Any,
    vlm: AutoModel,
    vla_dataset: TorchIterableDataset,
    collator,
    tokenizer,
    run_dir: Path,
    overwatch,
) -> None:
    """使用 transformers.Trainer 进行潜变量动作训练。

    必要的 cfg 字段（YAML 驱动）：
      - per_device_batch_size, epochs, max_steps, learning_rate, weight_decay,
        lr_scheduler_type, warmup_ratio, enable_mixed_precision_training
      - save_interval
      - 可选：gradient_accumulation_steps, wandb_project, wandb_entity,
        action_token_begin_id, fsdp, fsdp_config, save_total_limit, optim
    """

    per_device_bsz: int = cfg.per_device_batch_size

    max_steps: Optional[int] = cfg.max_steps
    epochs: int = cfg.epochs
    lr: float = cfg.learning_rate
    wd: float = cfg.weight_decay
    lr_type: str = cfg.lr_scheduler_type
    warmup_ratio: float = float(cfg.warmup_ratio)
    use_mp: bool = bool(getattr(cfg, "enable_mixed_precision_training", True))
    max_grad_norm: float = float(getattr(cfg, "max_grad_norm", 1.0))

    save_interval: int = int(getattr(cfg, "save_interval", 2500))
    grad_accum_steps: int = int(getattr(cfg, "gradient_accumulation_steps", 1))
    action_begin: int = int(getattr(cfg, "action_token_begin_id", 32000))
    logging_steps: int = int(getattr(cfg, "logging_steps", 1))
    dataloader_num_workers: int = int(getattr(cfg, "dataloader_num_workers", 0))
    dataloader_pin_memory: bool = bool(getattr(cfg, "dataloader_pin_memory", True))
    seed: int = int(getattr(cfg, "seed", 42))

    # 估算 steps（IterableDataset 常无限；以 max_steps 为准，若为空则按 epochs * 10000 近似）
    if max_steps is None:
        max_steps = epochs * 10000

    # W&B 项目与实体（Trainer 内部初始化前设置 env）
    if getattr(cfg, "wandb_project", None):
        os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb_project))
    if getattr(cfg, "wandb_entity", None):
        os.environ.setdefault("WANDB_ENTITY", str(cfg.wandb_entity))



    output_dir = Path(run_dir) / "hf_trainer"
    # lr_scheduler_type 映射：将自定义字符串映射为 HF 支持的类型
    if lr_type == "linear-warmup+cosine-decay":
        hf_lr_type = "cosine"
    else:
        hf_lr_type = lr_type
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_bsz,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=False,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=max_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        lr_scheduler_type=hf_lr_type,  # type: ignore[arg-type]
        warmup_ratio=warmup_ratio,
        bf16=use_mp,
        fp16=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_interval,
        save_total_limit=int(getattr(cfg, "save_total_limit", 3)),
        report_to=["wandb"],
        run_name=str(getattr(cfg, "run_id", "run")),
        max_grad_norm=max_grad_norm,
        optim=str(getattr(cfg, "optim", "adamw_torch")),
        save_safetensors=True,
        seed=seed,
        ddp_find_unused_parameters=False,
        fsdp=getattr(cfg, "fsdp", None),  # 例如："full_shard auto_wrap" 或 None
        fsdp_config=getattr(cfg, "fsdp_config", None),
    )

    callbacks = [
        SaveProcessorCallback(processor=getattr(vlm, "processor", None), tokenizer=tokenizer),
    ]

    trainer = ActionAccuracyTrainer(
        model=vlm,
        args=args,
        train_dataset=vla_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        action_token_begin_id=action_begin,
    )

    trainer.train()
    


