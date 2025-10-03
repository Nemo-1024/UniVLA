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
from typing import Optional, Any, cast, Dict

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import AutoModel
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import wandb

from prismatic.models.vlms import PrismaticVLM


class SaveProcessorCallback(TrainerCallback):
    """在保存 checkpoint 与训练结束时，额外保存 processor/tokenizer。"""

    def __init__(self, *, processor: Optional[Any], tokenizer: Optional[Any]):
        self.processor = processor
        self.tokenizer = tokenizer

    def on_save(self, args: TrainingArguments, state, control, **kwargs):  # type: ignore[override]
        # 仅 rank0 保存，避免每个进程都写入
        try:
            from prismatic.overwatch import initialize_overwatch

            ow = initialize_overwatch(__name__)
            if not ow.is_rank_zero():
                return
            ckpt_dir = os.path.join(cast(str, args.output_dir), f"checkpoint-{state.global_step}")
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(ckpt_dir)
            elif self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(ckpt_dir)
        except Exception:
            pass

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):  # type: ignore[override]
        # 仅 rank0 保存，避免每个进程都写入
        try:
            from prismatic.overwatch import initialize_overwatch

            ow = initialize_overwatch(__name__)
            if not ow.is_rank_zero():
                return
            out_dir = cast(str, args.output_dir)
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(out_dir)
            elif self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(out_dir)
        except Exception:
            pass


class BatchActionAccuracyAndBestCallback(TrainerCallback):
    """累积 action token 精度，并记录最佳 eval_action_accuracy 到 JSON。"""

    def __init__(self, action_token_begin_id: int, run_dir: Path, overwatch):
        self.action_token_begin_id = action_token_begin_id
        self.run_dir = Path(run_dir)
        self.overwatch = overwatch
        self.reset()
        self.best_acc: float = float("-inf")
        self.best_step: Optional[int] = None
        self._pending_best_step: Optional[int] = None

    def reset(self):
        return

    def on_prediction_step(self, args, state, control, model=None, inputs=None, outputs=None, **kwargs):
        # 训练步不再做累计，统一由 compute_metrics 在评估阶段统计
        return control


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_action_accuracy" not in metrics:
            if self.overwatch.is_rank_zero():
                self.overwatch.info("eval_action_accuracy 缺失，请启用 eval_strategy 与 compute_metrics")
            return control

        try:
            acc = float(metrics["eval_action_accuracy"])  # type: ignore[assignment]
        except Exception:
            acc = 0.0

        if self.overwatch.is_rank_zero():
            self.overwatch.info(f"eval_action_accuracy={acc:.6f} @ step={state.global_step}")

        # 更新最佳
        if acc > self.best_acc:
            self.best_acc = float(acc)
            self.best_step = int(state.global_step)
            self._pending_best_step = self.best_step
            if self.overwatch.is_rank_zero():
                self.overwatch.info(f"发现新的最佳 eval_action_accuracy={self.best_acc:.6f} @ step={self.best_step}")

        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.overwatch.is_rank_zero():
            return
        if self._pending_best_step is None or int(state.global_step) != int(self._pending_best_step):
            return
        try:
            import json
            with open(self.run_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"best_eval_action_accuracy": self.best_acc, "best_step": self.best_step}, f)
            self.overwatch.info(f"更新 best_metrics.json: step={self.best_step}, acc={self.best_acc:.6f}")
        except Exception:
            pass
        self._pending_best_step = None



class LoggingCallback(TrainerCallback):
    """同时写入本地日志与 wandb，并处理 train/val 前缀。"""

    def __init__(self, overwatch=None, log_to_local=True):
        self.overwatch = overwatch

        self.log_to_local = log_to_local

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        if logs is None:
            return

        # 分布式只在 rank0 写
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        # 本地日志
        if self.log_to_local and self.overwatch is not None:
            try:
                parts = [f"step={int(state.global_step)}"]
                for k in sorted(logs.keys()):
                    v = logs[k]
                    try:
                        fv = float(v) if isinstance(v, (int, float)) else float(v)
                        parts.append(f"{k}={fv:.6f}")
                    except Exception:
                        parts.append(f"{k}={v}")
                msg = " | ".join(parts)
                self.overwatch.info(msg)
            except Exception:
                pass



def run_latent_action_training(
    *,
    cfg: Optional[Any],
    vlm: AutoModel,
    vla_dataset: TorchIterableDataset,
    eval_dataset: Optional[Any] = None,
    collator,
    tokenizer,
    run_dir: Path,
    overwatch,
) -> None:
    """使用 transformers.Seq2SeqTrainer 进行潜变量动作训练。"""


    max_steps: Optional[int] = cfg.max_steps
    epochs: int = cfg.epochs
    lr: float = cfg.learning_rate
    wd: float = cfg.weight_decay
    use_mp: bool = bool(getattr(cfg, "enable_mixed_precision_training", True))
    max_grad_norm: float = float(getattr(cfg, "max_grad_norm", 1.0))

    save_interval: int = int(getattr(cfg, "save_interval", 2500))
    grad_accum_steps: int = int(getattr(cfg, "gradient_accumulation_steps", 1))
    action_begin: int = int(getattr(cfg, "action_token_begin_id", 32000))
    logging_steps: int = int(getattr(cfg, "logging_steps", 1))
    per_device_eval_bsz: int = cfg.per_device_eval_batch_size
    dataloader_num_workers: int = int(getattr(cfg, "dataloader_num_workers", 0))
    dataloader_pin_memory: bool = bool(getattr(cfg, "dataloader_pin_memory", True))
    seed: int = int(getattr(cfg, "seed", 42))

    if max_steps is None:
        max_steps = epochs * 10000

    if getattr(cfg, "wandb_project", None):
        os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb_project))
    if getattr(cfg, "wandb_entity", None):
        os.environ.setdefault("WANDB_ENTITY", str(cfg.wandb_entity))

    # 所有 checkpoint 写入专属子目录
    ckpt_output_dir = Path(run_dir) / "checkpoints"
    args = Seq2SeqTrainingArguments(
        output_dir=ckpt_output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_bsz,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=False,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=max_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        lr_scheduler_type=cfg.lr_scheduler_type,  # type: ignore[arg-type]
        warmup_steps=cfg.warmup_steps,
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
        fsdp=getattr(cfg, "fsdp", None),
        fsdp_config=getattr(cfg, "fsdp_config", None),
        eval_strategy=str(getattr(cfg, "eval_strategy", "no")),  
        eval_steps=int(getattr(cfg, "eval_interval", save_interval)),
        eval_accumulation_steps=int(getattr(cfg, "eval_accumulation_steps", 1)),
        prediction_loss_only=False,
        # prediction_loss_only = False
        # predict_with_generate=True,  # 🔑 启用 generate
        # generation_num_beams=int(getattr(cfg, "generation_num_beams", 1)),      # 可调
    )

    callbacks = [
        SaveProcessorCallback(processor=getattr(vlm, "processor", None), tokenizer=tokenizer),
        BatchActionAccuracyAndBestCallback(action_token_begin_id=action_begin, run_dir=ckpt_output_dir, overwatch=overwatch),
        # 避免与 report_to=["wandb"] 重复上报，这里仅做本地日志
        LoggingCallback(overwatch=overwatch, log_to_local=True),
    ]



    # 评估指标：基于 token id/labels 计算 action accuracy
    # 为减少内存与通信，将 logits 在设备上转为 token id 再传入 metrics
    def preprocess_logits_for_metrics_fn(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred_ids = logits.argmax(dim=-1)

        if labels is None:
            return pred_ids

        mask = (labels != -100) & (labels >= action_begin)
        correct = ((pred_ids == labels) & mask).sum(dim=-1)  # [B]
        total = mask.sum(dim=-1)                              # [B]

        return torch.stack([correct, total], dim=-1)  # [B, 2]


    def compute_metrics_fn(eval_pred: EvalPrediction):
        preds = eval_pred.predictions  # numpy array, shape [N, 2]
        correct = preds[:, 0].sum(dtype=np.int64)
        total = preds[:, 1].sum(dtype=np.int64)
        acc = (correct / total) if total > 0 else 0.0
        return {"action_accuracy": float(acc)}

    trainer = Seq2SeqTrainer(
        model=vlm,
        args=args,
        train_dataset=vla_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_fn,
        compute_metrics=compute_metrics_fn if str(getattr(cfg, "eval_strategy", "no")) != "no" else None,
        callbacks=callbacks,
    )

    trainer.train()

    


