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
    """
    一个专门用于在保存模型检查点时一同保存 Processor 的回调。
    
    Hugging Face Trainer 默认不会保存 Processor 对象，此回调弥补了这一点，
    确保了 image_processor 和 tokenizer 的配置与模型权重一同被保存。
    """
    def __init__(self, processor: Any):
        # 构造函数只接收一个 processor 对象，意图非常明确
        self.processor = processor

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        """在每次 `trainer.save_model()` 或达到 checkpoint 时触发。"""
        # 确保只在主进程执行保存操作
        if args.should_save and state.is_world_process_zero:
            # checkpoint 的目录是 output_dir + "checkpoint-XXXX"
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            # 创建目录以防万一（虽然 Trainer 通常会创建）
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(checkpoint_dir)

    # on_train_end 逻辑非常相似，但保存到最终的 output_dir
    # Trainer 在训练结束后会自动调用 on_save，所以这个方法甚至是可选的，
    # 但为了明确起见，可以保留。
    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        """在训练完全结束时触发。"""
        if state.is_world_process_zero:
            final_output_dir = args.output_dir
            os.makedirs(final_output_dir, exist_ok=True)
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(final_output_dir)


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
    processor,
    run_dir: Path,
    overwatch,
) -> None:
    """使用 transformers.Seq2SeqTrainer 进行潜变量动作训练。"""


    max_steps: Optional[int] = cfg.max_steps
    epochs: int = cfg.epochs
    lr: float = cfg.learning_rate
    wd: float = cfg.weight_decay
    use_mp: bool = bool(cfg.enable_mixed_precision_training)
    max_grad_norm: float = float(cfg.max_grad_norm)

    save_interval: int = int(cfg.save_interval)
    grad_accum_steps: int = int(cfg.gradient_accumulation_steps)
    action_begin: int = int(cfg.action_token_begin_id)
    action_end: int = action_begin + int(getattr(cfg, "codebook_size", 16))
    logging_steps: int = int(cfg.logging_steps)
    per_device_eval_bsz: int = cfg.per_device_eval_batch_size
    dataloader_num_workers: int = int(cfg.dataloader_num_workers)
    dataloader_pin_memory: bool = bool(cfg.dataloader_pin_memory)
    seed: int = int(cfg.seed)

    if max_steps is None:
        max_steps = epochs * 10000

    if cfg.wandb_project:
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
        save_total_limit=int(cfg.save_total_limit),
        report_to=["wandb"],
        run_name=str(cfg.run_id),
        max_grad_norm=max_grad_norm,
        optim=str(cfg.optim),
        save_safetensors=True,
        seed=seed,
        ddp_find_unused_parameters=False,
        fsdp=cfg.fsdp,
        fsdp_config=cfg.fsdp_config,
        eval_strategy=str(cfg.eval_strategy),  
        eval_steps=int(cfg.eval_interval),
        eval_accumulation_steps=int(cfg.eval_accumulation_steps),
        prediction_loss_only=False,
        # prediction_loss_only = False
        # predict_with_generate=True,  # 🔑 启用 generate
        # generation_num_beams=int(cfg.generation_num_beams),      # 可调
    )

    callbacks = [
        SaveProcessorCallback(processor=processor),
        BatchActionAccuracyAndBestCallback(action_token_begin_id=action_begin, run_dir=ckpt_output_dir, overwatch=overwatch),
        # 避免与 report_to=["wandb"] 重复上报，这里仅做本地日志
        LoggingCallback(overwatch=overwatch, log_to_local=True),
    ]



    # 评估指标：基于 token id/labels 计算 action accuracy
    # 为减少内存与通信，将 logits 在设备上转为 token id 再传入 metrics
    def preprocess_logits_for_metrics_fn(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # 仅在 <ACT_*> 子词表内进行分类以评估动作精度
        act_ids = torch.arange(action_begin, action_end, device=logits.device)
        act_logits = logits.index_select(dim=-1, index=act_ids)
        pred_rel = act_logits.argmax(dim=-1)
        pred_ids = act_ids[pred_rel]

        if labels is None:
            return pred_ids

        # 仅统计 labels 位于 <ACT_*> 范围内的位置，避免把非动作 token 计入总数
        mask = (labels != -100) & (labels >= action_begin) & (labels < action_end)
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
        tokenizer=processor.tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_fn,
        compute_metrics=compute_metrics_fn if str(cfg.eval_strategy) != "no" else None,
        callbacks=callbacks,
    )

    trainer.train()

    


