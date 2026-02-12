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
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import DataLoader
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_utils import unwrap_model
import numpy as np
import wandb
from peft import PeftModel

from prismatic.models import _unfreeze_last_n_llm_layers, _resolve_llm_module


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


class LoggingCallback(TrainerCallback):
    """同时写入本地日志与 wandb，并处理 train/val 前缀。"""

    def __init__(self, overwatch=None, log_to_local=True, trainer_ref=None):
        self.overwatch = overwatch
        self.log_to_local = log_to_local
        self.trainer_ref = trainer_ref  # 引用 Trainer 实例以访问累积的 accuracy

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        if logs is None:
            return

        # 分布式只在 rank0 写
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        # 合并 trainer 缓存的额外日志（loss 分量 / action accuracy / identity shortcut）
        if self.trainer_ref is not None:
            # loss 分量（来自模型输出 dict）
            for key in ("loss_main", "loss_distill", "loss_perceptual"):
                pending_key = f"_pending_{key}"
                if hasattr(self.trainer_ref, pending_key):
                    val = getattr(self.trainer_ref, pending_key)
                    if val is not None:
                        logs[f"train_{key}"] = float(val)
                        setattr(self.trainer_ref, pending_key, None)
            
            # identity shortcut 指标
            if hasattr(self.trainer_ref, "_pending_identity_shortcut"):
                if self.trainer_ref._pending_identity_shortcut is not None:
                    logs["train_identity_shortcut"] = float(self.trainer_ref._pending_identity_shortcut)
                    self.trainer_ref._pending_identity_shortcut = None

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

        # 注意：不需要手动调用 wandb.log，因为 Trainer 配置了 report_to=["wandb"]
        # Trainer 会自动将 logs 字典发送到 wandb，包括我们在 callback 中修改的内容



def run_latent_action_training(
    *,
    cfg: Optional[Any],
    model: Any,
    vla_dataset: TorchIterableDataset,
    eval_dataset: Optional[Any] = None,
    collator=None,
    processor=None,
    run_dir: Path = Path("."),
    overwatch=None,
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
    logging_steps: int = int(cfg.logging_steps)
    per_device_eval_bsz: int = cfg.per_device_eval_batch_size
    dataloader_num_workers: int = int(cfg.dataloader_num_workers)
    dataloader_pin_memory: bool = bool(cfg.dataloader_pin_memory)
    seed: int = int(cfg.seed)
    merge_lora_before_save: bool = bool(getattr(cfg, "lora_merge_before_save", False))
    unfreeze_llm_last_n_layers: Optional[int] = getattr(cfg, "unfreeze_llm_last_n_layers", None)
    unfreeze_lam_decoder: bool = bool(getattr(cfg, "unfreeze_lam_decoder", False))

    if max_steps is None:
        max_steps = epochs * 10000

    # 在 Trainer 创建前解冻 LLM 最后 n 层，确保优化器能包含这些参数
    if unfreeze_llm_last_n_layers is not None and int(unfreeze_llm_last_n_layers) > 0:
        try:
            base_model = unwrap_model(model)
            # LatentVLAModel wrapper -> VLM is at base_model.vlm
            if hasattr(base_model, "vlm"):
                vlm = base_model.vlm
            elif hasattr(base_model, "model") and hasattr(base_model.model, "vlm"):
                vlm = base_model.model.vlm
            else:
                vlm = base_model

            llm_module = _resolve_llm_module(vlm)
            if llm_module is None:
                if overwatch is not None:
                    overwatch.warning(
                        "[run_latent_action_training] Failed to resolve LLM module; "
                        "skip unfreeze (training may have 0 trainable params)."
                    )
            else:
                ok = _unfreeze_last_n_llm_layers(llm_module, int(unfreeze_llm_last_n_layers))
                if overwatch is not None:
                    overwatch.info(
                        f"[run_latent_action_training] Unfroze last {int(unfreeze_llm_last_n_layers)} LLM layers "
                        f"before optimizer creation (ok={ok})"
                    )
                # Log trainable parameter count AFTER unfreeze (more reliable than the earlier log in vla_scripts/train.py)
                try:
                    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    num_total = sum(p.numel() for p in model.parameters())
                    if overwatch is not None:
                        overwatch.info(
                            f"[run_latent_action_training] params: {num_total/1e6:.3f}M total, {num_trainable/1e6:.3f}M trainable (after pre-unfreeze)"
                        )
                except Exception:
                    pass
        except Exception as e:
            if overwatch is not None:
                overwatch.warning(f"[run_latent_action_training] LLM unfreeze failed: {e}")

    # 在 Trainer 创建前解冻 LAM decoder，确保优化器能包含这些参数
    if unfreeze_lam_decoder:
        try:
            base_model = unwrap_model(model)
            # LatentVLAModel wrapper -> LAM is at base_model.lam
            lam = None
            if hasattr(base_model, "lam"):
                lam = base_model.lam
            elif hasattr(base_model, "model") and hasattr(base_model.model, "lam"):
                lam = base_model.model.lam
            
            if lam is None:
                if overwatch is not None:
                    overwatch.warning(
                        "[run_latent_action_training] Failed to find LAM module; "
                        "skip decoder unfreeze."
                    )
            else:
                dec = getattr(lam, "decoder", None)
                if dec is None:
                    if overwatch is not None:
                        overwatch.warning(
                            "[run_latent_action_training] LAM has no decoder; "
                            "skip decoder unfreeze."
                        )
                else:
                    # 检查 decoder 是否已经被解冻（理论上不应该，除非手动解冻）
                    dec_params = sum(p.numel() for p in dec.parameters())
                    already_unfrozen = any(p.requires_grad for p in dec.parameters())
                    
                    if already_unfrozen:
                        if overwatch is not None:
                            overwatch.info(
                                f"[run_latent_action_training] LAM decoder ({dec_params/1e6:.3f}M params) already unfrozen; "
                                "skipping redundant unfreeze"
                            )
                    else:
                        # 执行解冻
                        for p in dec.parameters():
                            p.requires_grad = True
                        if overwatch is not None:
                            overwatch.info(
                                f"[run_latent_action_training] Unfroze LAM decoder ({dec_params/1e6:.3f}M params) "
                                f"before optimizer creation"
                            )
                    
                    # Log trainable parameter count (无论是否已经解冻，都统计一次以供参考)
                    try:
                        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        num_total = sum(p.numel() for p in model.parameters())
                        if overwatch is not None:
                            status = "already unfrozen" if already_unfrozen else "after LAM decoder pre-unfreeze"
                            overwatch.info(
                                f"[run_latent_action_training] params: {num_total/1e6:.3f}M total, {num_trainable/1e6:.3f}M trainable ({status})"
                            )
                    except Exception:
                        pass
        except Exception as e:
            if overwatch is not None:
                overwatch.warning(f"[run_latent_action_training] LAM decoder immediate unfreeze failed: {e}")

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

    # 创建 LoggingCallback（稍后会设置 trainer_ref）
    logging_callback = LoggingCallback(overwatch=overwatch, log_to_local=True)
    
    callbacks = [
        logging_callback,
    ]







    class VLATrainer(Seq2SeqTrainer):
        """VLA Trainer，支持 loss 分量记录。"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 用于累积梯度累积步骤中的 loss 分量记录
            self._accum_step_count_loss = 0
            # 额外 loss 分量缓存（在 on_log 时合并）
            self._pending_loss_main = None
            self._pending_loss_distill = None
            self._pending_loss_perceptual = None
            # identity shortcut 指标缓存
            self._pending_identity_shortcut = None
            # eval 阶段的 loss 分量累计（用于 wandb 记录）
            self._reset_eval_loss_components()


        def get_train_dataloader(self):
            """
            自定义 dataloader，避免对已经在 TF 数据管线中完成分片的 IterableDataset
            再次套用 IterableDatasetShard（否则会重复按 world_size 切分）。
            """
            if not isinstance(self.train_dataset, TorchIterableDataset):
                return super().get_train_dataloader()

            generator = torch.Generator()
            if getattr(self.args, "seed", None) is not None:
                generator.manual_seed(int(self.args.seed) + int(getattr(self.state, "epoch", 0) or 0))

            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
                persistent_workers=False,
                generator=generator,
            )

        def get_eval_dataloader(self, eval_dataset=None):
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            if not isinstance(eval_dataset, TorchIterableDataset):
                return super().get_eval_dataloader(eval_dataset)

            # 分布式 eval 时必须 drop_last=True，否则不同 rank 的 batch 数可能不一致，
            # 导致先完成的 rank 在 collective 操作处死锁等待（如 metrics 聚合）。
            # 单卡时可保留 drop_last=False 以评估全部样本。
            is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=is_distributed,  # 分布式时丢弃末尾不完整 batch，确保所有 rank batch 数一致
                persistent_workers=False,
            )

        def _clone_tied_weights_for_safetensors(self, model):
            """
            safetensors 不支持共享存储的权重，这里在保存前克隆 lm_head 与 embed_tokens 的共享存储。
            仅在保存前调用，不影响训练期间的权重绑定。
            """
            try:
                base = unwrap_model(model)

                def _maybe_clone(head, embed):
                    if head is None or embed is None:
                        return
                    if not (hasattr(head, "weight") and hasattr(embed, "weight")):
                        return
                    w_head = head.weight
                    w_embed = embed.weight
                    # 判断是否共享同一存储
                    if w_head.data_ptr() == w_embed.data_ptr():
                        head.weight = torch.nn.Parameter(w_head.detach().clone())

                # 顶层 lm_head 与 input embeddings
                lm_head = getattr(base, "lm_head", None)
                embed_tokens = None
                if hasattr(base, "get_input_embeddings"):
                    embed_tokens = base.get_input_embeddings()
                _maybe_clone(lm_head, embed_tokens)

                # 若模型有 language_model 子模块，额外处理内部绑定
                if hasattr(base, "language_model"):
                    lm = base.language_model
                    lm_embed = getattr(lm, "embed_tokens", None)
                    lm_head_sub = getattr(lm, "lm_head", None)
                    _maybe_clone(lm_head_sub, lm_embed)
            except Exception:
                # 仅用于保存前处理，失败不应影响训练
                pass
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """重写 compute_loss 以记录 loss 分量。"""
            # 训练时获取 outputs 用于记录 loss 分量
            need_outputs_for_logging = self.model.training
            should_return_outputs = return_outputs or need_outputs_for_logging
            
            # 调用父类的 compute_loss 获取损失和输出
            if should_return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
                outputs = None

            # 记录模型输出中的额外 loss 分量（wandb/console）
            if self.model.training and outputs is not None and isinstance(outputs, dict):
                try:
                    def _as_float(x):
                        if x is None:
                            return None
                        if torch.is_tensor(x):
                            return float(x.detach().float().mean().cpu())
                        return float(x)

                    loss_main = _as_float(outputs.get("loss_main"))
                    loss_distill = _as_float(outputs.get("loss_distill"))
                    loss_perceptual = _as_float(outputs.get("loss_perceptual"))
                    identity_shortcut = outputs.get("identity_shortcut")
                    
                    # 转换为 float（如果是 tensor）
                    if identity_shortcut is not None:
                        if isinstance(identity_shortcut, (int, float)):
                            identity_shortcut = float(identity_shortcut)
                        elif torch.is_tensor(identity_shortcut):
                            identity_shortcut = float(identity_shortcut.item())
                        else:
                            identity_shortcut = None
                    
                    # 缓存到 pending 中，在最后一个累积步骤时通过 self.log() 记录
                    self._pending_loss_main = loss_main
                    self._pending_loss_distill = loss_distill
                    self._pending_loss_perceptual = loss_perceptual
                    self._pending_identity_shortcut = identity_shortcut
                    
                    self._accum_step_count_loss += 1
                    is_last_accum_step = (self._accum_step_count_loss >= self.args.gradient_accumulation_steps)
                    
                    if is_last_accum_step:
                        log_dict = {}
                        if loss_main is not None:
                            log_dict["loss_main"] = loss_main
                        if loss_distill is not None:
                            log_dict["loss_distill"] = loss_distill
                        if loss_perceptual is not None:
                            log_dict["loss_perceptual"] = loss_perceptual
                        if identity_shortcut is not None:
                            log_dict["identity_shortcut"] = identity_shortcut
                        
                        if log_dict:
                            self.log(log_dict)
                        
                        self._accum_step_count_loss = 0
                except Exception:
                    pass
            
            if return_outputs:
                return loss, outputs
            else:
                return loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            """
            在评估前重置 loss 分量累计，评估后写入 eval_loss_* 到 metrics 并主动 log。
            分布式下先 barrier 再跑 eval，避免部分 rank 晚进 evaluation_loop 导致 collective 死锁。
            
            注意：eval 数据集不进行分布式 sharding，所有 rank 使用相同的数据（通过固定随机种子保证）。
            """
            self._reset_eval_loss_components()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            extra_metrics = self._finalize_eval_loss_components(metric_key_prefix=metric_key_prefix)
            if extra_metrics:
                metrics.update(extra_metrics)
                try:
                    self.log(extra_metrics)
                except Exception:
                    pass
            return metrics

        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            """
            显式前向，确保返回 3D logits；发现异常直接报错，避免后续兜底。
            """
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels", None)

            with torch.no_grad():
                outputs = model(**inputs)

            # 解析 loss 与 logits
            loss = None
            logits = None
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                logits = outputs.get("logits", None)
            elif hasattr(outputs, "loss") or hasattr(outputs, "logits"):
                loss = getattr(outputs, "loss", None)
                logits = getattr(outputs, "logits", None)
            elif isinstance(outputs, (tuple, list)):
                # 约定: (loss, logits, *rest) 或 (logits, ...)
                if len(outputs) >= 2 and torch.is_tensor(outputs[0]):
                    loss = outputs[0]
                    logits = outputs[1]
                elif len(outputs) >= 1:
                    logits = outputs[0]

            # 强校验 logits
            assert logits is not None and torch.is_tensor(logits), (
                f"prediction_step: logits missing/invalid, got type={type(logits)}"
            )
            assert logits.ndim == 3, f"prediction_step: logits ndim={logits.ndim}, shape={tuple(logits.shape)}"

            # 处理 loss
            if loss is None and labels is not None and torch.is_tensor(labels):
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, labels)
            if loss is not None and torch.is_tensor(loss):
                loss = loss.mean().detach()

            # 评估时累计各 loss 分量，便于日志记录 eval_loss_*
            if (not self.model.training) and isinstance(outputs, dict):
                try:
                    for key in ("loss_main", "loss_distill", "loss_perceptual"):
                        val = self._to_float(outputs.get(key))
                        if val is not None:
                            self._eval_loss_sums[key] += float(val)
                            self._eval_loss_counts[key] += 1
                except Exception:
                    pass

            if prediction_loss_only:
                return loss, None, None

            return loss, logits.detach(), labels

        # ------ helpers for eval loss logging ------
        def _reset_eval_loss_components(self):
            self._eval_loss_sums = {
                "loss_main": 0.0,
                "loss_distill": 0.0,
                "loss_perceptual": 0.0,
            }
            self._eval_loss_counts = {k: 0 for k in self._eval_loss_sums}

        def _finalize_eval_loss_components(self, *, metric_key_prefix: str = "eval"):
            out = {}
            for k, s in self._eval_loss_sums.items():
                c = self._eval_loss_counts.get(k, 0)
                if c > 0:
                    out[f"{metric_key_prefix}_{k}"] = s / float(c)
            return out

        @staticmethod
        def _to_float(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return float(x.detach().float().mean().cpu())
            try:
                return float(x)
            except Exception:
                return None

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            """
            使用 HF 官方 `save_pretrained(..., safe_serialization=True)` 来处理 tied weights，
            避免 safetensors 直接保存 state_dict 报共享存储错误。
            参考官方建议：https://huggingface.co/docs/safetensors/torch_shared_tensors
            """
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            base_model = unwrap_model(self.model)

            def _save_processor(target_dir):
                processor_obj = getattr(self, "processing_class", None)
                if processor_obj is None:
                    processor_obj = getattr(self, "tokenizer", None)
                if processor_obj is not None and hasattr(processor_obj, "save_pretrained"):
                    try:
                        processor_obj.save_pretrained(target_dir)
                    except Exception:
                        pass

            # LoRA: 在保存前将 LoRA 权重合并到基础模型，再进行持久化
            if isinstance(base_model, PeftModel) and merge_lora_before_save:
                try:
                    merged_model = copy.deepcopy(base_model).to("cpu")
                    merged_model = merged_model.merge_and_unload(progressbar=False)
                    merged_model.save_pretrained(output_dir, safe_serialization=True)
                    _save_processor(output_dir)
                    return
                except Exception:
                    # 回退到常规保存，避免训练中断
                    pass

            # 让 HF 内部处理权重共享与 safetensors 保存
            base_model.save_pretrained(output_dir, safe_serialization=True)
            _save_processor(output_dir)

            # 其余状态（optimizer/scheduler/scaler）由基类在 save_checkpoint 时处理
            return

    trainer = VLATrainer(
        model=model,
        args=args,
        train_dataset=vla_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        # 传递完整 processor，保证 image_processor + tokenizer 一起保存
        processing_class=processor,
        compute_metrics=None,
        callbacks=callbacks,
    )

    # 设置 LoggingCallback 的 trainer 引用，以便访问 _pending_action_acc
    logging_callback.trainer_ref = trainer

    trainer.train()

    

