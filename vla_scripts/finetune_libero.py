import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import draccus
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
 
from prismatic.overwatch import initialize_overwatch
import wandb
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.datasets import RLDSBatchTransformLIBERO
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
 

from datetime import datetime
import logging
import sys
import json

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)

def _install_global_exception_logger() -> None:
    def _handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        logging.getLogger().exception("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = _handler

home_path = "/mnt/mnt/public/jlchen"
@dataclass
class FinetuneConfig:
    # fmt: off
    # Base VLM & LAM
    # 已迁移到 LatentWorldVLAConfig：model_id, hf_cache_dir, lam_ckpt_path, lam_yaml_path
    # Dataset
    data_root_dir: Path = Path(home_path + "/datasets")
    data_mix: str = "libero_object_no_noops"
    image_resolution: int = 448
    shuffle_buffer_size: int = 2000
    image_aug: bool = False

    # Run & IO
    run_root_dir: Path = Path(__file__).resolve().parent / "world_vla_log"
    # adapter_tmp_dir: Path = Path("adapter-tmp")
    # save_latest_checkpoint_only: bool = True

    # Optimization
    batch_size: int = 64
    max_steps: int = 40000
    warmup_steps: int = 200
    save_steps: int = 1000
    eval_steps: int = 1000
    eval_batches: int = 100
    learning_rate: float = 1e-4
    # 独立的 VLM 学习率与调度超参
    vlm_learning_rate: float = 1e-5
    vlm_warmup_steps: int = 200
    grad_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4
    vlm_loss_weight: float = 1.0
    # 冻结策略（全量微调切换）：当 freeze_vlm=True 时冻结整个 VLM；否则全参数更新 VLM
    freeze_vlm: bool = True

    # Seeding & dtype
    seed: int = 42

    # 已迁移到 LatentWorldVLAConfig：codebook_size

    # Tracking
    wandb_project: str = "finetune-LIBERO"
    wandb_entity: Optional[str] = None
    run_id_note: Optional[str] = None
    run_time: Optional[str] = datetime.now().strftime("%m%d_%H%M%S")



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # 构建模型配置（包含 VLM/LAM 加载所需参数，已迁移至 LatentWorldVLAConfig）
    model_cfg = LatentWorldVLAConfig()
    overwatch.info(f"Fine-tuning LatentWorldVLA on `{cfg.data_mix}` with base `{model_cfg.model_id}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    # 允许 cudnn 根据输入大小自调最优算法（对固定/少量形状有利）
    torch.backends.cudnn.benchmark = True
    # 允许 TF32（Ampere+ 上通常带来明显加速，数值影响可接受）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Seeding
    torch.manual_seed(cfg.seed + device_id)
    torch.cuda.manual_seed_all(cfg.seed + device_id)
    random.seed(cfg.seed + device_id)

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.run_time}+{model_cfg.model_id.split('/')[-1]}+{cfg.data_mix}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.freeze_vlm:
        exp_id += "+freeze_vlm"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories (hierarchical)
    run_dir = cfg.run_root_dir / exp_id
    ckpt_root = run_dir / "checkpoints"
    if distributed_state.is_main_process:
        os.makedirs(ckpt_root, exist_ok=True)

    # Configure logging to file (main process only; avoid duplicate handlers)
    if distributed_state.is_main_process:
        log_file_path = run_dir / f"{cfg.run_id_note}.log"
        root_logger = logging.getLogger()
        already_attached = False
        for h in list(root_logger.handlers):
            try:
                if hasattr(h, "baseFilename") and getattr(h, "baseFilename") == str(log_file_path):
                    already_attached = True
                    break
            except Exception:
                continue
        if not already_attached:
            try:
                try:
                    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
                except TypeError:
                    file_handler = logging.FileHandler(str(log_file_path))
                formatter = logging.Formatter(
                    fmt="% (asctime)s | % (levelname)s | % (name)s: % (message)s".replace(" ", ""),
                    datefmt="%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                root_logger.addHandler(file_handler)
            except Exception:
                pass
        overwatch.info(f"📝 日志将写入 `{log_file_path}`")
        _install_global_exception_logger()
        overwatch.info("✅ 已安装全局异常捕获（未捕获异常将写入日志）")

    # 构建 LatentWorldVLA（内部自洽加载 VLM/LAM，并完成 tokenizer 扩展与冻结策略）
    overwatch.info("🔄 构建 LatentWorldVLA（内部加载VLM & LAM）")
    lwvla = LatentWorldVLA(model_cfg=model_cfg)

    num_params = sum(p.numel() for p in lwvla.parameters())
    num_trainable_params = sum(p.numel() for p in lwvla.parameters() if p.requires_grad)
    overwatch.info(
        f"# LatentWorldVLA Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator(mixture=`{cfg.data_mix}`, image_res={cfg.image_resolution})"
    )
    processor = lwvla.processor
    train_dataset, val_dataset, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        lwvla.lam,
        processor=processor,
        default_image_resolution=cfg.image_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        training_phase='post-training', 
        data_transform_fn=RLDSBatchTransformLIBERO,
        collator_fn=PaddedCollatorForActionPrediction_LIBERO,
    )
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # DDP
    lwvla = lwvla.to(device_id)
    wrapped_model = DDP(lwvla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Optimizer & LR scheduler
    # 按模块区分参数组（VLM 与非 VLM），以便为 VLM 设置独立 LR 与调度
    base_params: List[torch.nn.Parameter] = []
    vlm_params: List[torch.nn.Parameter] = []
    # 通过模块引用来界定 VLM 参数（不要依赖参数名）
    vlm_param_ids = {id(p) for p in wrapped_model.module.vlm.parameters()}
    for param in wrapped_model.module.parameters():
        if not param.requires_grad:
            continue
        if id(param) in vlm_param_ids:
            vlm_params.append(param)
        else:
            base_params.append(param)
    param_groups = [
        {"params": base_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ]
    if (not cfg.freeze_vlm) and len(vlm_params) > 0:
        param_groups.append({"params": vlm_params, "lr": cfg.vlm_learning_rate, "weight_decay": cfg.weight_decay})
    optimizer = AdamW(param_groups)
    # 训练中用于裁剪的可训练参数集合
    trainable_params = base_params + vlm_params

    # 调度器（为不同参数组提供独立的 lr lambda）
    base_decay_step = int(cfg.max_steps * 0.8)     # 原来的 StepLR step（基础网络）
    vlm_decay_step = int(cfg.max_steps * 0.8)      # VLM 的 step，可按需暴露更多配置

    def lr_lambda_base(current_step: int):
        if current_step < cfg.warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, cfg.warmup_steps))
        elif current_step < base_decay_step:
            # 保持原始学习率
            return 1.0
        else:
            # step decay
            return 0.1

    def lr_lambda_vlm(current_step: int):
        if current_step < cfg.vlm_warmup_steps:
            # linear warmup（VLM 独立 warmup）
            return float(current_step) / float(max(1, cfg.vlm_warmup_steps))
        elif current_step < vlm_decay_step:
            return 1.0
        else:
            return 0.1

    if len(param_groups) == 2:
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_base, lr_lambda_vlm])
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_base)

    # DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # 注意：collator 含 GPU 计算（LAM.vq_encode），不可开启多进程
        pin_memory=True,
    )
    val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    sampler=None,
    collate_fn=collator,
    num_workers=0,
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
        # push full config to W&B
        try:
            wandb.config.update({k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}, allow_val_change=True)
        except Exception:
            pass

    # Train!
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    with tqdm.tqdm(total=cfg.max_steps, desc=f"🚀 {cfg.run_id_note or 'Training'}", leave=False, disable=not distributed_state.is_main_process, ncols=80) as progress:
        wrapped_model.train()
        if cfg.freeze_vlm:
            wrapped_model.module.vlm.eval()
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {
                k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k in {"pixel_values","input_ids","labels","attention_mask","actions","latent_action_idx","proprio","image_features"}
            }
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            # Forward

            # 在训练循环中替换 forward/backward/step 区块：
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # 或者 dtype=torch.float16 取决于硬件
                out = wrapped_model(**batch)
                flow_loss = out["loss_flow"]
                vlm_loss = out.get("loss_vlm", torch.tensor(0.0, device=device_id, dtype=flow_loss.dtype))
                if cfg.freeze_vlm:
                    loss = flow_loss
                else:
                    loss = flow_loss + cfg.vlm_loss_weight * vlm_loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            recent_losses.append(loss.item())
            smoothened_loss = sum(recent_losses) / len(recent_losses)



            # 在 optimizer step 时（当满足 grad_accumulation 条件）:
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # clip grads
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                global_step += 1
                if distributed_state.is_main_process:
                    progress.update()

                # Logging per optimizer step
                if distributed_state.is_main_process:
                    progress.set_postfix({"loss": f"{smoothened_loss:.6f}"})
                    log_payload = {
                        "train_loss": smoothened_loss,
                        "train_flow_loss": float(flow_loss.item()),
                        # 记录基础参数组与（若有）VLM 参数组的学习率
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    if len(optimizer.param_groups) > 1:
                        log_payload["vlm_lr"] = optimizer.param_groups[1]['lr']
                    if "vlm_action_accuracy" in out:
                        log_payload["train_action_accuracy"] = float(out["vlm_action_accuracy"].item())
                    if not cfg.freeze_vlm:
                        log_payload["train_vlm_loss"] = float(vlm_loss.item())
                    wandb.log(log_payload, step=global_step)

                # Checkpoint on schedule (skip step 0)
                if global_step > 0 and global_step % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        step_dir = ckpt_root / f"step-{global_step:06d}"
                        os.makedirs(step_dir, exist_ok=True)
                        overwatch.info(f"💾 保存检查点 @ step={global_step}: 目录 `{step_dir}`")

                        # Save weights:
                        # - 当冻结 VLM：仅保存 Flow；
                        # - 当不冻结 VLM（全参数微调）：分别保存 VLM 与 Flow。
                        if cfg.freeze_vlm:
                            # 仅保存 Flow（使用 safetensors）
                            torch.save(wrapped_model.module.flow.state_dict(), step_dir / "flow.pt")
                            overwatch.info("✅ 已保存 Flow(safetensors)")
                        else:
                            # 保存 Flow（safetensors）
                            torch.save(wrapped_model.module.flow.state_dict(), step_dir / "flow.pt")
                            # 保存 VLM（HuggingFace save_pretrained 目录结构）
                            try:
                                vlm_out_dir = step_dir / "vlm"
                                os.makedirs(vlm_out_dir, exist_ok=True)
                                if hasattr(wrapped_model.module.vlm, "save_pretrained"):
                                    wrapped_model.module.vlm.save_pretrained(vlm_out_dir)
                                    overwatch.info("✅ 已保存 VLM(save_pretrained) 与 Flow(safetensors)")
                                else:
                                    overwatch.warning("VLM 缺少 save_pretrained()，已仅保存 Flow(safetensors)")
                            except Exception:
                                overwatch.warning("保存 VLM(save_pretrained) 失败，已仅保存 Flow(safetensors)")

                        # 始终保存优化器/调度器状态
                        torch.save(optimizer.state_dict(), step_dir / "optim.pt")
                        torch.save(scheduler.state_dict(), step_dir / "sched.pt")
                        overwatch.info("✅ 已保存优化器/调度器状态")

                        # 无 LoRA：不再导出 LoRA 适配器

                        # Save dataset statistics once is sufficient but safe to overwrite
                        save_dataset_statistics(train_dataset.dataset_statistics, step_dir)
                        overwatch.info("📦 已保存数据统计（用于推理反归一化）")

                    dist.barrier()

                # Stop training when max_steps is reached
                if global_step >= cfg.max_steps:
                    if distributed_state.is_main_process:
                        overwatch.info(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break

                # Periodic eval
                if global_step > 0 and global_step % cfg.eval_steps == 0:
                    wrapped_model.eval()
                    if distributed_state.is_main_process:
                        overwatch.info(f"🔎 开始评估 @ step={global_step}（最多 {cfg.eval_batches} 批次）")
                    eval_loss_accum, eval_count = 0.0, 0
                    with torch.no_grad():
                        for eval_i, eval_batch in enumerate(val_loader):
                            if eval_i >= cfg.eval_batches:
                                break
                            eval_batch = {
                                k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
                                for k, v in eval_batch.items()
                                if k in {"pixel_values","input_ids","labels","attention_mask","actions","latent_action_idx","proprio","image_features"}
                            }
                            if "pixel_values" in eval_batch:
                                eval_batch["pixel_values"] = eval_batch["pixel_values"].to(torch.bfloat16)

                            # 使用 predict_action 重建动作，并与 GT 动作计算 MSE
                            window_size = int(eval_batch["actions"].shape[1])
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                pred_actions = wrapped_model.module.predict_action(
                                    pixel_values=eval_batch["pixel_values"],
                                    input_ids=eval_batch["input_ids"],
                                    proprio=eval_batch["proprio"],
                                    image_feat_4_lam=eval_batch["image_features"],
                                    window_size=window_size,
                                    guidance_scale=1.0,
                                )
                                gt_actions = eval_batch["actions"]
                                # 对齐 dtype 以避免精度/类型不匹配
                                pred_actions = pred_actions.to(dtype=gt_actions.dtype)
                                eval_mse = F.mse_loss(pred_actions, gt_actions)

                            eval_loss_accum += float(eval_mse.item())
                            eval_count += 1

                    mean_eval_loss = eval_loss_accum / max(eval_count, 1)
                    if distributed_state.is_main_process:
                        wandb.log({
                            "eval_loss": mean_eval_loss,  # 使用 MSE 作为评估损失
                            "global_step": global_step,
                        }, step=global_step)
                        overwatch.info(
                            f"📊 评估完成 @ step={global_step}: eval_loss(MSE)={mean_eval_loss:.6f}"
                        )

                        # Maintain best checkpoint using metrics.json only
                        try:
                            metrics_path = ckpt_root / "metrics.json"
                            metrics = {"best": None, "history": []}
                            if metrics_path.exists():
                                try:
                                    with open(metrics_path, "r", encoding="utf-8") as f:
                                        metrics = json.load(f)
                                except Exception:
                                    metrics = {"best": None, "history": []}

                            if not isinstance(metrics, dict):
                                metrics = {"best": None, "history": []}
                            if "history" not in metrics or not isinstance(metrics.get("history"), list):
                                metrics["history"] = []

                            now_iso = datetime.now().isoformat()

                            prev_best = None
                            prev_best_step = None
                            if isinstance(metrics.get("best"), dict):
                                try:
                                    prev_best = float(metrics["best"].get("value"))
                                    prev_best_step = int(metrics["best"].get("step"))
                                except Exception:
                                    prev_best = None
                                    prev_best_step = None

                            is_new_best = (prev_best is None) or (mean_eval_loss < prev_best)

                            # Update best symlink if improved
                            if is_new_best:
                                best_link = ckpt_root / "best"
                                if best_link.exists() or best_link.is_symlink():
                                    try:
                                        best_link.unlink()
                                    except Exception:
                                        pass
                                target_dir = ckpt_root / f"step-{global_step:06d}"
                                if not target_dir.exists():
                                    os.makedirs(target_dir, exist_ok=True)
                                try:
                                    best_link.symlink_to(target_dir.name)
                                except Exception:
                                    with open(ckpt_root / "BEST_STEP", "w") as f:
                                        f.write(target_dir.name)
                                # Log best update
                                if prev_best is None:
                                    overwatch.info(
                                        f"🏆 首次创建最佳 mean_eval_loss={mean_eval_loss:.6f} @ step={global_step} -> `{target_dir}`"
                                    )
                                else:
                                    overwatch.info(
                                        f"🏆 刷新最佳 mean_eval_loss: {prev_best:.6f} -> {mean_eval_loss:.6f} @ step={global_step}; best -> `{target_dir}`"
                                    )
                            else:
                                if prev_best is not None:
                                    overwatch.info(
                                        f"ℹ️ 未刷新最佳 mean_eval_loss（best={prev_best:.6f}, best_step={prev_best_step}, current={mean_eval_loss:.6f}）"
                                    )
                                else:
                                    overwatch.info(
                                        f"ℹ️ 未刷新最佳 mean_eval_loss（当前={mean_eval_loss:.6f}）"
                                    )

                            # Append current eval to history and update best when improved
                            metrics["history"].append({
                                "step": int(global_step),
                                "train_loss": float(smoothened_loss),
                                "action_eval_loss": float(mean_eval_loss),
                                "time": now_iso,
                            })
                            if is_new_best:
                                metrics["best"] = {
                                    "step": int(global_step),
                                    "value": float(mean_eval_loss),
                                    "time": now_iso,
                                }

                            with open(metrics_path, "w", encoding="utf-8") as f:
                                json.dump(metrics, f, ensure_ascii=False, indent=2)
                        except Exception:
                            # Silent fail for metrics.json to avoid disrupting training
                            pass

                    wrapped_model.train()
                    if cfg.freeze_vlm:
                        wrapped_model.module.vlm.eval()



if __name__ == "__main__":
    finetune()

