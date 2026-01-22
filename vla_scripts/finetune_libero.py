import os
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union

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
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig, SimpleLatentWorldVLA
from prismatic.util.data_utils import PaddedCollatorForLatentWorldVLA_LIBERO
from prismatic.vla.datasets import RLDSBatchTransformLIBERO
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
 

from datetime import datetime
import logging
import sys
import json
import shutil

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)

def _install_global_exception_logger() -> None:
    def _handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        logging.getLogger().exception("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = _handler


class _TqdmLoggingHandler(logging.Handler):
    """让 logging 输出不破坏 tqdm 进度条（通过 tqdm.write）。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except Exception:
            # 不要让日志影响训练
            pass

home_path = "/mnt/project_rlinf/jlchen"
@dataclass
class FinetuneConfig:
    # fmt: off
    # 模型配置（装配由 LatentWorldVLA.from_config 内部完成）
    model_cfg: LatentWorldVLAConfig = field(default_factory=LatentWorldVLAConfig)
    # Dataset
    data_root_dir: Path = Path(home_path + "/datasets")
    data_mix: str = "libero_object_no_noops"
    image_resolution: int = 256
    shuffle_buffer_size: int = 2000
    image_aug: bool = False
    # debug_repeat_batch 支持 bool 或 int：传入正整数 k 时，会缓存前 k 个样本并循环返回
    debug_repeat_batch: Union[bool, int] = False
    use_history_frame: bool = True
    # Run & IO
    run_root_dir: Path = Path(__file__).resolve().parent / "world_vla_log"
    # adapter_tmp_dir: Path = Path("adapter-tmp")
    # save_latest_checkpoint_only: bool = True
    use_simple_model: bool = False
    # Optimization
    batch_size: int = 64
    max_steps: int = 40000
    warmup_steps: int = 200
    save_steps: int = 5000
    eval_steps: int = 1000
    eval_batches: int = 100
    log_every_steps: int = 50  # 每多少个 optimizer step 在命令行输出一次关键指标
    learning_rate: float = 1e-4
    window_size: int = 10
    # 独立的 VLM 学习率与调度超参
    vlm_learning_rate: float = 1e-5
    vlm_warmup_steps: int = 200
    grad_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4
    # vlm_loss_weight: float = 1.0

    # Seeding & dtype
    seed: int = 42

    # 已迁移到 LatentWorldVLAConfig：codebook_size

    # Tracking
    wandb_project: str = "finetune-LIBERO"
    wandb_entity: Optional[str] = None
    # 将每次 finetune 的 wandb 文件存储在本次 run_dir 内部（run_dir/wandb）
    wandb_dir: Optional[Path] = None
    run_id_note: Optional[str] = None
    run_time: Optional[str] = datetime.now().strftime("%m%d_%H%M%S")



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # 构建模型配置（包含 VLM/LAM 加载所需参数，已迁移至 LatentWorldVLAConfig）
    model_cfg = cfg.model_cfg
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
        f"{cfg.run_time}+{model_cfg.model_id.split('/')[-2]}+{cfg.data_mix}"
        f"+lr-{cfg.learning_rate}"
    )
    # 添加冻结策略标记
    freeze_tags = []
    if model_cfg.freeze_vision_backbone:
        freeze_tags.append("frzVis")
    if model_cfg.freeze_llm_backbone:
        freeze_tags.append("frzLLM")
        if model_cfg.unfreeze_llm_last_n_layers:
            freeze_tags.append(f"unfrzLast{model_cfg.unfreeze_llm_last_n_layers}")
    if model_cfg.freeze_embedding:
        freeze_tags.append("frzEmb")
    if freeze_tags:
        exp_id += "+" + "+".join(freeze_tags)
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories (hierarchical)
    run_dir = cfg.run_root_dir / exp_id
    ckpt_root = run_dir / "checkpoints"
    if distributed_state.is_main_process:
        os.makedirs(ckpt_root, exist_ok=True)

    # 保存本次运行的配置文件到日志目录（main process only）
    if distributed_state.is_main_process:
        try:
            # 1) 保存命令行参数
            (run_dir / "argv.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

            # 2) 若通过 draccus `--config` 指定了 YAML/JSON/TOML，则复制原始文件
            src_cfg = None
            if "--config" in sys.argv:
                i = sys.argv.index("--config")
                if i + 1 < len(sys.argv):
                    src_cfg = sys.argv[i + 1]
            # 兼容常见短参数
            if src_cfg is None and "-c" in sys.argv:
                i = sys.argv.index("-c")
                if i + 1 < len(sys.argv):
                    src_cfg = sys.argv[i + 1]
            if src_cfg is not None:
                src_path = Path(src_cfg).expanduser()
                if src_path.exists() and src_path.is_file():
                    # 使用原配置文件名保存到 run_dir
                    shutil.copy2(str(src_path), str(run_dir / src_path.name))
                    overwatch.info(f"📋 配置文件已保存到：{run_dir / src_path.name}")
        except Exception as e:
            overwatch.warning(f"保存配置文件失败（不影响训练）：{e}")

    # Configure logging to file (main process only; avoid duplicate handlers)
    if distributed_state.is_main_process:
        # Ensure a stable log filename even when run_id_note is None
        log_name = f"{cfg.run_id_note}.log" if cfg.run_id_note is not None else "train.log"
        log_file_path = run_dir / log_name
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="% (asctime)s | % (levelname)s | % (name)s: % (message)s".replace(" ", ""),
            datefmt="%m-%d %H:%M:%S",
        )
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
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                root_logger.addHandler(file_handler)
            except Exception:
                pass
        # tqdm 训练时，RichHandler/普通 StreamHandler 往往会被进度条覆盖；这里统一替换为 tqdm.write handler。
        for h in list(root_logger.handlers):
            try:
                if isinstance(h, logging.FileHandler):
                    continue
                if isinstance(h, _TqdmLoggingHandler):
                    continue
                root_logger.removeHandler(h)
            except Exception:
                continue
        # 确保 stdout 有 handler（否则命令行可能只有 tqdm 一行）
        has_tqdm_handler = any(isinstance(h, _TqdmLoggingHandler) for h in root_logger.handlers)
        if not has_tqdm_handler:
            stream_handler = _TqdmLoggingHandler()
            stream_handler.setLevel(logging.INFO)
            try:
                stream_handler.setFormatter(formatter)
            except Exception:
                pass
            root_logger.addHandler(stream_handler)
        overwatch.info(f"📝 日志将写入 `{log_file_path}`")
        _install_global_exception_logger()
        overwatch.info("✅ 已安装全局异常捕获（未捕获异常将写入日志）")

    # 构建 LatentWorldVLA（内部自洽加载 VLM/LAM，并完成 tokenizer 扩展与冻结策略）
    overwatch.info("🔄 构建 LatentWorldVLA（内部加载VLM & LAM）")
    if cfg.use_simple_model:
        lwvla, processor = SimpleLatentWorldVLA.from_config(cfg=model_cfg)
    else:
        lwvla, processor = LatentWorldVLA.from_config(cfg=model_cfg)
    num_params = sum(p.numel() for p in lwvla.parameters())
    num_trainable_params = sum(p.numel() for p in lwvla.parameters() if p.requires_grad)
    overwatch.info(
        f"# LatentWorldVLA Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # ===== 在 DDP 包装前解冻 LAM decoder（如果配置要求） =====
    if model_cfg.unfreeze_lam_decoder:
        try:
            lam = lwvla.lam
            dec = getattr(lam, "decoder", None)
            if dec is None:
                overwatch.warning(
                    "[finetune_libero] unfreeze_lam_decoder=True but LAM has no decoder; "
                    "skip decoder unfreeze."
                )
            else:
                # 检查 decoder 是否已经被解冻
                dec_params = sum(p.numel() for p in dec.parameters())
                already_unfrozen = any(p.requires_grad for p in dec.parameters())
                
                if already_unfrozen:
                    overwatch.info(
                        f"[finetune_libero] LAM decoder ({dec_params/1e6:.3f}M params) already unfrozen; "
                        "skipping redundant unfreeze"
                    )
                else:
                    # 执行解冻
                    for p in dec.parameters():
                        p.requires_grad = True
                    overwatch.info(
                        f"[finetune_libero] Unfroze LAM decoder ({dec_params/1e6:.3f}M params) "
                        f"before optimizer creation"
                    )
                
                # 重新统计可训练参数
                num_trainable_params_after = sum(p.numel() for p in lwvla.parameters() if p.requires_grad)
                overwatch.info(
                    f"[finetune_libero] Trainable params after LAM decoder unfreeze: "
                    f"{num_trainable_params_after / 10**6:.3f}M "
                    f"(+{(num_trainable_params_after - num_trainable_params) / 10**6:.3f}M)"
                )
        except Exception as e:
            overwatch.warning(f"[finetune_libero] LAM decoder unfreeze failed: {e}")

    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator(mixture=`{cfg.data_mix}`, image_res={cfg.image_resolution})"
    )
    train_dataset, val_dataset, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        processor,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size,
        training_phase='post-training', 
        data_transform_fn=RLDSBatchTransformLIBERO,
        collator_fn=PaddedCollatorForLatentWorldVLA_LIBERO,
        latent_action_num_queries=lwvla.lam.num_queries,
        debug_repeat_batch=cfg.debug_repeat_batch,
        use_history_frame=cfg.use_history_frame,
        target_seq_len=180 if cfg.use_history_frame else 120,
    )
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # DDP
    lwvla = lwvla.to(device_id)
    wrapped_model = DDP(lwvla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Optimizer & LR scheduler
    # 按模块区分参数组（VLM 与非 VLM），以便为 VLM 设置独立 LR 与调度
    # 冻结策略已在 LatentWorldVLA.__init__ 中应用，此处无需额外冻结
    base_params: List[torch.nn.Parameter] = []
    latent_vla_params: List[torch.nn.Parameter] = []
    # 通过模块引用来界定 VLM 与 LAM 参数（不要依赖参数名）
    # 将 VLM 和 LAM 的参数都归入 vlm_params 组，以使用相同的学习率调度
    latent_vla_param_ids = {id(p) for p in wrapped_model.module.latent_vla.parameters()}
    # lam_param_ids = {id(p) for p in wrapped_model.module.lam.parameters()}
    for param in wrapped_model.module.parameters():
        if not param.requires_grad:
            continue
        if id(param) in latent_vla_param_ids:
            latent_vla_params.append(param)
        else:
            base_params.append(param)
    param_groups = [
        {"params": base_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ]
    # 如果 VLM 有可训练参数，为其添加独立的参数组
    if len(latent_vla_params) > 0:
        param_groups.append({"params": latent_vla_params, "lr": cfg.vlm_learning_rate, "weight_decay": cfg.weight_decay})
    optimizer = AdamW(param_groups)
    # 训练中用于裁剪的可训练参数集合
    trainable_params = base_params + latent_vla_params

    # 调度器（为不同参数组提供独立的 lr lambda）
    def lr_lambda_base(current_step: int):
        if current_step < cfg.warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, cfg.warmup_steps))
        else:
            # cosine decay
            progress = float(current_step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-6)

    def lr_lambda_vlm(current_step: int):
        if current_step < cfg.vlm_warmup_steps:
            # linear warmup（VLM 独立 warmup）
            return float(current_step) / float(max(1, cfg.vlm_warmup_steps))
        else:
            # cosine decay
            progress = float(current_step - cfg.vlm_warmup_steps) / float(max(1, cfg.max_steps - cfg.vlm_warmup_steps))
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-6)

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
        num_workers=0,  # collator 已不再调用 vq_encode，默认单进程以保持确定性
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
        # Separate wandb storage for finetune runs
        # wb_dir = run_dir / "wandb"
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception:
            pass
        # ensure wandb honors directory even if it spawns processes
        os.environ["WANDB_DIR"] = str(run_dir)
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}", dir=str(run_dir))
        # push full config to W&B
        try:
            wandb.config.update({k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}, allow_val_change=True)
        except Exception:
            pass

    # Train!
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    with tqdm.tqdm(
        total=cfg.max_steps,
        desc=f"🚀 {cfg.run_id_note or 'Training'}",
        leave=True,
        disable=not distributed_state.is_main_process,
    ) as progress:
        wrapped_model.train()
        # VLM eval 模式由 LatentWorldVLA.train() 自动管理
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {
                k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k in {"pixel_values","input_ids","attention_mask","act_placeholder_mask","lam_videos","lam_states","actions","proprio","image_grid_thw"}
            }
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            # Forward

            # 在训练循环中替换 forward/backward/step 区块：
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # 或者 dtype=torch.float16 取决于硬件
                out = wrapped_model(**batch)
                flow_loss = out["loss_flow"]
                perceptual_loss = out.get("loss_perceptual", torch.tensor(0.0, device=device_id, dtype=flow_loss.dtype))
                loss = out["loss_total"]

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
                    postfix = {
                        "loss": f"{smoothened_loss:.6f}",
                        "flow": f"{float(flow_loss.item()):.6f}",
                        "perc": f"{float(perceptual_loss.item()):.6f}",
                    }
                    if "vlm_action_accuracy" in out:
                        postfix["acc"] = f"{float(out['vlm_action_accuracy'].item()):.4f}"
                    progress.set_postfix(postfix)
                    log_payload = {
                        "train_loss": smoothened_loss,
                        "train_flow_loss": float(flow_loss.item()),
                        "train_perceptual_loss": float(perceptual_loss.item()),
                        # 记录基础参数组与（若有）VLM 参数组的学习率
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    if len(optimizer.param_groups) > 1:
                        log_payload["vlm_lr"] = optimizer.param_groups[1]['lr']
                    if "vlm_action_accuracy" in out:
                        log_payload["train_action_accuracy"] = float(out["vlm_action_accuracy"].item())
                    wandb.log(log_payload, step=global_step)

                    # Also persist key iteration stats to the run log file (tqdm UI does not go to FileHandler)
                    if (global_step % int(max(1, cfg.log_every_steps))) == 0 or global_step == 1:
                        try:
                            overwatch.info(
                                f"[step {global_step}/{cfg.max_steps}] "
                                f"loss={smoothened_loss:.6f} "
                                f"flow={float(flow_loss.item()):.6f} "
                                f"perc={float(perceptual_loss.item()):.6f} "
                                f"lr={optimizer.param_groups[0]['lr']:.3e}"
                            )
                        except Exception:
                            pass

                # Checkpoint on schedule (skip step 0)
                if global_step > 0 and global_step % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        step_dir = ckpt_root / f"step-{global_step:06d}"
                        os.makedirs(step_dir, exist_ok=True)
                        overwatch.info(f"💾 保存检查点 @ step={global_step}: 目录 `{step_dir}`")


                        # 保存完整 LatentWorldVLA（包含 VLM+latent_vla_extra + Flow）
                        wrapped_model.module.save_pretrained(step_dir)
                        overwatch.info("✅ 已保存 LatentWorldVLA(checkpoint dir)（VLM+latent_vla_extra+Flow）")

                        # 保存 processor（image_processor + tokenizer）
                        if processor is not None and hasattr(processor, "save_pretrained"):
                            processor.save_pretrained(step_dir)
                            overwatch.info("✅ 已保存 Processor（image_processor + tokenizer）")

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
                                if k in {"pixel_values","input_ids","attention_mask","act_placeholder_mask","lam_videos","lam_states","actions","proprio","image_grid_thw"}
                            }
                            if "pixel_values" in eval_batch:
                                eval_batch["pixel_values"] = eval_batch["pixel_values"].to(torch.bfloat16)

                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                predicted_actions = wrapped_model.module.predict_action(
                                    pixel_values=eval_batch["pixel_values"],
                                    input_ids=eval_batch["input_ids"],
                                    attention_mask=eval_batch["attention_mask"],
                                    act_placeholder_mask=eval_batch["act_placeholder_mask"],
                                    lam_videos=eval_batch["lam_videos"],
                                    lam_states=eval_batch.get("lam_states", None),
                                    proprio=eval_batch.get("proprio", None),
                                    image_grid_thw=eval_batch.get("image_grid_thw", None),
                                )
                                # 计算预测动作与真实动作的 MSE
                                gt_actions = eval_batch["actions"]  # [B, T, Da]
                                eval_loss = F.mse_loss(predicted_actions, gt_actions)

                            eval_loss_accum += float(eval_loss.item())
                            eval_count += 1

                    mean_eval_loss = eval_loss_accum / max(eval_count, 1)
                    if distributed_state.is_main_process:
                        wandb.log({
                            "eval_loss": mean_eval_loss,
                            "global_step": global_step,
                        }, step=global_step)
                        overwatch.info(
                            f"📊 评估完成 @ step={global_step}: eval_loss(MSE)={mean_eval_loss:.6f}"
                        )
                        # Also persist eval stats explicitly (mirrors train step logging)
                        try:
                            overwatch.info(
                                f"[eval step {global_step}] "
                                f"eval_loss={mean_eval_loss:.6f} "
                                f"eval_batches={int(eval_count)}"
                            )
                        except Exception:
                            pass

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
                                # 只有在目标目录确实存在时才创建符号链接
                                # 不再主动创建目录，只有在保存检查点时才创建
                                if target_dir.exists():
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
                    # VLM eval 模式由 LatentWorldVLA.train() 自动管理



if __name__ == "__main__":
    finetune()

