import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import draccus
import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from prismatic.overwatch import initialize_overwatch
import wandb
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.datasets import RLDSBatchTransformLIBERO
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from latent_action_model.core.lam_model import load_latent_action_model
from prismatic.models.load import load_InternVL, freeze_internvl
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)


@dataclass
class FinetuneConfig:
    # fmt: off
    # Base VLM & LAM
    model_id: str = '/mnt/public_zgc/home/jlchen/weights/InternVL3_5-1B-Instruct-HF'
    hf_cache_dir: Optional[str] = None
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt"

    # Dataset
    data_root_dir: Path = Path("/mnt/public_zgc/home/jlchen/datasets")
    data_mix: str = "droid_100"
    image_resolution: int = 448
    shuffle_buffer_size: int = 2000
    image_aug: bool = True

    # Run & IO
    run_root_dir: Path = Path("vla_log")
    # adapter_tmp_dir: Path = Path("adapter-tmp")
    # save_latest_checkpoint_only: bool = True

    # Optimization
    batch_size: int = 8
    max_steps: int = 200000
    save_steps: int = 30000
    eval_steps: int = 1000
    eval_batches: int = 100
    learning_rate: float = 1e-5
    grad_accumulation_steps: int = 16
    gradient_clip: float = 1.0
    vlm_loss_weight: float = 1.0

    # Seeding & dtype
    seed: int = 42

    # LAM/codebook
    codebook_size: int = 16

    # Freezing policy (applies to InternVL)
    freeze_vision_backbone: bool = False
    freeze_projector: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False

    # LoRA (optional)
    use_lora: bool = False
    lora_rank: int = 32
    lora_dropout: float = 0.05
    lora_alpha: int = 32
    lora_target_modules: List[str] = None  # if None, use a sensible default
    # 是否在保存时同时导出 LoRA 适配器（便于后处理融合）
    save_lora_adapter: bool = True

    # Tracking
    wandb_project: str = "finetune-LIBERO"
    wandb_entity: str = "test-project"
    run_id_note: Optional[str] = None



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning LatentWorldVLA on `{cfg.data_mix}` with base `{cfg.model_id}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Seeding
    torch.manual_seed(cfg.seed + device_id)
    torch.cuda.manual_seed_all(cfg.seed + device_id)
    random.seed(cfg.seed + device_id)

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.model_id.split('/')[-1]}+{cfg.data_mix}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories (hierarchical)
    run_dir = cfg.run_root_dir / exp_id
    ckpt_root = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load InternVL model & tokenizer
    overwatch.info(f"🔄 加载基础 InternVL `{cfg.model_id}`（HF from_pretrained）")
    vlm, tokenizer = load_InternVL(cfg.model_id, cfg.hf_cache_dir, dtype=torch.bfloat16)

    # Extend tokenizer with latent action special tokens and resize embeddings
    overwatch.info("🔧 扩充 LLM 词表以注入动作离散 token")
    special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    try:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
    except Exception:
        num_added_toks = 0
    if num_added_toks > 0 and hasattr(vlm, 'resize_token_embeddings'):
        vlm.resize_token_embeddings(len(tokenizer))

    # Freeze InternVL as configured
    freeze_internvl(vlm, cfg.freeze_vision_backbone, cfg.freeze_projector, cfg.freeze_llm_backbone, cfg.freeze_last_llm_layer)

    # Optional LoRA: attach to language_model submodules
    if cfg.use_lora:
        target_modules = cfg.lora_target_modules or [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
        ]
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # 将 LoRA 应用于整个 VLM（内部仅会匹配 target_modules）
        vlm = get_peft_model(vlm, lora_cfg)

    # Load LAM
    overwatch.info(
        f"🔄 加载 V-JEPA2 动作编码器与码本（ckpt=`{cfg.lam_path}`，K={cfg.codebook_size}）"
    )
    latent_action_model = load_latent_action_model(cfg.lam_path)
    latent_action_model = latent_action_model.to(device_id).eval()

    # Build LatentWorldVLA with config
    model_cfg = LatentWorldVLAConfig()
    # 根据 tokenizer 中 ACT_ token 的起始 id 校准 action_token_begin_id
    act_tokens = [f"<ACT_{i}>" for i in range(cfg.codebook_size)]
    act_ids = tokenizer.convert_tokens_to_ids(act_tokens)
    if isinstance(act_ids, list) and len(act_ids) > 0 and min(act_ids) != -1:
        model_cfg.action_token_begin_id = min(act_ids)

    overwatch.info("🔄 构建 LatentWorldVLA")
    lwvla = LatentWorldVLA(
        lam=latent_action_model,
        model_cfg=model_cfg,
        vlm=vlm,
    )

    # 如果使用 LoRA，LatentWorldVLA 内部会对 VLM requires_grad_(False)，此处重新开启 LoRA 权重
    if cfg.use_lora:
        for name, param in lwvla.vlm.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(True)
        # 训练阶段启用 VLM 的 train 模式以激活 LoRA 分支（主体权重仍冻结）
        lwvla.vlm.train()

    num_params = sum(p.numel() for p in lwvla.parameters())
    num_trainable_params = sum(p.numel() for p in lwvla.parameters() if p.requires_grad)
    overwatch.info(
        f"# LatentWorldVLA Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator(mixture=`{cfg.data_mix}`, image_res={cfg.image_resolution})"
    )
    train_dataset, val_dataset, tokenizer, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        latent_action_model,
        tokenizer=tokenizer,
        default_image_resolution=cfg.image_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
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
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)

    # DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # RLDS 内部并行
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
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad(set_to_none=True)
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
            out = wrapped_model(**batch)
            flow_loss = out["loss_flow"]
            if cfg.use_lora:
                vlm_loss = out.get("loss_vlm", torch.tensor(0.0, device=device_id, dtype=flow_loss.dtype))
                loss = flow_loss + cfg.vlm_loss_weight * vlm_loss
            else:
                vlm_loss = torch.tensor(0.0, device=device_id, dtype=flow_loss.dtype)
                loss = flow_loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Grad clip at micro step
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.gradient_clip)

            # Metrics smoothing
            recent_losses.append(loss.item())
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                progress.update()

            # Logging every 5 steps
            if distributed_state.is_main_process and (batch_idx+1) % cfg.grad_accumulation_steps == 0:
                log_payload = {
                    "train_loss": smoothened_loss,
                    "train_flow_loss": float(flow_loss.item()),
                    "lr": optimizer.param_groups[0]['lr'],
                }
                if "vlm_action_accuracy" in out:
                    log_payload["train_action_accuracy"] = float(out["vlm_action_accuracy"].item())
                if cfg.use_lora:
                    log_payload["train_vlm_loss"] = float(vlm_loss.item())
                wandb.log(log_payload, step=gradient_step_idx)

            # Checkpoint
            if gradient_step_idx % cfg.save_steps == 0 and (batch_idx+1) % cfg.grad_accumulation_steps == 0:
                if distributed_state.is_main_process:
                    step_dir = ckpt_root / f"step-{gradient_step_idx:06d}"
                    adapters_dir = step_dir / "adapters"
                    os.makedirs(step_dir, exist_ok=True)

                    # Save model weights (state_dict) + optim/sched
                    torch.save(wrapped_model.module.state_dict(), step_dir / "lwvla.pt")
                    torch.save(optimizer.state_dict(), step_dir / "optim.pt")
                    torch.save(scheduler.state_dict(), step_dir / "sched.pt")

                    # Save LoRA adapter if used
                    if cfg.use_lora and cfg.save_lora_adapter and hasattr(wrapped_model.module.vlm, "save_pretrained"):
                        os.makedirs(adapters_dir, exist_ok=True)
                        wrapped_model.module.vlm.save_pretrained(adapters_dir)

                    # Save dataset statistics once is sufficient but safe to overwrite
                    save_dataset_statistics(train_dataset.dataset_statistics, step_dir)

                    # Maintain latest symlink/text pointer
                    latest_link = ckpt_root / "latest"
                    if latest_link.exists() or latest_link.is_symlink():
                        try:
                            latest_link.unlink()
                        except Exception:
                            pass
                    try:
                        latest_link.symlink_to(step_dir.name)
                    except Exception:
                        with open(ckpt_root / "LATEST", "w") as f:
                            f.write(step_dir.name)

                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx >= cfg.max_steps:
                if distributed_state.is_main_process:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

            # Periodic eval
            if gradient_step_idx % cfg.eval_steps == 0 and (batch_idx+1) % cfg.grad_accumulation_steps == 0:
                wrapped_model.eval()
                eval_loss_accum, eval_count = 0.0, 0
                eval_acc_accum = 0.0
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=cfg.batch_size,
                    sampler=None,
                    collate_fn=collator,
                    num_workers=0,
                )
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
                        eval_out = wrapped_model(**eval_batch)
                        eval_flow = eval_out["loss_flow"]
                        eval_total = eval_flow
                        if cfg.use_lora:
                            eval_vlm = eval_out.get("loss_vlm", torch.tensor(0.0, device=device_id, dtype=eval_flow.dtype))
                            eval_total = eval_total + cfg.vlm_loss_weight * eval_vlm
                        eval_loss_accum += float(eval_total.item())
                        if "vlm_action_accuracy" in eval_out:
                            eval_acc_accum += float(eval_out["vlm_action_accuracy"].item())
                        eval_count += 1
                mean_eval_loss = eval_loss_accum / max(eval_count, 1)
                mean_eval_acc = eval_acc_accum / max(eval_count, 1)
                if distributed_state.is_main_process:
                    wandb.log({
                        "eval_loss": mean_eval_loss,
                        "eval_action_accuracy": mean_eval_acc,
                        "global_step": gradient_step_idx,
                    }, step=gradient_step_idx)

                    # Maintain best checkpoint by eval_loss
                    best_file = ckpt_root / "BEST.txt"
                    prev_best = None
                    if best_file.exists():
                        try:
                            prev_best = float(open(best_file).read().strip().split()[0])
                        except Exception:
                            prev_best = None
                    if (prev_best is None) or (mean_eval_loss < prev_best):
                        with open(best_file, "w") as f:
                            f.write(f"{mean_eval_loss} step-{gradient_step_idx:06d}\n")
                        best_link = ckpt_root / "best"
                        if best_link.exists() or best_link.is_symlink():
                            try:
                                best_link.unlink()
                            except Exception:
                                pass
                        target_dir = ckpt_root / f"step-{gradient_step_idx:06d}"
                        if not target_dir.exists():
                            os.makedirs(target_dir, exist_ok=True)
                        try:
                            best_link.symlink_to(target_dir.name)
                        except Exception:
                            with open(ckpt_root / "BEST_STEP", "w") as f:
                                f.write(target_dir.name)
                wrapped_model.train()

                # Optionally export merged LoRA weights for the base VLM
                if cfg.use_lora and cfg.save_lora_adapter and distributed_state.is_main_process:
                    try:
                        if hasattr(wrapped_model.module.vlm, "merge_and_unload"):
                            merged_vlm = wrapped_model.module.vlm.merge_and_unload()
                            out_dir = Path(run_dir) / "vlm-merged"
                            os.makedirs(out_dir, exist_ok=True)
                            if hasattr(merged_vlm, "save_pretrained"):
                                merged_vlm.save_pretrained(out_dir)
                    except Exception as e:
                        print(f"[Warn] LoRA merge-and-unload failed: {e}")


if __name__ == "__main__":
    finetune()
