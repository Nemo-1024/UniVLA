import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from latent_action_model.core.lam_model import load_latent_action_model
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.vla.datasets.datasets import RLDSDataset
from prismatic.models import load_InternVL,freeze_internvl
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from typing import cast
from prismatic.training.accelerate_fsdp_trainer import run_latent_action_training
from transformers import AutoProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# 📝 动作 token 列表: ['<ACT_0>', '<ACT_1>', '<ACT_2>', '<ACT_3>', '<ACT_4>', '<ACT_5>', '<ACT_6>', '<ACT_7>', '<ACT_8>', '<ACT_9>', '<ACT_10>', '<ACT_11>', '<ACT_12>', '<ACT_13>', '<ACT_14>', '<ACT_15>']
# 🔢 对应的 token ID: [151679, 151680, 151681, 151682, 151683, 151684, 151685, 151686, 151687, 151688, 151689, 151690, 151691, 151692, 151693, 151694]
# 🎯 action_token_begin_id = 151679
# 📊 ID 范围: 151679 - 151694

@dataclass
class TrainConfig:
    # fmt: off

    # =========================
    # 路径与资源
    # =========================
    # Directory Paths
    data_root_dir: Path = Path("/mnt/public_zgc/home/jlchen/datasets")
    run_root_dir: Path = Path(__file__).resolve().parent / "vla_log"    # Store logs & checkpoints under vla_scripts/

    # Hugging Face 模型标识（或本地权重目录）；用于 PrismaticVLM.from_pretrained()
    model_id: str = '/mnt/public_zgc/home/jlchen/weights/InternVL3_5-1B-Instruct-HF'
    hf_cache_dir: Optional[Path] = None
    lam_path: str = "/mnt/public_zgc/home/jlchen/code/UniVLA/latent_action_model/logs/version_1/checkpoints/epoch=49_step=50000.ckpt"

    # =========================
    # 数据与预处理
    # =========================
    # 数据混合与缓冲
    data_mix: str = "droid_100"
    shuffle_buffer_size: int = 20_000
    image_resolution: int = 448
    image_aug: bool = True                                          # Whether to enable image augmentations

    # =========================
    # 模型冻结策略
    # =========================
    # 冻结策略
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_projector: bool = False

    # =========================
    # LAM / 动作离散参数
    # =========================
    action_token_begin_id: int = 151679
    # VJEPA_LAM 模型架构参数
    vision_model_id: str = "/mnt/public_zgc/home/jlchen/weights/vjepa2-vitl-fpc64-256"
    codebook_size: int = 16  #此处修改无效，仅作为标记
    # =========================
    # 训练设置
    # =========================
    # 训练超参
    epochs: Optional[int] = 10
    max_steps: Optional[int] = 200000  #以max_steps为准，若为空则按epochs * 10000近似
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-6
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant_with_warmup"   #constant_with_warmup
    # 训练加速
    enable_mixed_precision_training: bool = True
    seed: int = 42                                                  # Random seed (for reproducibility)

    # =========================
    # 评估与保存
    # =========================
    eval_strategy: str = "steps"
    save_total_limit: int = -1
    eval_interval: int = 1000
    eval_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 24
    save_interval: int = 1000                                    # Interval for saving checkpoints (in steps

    # =========================
    # 分布式 / FSDP
    # =========================
    fsdp: Optional[str] = "full_shard"                     # 示例："full_shard auto_wrap" 或 None 关闭
    fsdp_config: Optional[Dict[str, Any]] = None   # 示例：{"fsdp_min_num_params": 1e7, "xla": False}

    # =========================
    # 运行与日志
    # =========================
    # Run Arguments
    run_id: Optional[str] =  None                                  # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = "run_01"                               # Extra note for logging, Weights & Biases
    # Tracking Parameters
    wandb_project: str = "vla_pretraining"                   # Name of W&B project to log to (use default!)
    # wandb_entity: str = "opendrivelab"                              # Name of entity to log under

    # =========================
    # 恢复 / 断点（仅用于日志标记，不再用于模型权重加载）
    # =========================
    # Resume (logging) Parameters -- 仅用于日志标记，不再用于模型权重加载
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None

    # =========================
    # HF Hub 凭据（如有门限模型）
    # =========================
    # HF Hub Credentials (for any gated models)
    hf_token: Optional[str] = None

    # fmt: on


def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_tag = f"{cfg.model_id.split('/')[-1]}+{cfg.data_mix}"
    world_size = overwatch.world_size() if dist.is_initialized() else max(torch.cuda.device_count(), 1)
    overwatch.info(f"Detected world_size = {world_size}")
    cfg.run_id = (
        f"{vla_tag}+n{world_size}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # cfg.run_id += '-Latent-Action-Pretraining'
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    hf_token = str(cfg.hf_token) if isinstance(cfg.hf_token, Path) else cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    # 统一时间戳并广播，确保所有进程共用同一目录
    if dist.is_initialized():
        if overwatch.is_rank_zero():
            ts: Optional[str] = datetime.now().strftime("%m%d_%H%M%S")
        else:
            ts = None
        obj_list = [ts]
        dist.broadcast_object_list(obj_list, src=0)
        timestamp = obj_list[0]
        assert isinstance(timestamp, str)
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")

    run_dir_name = f"{timestamp}+{cfg.run_id}"
    run_dir = (cfg.run_root_dir / run_dir_name)
    # 仅 rank0 创建目录，其余进程等待
    if (not dist.is_initialized()) or overwatch.is_rank_zero():
        os.makedirs(run_dir, exist_ok=True)
        try:
            os.makedirs(run_dir / "checkpoints", exist_ok=True)
        except Exception:
            pass
    if dist.is_initialized():
        dist.barrier()

    # 仅 rank0 写入文件日志，并避免重复添加 FileHandler
    if (not dist.is_initialized()) or overwatch.is_rank_zero():
        try:
            import logging

            log_path = str(run_dir / "train.log")
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

    # os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)


 
    # 直接通过 HF ID/Path 加载 InternVL 模型与处理器
    overwatch.info(f"🔄 加载基础 InternVL `{cfg.model_id}`（HF from_pretrained）")
    vlm, tokenizer = load_InternVL(cfg.model_id, cfg.hf_cache_dir, dtype=torch.bfloat16) 
    vlm.generation_config.max_new_tokens = int(getattr(cfg, "max_new_tokens", 4))
    vlm.generation_config.pad_token_id=int(tokenizer.eos_token_id)
    vlm.config.loss_type = str(getattr(cfg, "loss_type", "ForCausalLMLoss"))
    vlm.config.use_cache = False

    # 直接按配置冻结模块（若可用）；HF-only InternVL 组件名：vision_tower / language_model / multi_modal_projector / lm_head
    freeze_internvl(vlm, cfg.freeze_vision_backbone, cfg.freeze_projector, cfg.freeze_llm_backbone, cfg.freeze_last_llm_layer)

    overwatch.info("🔧 扩充 LLM 词表以注入动作离散 token")
    special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    try:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
        overwatch.info(f"num_added_toks={num_added_toks}")
    except Exception:
        num_added_toks = 0
   
    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    
    # Get VLA Dataset & Collator

    overwatch.info(
        f"🔄 加载 V-JEPA2 动作编码器与码本（ckpt=`{cfg.lam_path}`，"
        f"K={cfg.codebook_size}）"
    )
    latent_action_model = load_latent_action_model(cfg.lam_path, vision_model_id=cfg.vision_model_id)  # default freeze all parameters
    latent_action_model = latent_action_model.to(device_id).eval()
    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator（mixture=`{cfg.data_mix}`，image_res={cfg.image_resolution}）"
    )
    # 类型提示规避：latent_action_tokenizer 需要 VQ 编码器，这里用 cast 静态规避
    train_dataset, val_dataset, tokenizer, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        latent_action_model,
        tokenizer=tokenizer,
        default_image_resolution=cfg.image_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )




    act_tokens = [f"<ACT_{i}>" for i in range(cfg.codebook_size)]
    act_ids = tokenizer.convert_tokens_to_ids(act_tokens)

    expected_begin_id = min(act_ids)
    expected_end_id = max(act_ids)

    assert cfg.action_token_begin_id == expected_begin_id, (
        f"cfg.action_token_begin_id={cfg.action_token_begin_id} "
        f"but tokenizer gives {expected_begin_id} "
        f"(range: {expected_begin_id}-{expected_end_id})"
    )


    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(cast(RLDSDataset, train_dataset).dataset_statistics, run_dir)

    # 使用 Accelerate + FSDP 的新训练器（直接传入 dataclass -> dict）
    overwatch.info("🚀 启动 VLA 训练循环（Accelerate+FSDP）；首次 step 可能较慢（初始化 FSDP/AMP）")

    run_latent_action_training(
        cfg=cfg,
        vlm=vlm,
        vla_dataset=cast(RLDSDataset, train_dataset),
        eval_dataset=cast(RLDSDataset, val_dataset),
        collator=collator,
        tokenizer=tokenizer,
        run_dir=run_dir,
        overwatch=overwatch,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    # And... we're done!
    overwatch.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    # 简化入口（原 draccus CLI 装饰器已移除）
    train(TrainConfig())