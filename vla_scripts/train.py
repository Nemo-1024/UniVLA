import json
import os
import sys
import shutil
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from datetime import datetime
import draccus
import torch
import torch.distributed as dist
import yaml
from latent_action_model.core.lam_model import load_latent_action_model
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.vla.datasets.datasets import RLDSDataset
from prismatic.models import load_vlm_auto, freeze_vlm_generic, freeze_qwen3vl
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from typing import cast
from prismatic.training.accelerate_fsdp_trainer import run_latent_action_training
from prismatic.vla.latent_vla_model import LatentVLAModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

home_path = "/mnt/project_rlinf/jlchen"
@dataclass
class TrainConfig:
    # fmt: off

    # =========================
    # 路径与资源
    # =========================
    # Directory Paths
    data_root_dir: Path = Path(home_path + "/datasets")
    run_root_dir: Path = Path(__file__).resolve().parent / "vla_log"    # Store logs & checkpoints under vla_scripts/

    # Hugging Face 模型标识（或本地权重目录）；用于 PrismaticVLM.from_pretrained()
    model_id: str = home_path + "/weights/InternVL3_5-1B-Instruct-HF"
    hf_cache_dir: Optional[Path] = None
    lam_ckpt_path: str = home_path + "/code/UniVLA/latent_action_model/logs/dino_base_4q_32_norep/version_0/checkpoints/epoch=4.ckpt"
    lam_yaml_path: str = home_path + "/code/UniVLA/latent_action_model/logs/dino_base_4q_32_norep/version_0/dino_base.yaml"

    # =========================
    # 数据与预处理
    # =========================
    # 数据混合与缓冲
    data_mix: str = "bridge_dataset"
    training_phase: str = "pre-training"
    shuffle_buffer_size: int = 10240
    image_resolution: int = 256
    image_aug: bool = False                                          # Whether to enable image augmentations
    # DataLoader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # =========================
    # 模型冻结策略
    # =========================
    # 冻结策略
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_projector: bool = False
    freeze_embedding: bool = False  # 冻结 embedding 层
    # 仅解冻视觉 backbone 中的 patch merger（如 Qwen3-VL 的 vision_model.merger / deepstack_merger_list）
    # 常用于 freeze_vision_backbone=True 时仍允许轻量适配视觉 patch 合并层。
    unfreeze_vision_merger: bool = False
    # 解冻 LLM 最后 n 层（需要配合 freeze_llm_backbone=True 使用）
    # 注意：解冻将在优化器创建前立即执行
    unfreeze_llm_last_n_layers: Optional[int] = None
    unfreeze_lam_decoder: bool = False
    # =========================
    # LoRA 低秩适配
    # =========================
    enable_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"  # ["none", "all", "lora_only"]
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_task_type: str = "CAUSAL_LM"
    lora_merge_before_save: bool = True

    # =========================
    # LAM / 动作离散参数
    # =========================
    latent_action_placeholder_token: str = "<ACT_PH>"
    # === 动作监督模式说明 ===
    # latent 回归监督：在 <ACT_PH> 位置注入可学习 query (act_query)，并监督回归 LAM latent
    # latent 回归损失类型：cosine 或 mse
    latent_loss_type: str = "mse"
    enable_lam_decoder_perceptual: bool = False
    lam_encoder_distill_weight: float = 1.0
    lam_decoder_perceptual_weight: float = 1.0
    # 对可学习 query / 回归头施加梯度放大（>1 放大，=1 不变）
    act_query_lr_scale: float = 1.0
    vlm_to_lam_lr_scale: float = 1.0
    use_latent_vla_model: bool = True
    lam_decoder_target: str = "teacher"
    # vision_model_id: str = home_path + "/weights/dinov3-vitl16-pretrain-lvd1689m"
    # =========================
    # 训练设置
    # =========================
    # 训练超参
    # debug_repeat_batch 支持 bool 或 int：传入正整数 k 时，会缓存前 k 个样本并循环返回
    debug_repeat_batch: Union[bool, int] = False
    epochs: Optional[int] = 10
    max_steps: Optional[int] = 200000  #以max_steps为准，若为空则按epochs * 10000近似
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant_with_warmup"   #constant_with_warmup
    optim: str = "adamw_torch"
    # 训练加速
    enable_mixed_precision_training: bool = False
    seed: int = 42                                                  # Random seed (for reproducibility)
    logging_steps: int = 50
    use_history_frame: bool = False
    window_size: int = 20
    # =========================
    # 评估与保存
    # =========================
    eval_strategy: str = "steps"
    save_total_limit: int = -1
    eval_interval: int = 1000
    eval_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 64
    save_interval: int = 10000                                   # Interval for saving checkpoints (in steps

    # =========================
    # 分布式 / FSDP
    # =========================
    fsdp: Optional[str] = None                     # 示例："full_shard auto_wrap" 或 None 关闭
    fsdp_config: Optional[Dict[str, Any]] = None   # 示例：{"fsdp_min_num_params": 1e7, "xla": False}

    # =========================
    # 运行与日志
    # =========================
    # Run Arguments
    run_id: Optional[str] =  "test"     # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    # Tracking Parameters
    wandb_project: str = "vla_pretraining"                   # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None                              # Name of entity to log under

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

@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    """训练入口（draccus 驱动 CLI）。

    用法示例：
    - 通过配置文件启动（draccus 将根据扩展名自动解析：.yaml/.yml/.json/.toml）：
      python -m vla_scripts.train --config /path/to/config.yaml
    - 也可直接传入单个参数覆盖配置文件对应字段（命令行实参优先生效）。
    """
    overwatch.info("OpenVLA Training :: Warming Up")

    # ---- Config info for action supervision ----
    overwatch.info(
        "[TrainConfig] using latent regression as main supervision (no token CE), with learnable queries at <ACT_PH>."
    )

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()
    # 尽量保持确定性，减少 LAM/latent 输出的随机波动
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Configure Unique Run Name & Save Directory
    vla_tag = f"{cfg.model_id.split('/')[-1]}+{cfg.data_mix}"
    world_size = overwatch.world_size() if dist.is_initialized() else max(torch.cuda.device_count(), 1)
    overwatch.info(f"Detected world_size = {world_size}")

    if cfg.run_id is None:
        cfg.run_id = f"{vla_tag}+n{world_size}+b{cfg.per_device_batch_size}+x{cfg.seed}"


    # cfg.run_id += '-Latent-Action-Pretraining'
    # Start =>> Build Directories and Set Randomness
    # overwatch.info('"Do or do not; there is no try."', ctx_level=1)
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

    # 保存本次运行的配置文件到日志目录（rank0）
    if (not dist.is_initialized()) or overwatch.is_rank_zero():
        try:
            # 1) 保存命令行
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
                    # 使用原配置文件名
                    shutil.copy2(str(src_path), str(run_dir / src_path.name))
        except Exception:
            pass

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


 
    # 直接在 LatentVLAModel 内部加载 VLM 与 LAM；当启用 debug_repeat_batch 时同步开启 debug_mode
    debug_mode = bool(cfg.debug_repeat_batch)
    latent_vla_model, processor = LatentVLAModel.from_config(cfg, overwatch=overwatch, debug_mode=debug_mode)

    # LoRA 适配（默认仅作用于 LLM/text backbone）
    if cfg.enable_lora:
        task_type = getattr(TaskType, str(cfg.lora_task_type).upper(), TaskType.CAUSAL_LM)
        lora_config = LoraConfig(
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            bias=str(cfg.lora_bias),
            target_modules=list(cfg.lora_target_modules) if cfg.lora_target_modules else None,
            task_type=task_type,
        )
        # 如果启用 LoRA，则确保 backbone 不被整体冻结
        if cfg.freeze_llm_backbone:
            overwatch.info("LoRA enabled: overriding freeze_llm_backbone=False to train adapters")
            cfg.freeze_llm_backbone = False

        latent_vla_model.vlm = get_peft_model(latent_vla_model.vlm, lora_config)
        try:
            latent_vla_model.vlm.print_trainable_parameters()
        except Exception:
            pass


    # 按配置冻结/解冻 VLM（内部已注册 tokenizer），保持显存与训练策略
    # Qwen3-VL: use explicit (non-generic) freezing logic for stability.

    freeze_qwen3vl(
        latent_vla_model.vlm,
        cfg.freeze_vision_backbone,
        cfg.freeze_llm_backbone,
        cfg.freeze_last_llm_layer,
        cfg.freeze_embedding,
        cfg.unfreeze_vision_merger,
    )
    num_params = sum(p.numel() for p in latent_vla_model.parameters())
    num_trainable_params = sum(p.numel() for p in latent_vla_model.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator（mixture=`{cfg.data_mix}`，image_res={cfg.image_resolution}）"
    )
    # 类型提示规避：latent_action_tokenizer 需要 LAM 编码器，这里用 cast 静态规避
    train_dataset, val_dataset, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        processor=processor,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        training_phase=cfg.training_phase,
        latent_action_num_queries=latent_vla_model.num_queries,
        debug_repeat_batch=cfg.debug_repeat_batch,
        target_seq_len=250 if cfg.use_history_frame else 200,
        use_history_frame=cfg.use_history_frame,
        window_size=cfg.window_size,
    )


    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(cast(RLDSDataset, train_dataset).dataset_statistics, run_dir)

    # 组装组合模型（VLM + LAM）
    # 使用 Accelerate + FSDP 的新训练器（直接传入 dataclass -> dict）
    overwatch.info("🚀 启动 VLA 训练循环（Accelerate+FSDP）；首次 step 可能较慢（初始化 FSDP/AMP）")
    # Avoid hanging when torch.distributed was not initialized by the launcher.
    if dist.is_initialized():
        dist.barrier()
    run_latent_action_training(
        cfg=cfg,
        model=latent_vla_model,
        vla_dataset=cast(RLDSDataset, train_dataset),
        eval_dataset=cast(RLDSDataset, val_dataset),
        collator=collator,
        processor=processor,
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
    # 入口：由 draccus.wrap 提供 CLI（支持 --config 指定 YAML/JSON/TOML 配置）
    train()
