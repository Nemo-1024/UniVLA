import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.distributed as dist
import yaml
from latent_action_model.core.lam_model import LatentLAMModel
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.vla.datasets.datasets import RLDSDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from typing import cast
from prismatic.training.accelerate_fsdp_trainer import run_latent_action_training
from transformers import AutoProcessor
from transformers.models.internvl.modeling_internvl import (
    InternVLForConditionalGeneration,
)

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

    # 冻结策略
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    freeze_last_llm_layer: bool = False
    freeze_projector: bool = True

    action_token_begin_id: int = 151679

    # 数据混合与缓冲
    data_mix: str = "droid_100"
    shuffle_buffer_size: int = 20_000
    image_resolution: int = 448

    # 训练超参
    epochs: int = 10
    max_steps: Optional[int] = None
    per_device_batch_size: int = 4
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    # 训练加速
    enable_mixed_precision_training: bool = True

    # 分布式 / FSDP
    fsdp: Optional[str] = "full_shard"                     # 示例："full_shard auto_wrap" 或 None 关闭
    fsdp_config: Optional[Dict[str, Any]] = None   # 示例：{"fsdp_min_num_params": 1e7, "xla": False}

    # Hugging Face 模型标识（或本地权重目录）；用于 PrismaticVLM.from_pretrained()
    model_id: str = '/data/home/jlchen/weights/InternVL3_5-1B-Instruct-HF'
    hf_cache_dir: Optional[Path] = None
    lam_path: str = "/data/home/jlchen/code/UniVLA/latent_action_model/logs/vjepa_lam/last.ckpt"

    # VJEPA_LAM 模型架构参数
    dim: int = 1024                    # 特征维度 (V-JEPA2 ViT Large 输出维度)
    enc_layers: int = 6                # LAM 编码器层数
    codebook_size: int = 16            # 码本大小
    code_dim: int = 256                # 码本维度
    dec_layers: int = 6                # 解码器层数
    dec_self_heads: int = 4            # 解码器自注意力头数
    dec_cross_heads: int = 4           # 解码器交叉注意力头数
    dropout: float = 0.1                 # Dropout 率
    num_queries: int = 4               # 查询向量数量

    # Directory Paths
    data_root_dir: Path = Path("/data/home/jlchen/datasets")
    run_root_dir: Path = Path("vla_log")                               # Path to directory to store logs & checkpoints

    # Resume (logging) Parameters -- 仅用于日志标记，不再用于模型权重加载
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 1                                      # Interval for saving checkpoints (in steps)
    image_aug: bool = True                                          # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Optional[str] = None

    # Tracking Parameters
    wandb_project: str = "test-project"                   # Name of W&B project to log to (use default!)
    # wandb_entity: str = "opendrivelab"                              # Name of entity to log under



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

    cfg.run_id += '-Latent-Action-Pretraining'
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    hf_token = str(cfg.hf_token) if isinstance(cfg.hf_token, Path) else cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)


 
    # 直接通过 HF ID/Path 加载 InternVL 模型与处理器
    overwatch.info(f"🔄 加载基础 InternVL `{cfg.model_id}`（HF from_pretrained）")
    vlm = InternVLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        token=hf_token,
        cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir is not None else None,
        trust_remote_code=True,
        device_map="cpu",  # 避免多进程默认加载到 cuda:0；后续由 Accelerate 迁移到各自 GPU
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        token=hf_token,
        cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir is not None else None,
        trust_remote_code=True,
    )
    hf_tokenizer = cast(PreTrainedTokenizerBase, processor.tokenizer)  # type: ignore[attr-defined]
    # 附加 processor 以便回调保存
    setattr(vlm, "processor", processor)

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype in (torch.float32, torch.bfloat16), f"Loaded VLM parameter has unexpected dtype: {param.dtype}"

    # 直接按配置冻结模块（若可用）；HF-only InternVL 组件名：vision_tower / language_model / multi_modal_projector / lm_head
    if cfg.freeze_vision_backbone and hasattr(vlm, "vision_tower"):
        vlm.vision_tower.requires_grad_(False)
    if cfg.freeze_projector and hasattr(vlm, "multi_modal_projector"):
        vlm.multi_modal_projector.requires_grad_(False)
    if cfg.freeze_llm_backbone and hasattr(vlm, "language_model"):
        vlm.language_model.requires_grad_(False)
    if cfg.freeze_last_llm_layer and hasattr(vlm, "lm_head"):
        vlm.lm_head.requires_grad_(False)
   
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
    latent_action_model = LatentLAMModel(
            dim=cfg.dim,
            enc_layers=cfg.enc_layers,
            codebook_size=cfg.codebook_size,
            code_dim=cfg.code_dim,
            dec_layers=cfg.dec_layers,
            dec_self_heads=cfg.dec_self_heads,
            dec_cross_heads=cfg.dec_cross_heads,
            dropout=cfg.dropout,
            num_queries=cfg.num_queries,
    )

    lam_ckpt = torch.load(cfg.lam_path, map_location="cpu")['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()
    overwatch.info(
        f"🔄 构建 RLDS 数据集与 Collator（mixture=`{cfg.data_mix}`，image_res={cfg.image_resolution}）"
    )
    # 类型提示规避：latent_action_tokenizer 需要 VQ 编码器，这里用 cast 静态规避
    vla_dataset, tokenizer, collator = get_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.data_mix,
        latent_action_model,
        tokenizer=hf_tokenizer,
        default_image_resolution=cfg.image_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    overwatch.info("🔧 扩充 LLM 词表以注入动作离散 token")
    special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    try:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
    except Exception:
        num_added_toks = 0


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
        save_dataset_statistics(cast(RLDSDataset, vla_dataset).dataset_statistics, run_dir)

    # 使用 Accelerate + FSDP 的新训练器（直接传入 dataclass -> dict）
    overwatch.info("🚀 启动 VLA 训练循环（Accelerate+FSDP）；首次 step 可能较慢（初始化 FSDP/AMP）")

    run_latent_action_training(
        cfg=cfg,
        vlm=vlm,
        vla_dataset=cast(RLDSDataset, vla_dataset),
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