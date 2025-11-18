from typing import Dict, Tuple, Optional, Callable, Iterable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from lightning import LightningModule
import lightning.pytorch as pl
# 定义优化器回调类型
OptimizerCallable = Callable[[Iterable], Optimizer]
from accelerate import PartialState
import wandb
# 导入 core 中的模型组件
from .lam_model import LatentLAMModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import os
import shutil
from .utils.utils import eef_reconstruction_loss
import importlib


class VJEPA_LAM(LightningModule):
    """
    V-JEPA2 版本的 Latent Action Model，适配 Lightning 框架
    
    基于 core/ 中的 LatentLAMModel 和 VJEPAEncoder，
    但采用 Lightning 的训练接口
    """
    
    def __init__(
        self,
        # 模型架构参数
        dim: int = 1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 2,
        enc_layers: int = 4,
        codebook_size: int = 16,
        code_dim: int = 128,
        num_frames: int = 5,
        ar_prediction: bool = False,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 4,
        dropout: float = 0.1,
        # 物理接地参数
        lambda_aux: float = 0.2,  # 辅助损失的总体权重
        loss_type: str = "l1",
        # 训练参数
        project: str = 'UniVLA-latent_action_model',
        task_name: str = 'vjepa_lam',
        wandb_offline: bool = False,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        weight_decay: float = 0.01,
        # 索引保存参数
        make_data_pair: bool = False,
        output_dir: str = "output_pairs",
        vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        # 学习率调度与预热
        warmup_steps: int = 0,
        lambda_diversity: float = 0.1,
        norm_latents: bool = False,
        disable_vq: bool = False,
        vq_type: str = "nsvq",
        **kwargs
    ):
        super().__init__()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('medium')

        # 保存超参数
        self.save_hyperparameters()
        
        # 初始化 LAM 模型
        self.lam = LatentLAMModel(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            enc_layers=enc_layers,
            codebook_size=codebook_size,
            code_dim=code_dim,
            num_frames=num_frames,
            ar_prediction=ar_prediction,
            dec_layers=dec_layers,
            dropout=dropout,
            vision_model_id=vision_model_id,
            vq_kwargs=vq_kwargs,
            norm_latents=norm_latents,
            disable_vq=disable_vq,
            vq_type=vq_type
        )
        
        # 训练参数
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.codebook_size = codebook_size
        self.warmup_steps = int(warmup_steps)
        
        # 物理接地参数
        self.lambda_aux = lambda_aux
        self.lambda_diversity = lambda_diversity

        # 索引保存参数
        self.make_data_pair = make_data_pair
        self.output_dir = output_dir
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(project=project, name=task_name, reinit=True, mode="offline" if wandb_offline else "online")

        self.loss_type = loss_type
    def shared_step(self, batch: Dict) -> Tuple[Tensor, Dict]:
        """共享的训练/验证步骤（训练分支）。"""
        return self._compute_step(batch=batch, vq_training=True)
    
    def _detect_robot_data(self, states: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
        """
        自动检测哪些样本包含有效的机械臂状态数据
        
        通过判断状态数据是否为零来区分机械臂数据和人类数据。
        人类数据的状态会在数据处理阶段被填充为零。
        
        Args:
            states: [B, T, state_dim] 状态张量
            threshold: 判断状态是否为零的阈值
            
        Returns:
            torch.Tensor: 包含机械臂数据的样本索引
        """
        # 计算每个样本在所有时间步和状态维度上的绝对值之和
        state_magnitudes = torch.sum(torch.abs(states), dim=(1, 2))  # [B]
        
        # 找到状态幅度大于阈值的样本（即非零填充的机械臂数据）
        robot_indices = (state_magnitudes > threshold).nonzero(as_tuple=True)[0]
        
        return robot_indices
    
    def shared_inference_step(self, batch: Dict) -> Tuple[Tensor, Dict]:
        """共享的推理步骤（验证/测试分支）。"""
        return self._compute_step(batch=batch, vq_training=False)

    def _compute_step(self, batch: Dict, vq_training: bool) -> Tuple[Tensor, Dict]:
        """
        统一的计算路径，仅在 VQ 调用上区分训练/推理。
        Args:
            batch: 输入 batch，需包含 'videos'，可选包含 'proprio'
            vq_training: True 使用训练 VQ；False 使用推理 VQ
        Returns:
            (loss, logs)
        """
        videos = batch["videos"]
        states = batch["proprio"]
        dec_videos = batch["dec_videos"]
        # print("videos shape:", videos.shape)
        # VQ 路径区分在模型内部（视觉编码也已迁移到 LAM 内部）
        if vq_training:
            recon, dec_in, tgt, perplexity, indices, delta_s_pred, features, _, entropy_loss, vq_loss = self.lam(videos, states, dec_videos)
        else:
            recon, dec_in, tgt, perplexity, indices, delta_s_pred, features, _, entropy_loss, vq_loss = self.lam.inference(videos, states, dec_videos)

        target = tgt
        # recon_loss = F.mse_loss(recon, target)
        if self.loss_type == "l1":
            recon_loss = F.l1_loss(recon, target)
            loss = recon_loss
        elif self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(recon, target, dim=-1).mean()
            recon_loss = F.l1_loss(recon, target)
            loss = recon_loss + (1 - cos_sim)
        elif self.loss_type == "delta":
            recon_loss = F.l1_loss(recon, target-dec_in)
            loss = recon_loss
        else:
            recon_loss = F.mse_loss(recon, target)
            loss = recon_loss
        entropy_loss = self.lambda_diversity * entropy_loss
        total_loss = loss + entropy_loss + vq_loss
        # total_loss = loss
        aux_loss = torch.tensor(0.0, device=self.device)
        aux_loss_logs: Dict[str, Tensor] = {}

        if "proprio" in batch and delta_s_pred is not None:
            states = batch["proprio"]
            robot_indices = self._detect_robot_data(states)
            if len(robot_indices) > 0:
                delta_s_pred_robot = delta_s_pred[robot_indices]
                states_robot = states[robot_indices]
                state_loss = eef_reconstruction_loss(states_robot, delta_s_pred_robot)
                aux_loss = self.lambda_aux * state_loss
                aux_loss_logs["state_loss"] = aux_loss.item()
                # aux_loss_logs["robot_data_count"] = torch.tensor(len(robot_indices), device=self.device)
                total_loss = total_loss + aux_loss

            logs: Dict[str, Tensor] = {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "perplexity": perplexity,
                **aux_loss_logs,
            }
            if self.loss_type == "cosine":
                logs["cos_sim"] = cos_sim
            if self.lambda_diversity >0:
                logs["entropy_loss"] = entropy_loss
        return total_loss, logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """训练步骤"""
        loss, aux_losses = self.shared_step(batch)
        
        # 记录训练损失 - Lightning 会自动将数据发送给配置的 WandbLogger
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        if self.distributed_state.is_main_process:
            # 将 tensors 转换为 Python 标量用于 wandb
            wandb_logs = {"train_loss": loss.item()}
            for k, v in aux_losses.items():
                if isinstance(v, torch.Tensor):
                    wandb_logs[f"train/{k}"] = v.item()
                else:
                    wandb_logs[f"train/{k}"] = v
            wandb.log(wandb_logs, step=self.global_step)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """验证步骤 - 采用推理模式，避免对数据管线新增依赖"""
        loss, aux_losses = self.shared_inference_step(batch)
        # 记录验证损失（建议仅在 epoch 级聚合，减少日志量）
        self.log_dict(
            {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self.distributed_state.is_main_process:
            # 将 tensors 转换为 Python 标量用于 wandb
            wandb_logs = {"val_loss": loss.item()}
            for k, v in aux_losses.items():
                if isinstance(v, torch.Tensor):
                    wandb_logs[f"val/{k}"] = v.item()
                else:
                    wandb_logs[f"val/{k}"] = v
            wandb.log(wandb_logs, step=self.global_step)
        return loss
    
    def on_after_backward(self) -> None:
        if not getattr(self.trainer, "is_global_zero", True):
            return
        unused = []
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is None:
                unused.append(name)
        if unused:
            self.print(f"UNUSED params ({len(unused)}): " + ", ".join(unused))
    
    # def on_train_epoch_end(self):
    #     """训练 epoch 结束时的回调"""
    #     # 在分布式场景下，仅在 rank0 执行维护逻辑，并将更新后的状态同步到所有 rank
    #     is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    #     if is_distributed:
    #         self.trainer.strategy.barrier()

    #     if getattr(self.trainer, "is_global_zero", True):
    #         with torch.no_grad():
    #             if hasattr(self.lam.vq, 'replace_unused_codebooks'):
    #                 self.lam.vq.replace_unused_codebooks()
    #             if hasattr(self.lam.vq, 'reset_node_count'):
    #                 self.lam.vq.reset_node_count()

    #     if is_distributed:
    #         state_list = [self.lam.vq.state_dict()] if getattr(self.trainer, "is_global_zero", True) else [None]
    #         torch.distributed.broadcast_object_list(state_list, src=0)
    #         with torch.no_grad():
    #             self.lam.vq.load_state_dict(state_list[0])
    #         self.trainer.strategy.barrier()

    def on_test_epoch_end(self):
        """测试 epoch 结束时的回调 - 保存索引和可视化"""
        if self.make_data_pair:
            # 创建输出目录
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 获取使用频率最高的码本索引
            if hasattr(self.lam.vq, 'node_count'):
                usage = self.lam.vq.node_count
                top_indices = torch.topk(usage, min(16, self.codebook_size), largest=True, sorted=True).indices
                
                # 保存 top latents
                top_latents = self.lam.vq.codebooks[top_indices]
                torch.save(top_latents, f"{self.output_dir}/top_16.pt")
                
                # 保存索引列表
                with open(f"{self.output_dir}/top_16.txt", "w") as f:
                    f.write(" ".join([str(i.item()) for i in top_indices]))
        
        # 绘制使用分布图
        if hasattr(self.lam.vq, 'node_count'):
            self.plot_usage_distribution(self.lam.vq.node_count, "unsorted_usage")
            sorted_usage, _ = torch.sort(self.lam.vq.node_count)
            self.plot_usage_distribution(sorted_usage, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        """绘制码本使用分布图"""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullLocator
        import numpy as np
        
        data = usage.cpu().numpy()
        
        # 计算合适的网格大小
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        
        # 重塑数据为方形矩阵
        data = data.reshape(2 ** n, -1)
        
        # 创建热力图
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Any:
        """配置优化器与可选的线性预热调度器。

        当 ``self.warmup_steps > 0`` 时，使用 ``LambdaLR`` 在前 ``warmup_steps`` 个
        优化步内将学习率从 0 线性提升到基础学习率，之后保持常数学习率。
        以 step 为粒度进行调度。
        """
        optim = self.optimizer(self.parameters())
        # optim = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()))
        if self.warmup_steps > 0:
            def lr_lambda(current_step: int) -> float:
                # 线性预热：从 0 -> 1.0
                if current_step < self.warmup_steps:
                    return float(current_step + 1) / float(self.warmup_steps)
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optim


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """测试步骤 - 使用推理模式"""
        loss, aux_losses = self.shared_inference_step(batch)
        
        # 记录测试损失
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        
        return loss
        

class CodebookMaintenanceCallback(pl.Callback):
    def __init__(self, interval_steps: int = 1000):
        super().__init__()
        self.interval_steps = interval_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step% self.interval_steps != 0:
            return

        # 额外检查：确保 kmeans 初始化已完成
        # replace_unused_codebooks 内部也会检查，但这里提前检查可以避免不必要的同步
        if hasattr(pl_module.lam.vq, 'initialized'):
            if not bool(pl_module.lam.vq.initialized.item()):
                return

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        # 使用 torch.distributed.barrier() 确保跨节点同步（支持多节点训练）
        if is_distributed:
            torch.distributed.barrier()

        # 重要：所有 rank 都需要调用，以便内部 all_reduce 能够正确聚合
        with torch.no_grad():
            if hasattr(pl_module.lam.vq, 'replace_unused_codebooks'):
                pl_module.lam.vq.replace_unused_codebooks()
            # 计数器在各 rank 本地清零，避免后续累计偏差
            if hasattr(pl_module.lam.vq, 'reset_node_count'):
                pl_module.lam.vq.reset_node_count()

        # 使用 torch.distributed.barrier() 确保跨节点同步（支持多节点训练）
        if is_distributed:
            torch.distributed.barrier()


class SaveConfigToCheckpointCallback(pl.Callback):
    def __init__(self, config_path: str = "", filename: str = "lam-vjepa.yaml"):
        super().__init__()
        # 若未显式传入：优先使用环境变量 LAM_CONFIG_PATH，其次回落到默认包内配置
        origin = "default"
        if not config_path:
            env_config_path = os.environ.get("LAM_CONFIG_PATH", "")
            if env_config_path:
                self.config_path = env_config_path
                origin = "env"
            else:
                from pathlib import Path
                base_dir = Path(__file__).resolve().parents[1]
                self.config_path = str(base_dir / "config" / "lam-vjepa.yaml")
                origin = "default"
        else:
            self.config_path = config_path
            origin = "arg"

        # 目标文件名：若来源为 env/arg 且未显式自定义文件名，则使用源配置名
        self.filename = filename
        if (not self.filename or self.filename == "lam-vjepa.yaml") and origin in ("env", "arg"):
            try:
                self.filename = os.path.basename(self.config_path)
            except Exception:
                self.filename = filename or "lam-vjepa.yaml"

    def _resolve_log_dir(self, trainer) -> Optional[str]:
        logger_obj = trainer.logger
        if logger_obj is None:
            return None
        if hasattr(logger_obj, 'log_dir') and logger_obj.log_dir is not None:
            return logger_obj.log_dir
        save_dir = getattr(logger_obj, 'save_dir', None)
        name = getattr(logger_obj, 'name', None)
        version = getattr(logger_obj, 'version', None)
        parts = [p for p in [save_dir, name, f"version_{version}" if version is not None else None] if p]
        if parts:
            return os.path.join(*parts)
        return None

    def on_fit_start(self, trainer, pl_module):
        # 基于 logger 管理的目录保存，不依赖 ModelCheckpoint.dirpath
        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            pl_module.print("无法解析 logger 保存目录，跳过保存配置文件")
            return

        ckpt_dir = log_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            dst_path = os.path.join(ckpt_dir, self.filename)
            shutil.copyfile(self.config_path, dst_path)
            pl_module.print(f"Saved config to {dst_path}")
        except Exception as e:
            pl_module.print(f"Failed to save config: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        # 在每个训练 epoch 结束时，将外部 shell 日志复制到与 checkpoints 同级的日志目录
        # 这会覆写之前的日志文件，确保始终保存最新的完整训练记录
        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            return
        self._copy_external_log(log_dir, pl_module)

    def _copy_external_log(self, log_dir: str, pl_module) -> None:
        src_log = os.environ.get("LAM_TRAIN_LOG_FILE", "")
        if not src_log:
            return
        try:
            if os.path.isfile(src_log):
                dst_log = os.path.join(log_dir, "train.log")
                shutil.copyfile(src_log, dst_log)
                pl_module.print(f"Copied training log to {dst_log}")
        except Exception:
            pass

    def on_exception(self, trainer, pl_module, exception):
        # 发生异常（包括 Ctrl-C）时，也复制训练日志
        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            return
        self._copy_external_log(log_dir, pl_module)