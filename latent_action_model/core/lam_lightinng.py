from typing import Dict, List, Tuple, Optional, Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from lightning import LightningModule

# 定义优化器回调类型
OptimizerCallable = Callable[[Iterable], Optimizer]
from accelerate import PartialState
import wandb
# 导入 core 中的模型组件
from .lam_model import LatentLAMModel, PhysicalGroundingLoss
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

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
        enc_layers: int = 4,
        codebook_size: int = 16,
        code_dim: int = 128,
        dec_layers: int = 4,
        dec_self_heads: int = 4,
        dec_cross_heads: int = 4,
        dropout: float = 0.1,
        num_queries: int = 4,
        # 物理接地参数
        enable_state_delta_prediction: bool = True,
        lambda_aux: float = 0.2,  # 辅助损失的总体权重
        lambda_dir: float = 1.0,  # 方向损失权重
        lambda_mag_reg: float = 0.1,  # 幅度正则化权重
        motion_threshold_beta: float = 0.01,  # 运动激活阈值 (1cm)
        motion_scale_alpha: float = 100.0,  # Sigmoid斜率
        # 训练参数
        task_name: str = 'vjepa_lam',
        optimizer: OptimizerCallable = torch.optim.AdamW,
        weight_decay: float = 0.01,
        # 索引保存参数
        make_data_pair: bool = False,
        output_dir: str = "output_pairs",
        **kwargs
    ):
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 初始化 LAM 模型
        self.lam = LatentLAMModel(
            dim=dim,
            enc_layers=enc_layers,
            codebook_size=codebook_size,
            code_dim=code_dim,
            dec_layers=dec_layers,
            dec_self_heads=dec_self_heads,
            dec_cross_heads=dec_cross_heads,
            dropout=dropout,
            num_queries=num_queries,
            enable_state_delta_prediction=enable_state_delta_prediction,
        )
        
        # 训练参数

        self.task_name = task_name
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.codebook_size = codebook_size
        
        # 物理接地参数
        self.enable_state_delta_prediction = enable_state_delta_prediction
        self.lambda_aux = lambda_aux
        
        # 初始化物理接地损失函数（如果启用）
        if enable_state_delta_prediction:
            self.physical_grounding_loss = PhysicalGroundingLoss(
                lambda_dir=lambda_dir,
                lambda_mag_reg=lambda_mag_reg,
                motion_threshold_beta=motion_threshold_beta,
                motion_scale_alpha=motion_scale_alpha
            )
        
        # 索引保存参数
        self.make_data_pair = make_data_pair
        self.output_dir = output_dir
        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)
    
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
        # VQ 路径区分在模型内部（视觉编码也已迁移到 LAM 内部）
        if vq_training:
            recon, perplexity, indices, delta_s_pred, features = self.lam(videos)
        else:
            recon, perplexity, indices, delta_s_pred, features = self.lam.inference(videos)

        target = features[:, 1]
        # recon_loss = F.mse_loss(recon, target)
        recon_loss = F.l1_loss(recon, target)
        total_loss = recon_loss

        aux_loss = torch.tensor(0.0, device=self.device)
        aux_loss_logs: Dict[str, Tensor] = {}

        if self.enable_state_delta_prediction and "proprio" in batch and delta_s_pred is not None:
            states = batch["proprio"]
            robot_indices = self._detect_robot_data(states)
            if len(robot_indices) > 0:
                delta_s_pred_robot = delta_s_pred[robot_indices]
                states_robot = states[robot_indices]
                s_t_robot = states_robot[:, 0, :3]
                s_t_plus_1_robot = states_robot[:, 1, :3]
                delta_s_gt = s_t_plus_1_robot - s_t_robot
                grounding_loss_dict = self.physical_grounding_loss(delta_s_pred_robot, delta_s_gt)
                aux_loss = self.lambda_aux * grounding_loss_dict["total_loss"]
                aux_loss_logs = {f"aux/{k}": v for k, v in grounding_loss_dict.items()}
                aux_loss_logs["aux/robot_data_count"] = torch.tensor(len(robot_indices), device=self.device)
                total_loss = recon_loss + aux_loss

        with torch.no_grad():
            unique, counts = torch.unique(indices, return_counts=True)
            index_counts = torch.zeros(self.codebook_size, dtype=torch.long, device=indices.device)
            index_counts[unique] = counts
            code_usage = (index_counts != 0).float().mean()

        logs: Dict[str, Tensor] = {
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "perplexity": perplexity,
            "code_usage": code_usage,
            **aux_loss_logs,
        }

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
    

    
    def on_train_epoch_end(self):
        """训练 epoch 结束时的回调"""
        # 1. 先替换未使用的码本条目（基于当前的使用统计）
        if hasattr(self.lam.vq, 'replace_unused_codebooks'):
            # 这里需要传入累计的批次数，可以根据实际情况调整
            self.lam.vq.replace_unused_codebooks()
        
        # 2. 然后重置码本使用统计（为下一个 epoch 做准备）
        if hasattr(self.lam.vq, 'reset_node_count'):
            self.lam.vq.reset_node_count()

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

    def configure_optimizers(self) -> Optimizer:
        """配置优化器"""
        optim = self.optimizer(self.parameters())
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
        
