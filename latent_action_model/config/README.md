# LAM 模型配置文件说明

本目录包含两种不同 LAM 模型的配置文件：

## 1. lam-stage-1.yaml (原始 LAM 模型)

适用于原始的 Latent Action Model，使用自定义的视觉编码器。

### 主要参数：
- `lam_model_dim`: 768 - 模型维度
- `lam_latent_dim`: 128 - 潜在空间维度  
- `lam_num_latents`: 16 - 潜在向量数量
- `lam_patch_size`: 14 - 图像补丁大小
- `lam_enc_blocks`: 12 - 编码器块数
- `lam_dec_blocks`: 12 - 解码器块数
- `lam_num_heads`: 12 - 注意力头数

### 使用场景：
- 使用自定义视觉编码器
- 需要更细粒度的模型控制
- 适合研究性质的实验

## 2. lam-vjepa.yaml (VJEPA_LAM 模型)

适用于基于 V-JEPA2 的 Latent Action Model，使用 Lightning CLI 框架。

### 主要参数：
- `dim`: 1024 - 特征维度 (V-JEPA2 ViT Large 输出维度)
- `num_layers`: 4 - LAM 编码器层数
- `codebook_size`: 16 - 码本大小
- `code_dim`: 128 - 码本维度
- `dec_layers`: 4 - 解码器层数
- `dec_self_heads`: 4 - 解码器自注意力头数
- `dec_cross_heads`: 4 - 解码器交叉注意力头数
- `dropout`: 0.1 - Dropout 率
- `num_queries`: 4 - 查询向量数量

### 使用场景：
- 使用 V-JEPA2 预训练视觉编码器
- 基于 Lightning CLI 框架的训练
- 适合生产环境的训练

## 主要区别

| 特性 | lam-stage-1.yaml | lam-vjepa.yaml |
|------|------------------|----------------|
| 视觉编码器 | 自定义编码器 | V-JEPA2 预训练编码器 |
| 训练框架 | 自定义训练循环 | Lightning CLI 框架 |
| 模型复杂度 | 较高 (12层编码器/解码器) | 中等 (4层编码器/解码器) |
| 参数量 | 较大 | 中等 |
| 训练稳定性 | 需要更多调优 | 更稳定 |
| 预训练特征 | 从头训练 | 利用 V-JEPA2 预训练特征 |

## 使用方法

### 使用原始 LAM 模型：
```bash
python main.py fit --config latent_action_model/config/lam-stage-1.yaml
```

### 使用 VJEPA_LAM 模型：
```bash
python main.py fit --config latent_action_model/config/lam-vjepa.yaml
```

### Lightning CLI 常用命令：

1. **训练模型**：
```bash
python main.py fit --config config/lam-vjepa.yaml
```

2. **测试模型**：
```bash
python main.py test --config config/lam-vjepa.yaml --ckpt_path logs/vjepa_lam/last.ckpt
```

3. **验证模型**：
```bash
python main.py validate --config config/lam-vjepa.yaml --ckpt_path logs/vjepa_lam/last.ckpt
```

4. **预测**：
```bash
python main.py predict --config config/lam-vjepa.yaml --ckpt_path logs/vjepa_lam/last.ckpt
```

## 配置建议

1. **首次使用**：建议使用 `lam-vjepa.yaml`，因为：
   - V-JEPA2 预训练特征更稳定
   - Lightning CLI 提供更好的训练监控
   - 代码结构更清晰

2. **研究实验**：可以使用 `lam-stage-1.yaml`，因为：
   - 提供更多模型架构控制
   - 适合探索不同的编码器设计

3. **生产部署**：推荐使用 `lam-vjepa.yaml`，因为：
   - 训练更稳定
   - 代码维护性更好
   - 与现有生态系统集成更好

## 配置文件结构

### VJEPA_LAM 配置文件结构：
```yaml
# 模型参数
model:
  dim: 1024                    # V-JEPA2 ViT Large 输出维度
  num_layers: 4                # LAM 编码器层数
  codebook_size: 16            # 码本大小
  code_dim: 128                # 码本维度
  # ... 其他模型参数

# 数据参数  
data:
  data_root: /path/to/data
  batch_size: 64
  # ... 其他数据参数

# 训练器参数
trainer:
  max_epochs: 20
  accelerator: gpu
  # ... 其他训练器参数
```

这种结构完全适配 Lightning CLI 的期望格式，可以直接使用。 