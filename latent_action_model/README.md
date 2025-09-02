# VJEPA_LAM 模型训练指南

本项目实现了基于 V-JEPA2 的 Latent Action Model (VJEPA_LAM)，使用 Lightning CLI 框架进行训练。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

编辑配置文件 `latent_action_model/config/lam-vjepa.yaml`，修改数据路径：

```yaml
data:
  data_root: /path/to/your/rlds_data_collection  # 修改为你的数据路径
  data_mix: omni_magic_soup_plus_plus
  batch_size: 64
  resolution: 224
  num_frames: 16
```

### 3. 开始训练

#### 方法一：使用训练脚本（推荐）

```bash
# 进入项目目录
cd latent_action_model

# 运行分布式训练
./train.sh
```

#### 方法二：直接使用 torchrun

```bash
# 单节点多 GPU 训练
torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config config/lam-vjepa.yaml \
    2>&1 | tee vjepa_lam.log
```

#### 方法三：使用 Lightning CLI

```bash
# 单 GPU 训练
python main.py fit --config config/lam-vjepa.yaml

# 多 GPU 训练（Lightning 自动处理）
python main.py fit --config config/lam-vjepa.yaml --trainer.devices 8
```

## 模型架构

VJEPA_LAM 模型包含以下组件：

1. **V-JEPA2 视觉编码器**: 预训练的视觉特征提取器
2. **LAM 编码器**: 将视觉特征编码为潜在表示
3. **向量量化 (VQ)**: 将连续特征离散化为码本索引
4. **LAM 解码器**: 从潜在表示重建视觉特征

## 配置文件说明

### 模型参数

```yaml
model:
  dim: 1024                    # 特征维度 (V-JEPA2 ViT Large 输出维度)
  num_layers: 4                # LAM 编码器层数
  codebook_size: 16            # 码本大小
  code_dim: 128                # 码本维度
  dec_layers: 4                # 解码器层数
  dec_self_heads: 4            # 解码器自注意力头数
  dec_cross_heads: 4           # 解码器交叉注意力头数
  dropout: 0.1                 # Dropout 率
  num_queries: 4               # 查询向量数量
  lr: 1e-4                     # 学习率
  vq_beta: 0.25               # VQ 损失权重
```

### 数据参数

```yaml
data:
  data_root: /path/to/data     # 数据根目录
  data_mix: omni_magic_soup_plus_plus  # 数据集混合
  batch_size: 64               # 批次大小
  resolution: 224              # 图像分辨率
  num_frames: 16               # 视频帧数
  episodic: false              # 是否使用情节数据
  shuffle_buffer_size: 45000   # 随机缓冲区大小
  image_aug: true              # 是否使用图像增强
```

### 训练器参数

```yaml
trainer:
  max_epochs: 20               # 最大训练轮数
  accelerator: gpu             # 加速器类型
  devices: 8                   # GPU 设备数量
  strategy: ddp_find_unused_parameters_false  # 分布式策略
  precision: 16-mixed          # 混合精度训练
```

## 分布式训练

### torchrun 命令详解

```bash
torchrun \
    --standalone \                    # 单节点模式
    --nnodes 1 \                     # 节点数量
    --nproc-per-node 8 \            # 每个节点的 GPU 数量
    main.py fit \                    # Lightning CLI 命令
    --config config/lam-vjepa.yaml \ # 配置文件
    2>&1 | tee vjepa_lam.log        # 日志输出
```

### 多节点训练

```bash
# 节点 0
torchrun \
    --nnodes 2 \
    --nproc-per-node 8 \
    --node-rank 0 \
    --master-addr "192.168.1.100" \
    --master-port 29500 \
    main.py fit \
    --config config/lam-vjepa.yaml

# 节点 1
torchrun \
    --nnodes 2 \
    --nproc-per-node 8 \
    --node-rank 1 \
    --master-addr "192.168.1.100" \
    --master-port 29500 \
    main.py fit \
    --config config/lam-vjepa.yaml
```

## 常用命令

### 训练

```bash
# 基本训练
python main.py fit --config config/lam-vjepa.yaml

# 指定 GPU 数量
python main.py fit --config config/lam-vjepa.yaml --trainer.devices 4

# 调试模式
python main.py fit --config config/lam-vjepa.yaml --trainer.fast_dev_run true
```

### 测试

```bash
# 测试模型
python main.py test --config config/lam-vjepa.yaml --ckpt_path logs/vjepa_lam/last.ckpt
```

### 验证

```bash
# 验证模型
python main.py validate --config config/lam-vjepa.yaml --ckpt_path logs/vjepa_lam/last.ckpt
```

## 模型检查点

训练过程中会自动保存检查点：

- `logs/vjepa_lam/last.ckpt`: 最新的检查点
- `logs/vjepa_lam/epoch_X-step_Y.ckpt`: 特定步骤的检查点

## 日志和监控

训练日志保存在以下位置：

- **TensorBoard**: `logs/vjepa_lam/`
- **WandB**: 自动上传到 Weights & Biases（如果配置了）

查看训练进度：

```bash
tensorboard --logdir logs/vjepa_lam/
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少 `batch_size`
   - 减少 `num_frames`
   - 使用梯度累积

2. **数据加载错误**
   - 检查 `data_root` 路径是否正确
   - 确保数据格式符合要求

3. **V-JEPA2 模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间

4. **分布式训练问题**
   - 确保所有 GPU 可用
   - 检查网络连接（多节点）
   - 确保端口未被占用

### 性能优化

1. **多 GPU 训练**
   ```bash
   # 使用 torchrun
   torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit --config config/lam-vjepa.yaml
   
   # 或使用 Lightning CLI
   python main.py fit --config config/lam-vjepa.yaml --trainer.devices 8
   ```

2. **混合精度训练**
   ```yaml
   trainer:
     precision: 16-mixed
   ```

3. **梯度累积**
   ```yaml
   trainer:
     accumulate_grad_batches: 2
   ```

## 自定义配置

你可以创建自己的配置文件来调整模型参数：

```bash
cp latent_action_model/config/lam-vjepa.yaml my_config.yaml
# 编辑 my_config.yaml
python main.py fit --config my_config.yaml
```

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！ 