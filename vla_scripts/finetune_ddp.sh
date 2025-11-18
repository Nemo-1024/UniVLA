#!/bin/bash

# --- 基本环境变量设置 ---
# 计算仓库根目录（本脚本位于 <repo>/vla_scripts/train.sh）
REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"

export OMP_NUM_THREADS=12
export TF_CPP_MIN_LOG_LEVEL=3

export WANDB_API_KEY="8d44fb58134f3f96e048d943a2543c51ff4f1d09"
export WANDB_MODE="offline"
# 将 W&B 输出目录设置为 vla_scripts 根目录（wandb 将在此自动创建单层 ./wandb）
export WANDB_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "📴 W&B 模式: ${WANDB_MODE}"

# --- 硬件和分布式环境检测 ---

# 1. 自动获取当前节点的 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # fallback: 使用 torch 获取 GPU 数量
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "🖥️  当前节点检测到 GPU 数量: ${NUM_GPUS}"

# 2. 从平台环境变量中获取多节点训练参数
#    如果环境变量不存在 (例如本地手动执行)，则使用默认值以兼容单节点训练。
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500} # 使用一个常用的默认端口

echo "🌐 分布式训练配置:"
echo "➡️  节点数量 (nnodes): ${NNODES}"
echo "🆔 当前节点排名 (node_rank): ${NODE_RANK}"
echo "🔗 主节点地址 (master_addr): ${MASTER_ADDR}"
echo "🔌 主节点端口 (master_port): ${MASTER_PORT}"

# --- 启动训练 ---
# 以模块方式运行，并传入所有分布式参数
torchrun --nproc_per_node ${NUM_GPUS} \
         --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         -m vla_scripts.finetune_libero