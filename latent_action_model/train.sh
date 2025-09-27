#!/bin/bash

# VJEPA_LAM 分布式训练脚本
# 使用 torchrun 进行多 GPU 训练

# 设置环境变量
# 计算仓库根目录（本脚本位于 <repo>/latent_action_model/train.sh）
REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"
export OMP_NUM_THREADS=12
export TF_CPP_MIN_LOG_LEVEL=3
# HuggingFace 缓存目录配置
export HF_HOME="/mnt/public_zgc/home/jlchen/.cache/huggingface"
export HF_HUB_CACHE="/mnt/public_zgc/home/jlchen/.cache/huggingface/models"
export HF_DATASETS_CACHE="/mnt/public_zgc/home/jlchen/.cache/huggingface/datasets"

# PyTorch Hub 缓存目录配置
export TORCH_HOME="/mnt/public_zgc/home/jlchen/.cache/torch"

# 训练参数
# CONFIG_FILE="config/lam-vjepa_large.yaml"
CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/lam-vjepa.yaml"
LOG_DIR="latent_action_model/logs/train_logs"
TIMESTAMP="$(date +%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vjepa_lam_${TIMESTAMP}.log"

# 可选：从检查点恢复（两种方式）
CKPT_PATH="${CKPT_PATH:-}"

# W&B 配置
export WANDB_API_KEY="8d44fb58134f3f96e048d943a2543c51ff4f1d09"
export WANDB_DIR="${REPO_ROOT_DIR}/latent_action_model"

# export WANDB_MODE="offline"

echo "🚀 美好的事情发生了！！！"
echo "🚀 开始 VJEPA_LAM 训练..."
echo "📋 配置文件: ${CONFIG_FILE}"
echo "📝 日志文件: ${LOG_FILE}"
echo ""
echo "📁 缓存目录配置:"
echo "💾 HF 缓存目录: ${HF_HOME}"
echo "🤗 模型缓存: ${HF_HUB_CACHE}"
echo "📊 数据集缓存: ${HF_DATASETS_CACHE}"
echo "🔥 PyTorch 缓存: ${TORCH_HOME}"
echo "📴 W&B 模式: ${WANDB_MODE}"
echo "🗂️ W&B 目录: ${WANDB_DIR}/wandb"
if [[ -n "${CKPT_PATH}" ]]; then
    echo "🔁 从检查点恢复: ${CKPT_PATH}"
fi

# 确保日志目录存在
mkdir -p "${LOG_DIR}"
mkdir -p "${WANDB_DIR}/wandb"

# 自动获取 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # fallback: 使用 torch 获取 GPU 数量
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "🖥️ 检测到 GPU 数量: ${NUM_GPUS}"

# 启动训练（以模块方式运行，避免相对导入问题）
torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} -m latent_action_model.main fit \
    --config ${CONFIG_FILE} \
    ${CKPT_PATH:+--ckpt_path ${CKPT_PATH}} \
    "$@" \
    2>&1 | tee ${LOG_FILE}
