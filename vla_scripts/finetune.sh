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

# 自动获取 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # fallback: 使用 torch 获取 GPU 数量
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "🖥️ 检测到 GPU 数量: ${NUM_GPUS}"

# 以模块方式运行，避免相对导入问题
torchrun --nproc_per_node ${NUM_GPUS} -m vla_scripts.finetune_libero 
                                