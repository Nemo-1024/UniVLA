#!/bin/bash

# VJEPA_LAM 分布式训练脚本
# 使用 torchrun 进行多 GPU 训练

# torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
#     --config config/lam-stage-1.yaml \
#     2>&1 | tee lam-stage-1.log

# 设置环境变量

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=16
# 训练参数
CONFIG_FILE="config/lam-vjepa.yaml"
LOG_FILE="vjepa_lam.log"

# 可选：从检查点恢复（两种方式）
# 1) 设置环境变量 CKPT_PATH 指向 .ckpt 文件
# 2) 直接在脚本后透传 Lightning CLI 原生参数，如：--ckpt_path logs/vjepa_lam/epoch=14-step=20000.ckpt
CKPT_PATH="${CKPT_PATH:-}"



echo "🚀 开始 VJEPA_LAM 训练..."
echo "📋 配置文件: ${CONFIG_FILE}"
echo "📝 日志文件: ${LOG_FILE}"

if [[ -n "${CKPT_PATH}" ]]; then
echo "🔁 从检查点恢复: ${CKPT_PATH}"
fi

torchrun --standalone --nnodes 1 --nproc-per-node 2 main.py fit \
    --config ${CONFIG_FILE} \
    ${CKPT_PATH:+--ckpt_path ${CKPT_PATH}} \
    "$@" \
    2>&1 | tee ${LOG_FILE}
