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



echo "🚀 开始 VJEPA_LAM 训练..."
echo "📋 配置文件: ${CONFIG_FILE}"
echo "📝 日志文件: ${LOG_FILE}"


torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config ${CONFIG_FILE} \
    2>&1 | tee ${LOG_FILE}
