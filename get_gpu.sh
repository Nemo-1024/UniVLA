#!/bin/bash

# ===================================================================================
#               交互式 GPU 会话申请脚本 (Slurm Wrapper)
#
# 功能:
#   此脚本用于通过 srun 快速申请一个交互式的 GPU 计算节点。
#   您可以在此节点上进行代码调试、环境配置或运行短期测试。
#
# 使用方法:
#   1. 根据您的需求，修改下面的 "====== 参数配置区域 ======" 中的变量。
#   2. 保存文件。
#   3. 在终端中给予脚本执行权限: chmod +x get_gpu.sh
#   4. 运行脚本: 
#      - 使用默认设置: ./get_gpu.sh
#      - 指定GPU数量: ./get_gpu.sh <GPU数量>
#      - 指定节点: ./get_gpu.sh -n <节点索引>
#      - 指定GPU数量和节点: ./get_gpu.sh <GPU数量> -n <节点索引>
#      - 查看帮助: ./get_gpu.sh -h
#   
#   节点索引对应关系:
#      0 = pm-9eea
#      1 = pm-9eea0001
#      2 = pm-9eea0002
#      3 = pm-9eea0003
#
#   使用示例:
#      ./get_gpu.sh           # 1个GPU，Slurm自动选择节点
#      ./get_gpu.sh 2         # 2个GPU，Slurm自动选择节点
#      ./get_gpu.sh -n 0      # 1个GPU，使用节点0 (pm-9eea)
#      ./get_gpu.sh 2 -n 2    # 2个GPU，使用节点2 (pm-9eea0002)
#
#   脚本执行后，会向 Slurm 提交请求。成功后，您的终端将直接连接到
#   计算节点上。工作完成后，输入 'exit' 即可退出并释放资源。
# ===================================================================================

# ============================ 参数配置区域 (请在此处修改) ============================

# -- 分区 (Partition) --
# 指定要提交到的计算节点分区。这通常取决于集群的配置，例如 'gpu' 'compute' 'debug' 等。
# 对应 srun 参数: -p
PARTITION="batch"

# -- 作业名称 (Job Name) --
# 为您的交互式作业设置一个名称，方便使用 squeue 命令查看。
# 对应 srun 参数: -J
JOB_NAME="comput_src"

# -- 时间限制 (Time Limit) --
# 您希望占用资源的最长时间。格式为 "小时:分钟:秒"。超过此时间后会话将自动终止。
# 建议根据实际需要设置，不要过长，以免长时间占用资源。
# 对应 srun 参数: -t
TIME="24:00:00"

# -- GPU 类型和数量 --
# 申请的 GPU 资源。格式为 "型号:数量"。
# 例如: "a800:1" (申请1块A800), "v100:2" (申请2块V100)。
# 如果对型号没有要求，在某些集群上可以只写数量，如 "gpu:1"。
# 对应 srun 参数: --gres
GPU_TYPE="a800"
# 默认GPU数量，可通过命令行参数覆盖
DEFAULT_GPU_COUNT=1

# -- CPU 核心数 --
# CPU核心数将根据GPU数量自动计算：8 * GPU数量，最少8个核心
# 对应 srun 参数: --cpus-per-task
# 注意：实际的CPUS值将在解析命令行参数后动态计算

# -- 内存大小 --
# 内存大小将根据GPU数量自动计算：128G * GPU数量，CPU-only节点也申请128G
# 对应 srun 参数: --mem
# 注意：实际的MEMORY值将在解析命令行参数后动态计算

# -- 可选计算节点列表 --
# 可以选择的计算节点
AVAILABLE_NODES=("pm-9eea" "pm-9eea0001" "pm-9eea0002" "pm-9eea0003")
# 节点索引对应关系：0=pm-9eea, 1=pm-9eea0001, 2=pm-9eea0002, 3=pm-9eea0003

# 移除原来的硬编码NODELIST
# NODELIST="pm-9eea0003"


# ============================ 脚本执行区域 (一般无需修改) ============================

# 函数：根据索引获取节点名
get_node_by_index() {
    local index=$1
    if [[ "$index" =~ ^[0-3]$ ]]; then
        echo "${AVAILABLE_NODES[$index]}"
        return 0
    else
        return 1
    fi
}

# 函数：验证节点是否有效
validate_node() {
    local node=$1
    for valid_node in "${AVAILABLE_NODES[@]}"; do
        if [[ "$node" == "$valid_node" ]]; then
            return 0
        fi
    done
    return 1
}

# 函数：显示帮助信息
show_usage() {
    echo "使用方法: $0 [GPU数量] [-n 节点索引]"
    echo ""
    echo "参数说明:"
    echo "  GPU数量      申请的GPU数量 (默认: 1)"
    echo "  -n 节点索引  指定节点 (0-3, 可选)"
    echo ""
    echo "节点索引对应关系:"
    echo "  0 = pm-9eea"
    echo "  1 = pm-9eea0001" 
    echo "  2 = pm-9eea0002"
    echo "  3 = pm-9eea0003"
    echo ""
    echo "使用示例:"
    echo "  $0                # 1个GPU，Slurm自动选择节点"
    echo "  $0 2              # 2个GPU，Slurm自动选择节点"
    echo "  $0 -n 0           # 1个GPU，使用节点0 (pm-9eea)"
    echo "  $0 2 -n 2         # 2个GPU，使用节点2 (pm-9eea0002)"
}

# 解析命令行参数
GPU_COUNT=$DEFAULT_GPU_COUNT
SELECTED_NODE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--node)
            if [[ -n "$2" ]] && get_node_by_index "$2" >/dev/null 2>&1; then
                SELECTED_NODE=$(get_node_by_index "$2")
                shift 2
            else
                echo "错误: -n 参数需要一个有效的节点索引 (0-3)"
                show_usage
                exit 1
            fi
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "错误: 未知参数 $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                GPU_COUNT=$1
                shift
            else
                echo "错误: GPU数量必须是正整数，得到: $1"
                show_usage
                exit 1
            fi
            ;;
    esac
done

# 根据GPU数量计算CPU核心数和内存大小
if [ "$GPU_COUNT" -eq 0 ]; then
    # CPU-only节点：8个核心，128G内存
    CPUS=16
    MEMORY="128G"
else
    # GPU节点：8 * GPU数量的核心，128G * GPU数量的内存
    CPUS=$((16 * GPU_COUNT))
    MEMORY="$((GPU_COUNT * 120))G"
fi

# 在终端打印出将要申请的资源信息，方便用户确认
echo "================================================="
echo "正在为您申请交互式 GPU 会话..."
echo "-------------------------------------------------"
echo "分区 (Partition): $PARTITION"
echo "作业名称 (Job Name):   $JOB_NAME"
echo "运行时长 (Time):     $TIME"
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "GPU 资源 (GRES):     无 (CPU-only节点)"
    echo "CPU 核心数 (CPUs):   $CPUS (最少16个核心)"
    echo "内存 (Memory):       $MEMORY (CPU-only节点固定128G)"
else
    echo "GPU 资源 (GRES):     $GPU_TYPE x $GPU_COUNT"
    echo "CPU 核心数 (CPUs):   $CPUS (16 x $GPU_COUNT GPU)"
    echo "内存 (Memory):       $MEMORY (128G x $GPU_COUNT GPU)"
fi
if [ -n "$SELECTED_NODE" ]; then
    # 找到节点对应的索引
    for i in "${!AVAILABLE_NODES[@]}"; do
        if [[ "${AVAILABLE_NODES[$i]}" == "$SELECTED_NODE" ]]; then
            echo "指定节点 (Node):     [$i] $SELECTED_NODE"
            break
        fi
    done
else
    echo "节点选择 (Node):     由Slurm自动选择"
fi
echo "================================================="
echo "请等待 Slurm 分配资源，这可能需要一些时间..."
echo ""

# 准备nodelist参数
if [ -n "$SELECTED_NODE" ]; then
    NODELIST_PARAM="--nodelist=$SELECTED_NODE"
else
    NODELIST_PARAM=""
fi

# 检查GPU数量是否为0
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "警告: GPU数量为0，将申请CPU-only节点"
    # 使用 srun 启动交互式作业 (无GPU)
    srun -p "$PARTITION" \
         -J "$JOB_NAME" \
         -N 1 \
         -t "$TIME" \
         --cpus-per-task="$CPUS" \
         --mem="$MEMORY" \
         $NODELIST_PARAM \
         --pty /bin/bash
else
    # 使用 srun 启动交互式作业 (有GPU)
    srun -p "$PARTITION" \
         -J "$JOB_NAME" \
         -N 1 \
         -t "$TIME" \
         --gres=gpu:"$GPU_TYPE":"$GPU_COUNT" \
         --cpus-per-task="$CPUS" \
         --mem="$MEMORY" \
         $NODELIST_PARAM \
         --pty /bin/bash
         
fi
echo "交互式会话已结束，资源已释放。"