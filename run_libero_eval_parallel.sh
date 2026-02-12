#!/bin/bash
# Parallel LIBERO evaluation script
# Each task suite runs in parallel on a different GPU
#
# Usage:
#   ./run_libero_eval_parallel.sh
#   ./run_libero_eval_parallel.sh --tasks libero_spatial,libero_goal
#   ./run_libero_eval_parallel.sh --gpu_list 0,1,2,3
#   ./run_libero_eval_parallel.sh --gpus 0,1,2,3
#
# Logs will be saved as log_<task_suite>.out

set -u
set -o pipefail

cleanup() {
    echo ""
    echo "🛑 Caught interrupt. Killing all child processes..."
    kill 0
}

trap cleanup INT TERM

########################################
# Project & environment setup
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Activate conda environment
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate vla

# PYTHONPATH
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${SCRIPT_DIR}"
fi

########################################
# Environment variables (IMPORTANT)
########################################

# In per-process CUDA_VISIBLE_DEVICES mode, robosuite/mujoco EGL index should be local 0.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export TOKENIZERS_PARALLELISM=false

########################################
# Default configuration
########################################

MODEL_ID="/mnt/project_rlinf/jlchen/code/UniVLA/vla_scripts/world_vla_log/0208_171019+libero4in1--vldit_aug_detach_distill00/checkpoints/step-030000"
NUM_TRIALS_PER_TASK=50
NUM_INFERENCE_STEPS=20
SAVE_VIDEO=False
SAVE_FAIL_VIDEO=True

# Default GPU list (one GPU per task)
GPU_LIST=(0 1 2 3)

# Default task suites
TASK_SUITES=(
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
)

########################################
# CLI argument parsing
########################################

ADDITIONAL_ARGS=()
TASK_SUITES_FROM_CLI=()
GPU_LIST_FROM_CLI=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --num_trials_per_task)
            NUM_TRIALS_PER_TASK="$2"
            shift 2
            ;;
        --num_inference_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --save_video)
            SAVE_VIDEO="$2"
            shift 2
            ;;
        --save_fail_video)
            SAVE_FAIL_VIDEO="$2"
            shift 2
            ;;
        --tasks)
            IFS=',' read -r -a TASK_SUITES_FROM_CLI <<< "$2"
            shift 2
            ;;
        --gpu_list|--gpus)
            IFS=',' read -r -a GPU_LIST_FROM_CLI <<< "$2"
            shift 2
            ;;
        *)
            ADDITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Override defaults if CLI provided
if [[ ${#TASK_SUITES_FROM_CLI[@]} -gt 0 ]]; then
    TASK_SUITES=("${TASK_SUITES_FROM_CLI[@]}")
fi

if [[ ${#GPU_LIST_FROM_CLI[@]} -gt 0 ]]; then
    GPU_LIST=("${GPU_LIST_FROM_CLI[@]}")
fi

########################################
# Build Python args
########################################

PYTHON_ARGS=(
    --model_id "$MODEL_ID"
    --num_trials_per_task "$NUM_TRIALS_PER_TASK"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --save_video "$SAVE_VIDEO"
    --save_fail_video "$SAVE_FAIL_VIDEO"
    "${ADDITIONAL_ARGS[@]}"
)

########################################
# Parallel execution
########################################

echo "========================================"
echo "Parallel LIBERO evaluation"
echo "Task suites: ${TASK_SUITES[*]}"
echo "GPU list:    ${GPU_LIST[*]}"
echo "========================================"

PIDS=()
LOG_DIR="${SCRIPT_DIR}/experiments/tmp_logs"
mkdir -p "${LOG_DIR}"

for IDX in "${!TASK_SUITES[@]}"; do
    TASK_SUITE="${TASK_SUITES[$IDX]}"
    GPU_ID="${GPU_LIST[$IDX % ${#GPU_LIST[@]}]}"

    LOG_FILE="${LOG_DIR}/log_${TASK_SUITE}.out"

    echo ""
    echo "----------------------------------------"
    echo "Launching ${TASK_SUITE} on GPU ${GPU_ID}"
    echo "Log file: ${LOG_FILE}"
    echo "----------------------------------------"

    env -u MUJOCO_EGL_DEVICE_ID \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        EGL_DEVICE_ID=0 \
        PYTHONFAULTHANDLER=1 \
        python -m experiments.robot.libero.run_libero_eval \
        "${PYTHON_ARGS[@]}" \
        --task_suite_name "${TASK_SUITE}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

########################################
# Wait for all jobs
########################################

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

if [[ ${FAIL} -ne 0 ]]; then
    echo "❌ One or more task suites failed. Check logs in ${LOG_DIR}"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ All task suites completed successfully!"
echo "Logs saved in: ${LOG_DIR}"
echo "========================================"
