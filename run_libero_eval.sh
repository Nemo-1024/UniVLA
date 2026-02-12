#!/bin/bash
# Equivalent bash script for test_libero.ipynb
# This script runs the LIBERO evaluation with the same configuration as the notebook
#
# Usage:
#   ./run_libero_eval.sh [OPTIONS]
#
# Examples:
#   ./run_libero_eval.sh --num_trials_per_task 10
#   ./run_libero_eval.sh --model_id /path/to/checkpoint
#   ./run_libero_eval.sh --gpu 1 --tasks libero_10
#   ./run_libero_eval.sh --tasks libero_10
#   ./run_libero_eval.sh --tasks libero_spatial,libero_goal
#
# All arguments are passed directly to the Python script.
# If no arguments are provided, default values from the notebook are used.
# Task suites run serially and can be overridden via CLI.

set -u
set -o pipefail

# Change to script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Activate conda environment
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate vla

# Set PYTHONPATH to include project root for proper module resolution
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${SCRIPT_DIR}"
fi

# Set environment variables (must be set BEFORE any mujoco/robosuite/OpenGL imports)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export EGL_PLATFORM=surfaceless
export TOKENIZERS_PARALLELISM=false

# Default values (will be used unless overridden by command line arguments)
MODEL_ID="/mnt/project_rlinf/jlchen/code/UniVLA/vla_scripts/world_vla_log/0208_170940+libero4in1--vldit_aug_detach_distill01/checkpoints/step-030000"
NUM_TRIALS_PER_TASK=50
NUM_INFERENCE_STEPS=20
SAVE_VIDEO=False
SAVE_FAIL_VIDEO=False
GPU_ID=0

# Task suites to run serially (modify this array to change which tasks to run)
# Available options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_SUITES=(
    # "libero_spatial"
    # "libero_object"
    # "libero_goal"
    "libero_10"
)

# Initialize array for additional arguments (unknown parameters)
ADDITIONAL_ARGS=()
TASK_SUITES_FROM_CLI=()

# Parse command line arguments and override defaults
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
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --tasks)
            IFS=',' read -r -a TASK_SUITES_FROM_CLI <<< "$2"
            shift 2
            ;;
        *)
            # Unknown argument, pass it through (for other parameters like --use_wandb, etc.)
            ADDITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Respect CLI-specified task suites if provided; otherwise keep defaults.
if [[ ${#TASK_SUITES_FROM_CLI[@]} -gt 0 ]]; then
    TASK_SUITES=("${TASK_SUITES_FROM_CLI[@]}")
fi

# Build Python command arguments with defaults (without task_suite_name, which is added per-task)
PYTHON_ARGS=(
    --model_id "$MODEL_ID"
    --num_trials_per_task "$NUM_TRIALS_PER_TASK"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --save_video "$SAVE_VIDEO"
    --save_fail_video "$SAVE_FAIL_VIDEO"
    "${ADDITIONAL_ARGS[@]}"
)

# Run evaluation for each task suite serially
echo "========================================"
echo "Running LIBERO evaluation for ${#TASK_SUITES[@]} task suites"
echo "GPU ID: ${GPU_ID}"
echo "========================================"

echo "Waiting for checkpoint: $MODEL_ID"
until [[ -e "$MODEL_ID" ]]; do
    sleep 600
done
echo "Checkpoint found, starting eval."

for TASK_SUITE in "${TASK_SUITES[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting task suite: $TASK_SUITE"
    echo "========================================"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python -m experiments.robot.libero.run_libero_eval "${PYTHON_ARGS[@]}" --task_suite_name "$TASK_SUITE"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "Task suite '${TASK_SUITE}' failed on GPU ${GPU_ID} (exit=${EXIT_CODE})."
        exit "${EXIT_CODE}"
    fi
    echo "Finished task suite: $TASK_SUITE"
done

echo ""
echo "========================================"
echo "All task suites completed!"
echo "========================================"
