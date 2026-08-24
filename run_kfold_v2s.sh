#!/usr/bin/env bash
# ==============================================================================
# 10-Fold Cross-Validation Runner for CAB1 & CAB2 (EfficientNetV2S)
# ==============================================================================
# Usage:
#   # Run both CAB1 and CAB2 in parallel on GPUs 0 and 1:
#   ./run_kfold_v2s.sh all 0 1
#
#   # Run only CAB1 on GPU 0:
#   ./run_kfold_v2s.sh cab1 0
#
#   # Run only CAB2 on GPU 1:
#   ./run_kfold_v2s.sh cab2 1
# ==============================================================================

set -euo pipefail

TARGET="${1:-all}"
GPU_CAB1="${2:-0}"
GPU_CAB2="${3:-1}"

mkdir -p logs results

echo "======================================================================"
echo " Starting 10-Fold Cross-Validation with EfficientNetV2S"
echo " Target: ${TARGET}"
echo "======================================================================"

if [[ "${TARGET}" == "cab1" || "${TARGET}" == "all" ]]; then
    echo "Launching CAB1 on GPU ${GPU_CAB1}..."
    CUDA_VISIBLE_DEVICES="${GPU_CAB1}" python train_config.py kfold_patch/eval_cab1_cubicasa.py > logs/kfold_cab1_v2s.log 2>&1 &
    PID_CAB1=$!
    echo "CAB1 started in background (PID ${PID_CAB1}). Logs: logs/kfold_cab1_v2s.log"
fi

if [[ "${TARGET}" == "cab2" || "${TARGET}" == "all" ]]; then
    echo "Launching CAB2 on GPU ${GPU_CAB2}..."
    CUDA_VISIBLE_DEVICES="${GPU_CAB2}" python train_config.py kfold_patch/eval_cab2_cubicasa.py > logs/kfold_cab2_v2s.log 2>&1 &
    PID_CAB2=$!
    echo "CAB2 started in background (PID ${PID_CAB2}). Logs: logs/kfold_cab2_v2s.log"
fi

if [[ "${TARGET}" == "all" ]]; then
    echo "Waiting for CAB1 (PID ${PID_CAB1}) and CAB2 (PID ${PID_CAB2}) to complete..."
    wait "${PID_CAB1}"
    echo "CAB1 finished."
    wait "${PID_CAB2}"
    echo "CAB2 finished."

    echo "Running aggregate 10-fold evaluation..."
    python kfold_patch/evaluate_kfold.py --models cab1 cab2
elif [[ "${TARGET}" == "cab1" ]]; then
    wait "${PID_CAB1}"
    echo "CAB1 finished. Running evaluation..."
    python kfold_patch/evaluate_kfold.py --models cab1
elif [[ "${TARGET}" == "cab2" ]]; then
    wait "${PID_CAB2}"
    echo "CAB2 finished. Running evaluation..."
    python kfold_patch/evaluate_kfold.py --models cab2
fi

echo "======================================================================"
echo " 10-Fold Training & Evaluation Completed Successfully!"
echo " Results are saved in the results/ directory."
echo "======================================================================"
