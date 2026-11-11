#!/usr/bin/env bash
# ==============================================================================
# 10-Fold Cross-Validation Runner for CAB1 & CAB2 (EfficientNetB4 Best Setup)
# ==============================================================================
# Usage:
#   # Run both CAB1 and CAB2 in parallel on GPUs 0 and 3 (default):
#   ./run_kfold_b4.sh all 0 3
#
#   # Run only CAB1 on GPU 0:
#   ./run_kfold_b4.sh cab1 0
#
#   # Run only CAB2 on GPU 3:
#   ./run_kfold_b4.sh cab2 3
# ==============================================================================

set -euo pipefail

TARGET="${1:-all}"
GPU_CAB1="${2:-0}"
GPU_CAB2="${3:-3}"

mkdir -p logs results

echo "======================================================================"
echo " Starting 10-Fold Cross-Validation with EfficientNetB4 (Best V1 Setup)"
echo " Target: ${TARGET}"
echo " CAB1 GPU: ${GPU_CAB1} | CAB2 GPU: ${GPU_CAB2}"
echo "======================================================================"

if [[ "${TARGET}" == "cab1" || "${TARGET}" == "all" ]]; then
    echo "Launching CAB1 (B4, HHDC=7, CAM=5) on GPU ${GPU_CAB1}..."
    CUDA_VISIBLE_DEVICES="${GPU_CAB1}" python train_config.py kfold_patch/eval_cab1_b4_cubicasa.py > logs/kfold_cab1_b4.log 2>&1 &
    PID_CAB1=$!
    echo "CAB1 started in background (PID ${PID_CAB1}). Logs: logs/kfold_cab1_b4.log"
fi

if [[ "${TARGET}" == "cab2" || "${TARGET}" == "all" ]]; then
    echo "Launching CAB2 (B4, HHDC=False, CAM=3) on GPU ${GPU_CAB2}..."
    CUDA_VISIBLE_DEVICES="${GPU_CAB2}" python train_config.py kfold_patch/eval_cab2_b4_cubicasa.py > logs/kfold_cab2_b4.log 2>&1 &
    PID_CAB2=$!
    echo "CAB2 started in background (PID ${PID_CAB2}). Logs: logs/kfold_cab2_b4.log"
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
echo " 10-Fold B4 Training & Evaluation Completed Successfully!"
echo " Results are saved in the results/ directory."
echo "======================================================================"
