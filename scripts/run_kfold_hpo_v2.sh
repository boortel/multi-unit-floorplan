#!/usr/bin/env bash
# ==============================================================================
# 10-Fold Cross-Validation Runner: Optimal EfficientNetV2 (Optuna Champions)
#   CAB1: Trial 21 (EfficientNetV2M, small decoder, CAM 5, AAF [4,8], plateau)
#   CAB2: Trial 32 (EfficientNetV2M, small decoder, HHDC 3, CAM 1, AAF [2,4,8], plateau)
# ==============================================================================
set -euo pipefail

TARGET="${1:-all}"
GPU_CAB1="${2:-0}"
GPU_CAB2="${3:-1}"

mkdir -p logs results

echo "Starting 10-Fold CV for Optimal EfficientNetV2 (Target: ${TARGET})"

if [[ "${TARGET}" == "cab1" || "${TARGET}" == "all" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_CAB1}" python train_config.py kfold_patch/eval_cab1_hpo_v2m.py > logs/kfold_cab1_hpo_v2m.log 2>&1 &
    PID_CAB1=$!
    echo "CAB1 V2M HPO started (PID ${PID_CAB1}). Log: logs/kfold_cab1_hpo_v2m.log"
fi

if [[ "${TARGET}" == "cab2" || "${TARGET}" == "all" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_CAB2}" python train_config.py kfold_patch/eval_cab2_hpo_v2m.py > logs/kfold_cab2_hpo_v2m.log 2>&1 &
    PID_CAB2=$!
    echo "CAB2 V2M HPO started (PID ${PID_CAB2}). Log: logs/kfold_cab2_hpo_v2m.log"
fi

if [[ "${TARGET}" == "all" ]]; then
    wait "${PID_CAB1}"
    wait "${PID_CAB2}"
    python kfold_patch/evaluate_kfold.py --models cab1 cab2
elif [[ "${TARGET}" == "cab1" ]]; then
    wait "${PID_CAB1}"
    python kfold_patch/evaluate_kfold.py --models cab1
elif [[ "${TARGET}" == "cab2" ]]; then
    wait "${PID_CAB2}"
    python kfold_patch/evaluate_kfold.py --models cab2
fi
