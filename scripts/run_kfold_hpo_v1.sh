#!/usr/bin/env bash
# ==============================================================================
# 10-Fold Cross-Validation Runner: Optimal EfficientNetV1 (Optuna Champions)
#   CAB1: Trial 52 (EfficientNetB4, small decoder, AAF [2,4,8], CAM 3, no HHDC)
#   CAB2: Option A: Trial 52/B4 Setup (eval_cab2_b4_cubicasa.py)
#         Option B: Trial 5 (EfficientNetB0, base decoder, AAF [2,4,8], HHDC 7)
# ==============================================================================
set -euo pipefail

TARGET="${1:-all}"
GPU_CAB1="${2:-0}"
GPU_CAB2="${3:-1}"

mkdir -p logs results

echo "Starting 10-Fold CV for Optimal EfficientNetV1 (Target: ${TARGET})"

if [[ "${TARGET}" == "cab1" || "${TARGET}" == "all" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_CAB1}" python train_config.py kfold_patch/eval_cab1_hpo_b4.py > logs/kfold_cab1_hpo_b4.log 2>&1 &
    PID_CAB1=$!
    echo "CAB1 B4 HPO started (PID ${PID_CAB1}). Log: logs/kfold_cab1_hpo_b4.log"
fi

if [[ "${TARGET}" == "cab2" || "${TARGET}" == "all" ]]; then
    # Runs the verified optimal CAB2 B4 config
    CUDA_VISIBLE_DEVICES="${GPU_CAB2}" python train_config.py kfold_patch/eval_cab2_b4_cubicasa.py > logs/kfold_cab2_b4.log 2>&1 &
    PID_CAB2=$!
    echo "CAB2 B4 started (PID ${PID_CAB2}). Log: logs/kfold_cab2_b4.log"
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
