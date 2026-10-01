#!/usr/bin/env bash
# ==============================================================================
# Run All 4 Optimal K-Fold Combinations:
#   1. CAB1 - EfficientNetV1 (Trial 52: B4, small decoder, AAF [2,4,8], CAM 3)
#   2. CAB1 - EfficientNetV2 (Trial 21: V2M, small decoder, AAF [4,8], CAM 5)
#   3. CAB2 - EfficientNetV1 (Ablation winner: B4, base decoder, AAF [2,4], CAM 3, no HHDC)
#   4. CAB2 - EfficientNetV2 (Trial 32: V2M, small decoder, AAF [2,4,8], HHDC 3, CAM 1)
# ==============================================================================
# Usage:
#   ./run_optimal_all_four.sh <mode: parallel | sequential> [GPU_1] [GPU_2]
# Default:
#   ./run_optimal_all_four.sh sequential 0
# ==============================================================================
set -euo pipefail

MODE="${1:-sequential}"
GPU1="${2:-0}"
GPU2="${3:-1}"

mkdir -p logs results

CONFIGS=(
    "kfold_patch/eval_cab1_hpo_b4.py:logs/kfold_cab1_v1_b4.log:CAB1-V1"
    "kfold_patch/eval_cab1_hpo_v2m.py:logs/kfold_cab1_v2_v2m.log:CAB1-V2"
    "kfold_patch/eval_cab2_b4_cubicasa.py:logs/kfold_cab2_v1_b4.log:CAB2-V1"
    "kfold_patch/eval_cab2_hpo_v2m.py:logs/kfold_cab2_v2_v2m.log:CAB2-V2"
)

if [[ "${MODE}" == "sequential" ]]; then
    echo "Running 4 optimal configurations sequentially on GPU ${GPU1}..."
    for item in "${CONFIGS[@]}"; do
        IFS=":" read -r cfg log label <<< "${item}"
        echo "Starting ${label} (${cfg})..."
        CUDA_VISIBLE_DEVICES="${GPU1}" python train_config.py "${cfg}" > "${log}" 2>&1
        echo "${label} finished. Log: ${log}"
    done
elif [[ "${MODE}" == "parallel" ]]; then
    echo "Running Pair 1 (CAB1-V1 on GPU ${GPU1}, CAB2-V1 on GPU ${GPU2})..."
    CUDA_VISIBLE_DEVICES="${GPU1}" python train_config.py kfold_patch/eval_cab1_hpo_b4.py > logs/kfold_cab1_v1_b4.log 2>&1 &
    PID1=$!
    CUDA_VISIBLE_DEVICES="${GPU2}" python train_config.py kfold_patch/eval_cab2_b4_cubicasa.py > logs/kfold_cab2_v1_b4.log 2>&1 &
    PID2=$!
    wait "${PID1}" "${PID2}"
    echo "Pair 1 finished."

    echo "Running Pair 2 (CAB1-V2 on GPU ${GPU1}, CAB2-V2 on GPU ${GPU2})..."
    CUDA_VISIBLE_DEVICES="${GPU1}" python train_config.py kfold_patch/eval_cab1_hpo_v2m.py > logs/kfold_cab1_v2_v2m.log 2>&1 &
    PID3=$!
    CUDA_VISIBLE_DEVICES="${GPU2}" python train_config.py kfold_patch/eval_cab2_hpo_v2m.py > logs/kfold_cab2_v2_v2m.log 2>&1 &
    PID4=$!
    wait "${PID3}" "${PID4}"
    echo "Pair 2 finished."
else
    echo "Unknown mode: ${MODE}. Choose 'sequential' or 'parallel'."
    exit 1
fi

echo "All runs finished. Running aggregate evaluation..."
python kfold_patch/evaluate_kfold.py --models cab1 cab2
