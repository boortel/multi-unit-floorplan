#!/usr/bin/env bash
# ==============================================================================
# Automated Test Set Evaluation Runner for CubiCasa5k (data/cubicasa5k/test.txt)
# ==============================================================================
# Usage:
#   # Evaluate both CAB1 and CAB2 with EfficientNetB4 on GPU 0 (Default & Recommended):
#   ./run_test_evaluation.sh b4 0
#
#   # Evaluate only CAB1 B4 on GPU 0:
#   ./run_test_evaluation.sh cab1_b4 0
#
#   # Evaluate only CAB2 B4 on GPU 3:
#   ./run_test_evaluation.sh cab2_b4 3
#
#   # Evaluate both CAB1 and CAB2 with EfficientNetV2S on GPU 0:
#   ./run_test_evaluation.sh v2s 0
#
#   # Evaluate all models (CAB1 B4, CAB2 B4, CubiCasa5k, Zeng):
#   ./run_test_evaluation.sh all 0
# ==============================================================================

set -euo pipefail

TARGET="${1:-b4}"
GPU_ID="${2:-0}"

mkdir -p logs results

echo "======================================================================"
echo " CubiCasa5k Test Set Evaluation Runner"
echo " Target: ${TARGET} | GPU: ${GPU_ID}"
echo " Test Split: data/cubicasa5k/test.txt (cubicasa5k_test.tfrecords, 400 images)"
echo "======================================================================"

# Ensure NVIDIA pip package libraries are discoverable in LD_LIBRARY_PATH
NV_LIBS=$(python -c "import site, glob; print(':'.join(glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')))" 2>/dev/null || true)
if [[ -n "${NV_LIBS}" ]]; then
    export LD_LIBRARY_PATH="${NV_LIBS}:/usr/lib/x86_64-linux-gnu:/opt/miniconda3/envs/main/lib:${LD_LIBRARY_PATH:-}"
fi

# Verify GPU accessibility
echo "Checking GPU accessibility on device ${GPU_ID}..."
GPU_CHECK=$(CUDA_VISIBLE_DEVICES="${GPU_ID}" python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))
" 2>/dev/null || echo "0")
GPU_CHECK=$(echo "${GPU_CHECK}" | tr -d '[:space:]')

if [[ "${GPU_CHECK}" == "0" || -z "${GPU_CHECK}" ]]; then
    echo "WARNING: No accessible GPU detected for CUDA_VISIBLE_DEVICES=${GPU_ID}."
    echo "Running on CPU is significantly slower (~25-30 min/fold vs ~1.5 min/fold on GPU)."
    echo "If you recently rebuilt the container, verify docker was launched with '--gpus all'."
    read -p "Continue on CPU? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting evaluation. Please ensure GPU passthrough is active."
        exit 1
    fi
else
    echo "GPU detected successfully! Proceeding with GPU accelerated evaluation."
fi

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="logs/test_eval_${TARGET}_${TIMESTAMP}.log"

echo "Evaluation log will be written to: ${LOG_FILE}"

case "${TARGET}" in
    b4|B4)
        echo "Evaluating CAB1 and CAB2 with EfficientNetB4 across all 10 folds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cab1 cab2 \
            --backbone EfficientNetB4 \
            --k_fold 10 2>&1 | tee "${LOG_FILE}"
        ;;
    cab1_b4|CAB1_B4)
        echo "Evaluating CAB1 with EfficientNetB4 across all 10 folds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cab1 \
            --backbone EfficientNetB4 \
            --k_fold 10 2>&1 | tee "${LOG_FILE}"
        ;;
    cab2_b4|CAB2_B4)
        echo "Evaluating CAB2 with EfficientNetB4 across all 10 folds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cab2 \
            --backbone EfficientNetB4 \
            --k_fold 10 2>&1 | tee "${LOG_FILE}"
        ;;
    v2s|V2S)
        echo "Evaluating CAB1 and CAB2 with EfficientNetV2S across all 10 folds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cab1 cab2 \
            --backbone EfficientNetV2S \
            --k_fold 10 2>&1 | tee "${LOG_FILE}"
        ;;
    all|ALL)
        echo "Evaluating all primary architectures (CAB1 B4, CAB2 B4, CubiCasa5k, Zeng)..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cab1 cab2 \
            --backbone EfficientNetB4 \
            --k_fold 10 2>&1 | tee "${LOG_FILE}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python kfold_patch/evaluate_kfold.py \
            --models cubicasa5k zeng \
            --k_fold 10 2>&1 | tee -a "${LOG_FILE}"
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Available targets: b4, cab1_b4, cab2_b4, v2s, all"
        exit 1
        ;;
esac

echo "======================================================================"
echo " Evaluation complete! Results saved in results/ directory."
echo " Review log at: ${LOG_FILE}"
echo "======================================================================"
