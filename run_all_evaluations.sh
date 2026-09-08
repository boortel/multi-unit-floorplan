#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

NV_LIBS=$(python -c "import site, glob; print(':'.join(glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')))" 2>/dev/null || true)
if [[ -n "${NV_LIBS}" ]]; then
    export LD_LIBRARY_PATH="${NV_LIBS}:/usr/lib/x86_64-linux-gnu:/opt/miniconda3/envs/main/lib:${LD_LIBRARY_PATH:-}"
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "======================================================================"
echo " Launching Full Cross-Validation & Test Evaluation Across 4 GPUs"
echo " Date: $(date)"
echo " Fixes from CODEBASE_AUDIT.md in effect (E-4, E-5, E-6, D-1, D-2)"
echo "======================================================================"

# GPU 0: CAB1 EfficientNetB4 (val + test)
echo "Launching GPU 0: CAB1 EfficientNetB4 (val + test)..."
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 \
    --backbone EfficientNetB4 \
    --k_fold 10 \
    --split both > logs/eval_cab1_b4.log 2>&1 &
PID_GPU0=$!

# GPU 1: CAB2 EfficientNetB4 (val + test)
echo "Launching GPU 1: CAB2 EfficientNetB4 (val + test)..."
CUDA_VISIBLE_DEVICES=1 python kfold_patch/evaluate_kfold.py \
    --models cab2 \
    --backbone EfficientNetB4 \
    --k_fold 10 \
    --split both > logs/eval_cab2_b4.log 2>&1 &
PID_GPU1=$!

# GPU 2: CubiCasa5k VGG16 (val + test) then CAB1 EfficientNetV2S (val + test)
echo "Launching GPU 2: CubiCasa5k VGG16 -> CAB1 EfficientNetV2S (val + test)..."
(
    CUDA_VISIBLE_DEVICES=2 python kfold_patch/evaluate_kfold.py \
        --models cubicasa5k \
        --backbone VGG16 \
        --k_fold 10 \
        --split both > logs/eval_cubicasa5k_vgg16.log 2>&1
    echo "GPU 2: CubiCasa5k VGG16 complete. Starting CAB1 EfficientNetV2S..."
    CUDA_VISIBLE_DEVICES=2 python kfold_patch/evaluate_kfold.py \
        --models cab1 \
        --backbone EfficientNetV2S \
        --k_fold 10 \
        --split both > logs/eval_cab1_v2s.log 2>&1
) &
PID_GPU2=$!

# GPU 3: Zeng VGG16 (val + test) then CAB2 EfficientNetV2S (val + test)
echo "Launching GPU 3: Zeng VGG16 -> CAB2 EfficientNetV2S (val + test)..."
(
    CUDA_VISIBLE_DEVICES=3 python kfold_patch/evaluate_kfold.py \
        --models zeng \
        --backbone VGG16 \
        --k_fold 10 \
        --split both > logs/eval_zeng_vgg16.log 2>&1
    echo "GPU 3: Zeng VGG16 complete. Starting CAB2 EfficientNetV2S..."
    CUDA_VISIBLE_DEVICES=3 python kfold_patch/evaluate_kfold.py \
        --models cab2 \
        --backbone EfficientNetV2S \
        --k_fold 10 \
        --split both > logs/eval_cab2_v2s.log 2>&1
) &
PID_GPU3=$!

echo "All 4 GPU pipelines started:"
echo "  GPU 0: PID ${PID_GPU0} (CAB1 EfficientNetB4)"
echo "  GPU 1: PID ${PID_GPU1} (CAB2 EfficientNetB4)"
echo "  GPU 2: PID ${PID_GPU2} (CubiCasa5k VGG16 -> CAB1 EfficientNetV2S)"
echo "  GPU 3: PID ${PID_GPU3} (Zeng VGG16 -> CAB2 EfficientNetV2S)"

wait ${PID_GPU0}
echo "GPU 0 pipeline complete!"

wait ${PID_GPU1}
echo "GPU 1 pipeline complete!"

wait ${PID_GPU2}
echo "GPU 2 pipeline complete!"

wait ${PID_GPU3}
echo "GPU 3 pipeline complete!"

echo "======================================================================"
echo " All evaluations completed successfully at $(date)!"
echo "======================================================================"
