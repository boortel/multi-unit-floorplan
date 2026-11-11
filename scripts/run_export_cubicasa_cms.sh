#!/usr/bin/env bash
# Re-export per-fold CubiCasa5k test confusion matrices for the six k-fold result sets used in the
# cross-dataset overview (CAB1/CAB2 x B4/V2M + CubiCasa5k/Zeng VGG16 baselines). GPU 0 by default.
cd "$(dirname "$0")/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true
T=results/official/test
for f in test_kfold_cab1_EfficientNetB4_20261014-140050 test_kfold_cab1_EfficientNetV2M_20261014-122132 \
         test_kfold_cab2_EfficientNetB4_20260908-075647; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} python kfold_patch/export_cubicasa_cms.py --source $T/$f.txt > logs/export_$f.log 2>&1 &
done
wait
for f in test_kfold_cab2_EfficientNetV2M_20261014-120852 test_kfold_cubicasa5k_VGG16_20260908-072527 \
         test_kfold_zeng_VGG16_20260908-071630; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} python kfold_patch/export_cubicasa_cms.py --source $T/$f.txt > logs/export_$f.log 2>&1 &
done
wait
echo "export done"
