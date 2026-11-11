#!/usr/bin/env bash
# Zero-shot per-class (walls / doors / windows) evaluation on WAFFLE, for comparison with the
# WAFFLE paper's Table 4. Needs: python kfold_patch/create_cross_tfrecords.py --dataset waffle5
cd "$(dirname "$0")/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true
GPU=${GPU:-0}
T=results/official/test
run() { CUDA_VISIBLE_DEVICES=$GPU python kfold_patch/eval_waffle_perclass.py --model $1 --source $T/$2 > logs/waffle5_$3.log 2>&1; }
run cab1 test_kfold_cab1_EfficientNetB4_20261014-140050.txt cab1_b4 &
run cab1 test_kfold_cab1_EfficientNetV2M_20261014-122132.txt cab1_v2m &
run cab2 test_kfold_cab2_EfficientNetB4_20260908-075647.txt cab2_b4 &
run cab2 test_kfold_cab2_EfficientNetV2M_20261014-120852.txt cab2_v2m &
wait
run cubicasa5k test_kfold_cubicasa5k_VGG16_20260908-072527.txt cubicasa5k &
run zeng test_kfold_zeng_VGG16_20260908-071630.txt zeng &
wait
echo "waffle perclass done"
