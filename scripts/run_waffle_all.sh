#!/usr/bin/env bash
# WAFFLE benchmark: zero-shot (all 10 folds of each CAB1/CAB2 x B4/V2M, same checkpoint sets as the
# CVC-FP zero-shot run) followed by fine-tuning (5-fold CV, LR 3e-5, max 300 epochs, as for CVC-FP).
# Everything on one GPU (default 0). Build the records first:
#   python kfold_patch/create_cross_tfrecords.py --dataset waffle
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs
declare -A SRC=(
  [cab1_b4]=results/official/test/test_kfold_cab1_EfficientNetB4_20261014-140050.txt
  [cab1_v2m]=results/official/test/test_kfold_cab1_EfficientNetV2M_20261014-122132.txt
  [cab2_b4]=results/official/test/test_kfold_cab2_EfficientNetB4_20260908-075647.txt
  [cab2_v2m]=results/official/test/test_kfold_cab2_EfficientNetV2M_20261014-120852.txt
)
if [ -z "$SKIP_ZEROSHOT" ]; then
  for k in cab1_b4 cab1_v2m cab2_b4 cab2_v2m; do
    CUDA_VISIBLE_DEVICES=$GPU python kfold_patch/evaluate_cross_dataset.py --models ${k%_*} --datasets waffle_all \
      --source ${SRC[$k]} > logs/cross_waffle_$k.log 2>&1 &
  done
  wait
  echo "zero-shot done"
fi
GPU_CAB1=$GPU GPU_CAB2=$GPU DATASETS=waffle LR=3e-5 EPOCHS=300 bash scripts/run_finetune_cross.sh
echo "waffle all done"
