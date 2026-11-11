#!/usr/bin/env bash
# Fine-tune the best-val CubiCasa5k fold of each CAB1/CAB2 x B4/V2M on R3D (official split)
# and CVC-FP / WAFFLE (5-fold CV, 60/20/20; DATASETS=waffle). One sequential lane per model x backbone (4 lanes in parallel).
# Env overrides: GPU_CAB1 / GPU_CAB2 (default 0 / 1), DATASETS (default "r3d cvc"), LR (default 1e-5),
# EPOCHS (default 150). Example (CVC only, everything on GPU 0):
#   GPU_CAB1=0 GPU_CAB2=0 DATASETS=cvc LR=3e-5 EPOCHS=300 bash scripts/run_finetune_cross.sh
cd "$(dirname "$0")/.."
mkdir -p logs/finetune
declare -A CK=(
  [cab1:EfficientNetB4]="models/cab1_cab1_hpo_trial52_b4_EfficientNetB4_16,32,64,128,256_cubicasa5k_20261004-120619/9"
  [cab1:EfficientNetV2M]="models/cab1_cab1_hpo_trial21_v2m_EfficientNetV2M_16,32,64,128,256_cubicasa5k_20261005-034025/1"
  [cab2:EfficientNetB4]="models/cab2_cab2_eval_cubicasa_b4_EfficientNetB4_32,64,128,256,512_cubicasa5k_20260831-123736/3"
  [cab2:EfficientNetV2M]="models/cab2_cab2_hpo_trial32_v2m_EfficientNetV2M_16,32,64,128,256_cubicasa5k_20261012-201652/4"
)
run() { # gpu model backbone dataset split  (skips runs that already have a result JSON)
  if compgen -G "results/finetune/$4_$2_$3_split$5_*.json" > /dev/null; then echo "skip $4 $2 $3 split$5"; return; fi
  CUDA_VISIBLE_DEVICES=$1 python kfold_patch/finetune_cross_dataset.py --model $2 --backbone $3 \
    --checkpoint "${CK[$2:$3]}" --dataset $4 --split $5 --lr ${LR:-1e-5} --epochs ${EPOCHS:-150} > logs/finetune/$4_$2_$3_split$5.log 2>&1
}
for key in cab1:EfficientNetB4 cab1:EfficientNetV2M cab2:EfficientNetB4 cab2:EfficientNetV2M; do
  m=${key%:*}; b=${key#*:}; gpu=$([ $m = cab1 ] && echo ${GPU_CAB1:-0} || echo ${GPU_CAB2:-1})
  ( for ds in ${DATASETS:-r3d cvc}; do
      if [ $ds = r3d ]; then run $gpu $m $b r3d 0; else for k in 0 1 2 3 4; do run $gpu $m $b $ds $k; done; fi
    done ) &
done
wait
echo "all fine-tuning runs finished"
