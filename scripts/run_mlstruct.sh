#!/usr/bin/env bash
# MLSTRUCT-FP evaluation (bg / wall, 5-fold CV, tiled inference, scored at native resolution).
# Order: prepare data -> zero-shot (s40 all 10 CubiCasa folds; px/m sensitivity s30-s60 and native with the
# fine-tune checkpoint) -> unfrozen fine-tuning at s40 -> unfrozen fine-tuning at native.
# Jobs go into one ordered queue served by LANES_PER_GPU lanes on each GPU in GPUS; jobs with a result
# already present are skipped, so the script can be re-run after an interruption.
# Env: GPUS (default "0 2 3"), LANES_PER_GPU (default 2), LR (3e-5), EPOCHS (300), STEPS (500 steps/epoch).
cd "$(dirname "$0")/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true
GPUS=${GPUS:-0 2 3}; LANES_PER_GPU=${LANES_PER_GPU:-2}
LOGS=logs/mlstruct; ZS=results/mlstruct/zeroshot; FT=results/mlstruct/finetune; MODELS_OUT=models/finetune_mlstruct
mkdir -p $LOGS $ZS $FT

declare -A CK=(  # best-val CubiCasa fold per model x backbone (same as run_finetune_cross.sh)
  [cab1:EfficientNetB4]="models/cab1_cab1_hpo_trial52_b4_EfficientNetB4_16,32,64,128,256_cubicasa5k_20261004-120619/9"
  [cab1:EfficientNetV2M]="models/cab1_cab1_hpo_trial21_v2m_EfficientNetV2M_16,32,64,128,256_cubicasa5k_20261005-034025/1"
  [cab2:EfficientNetB4]="models/cab2_cab2_eval_cubicasa_b4_EfficientNetB4_32,64,128,256,512_cubicasa5k_20260831-123736/3"
  [cab2:EfficientNetV2M]="models/cab2_cab2_hpo_trial32_v2m_EfficientNetV2M_16,32,64,128,256_cubicasa5k_20261012-201652/4"
)
declare -A SRC=(  # official k-fold result files (fold checkpoints), same as the CVC/WAFFLE zero-shot runs
  [cab1:EfficientNetB4]=results/official/test/test_kfold_cab1_EfficientNetB4_20261014-140050.txt
  [cab1:EfficientNetV2M]=results/official/test/test_kfold_cab1_EfficientNetV2M_20261014-122132.txt
  [cab2:EfficientNetB4]=results/official/test/test_kfold_cab2_EfficientNetB4_20260908-075647.txt
  [cab2:EfficientNetV2M]=results/official/test/test_kfold_cab2_EfficientNetV2M_20261014-120852.txt
)
KEYS="cab2:EfficientNetV2M cab1:EfficientNetV2M cab1:EfficientNetB4 cab2:EfficientNetB4"

# 1. data (CPU)
[ -f data/mlstruct_proc/meta.json ] || python kfold_patch/prepare_mlstruct.py > $LOGS/prepare.log 2>&1 || exit 1

# 2. job queue: "<done-glob>|<log name>|<command>"
Q=$LOGS/queue.txt; : > $Q
for key in $KEYS; do m=${key%:*}; b=${key#*:}
  echo "$ZS/s40_${m}_${b}_allfolds_2*.json|zs_s40_${m}_${b}|python kfold_patch/evaluate_mlstruct.py --model $m --scale s40 --source ${SRC[$key]}" >> $Q
done
for s in s30 s50 s60 s40 native; do for key in $KEYS; do m=${key%:*}; b=${key#*:}
  echo "$ZS/${s}_${m}_${b}_single_2*.json|zs_${s}_${m}_${b}|python kfold_patch/evaluate_mlstruct.py --model $m --scale $s --checkpoint ${CK[$key]} --backbone $b" >> $Q
done; done
for s in s40 native; do for k in 0 1 2 3 4; do for key in $KEYS; do m=${key%:*}; b=${key#*:}
  extra=$([ $s = native ] && echo --skip_adabn)
  echo "$FT/mlstruct-${s}_${m}_${b}_split${k}_*.json|ft_${s}_${m}_${b}_split${k}|python kfold_patch/finetune_cross_dataset.py --model $m --backbone $b --checkpoint ${CK[$key]} --dataset mlstruct --scale $s --split $k --unfreeze --lr ${LR:-3e-5} --epochs ${EPOCHS:-300} --steps_per_epoch ${STEPS:-500} --out $FT --models_out $MODELS_OUT $extra" >> $Q
done; done; done
echo "$(wc -l < $Q) jobs queued"

# 3. lanes pull the next job index under a lock
NEXT=$LOGS/queue.next; echo 0 > $NEXT
lane() { # gpu
  while true; do
    i=$(flock $NEXT -c "i=\$(cat $NEXT); echo \$((i+1)) > $NEXT; echo \$i")
    line=$(sed -n "$((i+1))p" $Q); [ -z "$line" ] && return
    IFS='|' read -r done name cmd <<< "$line"
    if compgen -G "$done" > /dev/null; then echo "skip $name"; continue; fi
    echo "$(date +%F' '%T) gpu$1 start $name"
    CUDA_VISIBLE_DEVICES=$1 $cmd > $LOGS/$name.log 2>&1 && st=ok || st=FAILED
    echo "$(date +%F' '%T) gpu$1 $st $name"
  done
}
for g in $GPUS; do for l in $(seq $LANES_PER_GPU); do lane $g & sleep 20; done; done
wait
python kfold_patch/summarize_mlstruct.py > /dev/null && echo "summary: results/mlstruct/SUMMARY.md"
echo "MLSTRUCT-FP runs finished"
