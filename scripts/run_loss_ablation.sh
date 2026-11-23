#!/usr/bin/env bash
# Loss ablation (retraining), option C: CAB1 + CAB2, folds FOLDS, variants auf / auf_aaf.
# One sequential queue per GPU: queue A on GPU_A trains CAB1, queue B on GPU_B trains CAB2.
# Fold-major order so that early folds of both variants finish first.
# Resumable: a (model,variant,fold) with a finished SavedModel is skipped.
# Usage: GPU_A=0 GPU_B=2 FOLDS="0 1 2 3 4" scripts/run_loss_ablation.sh [cab1|cab2|all]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-2}"; FOLDS="${FOLDS:-0 1 2 3 4}"
VARIANTS="${VARIANTS:-auf auf_aaf}"; TARGET="${1:-all}"
GEN=kfold_patch/loss_ablation/generated; LOGS=logs/loss_ablation
mkdir -p "$GEN" "$LOGS"

done_already() {  # $1=model $2=variant $3=fold
  local c
  for c in models/${1}_lossabl_${1}_${2}_v2m_*/"$3"/training_cfg.py; do
    [[ -f "$c" && -f "$(dirname "$c")/saved_model.pb" ]] && grep -q "epochs=100," "$c" && return 0
  done
  return 1
}

queue() {  # $1=model $2=gpu
  local m=$1 gpu=$2
  for f in $FOLDS; do for v in $VARIANTS; do
    if done_already "$m" "$v" "$f"; then echo "[skip] $m $v fold $f"; continue; fi
    local cfg="$GEN/${m}_${v}_f${f}.py"
    printf "_base_ = ['../%s_%s.py']\ntarget_fold = %d\n" "$m" "$v" "$f" > "$cfg"
    echo "[$(date '+%F %T')] start $m $v fold $f on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python train_config.py "$cfg" > "$LOGS/${m}_${v}_f${f}.log" 2>&1
    echo "[$(date '+%F %T')] end   $m $v fold $f rc=$?"
  done; done
}

[[ "$TARGET" == cab1 || "$TARGET" == all ]] && { queue cab1 "$GPU_A" > "$LOGS/queue_cab1.log" 2>&1 & }
[[ "$TARGET" == cab2 || "$TARGET" == all ]] && { queue cab2 "$GPU_B" > "$LOGS/queue_cab2.log" 2>&1 & }
wait
