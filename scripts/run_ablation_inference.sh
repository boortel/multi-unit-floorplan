#!/usr/bin/env bash
# Inference-time ablation of the CAB1/CAB2 V2M champions on CPU (no retraining).
#
#   scripts/run_ablation_inference.sh screen            # fold 0, 100 test images, all variants + single edges + AC
#   scripts/run_ablation_inference.sh main              # 10 folds x 400 test images, main variants
#   JOBS=10 THREADS=12 scripts/run_ablation_inference.sh main        # CPU
#   DEVICES=0 JOBS=1 scripts/run_ablation_inference.sh main           # one GPU
#   DEVICES=0,1 JOBS=2 scripts/run_ablation_inference.sh main         # jobs spread round-robin over GPUs 0 and 1
#
# Each job = one (model, fold); jobs are resumable (finished variants are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:-screen}
JOBS=${JOBS:-8}
THREADS=${THREADS:-14}
DEVICES=${DEVICES:--1}   # -1 = CPU; otherwise comma-separated GPU ids
LOG_DIR=results/ablation_inference/logs
mkdir -p "$LOG_DIR"

if [[ "$PHASE" == "screen" ]]; then
  FOLDS="0"
else
  FOLDS="0 1 2 3 4 5 6 7 8 9"
fi

IFS=, read -ra DEV <<< "$DEVICES"
jobs_list=()
n=0
for f in $FOLDS; do
  for m in cab1 cab2; do
    jobs_list+=("$m $f ${DEV[$((n % ${#DEV[@]}))]}")
    n=$((n + 1))
  done
done

printf '%s\n' "${jobs_list[@]}" | xargs -P "$JOBS" -L 1 bash -c '
  m=$0; f=$1; d=$2
  CUDA_VISIBLE_DEVICES=$d python kfold_patch/ablation_inference.py --model "$m" --fold "$f" \
    --phase '"$PHASE"' --threads '"$THREADS"' >> '"$LOG_DIR"'/'"$PHASE"'_${m}_f${f}.log 2>&1 \
    && echo "done $m fold $f" || echo "FAILED $m fold $f"
'
python kfold_patch/summarize_ablation.py --phase "$PHASE"
