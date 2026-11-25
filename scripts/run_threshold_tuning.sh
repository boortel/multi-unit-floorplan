#!/usr/bin/env bash
# Validation-tuned decision thresholds for every fine-tuned run (kfold_patch/tune_threshold.py; inference only,
# no retraining). Order: R3D / CVC-FP / WAFFLE (frozen, then unfrozen backbone) -> MLSTRUCT-FP s40 -> MLSTRUCT-FP
# native (slowest, last). One queue served by LANES_PER_GPU lanes on each GPU in GPUS; finished runs are skipped,
# so the script can be re-run after an interruption. Ends with kfold_patch/summarize_threshold.py.
# Env: GPUS (default "0 2"), LANES_PER_GPU (default 2), SKIP_NATIVE=1 to leave out MLSTRUCT-FP native.
cd "$(dirname "$0")/.."
export TF_FORCE_GPU_ALLOW_GROWTH=true
GPUS=${GPUS:-0 2}; LANES_PER_GPU=${LANES_PER_GPU:-2}
OUT=results/threshold_tuning; LOGS=logs/threshold_tuning; mkdir -p $OUT $LOGS

Q=$LOGS/queue.txt; : > $Q
srcs="results/finetune/*.json results/finetune_unfrozen/*.json results/mlstruct/finetune/mlstruct-s40_*.json"
[ -z "$SKIP_NATIVE" ] && srcs="$srcs results/mlstruct/finetune/mlstruct-native_*.json"
for f in $srcs; do
  sub=$(basename $(dirname $f)); [ $sub = finetune ] && [[ $f == results/mlstruct/* ]] && sub=mlstruct
  echo "$OUT/$sub/$(basename $f)|${sub}_$(basename $f .json)|python kfold_patch/tune_threshold.py $f --out $OUT/$sub" >> $Q
done
echo "$(wc -l < $Q) jobs queued"

NEXT=$LOGS/queue.next; echo 0 > $NEXT
lane() { # gpu
  while true; do
    i=$(flock $NEXT -c "i=\$(cat $NEXT); echo \$((i+1)) > $NEXT; echo \$i")
    line=$(sed -n "$((i+1))p" $Q); [ -z "$line" ] && return
    IFS='|' read -r done name cmd <<< "$line"
    if [ -f "$done" ]; then echo "skip $name"; continue; fi
    echo "$(date +%F' '%T) gpu$1 start $name"
    CUDA_VISIBLE_DEVICES=$1 $cmd > $LOGS/$name.log 2>&1 && st=ok || st=FAILED
    echo "$(date +%F' '%T) gpu$1 $st $name"
  done
}
for g in $GPUS; do for l in $(seq $LANES_PER_GPU); do lane $g & sleep 10; done; done
wait
python kfold_patch/summarize_threshold.py > /dev/null && echo "summary: $OUT/SUMMARY.md"
echo "threshold tuning finished"
