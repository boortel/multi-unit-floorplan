#!/usr/bin/env bash
# Backbone-unfrozen fine-tuning of CAB1/CAB2 x B4/V2M on R3D, CVC-FP and WAFFLE, all on one GPU (default 0).
# Same schedule as the frozen runs (R3D: LR 1e-5, 150 epochs; CVC-FP / WAFFLE: LR 3e-5, 300 epochs), so the
# only change is the trainable backbone. Results: results/finetune_unfrozen/, checkpoints: models/finetune_unfrozen/.
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true UNFREEZE=1 GPU_CAB1=$GPU GPU_CAB2=$GPU
DATASETS=r3d LR=1e-5 EPOCHS=150 bash scripts/run_finetune_cross.sh
DATASETS="cvc waffle" LR=3e-5 EPOCHS=300 bash scripts/run_finetune_cross.sh
python kfold_patch/summarize_finetune.py results/finetune_unfrozen > results/finetune_unfrozen/SUMMARY.md
echo "unfrozen fine-tuning done"
