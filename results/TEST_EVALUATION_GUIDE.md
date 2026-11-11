# Test Evaluation Guide

**Project:** Multi-Unit Floorplan Segmentation  
**Test Set:** 400 CubiCasa5k floorplans (`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`)  
**Last Evaluated:** October 14, 2026 (October runs: V2M HPO, B4 HPO t52, CAB2 B4 retrain) · September 8, 2026 (B4 / V2S / VGG16)

---

## Quick Start

All scripts live in `scripts/`. They use repo-relative paths (`kfold_patch/…`, `logs/`, `results/`), so **run them from the repository root**.

### Full 4-GPU Parallel Evaluation (~59 min)

Runs validation (4,600 images) + test (400 images) for the 6 Sept-8 architectures:

```bash
./scripts/run_all_evaluations.sh
```

**GPU Allocation:**
| GPU | Architecture |
|:---:|:-------------|
| 0 | CAB1 EfficientNetB4 |
| 1 | CAB2 EfficientNetB4 |
| 2 | CubiCasa5k VGG16 → CAB1 EfficientNetV2S |
| 3 | Zeng VGG16 → CAB2 EfficientNetV2S |

### Single Architecture

```bash
# CAB1 + CAB2 with EfficientNetB4 on GPU 0:
./scripts/run_test_evaluation.sh b4 0

# All architectures sequentially on GPU 0:
./scripts/run_test_evaluation.sh all 0
```

### Direct Python CLI

```bash
# CAB1 + CAB2 V2M HPO models, val + test (how the Oct 14 results were produced):
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 cab2 --backbone EfficientNetV2M --k_fold 10 --split both

# CAB1 B4, both val + test splits:
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 --backbone EfficientNetB4 --k_fold 10 --split both
```

`--split` defaults to `test`. Results are written to the `results/` root as `{split}_kfold_{model}_{backbone}_{timestamp}.txt`.

### ⚠️ Checkpoint selection: which run gets evaluated

`evaluate_kfold.py` scans `models/{model}_*`, filters by the `--backbone` substring, and takes the **latest** checkpoint per fold. Since October there are two 10-fold runs for each B4 setup:

| `--models … --backbone EfficientNetB4` | Sept 8 official (older) | Now picked (newer) |
|:---|:---|:---|
| `cab1` | `cab1_cab1_eval_cubicasa_b4_*` (Aug 30 – Sep 1) | `cab1_cab1_hpo_trial52_b4_*` (Oct 1 – 4) |
| `cab2` | `cab2_cab2_eval_cubicasa_b4_*` (Aug 30 – Sep 1) | `cab2_cab2_eval_cubicasa_b4_*` (Oct 8 – 11 retrain) |

So **re-running `run_all_evaluations.sh` / `run_test_evaluation.sh b4` now evaluates the October B4 runs and won't reproduce the Sept 8 B4 numbers.** (The October B4 runs were evaluated this way on Oct 14. Running the val and test splits as separate jobs, one per GPU, takes ~25 min total on 4 GPUs. Each split takes ~22 min per model on one GPU.) Every result file lists the exact checkpoint directory per fold under `Fold Checkpoint Sources`, so check that section to see which run a file came from.

---

## GPU Setup

### Dev Containers (Recommended)

Ensure `.devcontainer/devcontainer.json` includes:
```json
{ "runArgs": ["--gpus", "all", "--ipc=host"] }
```
Then: `F1` → `Dev Containers: Rebuild Container`

### Docker CLI
```bash
docker run --gpus all --ipc=host -it ...
```

### Verification
```bash
nvidia-smi
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## Latest Results Summary

V2M rows were trained after the Sept 8 audit fixes and with HPO configs. B4 / V2S rows were trained before them. See the caveat in [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md#1-official-results).

### Test Set (400 images)

| Architecture | Backbone | Test Acc | No-BG Acc | Walls IoU | Windows | Doors | Stairs | Railings | Macro IoU |
|:-------------|:---------|:--------:|:---------:|:---------:|:-------:|:-----:|:------:|:--------:|:---------:|
| **CAB1 V2M (t21)** | V2M | **96.37%** | **80.16%** | **72.82%** | 67.26% | 56.18% | 47.90% | 27.48% | 61.37% |
| **CAB2 V2M (t32)** | V2M | 96.14% | 76.39% | 68.66% | **69.91%** | **59.17%** | **49.93%** | **28.03%** | **61.99%** |
| CAB1 B4 HPO t52 (Oct) | B4 | 95.86% | 74.96% | 67.77% | 63.04% | 53.50% | 46.07% | 27.48% | 58.98% |
| CAB2 B4 retrain (Oct) ⚠️ | B4 | 92.70% | 57.40% | 48.53% | 36.13% | 16.51% | 8.63% | 6.23% | 34.89% |
| CAB1 B4 | B4 | 94.52% | 69.19% | 60.63% | 53.92% | 27.43% | 14.69% | 9.16% | 43.47% |
| CAB2 B4 | B4 | 94.14% | 66.45% | 59.18% | 47.80% | 19.35% | 9.87% | 7.47% | 39.72% |
| CAB1 V2S | V2S | 93.79% | 63.88% | 56.11% | 46.57% | 6.86% | 6.03% | 3.81% | 35.63% |
| CAB2 V2S | V2S | 93.95% | 63.05% | 57.04% | 42.91% | 26.24% | 15.82% | 10.65% | 41.17% |
| CubiCasa5k | VGG16 | 95.61% | 61.65% | 63.46% | 59.82% | 41.95% | 36.56% | 13.92% | 51.89% |
| Zeng | VGG16 | 95.19% | 58.08% | 59.33% | 56.47% | 43.34% | 38.97% | 8.97% | 50.37% |

### Validation Set (4,600 out-of-fold images)

| Architecture | Backbone | Val Acc | No-BG Acc | Walls IoU | Windows | Doors | Stairs | Railings | Macro IoU |
|:-------------|:---------|:-------:|:---------:|:---------:|:-------:|:-----:|:------:|:--------:|:---------:|
| **CAB1 V2M (t21)** | V2M | **96.43%** | **79.68%** | **71.63%** | 66.21% | 56.03% | 46.61% | 29.53% | 61.10% |
| **CAB2 V2M (t32)** | V2M | 96.15% | 75.36% | 66.90% | **68.81%** | 59.25% | 48.60% | **29.72%** | **61.58%** |
| CAB1 B4 HPO t52 (Oct) | B4 | 95.89% | 74.07% | 66.15% | 61.95% | 53.26% | 46.05% | 28.50% | 58.66% |
| CAB2 B4 retrain (Oct) ⚠️ | B4 | 92.90% | 57.73% | 47.88% | 35.22% | 16.72% | 9.07% | 6.71% | 34.85% |
| CAB1 B4 | B4 | 94.64% | 69.55% | 59.75% | 53.06% | 27.37% | 14.53% | 9.93% | 43.29% |
| CAB2 B4 | B4 | 94.26% | 66.68% | 58.09% | 46.81% | 19.57% | 9.04% | 7.26% | 39.26% |
| CAB1 V2S | V2S | 93.99% | 64.72% | 55.82% | 45.29% | 6.99% | 6.36% | 4.24% | 35.55% |
| CAB2 V2S | V2S | 94.10% | 63.42% | 56.10% | 42.13% | 26.49% | 15.55% | 11.49% | 41.05% |
| CubiCasa5k | VGG16 | 96.42% | 66.33% | 67.52% | 66.07% | 47.58% | **57.13%** | 18.97% | 58.95% |
| Zeng | VGG16 | 95.79% | 60.86% | 61.93% | 60.47% | 45.79% | 49.12% | 11.39% | 54.07% |

---

## Evaluated Models

| Model | Backbone | Folds | Config | Hyperparameters |
|:------|:---------|:-----:|:-------|:----------------|
| CAB1 V2M (HPO t21) | EfficientNetV2M | 10 | `kfold_patch/eval_cab1_hpo_v2m.py` | filters small, hhdc=False, cam=5, aaf=[4,8], plateau |
| CAB2 V2M (HPO t32) | EfficientNetV2M | 10 | `kfold_patch/eval_cab2_hpo_v2m.py` | filters small, hhdc=3, cam=1, aaf=[2,4,8], plateau |
| CAB1 B4 HPO t52 (Oct) | EfficientNetB4 | 10 | `kfold_patch/eval_cab1_hpo_b4.py` | filters small, hhdc=False, cam=3, aaf=[2,4,8], cosine-warmup |
| CAB2 B4 retrain (Oct) | EfficientNetB4 | 10 | `kfold_patch/eval_cab2_b4_cubicasa.py` | hhdc=False, cam=3, aaf=[2,4] — 5 folds ≤91.5% test acc |
| CAB1 B4 | EfficientNetB4 | 10 | `kfold_patch/eval_cab1_b4_cubicasa.py` | hhdc=7, cam=5, aaf=[2,4] |
| CAB2 B4 | EfficientNetB4 | 10 | `kfold_patch/eval_cab2_b4_cubicasa.py` | hhdc=False, cam=3, aaf=[2,4] |
| CAB1 V2S | EfficientNetV2S | 10 | `kfold_patch/eval_cab1_cubicasa.py` | hhdc=7, cam=5, aaf=[2,4] |
| CAB2 V2S | EfficientNetV2S | 10 | `kfold_patch/eval_cab2_cubicasa.py` | hhdc=False, cam=3, aaf=[2,4] |
| CubiCasa5k | VGG16 | 10 | Pre-trained reference | Multi-task heatmap heads |
| Zeng | VGG16 | 10 | Pre-trained reference | Multi-dilation aggregation |

> **Audit Fixes Applied:** E-4 (batch aggregation), E-5 (ε division guards), E-6 (frequency-weight normalization), D-1/D-2 (nearest-neighbor interpolation). See [`docs/CODEBASE_AUDIT.md`](../docs/CODEBASE_AUDIT.md).
