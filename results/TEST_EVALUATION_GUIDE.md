# Test Evaluation Guide

**Project:** Multi-Unit Floorplan Segmentation  
**Test Set:** 400 CubiCasa5k floorplans (`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`)  
**Last Evaluated:** September 8, 2026

---

## Quick Start

### Full 4-GPU Parallel Evaluation (~59 min)

Runs validation (4,600 images) + test (400 images) across all 6 architectures:

```bash
./run_all_evaluations.sh
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
./run_test_evaluation.sh b4 0

# All architectures sequentially on GPU 0:
./run_test_evaluation.sh all 0
```

### Direct Python CLI

```bash
# CAB1 B4, both val + test splits:
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 --backbone EfficientNetB4 --k_fold 10 --split both

# CAB2 B4, test only:
CUDA_VISIBLE_DEVICES=1 python kfold_patch/evaluate_kfold.py \
    --models cab2 --backbone EfficientNetB4 --k_fold 10 --split test
```

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

### Test Set (400 images)

| Architecture | Backbone | Test Acc | No-BG Acc | Walls IoU | Windows | Doors | Stairs | Railings | Macro IoU |
|:-------------|:---------|:--------:|:---------:|:---------:|:-------:|:-----:|:------:|:--------:|:---------:|
| **CAB1 B4** | B4 | 94.52% | **69.19%** | 60.63% | 53.92% | 27.43% | 14.69% | 9.16% | 43.47% |
| **CAB2 B4** | B4 | 94.14% | 66.45% | 59.18% | 47.80% | 19.35% | 9.87% | 7.47% | 39.72% |
| CAB1 V2S | V2S | 93.79% | 63.88% | 56.11% | 46.57% | 6.86% | 6.03% | 3.81% | 35.63% |
| CAB2 V2S | V2S | 93.95% | 63.05% | 57.04% | 42.91% | 26.24% | 15.82% | 10.65% | 41.17% |
| CubiCasa5k | VGG16 | **95.61%** | 61.65% | **63.46%** | **59.82%** | 41.95% | 36.56% | **13.92%** | **51.89%** |
| Zeng | VGG16 | 95.19% | 58.08% | 59.33% | 56.47% | **43.34%** | **38.97%** | 8.97% | 50.37% |

### Validation Set (4,600 out-of-fold images)

| Architecture | Backbone | Val Acc | No-BG Acc | Walls IoU | Windows | Doors | Stairs | Railings | Macro IoU |
|:-------------|:---------|:-------:|:---------:|:---------:|:-------:|:-----:|:------:|:--------:|:---------:|
| **CAB1 B4** | B4 | 94.64% | **69.55%** | 59.75% | 53.06% | 27.37% | 14.53% | 9.93% | 43.29% |
| **CAB2 B4** | B4 | 94.26% | 66.68% | 58.09% | 46.81% | 19.57% | 9.04% | 7.26% | 39.26% |
| CAB1 V2S | V2S | 93.99% | 64.72% | 55.82% | 45.29% | 6.99% | 6.36% | 4.24% | 35.55% |
| CAB2 V2S | V2S | 94.10% | 63.42% | 56.10% | 42.13% | 26.49% | 15.55% | 11.49% | 41.05% |
| CubiCasa5k | VGG16 | **96.42%** | 66.33% | **67.52%** | **66.07%** | **47.58%** | **57.13%** | **18.97%** | **58.95%** |
| Zeng | VGG16 | 95.79% | 60.86% | 61.93% | 60.47% | 45.79% | 49.12% | 11.39% | 54.07% |

---

## Evaluated Models

| Model | Backbone | Folds | Config | Hyperparameters |
|:------|:---------|:-----:|:-------|:----------------|
| CAB1 B4 | EfficientNetB4 | 10 | `kfold_patch/eval_cab1_b4_cubicasa.py` | hhdc=7, cam=5, aaf=[2,4] |
| CAB2 B4 | EfficientNetB4 | 10 | `kfold_patch/eval_cab2_b4_cubicasa.py` | hhdc=False, cam=3, aaf=[2,4] |
| CAB1 V2S | EfficientNetV2S | 10 | `kfold_patch/eval_cab1_cubicasa.py` | hhdc=7, cam=5, aaf=[2,4] |
| CAB2 V2S | EfficientNetV2S | 10 | `kfold_patch/eval_cab2_cubicasa.py` | hhdc=False, cam=3, aaf=[2,4] |
| CubiCasa5k | VGG16 | 10 | Pre-trained reference | Multi-task heatmap heads |
| Zeng | VGG16 | 10 | Pre-trained reference | Multi-dilation aggregation |

> **Audit Fixes Applied:** E-4 (batch aggregation), E-5 (ε division guards), E-6 (frequency-weight normalization), D-1/D-2 (nearest-neighbor interpolation). See `CODEBASE_AUDIT.md`.
