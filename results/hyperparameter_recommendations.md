# Hyperparameter Recommendations

**Project:** Multi-Unit Floorplan Segmentation · CubiCasa5k  
**Models:** CAB1 & CAB2 · EfficientNetB4 / EfficientNetV2S / EfficientNetV2M  
**Date:** September 24, 2026 · **Updated:** October 14, 2026

---

## Recommended Configurations

### Best 10-Fold Result: HPO Champions on EfficientNetV2M (Oct 14, 2026)

The two V2M HPO configs below have been promoted to full 10-fold CV and now hold the best measured results (CAB1 V2M: 96.37% test acc / 80.16% no-BG; CAB2 V2M: 96.14% / 61.99% macro IoU). See [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) for numbers and caveats: they were trained after the audit fixes, while the B4 baseline below was trained before them.

| Parameter | CAB1 (Trial 21) | CAB2 (Trial 32) |
|:----------|:-----------------|:-----------------|
| **Config** | `kfold_patch/eval_cab1_hpo_v2m.py` | `kfold_patch/eval_cab2_hpo_v2m.py` |
| **Backbone** | EfficientNetV2M | EfficientNetV2M |
| **HHDC** | False | 3 |
| **CAM** | 5 | 1 |
| **AAF** | [4, 8] | [2, 4, 8] |
| **Decoder Filters** | [16, 32, 64, 128, 256] (small) | [16, 32, 64, 128, 256] (small) |
| **Scheduler** | reduce-lr-on-plateau (2 warmup) | reduce-lr-on-plateau (10 warmup) |
| **Min LR** | 1.83e-7 | 9.64e-6 |
| **Single-Fold Val Loss** | -11.65 | -16.67 |

### Previous Production Setup (B4, 10-Fold Verified Sept 8, pre-audit training)

| Parameter | CAB1 | CAB2 |
|:----------|:-----|:-----|
| **Backbone** | EfficientNetB4 | EfficientNetB4 |
| **HHDC** | 7 (expanded receptive field) | False (skip redundancy removal) |
| **CAM** | 5 (channel recalibration) | 3 (baseline optimal) |
| **AAF** | [2, 4] | [2, 4] |
| **Decoder Filters** | [32, 64, 128, 256, 512] | [32, 64, 128, 256, 512] |
| **Loss** | Unified Focal + Heatmap + AAF + AWL | Unified Focal + Heatmap + AAF + AWL |
| **Optimizer** | Adam, lr=1e-4 | Adam, lr=1e-4 |
| **Scheduler** | cosine-decay-warmup (5 warmup, min 1e-6) | cosine-decay-warmup (5 warmup, min 1e-6) |
| **Epochs** | 100 (early stopping ~32–76) | 100 (early stopping ~31–76) |
| **Batch Size** | 4 | 4 |

### CAB1 B4 HPO Champion (Trial 52) — 10-Fold Evaluated Oct 14

Best single-fold CAB1 trial overall (val loss **-11.9251**). 10-fold Oct 1–4 (`kfold_patch/eval_cab1_hpo_b4.py`). **Test: 95.86% ±1.18 acc, 74.96% no-BG, 58.98% macro IoU.** That is second only to the V2M configs. Fold 4 lags (92.30%) and the other 9 folds score ~96.25%. A lower single-fold val loss (-11.93 vs -11.65 for V2M t21) did **not** translate into better 10-fold test metrics than V2M.

| Parameter | CAB1 (Trial 52) |
|:----------|:-----------------|
| **Backbone** | EfficientNetB4 |
| **HHDC / CAM** | False / 3 |
| **AAF** | [2, 4, 8] |
| **Decoder Filters** | [16, 32, 64, 128, 256] (small) |
| **Scheduler** | cosine-decay-warmup (4 warmup), min LR 7.08e-7 |

---

## 1. Ablation Results (Fold 0, 40 epochs, EfficientNetV1)

### CAB1

| Variant | Val Loss | Val Acc | Δ vs Baseline |
|:--------|:--------:|:-------:|:---------------|
| **Baseline** (B2, hhdc=5, cam=3) | 1.5629 | 95.73% | — |
| B0_small | 1.6051 | 95.31% | +0.042 loss ↓ |
| B3 | 1.5511 | 95.86% | -0.012 loss ↑ |
| **B4** | **1.5401** | **95.92%** | **-0.023 loss ↑ Best backbone** |
| hhdc=7 | 1.5576 | 95.73% | -0.005 loss ↑ Best HHDC |
| cam=5 | 1.5552 | 95.74% | -0.008 loss ↑ Best CAM |

### CAB2

| Variant | Val Loss | Val Acc | Δ vs Baseline |
|:--------|:--------:|:-------:|:---------------|
| **Baseline** (B2, hhdc=5, cam=3) | 1.5481 | 95.84% | — |
| B0_small | 1.5959 | 95.48% | +0.048 loss ↓ |
| B3 | 1.5359 | 95.98% | -0.012 loss ↑ |
| **B4** | **1.5328** | **95.99%** | **-0.015 loss ↑ Best backbone** |
| **no_hhdc** | **1.5462** | **95.87%** | **-0.002 loss ↑ Best (removes redundancy)** |
| cam=3 (baseline) | 1.5481 | 95.84% | Best CAM (any change degrades) |

---

## 2. EfficientNetV1 vs V2 — 10-Fold Test Comparison

*B4 vs V2S with the fixed ablation configs (Sept 8). For the V2M HPO results see [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md).*

| Metric | CAB1: B4 | CAB1: V2S | Δ | CAB2: B4 | CAB2: V2S | Δ |
|:-------|:--------:|:---------:|:-:|:--------:|:---------:|:-:|
| Test Acc | **94.52%** | 93.79% | +0.73 | **94.14%** | 93.95% | +0.19 |
| No-BG Acc | **69.19%** | 63.88% | **+5.31** | **66.45%** | 63.05% | **+3.40** |
| Walls IoU | **60.63%** | 56.11% | +4.52 | **59.18%** | 57.04% | +2.14 |
| Windows IoU | **53.92%** | 46.57% | +7.35 | **47.80%** | 42.91% | +4.89 |
| Doors IoU | **27.43%** | 6.86% | **+20.57 (4×)** | 19.35% | **26.24%** | -6.89 |
| Stairs IoU | **14.69%** | 6.03% | +8.66 | 9.87% | **15.82%** | -5.95 |
| Fold Std | ±0.94% | ±0.99% | — | **±0.86%** | ±1.69% | **2× lower** |
| Parameters | **8.75M** | 9.94M | -12% | **8.04M** | 9.34M | -14% |

### V1 vs V2 Architecture Differences

| Feature | EfficientNetV1 (B4) | EfficientNetV2 (V2S) |
|:--------|:---------------------|:----------------------|
| Encoder params | 17.67M | 20.33M |
| Early blocks | Depthwise separable (MBConv) | Fused-MBConv (conv 3×3 + 1×1) |
| Preprocessing | Scaled [0, 255] | Rescaling to [-1, 1] |

---

## 3. Optuna HPO Results

### CAB1 — Top 6 Completed Trials

| Rank | Trial | Backbone | Val Loss | Filters | HHDC | CAM | AAF | Scheduler |
|:----:|:-----:|:---------|:--------:|:--------|:----:|:---:|:---:|:----------|
| **1** | **52** | **EfficientNetB4** | **-11.93** | small | False | 3 | [2,4,8] | cosine-warmup |
| 2 | 21 | EfficientNetV2M | -11.65 | small | False | 5 | [4,8] | plateau |
| 3 | 12 | EfficientNetV2M | -11.39 | small | False | 5 | [4,8] | cosine-warmup |
| 4 | 9 | EfficientNetV2M | -11.33 | small | False | 5 | [4,8] | cosine-warmup |
| 5 | 13 | EfficientNetV2M | -11.21 | small | 5 | 5 | [4,8] | cosine-warmup |
| 6 | 10 | EfficientNetB3 | -10.14 | small | False | 3 | [4,8] | cosine-warmup |

*53 total trials (12 completed, 32 pruned, 9 failed). Source: `optuna_cab1.db`, summary `hpo/hpo_cab1_20260921-133827.txt`*

### CAB2 — Multi-Backbone Study (Top 3)

| Rank | Trial | Backbone | Val Loss | Filters | HHDC | CAM | AAF | Scheduler |
|:----:|:-----:|:---------|:--------:|:--------|:----:|:---:|:---:|:----------|
| **1** | **5** | **EfficientNetB0** | **-16.86** | base | 7 | False | [2,4,8] | plateau |
| 2 | 2 | EfficientNetV2S | -16.43 | large | 5 | False | [2,4,8] | cosine |
| 3 | 8 | EfficientNetV2M | -11.41 | small | False | 5 | [4,8] | cosine-warmup |

*10 total trials (6 completed). Source: `optuna_cab2.db`*

### CAB2 — Dedicated V2M Study (Top 5 of 23 completed)

| Rank | Trial | Val Loss | Filters | HHDC | CAM | AAF | Scheduler |
|:----:|:-----:|:--------:|:--------|:----:|:---:|:---:|:----------|
| **1** | **34** | **-16.680** | small | 3 | 1 | [2,4,8] | plateau |
| 2 | 32 ← used for 10-fold | -16.673 | small | 3 | 1 | [2,4,8] | plateau |
| 3 | 33 | -16.658 | small | 3 | 1 | [2,4,8] | plateau |
| 4 | 30 | -16.652 | small | 3 | 1 | [2,4,8] | plateau |
| 5 | 35 | -16.632 | small | 3 | 1 | [2,4,8] | plateau |

*37 total trials (23 completed, 13 pruned, 1 failed). Source: `optuna_cab2_v2m.db`, summary `hpo/hpo_cab2_20260924-220154.txt`*  
*Top 5 differ only in `lr_min` / `warmup_epochs`. The top 12 all share small filters, hhdc=3, cam=1, aaf=[2,4,8]. Plateau (≈ -16.6) clearly beats cosine-decay (≈ -12.8).*

### Key HPO Insights

- **Decoder footprint matters:** Deep encoders (B4, V2M) perform best with `small` filters [16..256], preventing overfitting.
- **Broad affinity supervision:** `aaf=[2,4,8]` consistently dominates over `[2,4]` in HPO.
- **Scheduler divergence:** CAB1 favors cosine-warmup; CAB2 benefits from reduce-on-plateau (+3.8 loss points).
- **Convergence reached:** Both CAB1 and CAB2 search spaces are fully converged — no further HPO needed.

---

## 4. Production Config Files

### CAB1 B4 — `kfold_patch/eval_cab1_b4_cubicasa.py`

```python
model_type = 'cab1'
backbone = 'EfficientNetB4'
filters = [32, 64, 128, 256, 512]
hhdc = 7          # Expanded receptive field
cam = 5           # Channel attention scale
aaf = [2, 4]
batch_size = 4
epochs = 100
lr_scheduler = 'cosine-decay-warmup'
lr_min = 1e-6
warmup_epochs = 5
kFold = 10
```

### CAB2 B4 — `kfold_patch/eval_cab2_b4_cubicasa.py`

```python
model_type = 'cab2'
backbone = 'EfficientNetB4'
filters = [32, 64, 128, 256, 512]
hhdc = False      # Disabled (skip redundancy)
cam = 3           # Baseline channel attention
aaf = [2, 4]
batch_size = 4
epochs = 100
lr_scheduler = 'cosine-decay-warmup'
lr_min = 1e-6
warmup_epochs = 5
kFold = 10
```

### CAB1 HPO Champion — Trial 52 (val_loss = -11.93)

```python
model_type = 'cab1'
backbone = 'EfficientNetB4'
filters = [16, 32, 64, 128, 256]  # small preset
hhdc = False
cam = 3
aaf = [2, 4, 8]                   # broad affinity
lr_scheduler = 'cosine-decay-warmup'
lr_min = 7.08e-07
warmup_epochs = 4
```

### CAB2 HPO Champion — Trial 32 (val_loss = -16.67)

```python
model_type = 'cab2'
backbone = 'EfficientNetV2M'
filters = [16, 32, 64, 128, 256]  # small preset
hhdc = 3                          # tuned context dilation
cam = 1                           # subtle attention
aaf = [2, 4, 8]                   # broad affinity
lr_scheduler = 'reduce-lr-on-plateau'  # critical for CAB2
lr_min = 9.64e-06
warmup_epochs = 10
```

---

## 5. Important: Why 10-Fold CV and HPO Results Differ

These two experimental regimes have different configurations and **should not be directly equated:**

The comparison below is between the Sept 8 B4/V2S 10-fold runs and the Optuna search. The October V2M 10-fold runs used the HPO configs directly.

| Aspect | 10-Fold CV (Sections 1–2) | Optuna HPO (Section 3) |
|:-------|:--------------------------|:-----------------------|
| Decoder | Fixed `base` [32..512] | Variable (`small`/`base`/`large`) |
| AAF | Fixed [2, 4] | Variable ([2,4], [4,8], [2,4,8]) |
| Scheduler | Fixed cosine-warmup | Variable (cosine/plateau/etc.) |
| Scope | 10 folds, 100 epochs, 5,000 images | 1 fold, 60 epochs, MedianPruner |
| Metric | Test accuracy, per-class IoU | Total val_loss (negative due to AWL uncertainty) |

> **Status (Oct 14, 2026):** The V2M champions (CAB1 t21, CAB2 t32) have been promoted to 10-fold and evaluated. The October B4 runs were evaluated the same day. Audit fixes + HPO explain most of the gain over the Sept 8 B4 baseline, and V2M adds a smaller margin on top (see the analysis doc, Section 0.3). The CAB2 B4 retrain used the old ablation config and was unstable (5 of 10 folds ≤91.5%), so it isn't a fair B4 reference for CAB2.
