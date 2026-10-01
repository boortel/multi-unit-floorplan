# Experiment Registry

**Project:** Multi-Unit Floorplan Segmentation · CubiCasa5k  
**Last Updated:** September 24, 2026

---

## Folder Structure

```
results/
├── official/              Audit-verified final evaluations (Sept 8, 2026)
│   ├── test/              400-image CubiCasa5k test set — 6 architectures
│   └── val/               4,600-image out-of-fold cross-validation — 6 architectures
├── ablation/              Single-fold (Fold 0) architecture ablation searches
├── hpo/                   Final Optuna HPO summary (cab1)
├── archive/               Pre-audit, baseline, and intermediate HPO snapshots
├── trash/                 Original verbose MD documents (kept for reference)
│
├── EXPERIMENT_REGISTRY.md           ← You are here
├── hyperparameter_recommendations.md
├── experiment_analysis.md
└── TEST_EVALUATION_GUIDE.md
```

---

## 1. Official Results — Audit-Verified (September 8, 2026)

Ran on 4× NVIDIA A100 GPUs via `run_all_evaluations.sh` (~59 min).  
Audit fixes: E-4 (confusion matrix), E-5 (ε-guards), E-6 (frequency weights), D-1/D-2 (interpolation).

### Test Set (400 images)

| File | Model | Backbone | Test Acc | No-BG Acc | Macro IoU |
|:-----|:------|:---------|:--------:|:---------:|:---------:|
| `test_…cab1_EfficientNetB4_20260908-075715` | **CAB1** | **B4** | **94.52% ±0.94** | **69.19%** ★ | **43.47%** |
| `test_…cab2_EfficientNetB4_20260908-075647` | CAB2 | B4 | 94.14% ±0.86 | 66.45% | 39.72% |
| `test_…cab1_EfficientNetV2S_20260908-080916` | CAB1 | V2S | 93.79% ±0.99 | 63.88% | 35.63% |
| `test_…cab2_EfficientNetV2S_20260908-075927` | CAB2 | V2S | 93.95% ±1.69 | 63.05% | 41.17% |
| `test_…cubicasa5k_VGG16_20260908-072527` | CubiCasa5k | VGG16 | 95.61% ±0.28 | 61.65% | 51.89% |
| `test_…zeng_VGG16_20260908-071630` | Zeng | VGG16 | 95.19% ±0.22 | 58.08% | 50.37% |

★ = Project Record

### Validation Set (4,600 out-of-fold images)

| File | Model | Backbone | Val Acc | No-BG Acc | Macro IoU |
|:-----|:------|:---------|:-------:|:---------:|:---------:|
| `val_…cab1_EfficientNetB4_20260908-073505` | **CAB1** | **B4** | **94.64% ±0.88** | **69.55%** | **43.29%** |
| `val_…cab2_EfficientNetB4_20260908-073515` | CAB2 | B4 | 94.26% ±0.80 | 66.68% | 39.26% |
| `val_…cab1_EfficientNetV2S_20260908-074843` | CAB1 | V2S | 93.99% ±0.89 | 64.72% | 35.55% |
| `val_…cab2_EfficientNetV2S_20260908-073942` | CAB2 | V2S | 94.10% ±1.56 | 63.42% | 41.05% |
| `val_…cubicasa5k_VGG16_20260908-071805` | CubiCasa5k | VGG16 | 96.42% ±0.36 | 66.33% | 58.95% |
| `val_…zeng_VGG16_20260908-071329` | Zeng | VGG16 | 95.79% ±0.32 | 60.86% | 54.07% |

---

## 2. Ablation Results (Fold 0, 40 epochs)

| File | Model | Scope | Winner |
|:-----|:------|:------|:-------|
| `ablation_cab1_fold0_20260817-…` | CAB1 | Backbone scaling | **B4** (val loss 1.5401) |
| `ablation_cab1_fold0_20260822-…` | CAB1 | HHDC & CAM modules | **hhdc=7, cam=5** |
| `ablation_cab2_fold0_20260818-…` | CAB2 | Backbone scaling | **B4** (val loss 1.5328) |
| `ablation_cab2_fold0_20260821-…` | CAB2 | HHDC & CAM modules | **no_hhdc, cam=3** |

---

## 3. Optuna HPO Studies

| Location | Model | Phase | Completed | Best Trial | Val Loss | Backbone |
|:---------|:------|:------|:---------:|:----------:|:--------:|:---------|
| `hpo/hpo_cab1_20260921-…` | CAB1 | Targeted B4 | 12 | **#52** | **-11.93** | **EfficientNetB4** |
| `optuna_cab1.db` | CAB1 | Multi-backbone | 11 | #21 | -11.65 | EfficientNetV2M |
| `optuna_cab2_v2m.db` | CAB2 | Dedicated V2M | 20 | **#32** | **-16.67** | **EfficientNetV2M** |
| `optuna_cab2.db` | CAB2 | Multi-backbone | 6 | #5 | -16.86 | EfficientNetB0 |

---

## 4. Training Runs (10-Fold Cross-Validation)

| Model | Backbone | Log | Duration | Val Acc | Val Loss |
|:------|:---------|:----|:--------:|:-------:|:--------:|
| CAB1 | B4 | `logs/kfold_cab1_b4.log` | 42.2 hrs | 94.86% ±0.71 | 1.91 ±0.21 |
| CAB2 | B4 | `logs/kfold_cab2_b4.log` | 45.0 hrs | 94.46% ±0.78 | 2.01 ±0.20 |
| CAB1 | V2S | `logs/kfold_cab1_v2s.log` | ~45.0 hrs | 94.23% ±0.79 | 2.03 ±0.19 |
| CAB2 | V2S | `logs/kfold_cab2_v2s.log` | ~47.8 hrs | 94.12% ±1.52 | 1.97 ±0.29 |

---

## 5. Archive Contents

Pre-audit evaluations, baselines, and duplicate HPO snapshots. See `archive/` folder.

| Date | Files | Description |
|:----:|:-----:|:------------|
| Aug 4 | 4 | B2 baselines (cab1, cab2, cubicasa5k, zeng) — 0% door/window IoU in CAB |
| Aug 29–30 | 3 | V2S evaluations (pre/post Fold 4 retrain) |
| Sep 2 | 2 | Automated post-training hook (wrong backbone, CPU) |
| Sep 7 | 4 | Pre-audit B4 evaluations |
| Sep 14–16 | 7 | Intermediate HPO snapshots (5 identical, 2 with Trial 21) |

---

## File Naming Convention

```
{split}_kfold_{model}_{backbone}_{YYYYMMDD-HHMMSS}.txt
```

| Field | Values |
|:------|:-------|
| `split` | `test` (400 samples) · `val` (4,600 out-of-fold) |
| `model` | `cab1` · `cab2` · `cubicasa5k` · `zeng` |
| `backbone` | `EfficientNetB4` · `EfficientNetV2S` · `VGG16` |
