# Experiment Analysis & 10-Fold Cross-Validation

**Project:** Multi-Unit Floorplan Segmentation · CubiCasa5k  
**Models:** CAB1 & CAB2 (EfficientNetV2M HPO / EfficientNetB4 / EfficientNetV2S) vs. CubiCasa5k & Zeng (VGG16)  
**Date:** September 8, 2026 · **Updated:** October 14, 2026 (Section 0)

---

## Key Findings

**Current (Oct 14, 2026):**

1. **The HPO-tuned EfficientNetV2M models lead on every aggregate test metric.** CAB1 V2M (trial 21): 96.37% test acc, **80.16% no-BG acc**. CAB2 V2M (trial 32): 96.14% test acc, **61.99% macro IoU**. Both beat CubiCasa5k (95.61% / 61.65% / 51.89%) on overall accuracy, no-BG accuracy and macro IoU.
2. **The weak small classes improved the most.** Test door IoU went from 27.43% → 56.18% / 59.17%, stairs 14.69% → 47.90% / 49.93%, railings 9.16% → 27.48% / 28.03% (CAB1 B4 → CAB1 / CAB2 V2M). These are the first CAB models to beat CubiCasa5k and Zeng on doors and stairs.
3. **Fold variance collapsed.** Test-acc std across folds is ±0.03% (CAB1) / ±0.10% (CAB2), versus ±0.86–1.69% for the B4/V2S runs. The Fold-4-style collapses are gone.
4. **Most of the gain comes from the audit fixes + HPO, not the backbone.** The post-audit CAB1 B4 HPO model (trial 52) reaches 95.86% / 74.96% no-BG / 58.98% macro IoU, versus 94.52% / 69.19% / 43.47% for the Sept 8 B4. So +15.5 macro-IoU points come without switching backbone. V2M adds another +2.4 macro IoU and +5.2 no-BG on top (Section 0.3). The CAB1 B4 and V2M HPO configs also differ, so this isn't a pure backbone comparison.

**Historical (Sept 8, 2026 — pre-audit-retraining models):**

1. CAB1 B4 reached 69.19% non-background test accuracy, +7.54% over CubiCasa5k and +11.11% over Zeng.
2. B4 outperformed V2S: +5.31% non-background accuracy on CAB1, +3.40% on CAB2, with 4× higher door IoU (27.43% vs 6.86%).
3. Val and test metrics agreed within <0.5% across all structural classes.
4. At that point CubiCasa5k still led on overall accuracy (95.61%) and macro IoU (51.89%), but it missed 33% of wall pixels.

---

## 0. HPO Champion Models — EfficientNetV2M (October 2026)

### 0.1 Setup

| | CAB1 V2M | CAB2 V2M |
|:--|:--|:--|
| Optuna source | `optuna_cab1.db`, trial 21 (val loss -11.65) | `optuna_cab2_v2m.db`, trial 32 (val loss -16.67) |
| Config | `kfold_patch/eval_cab1_hpo_v2m.py` | `kfold_patch/eval_cab2_hpo_v2m.py` |
| Decoder filters | [16, 32, 64, 128, 256] | [16, 32, 64, 128, 256] |
| HHDC / CAM | False / 5 | 3 / 1 |
| AAF | [4, 8] | [2, 4, 8] |
| Scheduler | reduce-lr-on-plateau, lr_min 1.83e-7, warmup 2 | reduce-lr-on-plateau, lr_min 9.64e-6, warmup 10 |
| Training | Oct 4 – Oct 7, 77.9 h, sequential (`scripts/run_optimal_all_four.sh`) | Oct 11 – Oct 13, folds 4–9 in parallel on 3 GPUs (`kfold_patch/parallel_cab2_orchestrator.py`) |
| Train-time val acc | 96.64% ±0.06 | — (orchestrator summary unreliable, see registry) |

### 0.2 Test Set (400 images) — Per-Class

| Class | CAB1 V2M Recall | CAB1 V2M IoU | CAB2 V2M Recall | CAB2 V2M IoU | CAB1 B4 IoU (Sep 8) | CubiCasa5k IoU |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| Background | 98.30% | 96.56% | 98.49% | 96.23% | 94.98% | — |
| Walls | **83.70%** | **72.82%** | 78.63% | 68.66% | 60.63% | 63.46% |
| Windows | 84.45% | 67.26% | 82.93% | **69.91%** | 53.92% | 59.82% |
| Doors | 75.92% | 56.18% | 73.53% | **59.17%** | 27.43% | 41.95% |
| Stairs | 56.31% | 47.90% | 58.35% | **49.93%** | 14.69% | 36.56% |
| Railings | 37.77% | 27.48% | 37.19% | **28.03%** | 9.16% | 13.92% |
| **Accuracy / No-BG** | | **96.37% / 80.16%** | | 96.14% / 76.39% | 94.52% / 69.19% | 95.61% / 61.65% |
| **Macro IoU** | | 61.37% | | **61.99%** | 43.47% | 51.89% |

CAB1 V2M is stronger on walls and no-BG accuracy, with higher recall overall. CAB2 V2M has higher precision and is stronger on windows, doors, stairs and railings. Val results (4,600 OOF images) agree with test within ~2 IoU points on every class (largest gap: railings). See `official/val/`.

### 0.3 October B4 Runs — Post-Audit B4 vs V2M (evaluated Oct 14)

| Test (400 images) | CAB1 B4 Sep 8 (pre-audit) | CAB1 B4 HPO t52 (Oct) | CAB1 V2M HPO t21 (Oct) | CAB2 B4 retrain (Oct) ⚠️ | CAB2 V2M HPO t32 (Oct) |
|:--|:--:|:--:|:--:|:--:|:--:|
| Test Acc | 94.52% ±0.94 | 95.86% ±1.18 | **96.37% ±0.03** | 92.70% ±2.18 | 96.14% ±0.10 |
| No-BG Acc | 69.19% | 74.96% | **80.16%** | 57.40% | 76.39% |
| Walls IoU | 60.63% | 67.77% | **72.82%** | 48.53% | 68.66% |
| Windows IoU | 53.92% | 63.04% | 67.26% | 36.13% | **69.91%** |
| Doors IoU | 27.43% | 53.50% | 56.18% | 16.51% | **59.17%** |
| Stairs IoU | 14.69% | 46.07% | 47.90% | 8.63% | **49.93%** |
| Railings IoU | 9.16% | 27.48% | 27.48% | 6.23% | **28.03%** |
| Macro IoU | 43.47% | 58.98% | 61.37% | 34.89% | **61.99%** |

* **CAB1 B4 t52:** 9 of 10 folds score 96.20–96.29% test acc. Fold 4 lags at 92.30%, the same fold that collapsed in the Aug V2S run, and that explains the ±1.18 std. Even with all folds, B4 t52 closes ~85% of the Sept-8-B4 → V2M macro-IoU gap. The remaining V2M advantage is mostly walls (+5.1) and no-BG accuracy (+5.2).
* **CAB2 B4 retrain:** test acc swings from 89.4% to 96.5% across folds, and 5 folds (0, 2, 4, 5, 6) are ≤91.5%. Fold 1 (96.5%) shows the setup can work but trains unstably. This run used the **pre-HPO ablation config** (base filters, aaf=[2,4], cosine-warmup) and is worse than the Sept 8 CAB2 B4 run. It doesn't measure the B4 backbone under the CAB2 HPO config, so a CAB2 B4 run with the trial-32-style config would be needed for a clean comparison.
* Val results agree with test for all three runs (`official/val/`).

### 0.4 Open Items

* **CAB1 fold 4** is a recurring weak fold (V2S collapse in Aug, B4 t52 lag in Oct). Its data split may be worth inspecting.
* **Reference baselines:** CubiCasa5k / Zeng weights predate the audit fixes too. For a strictly fair comparison they would need retraining under the current pipeline.
* **Evaluator ambiguity:** `--backbone EfficientNetB4` now resolves to the October runs. See [TEST_EVALUATION_GUIDE.md](TEST_EVALUATION_GUIDE.md) ("Checkpoint selection").

---

> Sections 1–5 below document the **September 8, 2026** state (B4 / V2S models trained before the audit fixes).

---

## 1. Training History

### CAB1 B4 — Mean Val Acc: **94.86% ±0.71%**, Val Loss: **1.91 ±0.21** (42.2 hrs)

| Fold | Stopped | Val Loss | Val Acc |
|:----:|:-------:|:--------:|:-------:|
| 0 | Ep 55 | 1.9463 | 94.56% |
| 1 | Ep 32 | 2.0506 | 94.56% |
| 2 | Ep 33 | 1.8893 | 94.90% |
| 3 | Ep 76 | 2.1479 | 93.96% |
| 4 | Ep 66 | 2.0723 | 94.66% |
| 5 | Ep 43 | 1.5101 | 96.31% |
| 6 | Ep 37 | 1.5307 | 96.03% |
| 7 | Ep 37 | 1.9983 | 94.56% |
| 8 | Ep 32 | 1.8592 | 94.84% |
| 9 | Ep 39 | 2.0676 | 94.24% |

### CAB2 B4 — Mean Val Acc: **94.46% ±0.78%**, Val Loss: **2.01 ±0.20** (45.0 hrs)

| Fold | Stopped | Val Loss | Val Acc |
|:----:|:-------:|:--------:|:-------:|
| 0 | Ep 51 | 1.9674 | 94.68% |
| 1 | Ep 59 | 2.0845 | 94.26% |
| 2 | Ep 48 | 2.1755 | 93.49% |
| 3 | Ep 76 | 1.4805 | 96.51% |
| 4 | Ep 32 | 1.8897 | 94.85% |
| 5 | Ep 43 | 2.0860 | 94.39% |
| 6 | Ep 48 | 2.1395 | 93.99% |
| 7 | Ep 31 | 2.1195 | 94.06% |
| 8 | Ep 32 | 1.9870 | 94.56% |
| 9 | Ep 60 | 2.1786 | 93.85% |

### CAB1 V2S — Mean Val Acc: **94.23% ±0.79%**, Val Loss: **2.03 ±0.19**

| Fold | Stopped | Val Loss | Val Acc | Note |
|:----:|:-------:|:--------:|:-------:|:-----|
| 0 | Ep 99 | 2.1605 | 93.77% | |
| 1 | Ep 31 | 2.0306 | 94.48% | |
| 2 | Ep 52 | 2.3140 | 93.13% | |
| 3 | Ep 32 | 2.0133 | 94.10% | |
| 4 | Ep 41 | 2.0727 | 94.42% | Retrained (Fold 4 collapse fix) |
| 5 | Ep 36 | 2.0586 | 93.84% | |
| 6 | Ep 43 | 1.9734 | 94.48% | |
| 7 | Ep 36 | 1.5295 | 96.15% | |
| 8 | Ep 31 | 2.1022 | 94.33% | |
| 9 | Ep 31 | 2.0285 | 93.63% | |

### CAB2 V2S — Mean Val Acc: **94.12% ±1.52%**, Val Loss: **1.97 ±0.29**

| Fold | Stopped | Val Loss | Val Acc | Note |
|:----:|:-------:|:--------:|:-------:|:-----|
| 0 | Ep 35 | 1.5400 | 95.95% | |
| 1 | Ep 34 | 1.5552 | 95.75% | |
| 2 | Max 100 | 2.3492 | 91.20% | No early stop |
| 3 | Ep 31 | 2.0865 | 94.21% | |
| 4 | Ep 53 | 2.0125 | 94.80% | |
| 5 | Ep 63 | 1.9825 | 94.53% | |
| 6 | Ep 31 | 1.7943 | 94.95% | |
| 7 | Ep 34 | 2.2493 | 92.84% | |
| 8 | Ep 31 | 1.7697 | 95.00% | |
| 9 | Max 100 | 2.3818 | 91.91% | No early stop |

---

## 2. Fold 4 Collapse — Root Cause & Fix

**Bug:** `tf.data.experimental.cardinality()` returned `UNKNOWN` on concatenated datasets, causing fallback to `train_dataset_size = 400` instead of the true **4,140 samples**.

**Effect:** `steps_per_epoch` = 100 (should be 1,035) → cosine decay exhausted LR to 1e-6 within ~10 epochs → optimizer couldn't escape poor initial gradient.

**Fix:** Hardcoded `4,140` samples in `train_config.py` and `kfold_patch/train_config_kfold.py` when cardinality is unknown. Fold 4 retrained successfully (Val Acc 94.42%).

---

## 3. Official Test Set Results (September 8, 2026)

### CAB1 EfficientNetB4 — **94.52% ±0.94%** Test Acc, **69.19% No-BG** ★

| Class | Recall | Precision | F1 | IoU |
|:------|:------:|:---------:|:--:|:---:|
| Background | 97.53% | 97.32% | 97.43% | 94.98% |
| **Walls** | **79.78%** | 71.64% | 75.49% | **60.63%** |
| Windows | 70.48% | 69.65% | 70.06% | 53.92% |
| Doors | 32.90% | 62.28% | 43.06% | 27.43% |
| Stairs | 16.53% | 56.97% | 25.62% | 14.69% |
| Railings | 10.64% | 39.79% | 16.79% | 9.16% |

### CAB2 EfficientNetB4 — **94.14% ±0.86%** Test Acc, **66.45% No-BG**

| Class | Recall | Precision | F1 | IoU |
|:------|:------:|:---------:|:--:|:---:|
| Background | 97.43% | 97.08% | 97.25% | 94.65% |
| **Walls** | **78.05%** | 70.99% | 74.36% | **59.18%** |
| Windows | 67.69% | 61.93% | 64.68% | 47.80% |
| Doors | 22.07% | 61.04% | 32.42% | 19.35% |
| Stairs | 11.08% | 47.62% | 17.97% | 9.87% |
| Railings | 8.46% | 38.91% | 13.90% | 7.47% |

### CAB1 V2S — **93.79% ±0.99%**, 63.88% No-BG | CAB2 V2S — **93.95% ±1.69%**, 63.05% No-BG

*(Detailed per-class metrics in `official/test/` files)*

---

## 4. Cross-Architecture Comparison

### B4 vs. V2S (within CAB models)

| Metric | CAB1 B4 | CAB1 V2S | Δ | CAB2 B4 | CAB2 V2S | Δ |
|:-------|:-------:|:--------:|:-:|:-------:|:--------:|:-:|
| Test Acc | 94.52% | 93.79% | **+0.73** | 94.14% | 93.95% | +0.19 |
| No-BG Acc | **69.19%** | 63.88% | **+5.31** | **66.45%** | 63.05% | **+3.40** |
| Walls IoU | 60.63% | 56.11% | +4.52 | 59.18% | 57.04% | +2.14 |
| Windows IoU | 53.92% | 46.57% | +7.35 | 47.80% | 42.91% | +4.89 |
| Doors IoU | 27.43% | 6.86% | **+20.57** | 19.35% | 26.24% | -6.89 |
| Stairs IoU | 14.69% | 6.03% | +8.66 | 9.87% | 15.82% | -5.95 |
| Fold Std | ±0.94% | ±0.99% | ≈ | ±0.86% | ±1.69% | **2× lower** |

### CAB Models vs. Reference Benchmarks

| Metric | CAB1 B4 | CubiCasa5k | Zeng | CAB1 Advantage |
|:-------|:-------:|:----------:|:----:|:---------------|
| No-BG Accuracy | **69.19%** | 61.65% | 58.08% | +7.54% / +11.11% |
| Walls Recall | **79.78%** | 66.54% | — | +13.24% (40% fewer missed walls) |
| Windows Recall | **70.49%** | 66.03% | — | +4.46% |
| Overall Test Acc | 94.52% | **95.61%** | 95.19% | CubiCasa leads (background bias) |
| Macro IoU | 43.47% | **51.89%** | 50.37% | CubiCasa leads (heatmap heads) |

---

## 5. Why CAB Models Are Superior for Real-World Use

Despite CubiCasa5k having higher overall accuracy (~88% of pixels are background), **CAB models win on what matters for downstream applications:**

1. **Wall completeness** — CubiCasa misses 33% of wall pixels; CAB1 B4 misses only 20% (40% reduction)
2. **Boundary continuity** — Adaptive Affinity Fields (`aaf=[2,4]`) enforce continuous wall segments, preventing pinhole gaps
3. **Foreground accuracy** — 69.19% vs 61.65% on structural elements
4. **Modern architecture** — EfficientNet replaces aging VGG16 (2014), with channel attention (CAM) and context aggregation (HHDC)

| Application | Best Model | Why |
|:------------|:-----------|:----|
| CAD vectorization & wall polygonization | **CAB1/CAB2** | High wall recall + AAF prevent disconnected boundaries |
| 3D mesh generation | **CAB1/CAB2** | Continuous walls without 33% gaps |
| Foreground completeness | **CAB1 B4** | 69.19% vs 61.65% |
| Conservative icon detection | **CubiCasa5k** | Auxiliary heatmap heads for small elements |

---

## Ablation Study Verification

All ablation recommendations were verified in production configs:

| Parameter | CAB1 Recommendation | CAB1 Config | CAB2 Recommendation | CAB2 Config |
|:----------|:-------------------:|:-----------:|:-------------------:|:-----------:|
| Backbone | EfficientNetB4 | ✓ B4 | EfficientNetB4 | ✓ B4 |
| HHDC | hhdc=7 | ✓ 7 | hhdc=False | ✓ False |
| CAM | cam=5 | ✓ 5 | cam=3 | ✓ 3 |
| AAF | [2, 4] | ✓ | [2, 4] | ✓ |
| Decoder | [32,64,128,256,512] | ✓ | [32,64,128,256,512] | ✓ |
| Scheduler | cosine-decay-warmup | ✓ | cosine-decay-warmup | ✓ |
