# Results & Experiment Registry Manifest
**Project:** Multi-Unit Floorplan Segmentation  
**Dataset:** CubiCasa5k  
**Last Updated:** September 8, 2026  

---

## 1. Quick Reference: Results File to Experiment Mapping

| Result File | Experiment Name | Model | Backbone | Split | Key Hyperparameters | Date Generated | Status / Key Metric |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| [`test_kfold_cab1_EfficientNetB4_20260908-075715.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260908-075715.txt) | **CAB1 B4 10-Fold Test (Audit Verified)** | CAB1 | `EfficientNetB4` | Test | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-08 | **94.52% ± 0.94%** Test Acc, **69.19% No-BG Acc** (Project Record!) |
| [`val_kfold_cab1_EfficientNetB4_20260908-073505.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetB4_20260908-073505.txt) | **CAB1 B4 10-Fold Val (Out-Of-Fold)** | CAB1 | `EfficientNetB4` | Val | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-08 | **94.64% ± 0.88%** Val Acc, **69.55% No-BG Acc** (4,600 samples) |
| [`test_kfold_cab2_EfficientNetB4_20260908-075647.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260908-075647.txt) | **CAB2 B4 10-Fold Test (Audit Verified)** | CAB2 | `EfficientNetB4` | Test | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-08 | **94.14% ± 0.86%** Test Acc, **66.45% No-BG Acc** |
| [`val_kfold_cab2_EfficientNetB4_20260908-073515.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetB4_20260908-073515.txt) | **CAB2 B4 10-Fold Val (Out-Of-Fold)** | CAB2 | `EfficientNetB4` | Val | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-08 | **94.26% ± 0.80%** Val Acc, **66.68% No-BG Acc** |
| [`test_kfold_cab1_EfficientNetV2S_20260908-080916.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetV2S_20260908-080916.txt) | **CAB1 V2S 10-Fold Test (Audit Verified)** | CAB1 | `EfficientNetV2S` | Test | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-08 | **93.79% ± 0.99%** Test Acc, **63.88% No-BG Acc** |
| [`val_kfold_cab1_EfficientNetV2S_20260908-074843.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetV2S_20260908-074843.txt) | **CAB1 V2S 10-Fold Val (Out-Of-Fold)** | CAB1 | `EfficientNetV2S` | Val | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-08 | **93.99% ± 0.89%** Val Acc, **64.72% No-BG Acc** |
| [`test_kfold_cab2_EfficientNetV2S_20260908-075927.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetV2S_20260908-075927.txt) | **CAB2 V2S 10-Fold Test (Audit Verified)** | CAB2 | `EfficientNetV2S` | Test | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-08 | **93.95% ± 1.69%** Test Acc, **63.05% No-BG Acc** |
| [`val_kfold_cab2_EfficientNetV2S_20260908-073942.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetV2S_20260908-073942.txt) | **CAB2 V2S 10-Fold Val (Out-Of-Fold)** | CAB2 | `EfficientNetV2S` | Val | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-08 | **94.10% ± 1.56%** Val Acc, **63.42% No-BG Acc** |
| [`test_kfold_cubicasa5k_VGG16_20260908-072527.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260908-072527.txt) | **CubiCasa5k VGG16 10-Fold Test (Audit)** | CubiCasa | `VGG16` | Test | Multi-task heatmap heads | 2026-09-08 | 95.61% ± 0.28% Test Acc, 61.65% No-BG Acc, 51.89% Macro IoU |
| [`val_kfold_cubicasa5k_VGG16_20260908-071805.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cubicasa5k_VGG16_20260908-071805.txt) | **CubiCasa5k VGG16 10-Fold Val (Out-Of-Fold)** | CubiCasa | `VGG16` | Val | Multi-task heatmap heads | 2026-09-08 | **96.42% ± 0.36%** Val Acc, **66.33% No-BG Acc**, 58.95% Macro IoU |
| [`test_kfold_zeng_VGG16_20260908-071630.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260908-071630.txt) | **Zeng VGG16 10-Fold Test (Audit)** | Zeng | `VGG16` | Test | Multi-dilation feature aggregation | 2026-09-08 | 95.19% ± 0.22% Test Acc, 58.08% No-BG Acc, 50.37% Macro IoU |
| [`val_kfold_zeng_VGG16_20260908-071329.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_zeng_VGG16_20260908-071329.txt) | **Zeng VGG16 10-Fold Val (Out-Of-Fold)** | Zeng | `VGG16` | Val | Multi-dilation feature aggregation | 2026-09-08 | **95.79% ± 0.32%** Val Acc, **60.86% No-BG Acc**, 54.07% Macro IoU |
| [`test_kfold_cab1_EfficientNetB4_20260907-075825.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt) | CAB1 B4 10-Fold Test (Pre-Audit) | CAB1 | `EfficientNetB4` | Test | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-07 | Pre-audit evaluation run |
| [`test_kfold_cab2_EfficientNetB4_20260907-082006.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt) | CAB2 B4 10-Fold Test (Pre-Audit) | CAB2 | `EfficientNetB4` | Test | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-07 | Pre-audit evaluation run |
| [`logs/kfold_cab1_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab1_b4.log) | **CAB1 B4 10-Fold Training** | CAB1 | `EfficientNetB4` | Train | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-01 | **10-Fold CV Finished**: 94.86% ± 0.71% Val Acc, 1.9073 ± 0.2106 Val Loss |
| [`logs/kfold_cab2_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab2_b4.log) | **CAB2 B4 10-Fold Training** | CAB2 | `EfficientNetB4` | Train | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-01 | **10-Fold CV Finished**: 94.46% ± 0.78% Val Acc, 2.0108 ± 0.1983 Val Loss |
| [`test_kfold_cab1_20260830-211306.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260830-211306.txt) | CAB1 V2S 10-Fold Test (Pre-Audit) | CAB1 | `EfficientNetV2S` | Test | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-08-30 | Pre-audit V2S evaluation |
| [`test_kfold_cab2_20260829-202421.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260829-202421.txt) | CAB2 V2S 10-Fold Test (Pre-Audit) | CAB2 | `EfficientNetV2S` | Test | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-08-29 | Pre-audit V2S evaluation |
| [`test_kfold_cab1_20260804-105154.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260804-105154.txt) | CAB1 B2 10-Fold (Baseline) | CAB1 | `EfficientNetB2` | Test | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-04 | Baseline: 94.64% Test Acc (0% door/window IoU) |
| [`test_kfold_cab2_20260804-112730.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260804-112730.txt) | CAB2 B2 10-Fold (Baseline) | CAB2 | `EfficientNetB2` | Test | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-04 | Baseline: 94.68% Test Acc |
| [`ablation_cab1_fold0_20260817-032305.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab1_fold0_20260817-032305.txt) | CAB1 Fold 0 Backbone Ablation | CAB1 | `B0, B2, B3, B4` | Val | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-17 | **B4 Best Backbone** (1.5401 val loss) |
| [`ablation_cab1_fold0_20260822-032911.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab1_fold0_20260822-032911.txt) | CAB1 Fold 0 Context & CAM Ablation | CAB1 | `EfficientNetB2` | Val | `hhdc={0,3,7}`, `cam={0,1,5}` | 2026-08-22 | **`hhdc=7`** (1.5576 loss), **`cam=5`** (1.5552 loss) Best |
| [`ablation_cab2_fold0_20260818-175528.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab2_fold0_20260818-175528.txt) | CAB2 Fold 0 Backbone Ablation | CAB2 | `B0, B2, B3, B4` | Val | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-18 | **B4 Best Backbone** (1.5328 val loss) |
| [`ablation_cab2_fold0_20260821-191722.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab2_fold0_20260821-191722.txt) | CAB2 Fold 0 Context & CAM Ablation | CAB2 | `EfficientNetB2` | Val | `hhdc={0,3,7}`, `cam={0,1,5}` | 2026-08-21 | **`no_hhdc`** (1.5462 loss), **`cam=3`** (1.5481 loss) Best |
| [`hyperparameter_recommendations.md`](file:///workspaces/multi-unit-floorplan/results/hyperparameter_recommendations.md) | V1/V2 Hyperparameter Analysis | All | `B2, B4, V2S` | All | Master recommendations & audit findings | 2026-09-08 | Master guide for V1 & V2 setups |
| [`experiment_analysis_and_kfold_evaluation.md`](file:///workspaces/multi-unit-floorplan/results/experiment_analysis_and_kfold_evaluation.md) | 10-Fold Post-Mortem & Multi-Model Comparison | All | `B4, V2S, VGG16` | Val & Test | Full class-by-class comparison | 2026-09-08 | Comprehensive performance analysis |
| [`TEST_EVALUATION_GUIDE.md`](file:///workspaces/multi-unit-floorplan/results/TEST_EVALUATION_GUIDE.md) | Evaluation Runbook & Rebuild Guide | All | `B4, V2S, B2, VGG16` | Val & Test | Execution instructions for 4 GPUs | 2026-09-08 | 4-GPU parallel evaluation runbook |

---

## 2. Detailed Experiment Profiles & Resource Tracking

### 2.1 Full 4-GPU Cross-Validation & Test Re-Evaluation with Audit Fixes (September 8, 2026)

* **Status:** **Completed Successfully** across all 6 architectures for both **Out-of-Fold Validation (4,600 floorplans)** and the **Official Test Set (400 floorplans)**.
* **Runner Script:** [`run_all_evaluations.sh`](file:///workspaces/multi-unit-floorplan/run_all_evaluations.sh) executing concurrently across all 4 NVIDIA A100 GPUs:
  * **GPU 0:** CAB1 EfficientNetB4 (Val & Test) → `logs/eval_cab1_b4.log`
  * **GPU 1:** CAB2 EfficientNetB4 (Val & Test) → `logs/eval_cab2_b4.log`
  * **GPU 2:** CubiCasa5k VGG16 → CAB1 EfficientNetV2S (Val & Test) → `logs/eval_cubicasa5k_vgg16.log`, `logs/eval_cab1_v2s.log`
  * **GPU 3:** Zeng VGG16 → CAB2 EfficientNetV2S (Val & Test) → `logs/eval_zeng_vgg16.log`, `logs/eval_cab2_v2s.log`
* **Fixes from CODEBASE_AUDIT.md in Effect:**
  * **E-4:** Confusion matrix batch aggregation without batch-0 truncation.
  * **E-5:** Epsilon division-by-zero guards ($\epsilon = 10^{-7}$).
  * **E-6:** Frequency weighting (`fw`) normalized using ground-truth totals ($\text{TP} + \text{FN}$) instead of $\text{TP}$ alone.
  * **D-1 & D-2:** Nearest-neighbor discrete mask interpolation and aspect-ratio preservation active in dataset pipelines.
* **Results Artifacts Generated:**
  * **CAB1 B4 Test:** [`results/test_kfold_cab1_EfficientNetB4_20260908-075715.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260908-075715.txt) — **94.52% ± 0.94%** Test Acc, **69.19% No-BG Acc**
  * **CAB1 B4 Val:** [`results/val_kfold_cab1_EfficientNetB4_20260908-073505.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetB4_20260908-073505.txt) — **94.64% ± 0.88%** Val Acc, **69.55% No-BG Acc**
  * **CAB2 B4 Test:** [`results/test_kfold_cab2_EfficientNetB4_20260908-075647.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260908-075647.txt) — **94.14% ± 0.86%** Test Acc, **66.45% No-BG Acc**
  * **CAB2 B4 Val:** [`results/val_kfold_cab2_EfficientNetB4_20260908-073515.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetB4_20260908-073515.txt) — **94.26% ± 0.80%** Val Acc, **66.68% No-BG Acc**
  * **CAB1 V2S Test:** [`results/test_kfold_cab1_EfficientNetV2S_20260908-080916.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetV2S_20260908-080916.txt) — **93.79% ± 0.99%** Test Acc, **63.88% No-BG Acc**
  * **CAB1 V2S Val:** [`results/val_kfold_cab1_EfficientNetV2S_20260908-074843.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetV2S_20260908-074843.txt) — **93.99% ± 0.89%** Val Acc, **64.72% No-BG Acc**
  * **CAB2 V2S Test:** [`results/test_kfold_cab2_EfficientNetV2S_20260908-075927.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetV2S_20260908-075927.txt) — **93.95% ± 1.69%** Test Acc, **63.05% No-BG Acc**
  * **CAB2 V2S Val:** [`results/val_kfold_cab2_EfficientNetV2S_20260908-073942.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetV2S_20260908-073942.txt) — **94.10% ± 1.56%** Val Acc, **63.42% No-BG Acc**
  * **CubiCasa5k Test:** [`results/test_kfold_cubicasa5k_VGG16_20260908-072527.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260908-072527.txt) — **95.61% ± 0.28%** Test Acc, **61.65% No-BG Acc**, 51.89% Macro IoU
  * **CubiCasa5k Val:** [`results/val_kfold_cubicasa5k_VGG16_20260908-071805.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cubicasa5k_VGG16_20260908-071805.txt) — **96.42% ± 0.36%** Val Acc, **66.33% No-BG Acc**, 58.95% Macro IoU
  * **Zeng Test:** [`results/test_kfold_zeng_VGG16_20260908-071630.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260908-071630.txt) — **95.19% ± 0.22%** Test Acc, **58.08% No-BG Acc**, 50.37% Macro IoU
  * **Zeng Val:** [`results/val_kfold_zeng_VGG16_20260908-071329.txt`](file:///workspaces/multi-unit-floorplan/results/val_kfold_zeng_VGG16_20260908-071329.txt) — **95.79% ± 0.32%** Val Acc, **60.86% No-BG Acc**, 54.07% Macro IoU

---

### 2.2 EfficientNetV1 Best Setup (B4) 10-Fold Cross-Validation Runs (August 30 – September 1, 2026)

* **Status:** **Completed Successfully** across all 10 folds for both CAB1 and CAB2.
* **Runner Script:** [`run_kfold_b4.sh`](file:///workspaces/multi-unit-floorplan/run_kfold_b4.sh) executed concurrently on **GPU 0** and **GPU 3**.
* **Configurations:**
  * **CAB1 B4:** [`kfold_patch/eval_cab1_b4_cubicasa.py`](file:///workspaces/multi-unit-floorplan/kfold_patch/eval_cab1_b4_cubicasa.py) (`backbone='EfficientNetB4'`, `hhdc=7`, `cam=5`, `aaf=[2,4]`, `batch_size=4`, `lr=1e-4`, `cosine-decay-warmup`)
  * **CAB2 B4:** [`kfold_patch/eval_cab2_b4_cubicasa.py`](file:///workspaces/multi-unit-floorplan/kfold_patch/eval_cab2_b4_cubicasa.py) (`backbone='EfficientNetB4'`, `hhdc=False`, `cam=3`, `aaf=[2,4]`, `batch_size=4`, `lr=1e-4`, `cosine-decay-warmup`)
* **Log Files:**
  * CAB1: [`logs/kfold_cab1_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab1_b4.log) (Training time: 2,534.55 min / 42.24 hrs)
  * CAB2: [`logs/kfold_cab2_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab2_b4.log) (Training time: 2,698.55 min / 44.98 hrs)
* **Cross-Validation Results:**
  * **CAB1 B4:** Mean Val Loss = **1.9073 ± 0.2106**, Mean Val Categorical Accuracy = **0.9486 ± 0.0071** (94.86% ± 0.71%)
  * **CAB2 B4:** Mean Val Loss = **2.0108 ± 0.1983**, Mean Val Categorical Accuracy = **0.9446 ± 0.0078** (94.46% ± 0.78%)
* **Model Checkpoints (All 20 folds saved with valid `saved_model.pb`):**
  * CAB1 B4 Folds 0–9: `models/cab1_cab1_eval_cubicasa_b4_EfficientNetB4_32,64,128,256,512_cubicasa5k_20260830-214426/0` through `models/..._20260901-122538/9`
  * CAB2 B4 Folds 0–9: `models/cab2_cab2_eval_cubicasa_b4_EfficientNetB4_32,64,128,256,512_cubicasa5k_20260830-214426/0` through `models/..._20260901-131551/9`

---

### 2.3 Historical Pre-Audit EfficientNetB4 10-Fold Test Set Evaluation (September 7, 2026)

* **Status:** **Completed Successfully** on NVIDIA A100 GPU across all 10 folds for CAB1 B4, CAB2 B4, CubiCasa5k (VGG16), and Zeng (VGG16).
* **Runner Script:** [`run_test_evaluation.sh all 0`](file:///workspaces/multi-unit-floorplan/run_test_evaluation.sh)
* **Log File:** [`logs/test_eval_all_20260907-073649.log`](file:///workspaces/multi-unit-floorplan/logs/test_eval_all_20260907-073649.log)
* **Official Test Set Artifacts:**
  * **CAB1 B4:** [`results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt)
    * **Mean Test Accuracy:** **0.9452 ± 0.0094** (94.52% ± 0.94%)
    * **Non-Background Accuracy:** **0.6919** (**69.19%** — *Highest across all architectures evaluated in this project*)
    * **Per-Fold Accuracy:** `[0.9427, 0.9414, 0.9478, 0.9404, 0.9395, 0.9635, 0.9602, 0.9381, 0.9473, 0.9313]`
    * **Key Class IoUs:** Walls: **60.64%**, Windows: **53.92%**, Doors: **27.44%**, Stairs: **14.69%**, Railings: **9.16%**
    * **Macro IoU:** **43.47%** (Excl. Background: **33.17%**)
    * **Macro F1:** **54.74%** (Excl. Background: **46.21%**)
  * **CAB2 B4:** [`results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt)
    * **Mean Test Accuracy:** **0.9414 ± 0.0086** (94.14% ± 0.86%)
    * **Non-Background Accuracy:** **0.6646** (**66.46%** — *Significantly outperforms V2S 63.06% and CubiCasa5k 61.65%*)
    * **Per-Fold Accuracy:** `[0.9449, 0.9374, 0.9349, 0.9637, 0.9419, 0.9411, 0.9365, 0.9399, 0.9443, 0.9298]`
    * **Key Class IoUs:** Walls: **59.19%**, Windows: **47.80%**, Doors: **19.35%**, Stairs: **9.88%**, Railings: **7.47%**
    * **Macro IoU:** **39.72%** (Excl. Background: **28.74%**)
    * **Macro F1:** **50.10%** (Excl. Background: **40.67%**)
  * **CubiCasa5k Reference (VGG16):** [`results/test_kfold_cubicasa5k_VGG16_20260907-082742.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260907-082742.txt)
    * Mean Test Accuracy: **0.9561 ± 0.0028**, Non-Background Accuracy: **61.65%**, Macro IoU: **51.89%**
  * **Zeng Reference (VGG16):** [`results/test_kfold_zeng_VGG16_20260907-083051.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260907-083051.txt)
    * Mean Test Accuracy: **0.9519 ± 0.0022**, Non-Background Accuracy: **58.09%**, Macro IoU: **50.37%**

---

### 2.3 Automated Post-Training Hook (September 2, 2026)

* **Context:** Ran automatically after training via unconstrained `python kfold_patch/evaluate_kfold.py --models cab1 cab2` without `--backbone EfficientNetB4`. Default model discovery fell back to the older EfficientNetV2S checkpoints on CPU before the dedicated GPU test harness was established.
* **Results Artifacts:**
  * CAB1: [`results/test_kfold_cab1_cubicasa_20260902-002106.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_cubicasa_20260902-002106.txt) (92.42% Test Acc, 43.00% No-BG Acc)
  * CAB2: [`results/test_kfold_cab2_cubicasa_20260902-063522.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_cubicasa_20260902-063522.txt) (93.33% Test Acc, 54.18% No-BG Acc)

---

### 2.4 EfficientNetV2S 10-Fold Cross-Validation Runs (August 26–30, 2026)

* **Runner Configuration:** `kfold_patch/eval_cab1_cubicasa.py` and `kfold_patch/eval_cab2_cubicasa.py`
* **Log Files:** 
  * CAB1: [`logs/kfold_cab1_v2s.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab1_v2s.log) (and Fold 4 retrain in `/root/.gemini/antigravity-cli/brain/.../task-159.log`)
  * CAB2: [`logs/kfold_cab2_v2s.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab2_v2s.log)
* **Model Checkpoints:**
  * CAB1: `models/cab1_cab1_eval_cubicasa_v2s_EfficientNetV2S_32,64,128,256,512_cubicasa5k_20260826-081010/0` through `models/.../9` (with retrained Fold 4 at `models/..._20260829-201005/4`)
  * CAB2: `models/cab2_cab2_eval_cubicasa_v2s_EfficientNetV2S_32,64,128,256,512_cubicasa5k_20260826-081009/0` through `models/.../9`
* **Results Artifacts:**
  * CAB1 Final Test: [`results/test_kfold_cab1_20260830-211306.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260830-211306.txt)
  * CAB2 Final Test: [`results/test_kfold_cab2_20260829-202421.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260829-202421.txt)

---

### 2.5 Single-Fold (Fold 0) Ablation Search Runs (August 15–22, 2026)

* **Script:** `kfold_patch/ablation_search.py`
* **Model Checkpoints:** `models/cab1_cab1_ablation_*` and `models/cab2_cab2_ablation_*`
* **Output Logs:** 
  * `results/ablation_cab1_fold0_20260817-032305.txt` (CAB1 Backbone Search: B0, B2, B3, B4)
  * `results/ablation_cab1_fold0_20260822-032911.txt` (CAB1 Attention Search: no_hhdc, hhdc_3, hhdc_7, no_cam, cam_1, cam_5)
  * `results/ablation_cab2_fold0_20260818-175528.txt` (CAB2 Backbone Search: B0, B2, B3, B4)
  * `results/ablation_cab2_fold0_20260821-191722.txt` (CAB2 Attention Search: no_hhdc, hhdc_3, hhdc_7, no_cam, cam_1, cam_5)

---

### 2.6 Baseline 10-Fold Cross-Validation Runs (August 4, 2026)

* **Checkpoints:** `models/cab1_cab1_eval_cubicasa_EfficientNetB2_*`, `models/cab2_cab2_eval_cubicasa_EfficientNetB2_*`, `models/cubicasa5k_*`, `models/zeng_*`
* **Output Logs:**
  * `results/test_kfold_cab1_20260804-105154.txt` (CAB1 B2 Baseline)
  * `results/test_kfold_cab2_20260804-112730.txt` (CAB2 B2 Baseline)
  * `results/test_kfold_cubicasa5k_20260804-115310.txt` (CubiCasa5k VGG16 Paper Baseline)
  * `results/test_kfold_zeng_20260804-121359.txt` (Zeng VGG16 Paper Baseline)

---

## 3. Results File Naming Convention Standard

To eliminate ambiguity across experiment runs, all evaluation outputs generated by `kfold_patch/evaluate_kfold.py` adhere to the self-describing schema:

```
results/{split}_kfold_{model}_{backbone}_{YYYYMMDD-HHMMSS}.txt
```

Where:
* `{split}`: Evaluation split (`test` for the 400-sample test set; `val` for 10-fold out-of-fold cross-validation on 4,600 samples)
* `{model}`: Model architecture type (`cab1`, `cab2`, `cubicasa5k`, `zeng`)
* `{backbone}`: Encoder network (`EfficientNetB4`, `EfficientNetV2S`, `EfficientNetB2`, `VGG16`)
* `{YYYYMMDD-HHMMSS}`: Timestamp of evaluation completion

Each generated file header explicitly includes:
* `Model`: Model architecture type (`cab1`, `cab2`, `cubicasa5k`, `zeng`)
* `Backbone`: Encoder network (`EfficientNetB4`, `EfficientNetV2S`, `EfficientNetB2`, `VGG16`)
* `Experiment Name`: Configured experiment label
* `Dataset`: Dataset evaluated (`cubicasa5k`)
* `Split`: Evaluated dataset split (`test` or `val`)
* `Fold Checkpoint Sources`: Full paths to each fold model directory evaluated.
