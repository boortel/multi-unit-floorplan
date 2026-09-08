# Results & Experiment Registry Manifest
**Project:** Multi-Unit Floorplan Segmentation  
**Dataset:** CubiCasa5k  
**Last Updated:** September 7, 2026  

---

## 1. Quick Reference: Results File to Experiment Mapping

| Result File | Experiment Name | Model | Backbone | Key Hyperparameters | Date Generated | Status / Key Metric |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| [`test_kfold_cab1_EfficientNetB4_20260907-075825.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt) | **CAB1 B4 10-Fold Test (Official)** | CAB1 | `EfficientNetB4` | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-07 | **Official 10-Fold Test**: **94.52% ± 0.94%** Test Acc, **69.19% No-BG Acc** (Project Record!) |
| [`test_kfold_cab2_EfficientNetB4_20260907-082006.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt) | **CAB2 B4 10-Fold Test (Official)** | CAB2 | `EfficientNetB4` | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-07 | **Official 10-Fold Test**: **94.14% ± 0.86%** Test Acc, **66.46% No-BG Acc** |
| [`test_kfold_cubicasa5k_VGG16_20260907-082742.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260907-082742.txt) | **CubiCasa5k VGG16 10-Fold Test** | CubiCasa | `VGG16` | Multi-task heatmap heads | 2026-09-07 | Verified Benchmark: 95.61% ± 0.28% Test Acc, 61.65% No-BG Acc |
| [`test_kfold_zeng_VGG16_20260907-083051.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260907-083051.txt) | **Zeng VGG16 10-Fold Test** | Zeng | `VGG16` | Multi-dilation feature aggregation | 2026-09-07 | Verified Benchmark: 95.19% ± 0.22% Test Acc, 58.09% No-BG Acc |
| [`logs/kfold_cab1_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab1_b4.log) | **CAB1 B4 10-Fold Training** | CAB1 | `EfficientNetB4` | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-01 | **10-Fold CV Finished**: 94.86% ± 0.71% Val Acc, 1.9073 ± 0.2106 Val Loss (All 10 Folds converged) |
| [`logs/kfold_cab2_b4.log`](file:///workspaces/multi-unit-floorplan/logs/kfold_cab2_b4.log) | **CAB2 B4 10-Fold Training** | CAB2 | `EfficientNetB4` | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-01 | **10-Fold CV Finished**: 94.46% ± 0.78% Val Acc, 2.0108 ± 0.1983 Val Loss (All 10 Folds converged) |
| [`test_kfold_cab1_cubicasa_20260902-002106.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_cubicasa_20260902-002106.txt) | CAB1 10-Fold Test (Sep 2 Post-Run Hook) | CAB1 | `EfficientNetV2S` | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-09-02 | Unfiltered hook evaluated older V2S: 92.42% Test Acc, 43.00% No-BG Acc |
| [`test_kfold_cab2_cubicasa_20260902-063522.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_cubicasa_20260902-063522.txt) | CAB2 10-Fold Test (Sep 2 Post-Run Hook) | CAB2 | `EfficientNetV2S` | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-09-02 | Unfiltered hook evaluated older V2S: 93.33% Test Acc, 54.18% No-BG Acc |
| [`test_kfold_cab1_20260830-211306.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260830-211306.txt) | **CAB1 V2S 10-Fold Test (Official)** | CAB1 | `EfficientNetV2S` | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-08-30 | **Official V2S**: 93.79% ± 0.98% Test Acc, 63.89% No-BG Acc |
| [`test_kfold_cab2_20260829-202421.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260829-202421.txt) | **CAB2 V2S 10-Fold Test (Official)** | CAB2 | `EfficientNetV2S` | `hhdc=False`, `cam=3`, `aaf=[2,4]` | 2026-08-29 | **Official V2S**: 93.95% ± 1.69% Test Acc, 63.06% No-BG Acc |
| [`test_kfold_cab1_20260829-204602.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260829-204602.txt) | CAB1 V2S 10-Fold (Intermediate) | CAB1 | `EfficientNetV2S` | `hhdc=7`, `cam=5`, `aaf=[2,4]` | 2026-08-29 | Intermediate (evaluated during Fold 4 retraining) |
| [`test_kfold_cab1_20260804-105154.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260804-105154.txt) | CAB1 B2 10-Fold (Baseline) | CAB1 | `EfficientNetB2` | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-04 | Baseline: 94.64% Test Acc (0% door/window IoU) |
| [`test_kfold_cab2_20260804-112730.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260804-112730.txt) | CAB2 B2 10-Fold (Baseline) | CAB2 | `EfficientNetB2` | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-04 | Baseline: 94.68% Test Acc |
| [`test_kfold_cubicasa5k_20260804-115310.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_20260804-115310.txt) | CubiCasa5k 10-Fold (Paper Ref) | CubiCasa | `VGG16` | Multi-task heatmap heads | 2026-08-04 | Benchmark Reference: 95.61% Test Acc, 61.65% No-BG Acc |
| [`test_kfold_zeng_20260804-121359.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_20260804-121359.txt) | Zeng 10-Fold (Paper Ref) | Zeng | `VGG16` | Multi-dilation feature aggregation | 2026-08-04 | Benchmark Reference: 95.19% Test Acc, 58.09% No-BG Acc |
| [`ablation_cab1_fold0_20260817-032305.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab1_fold0_20260817-032305.txt) | CAB1 Fold 0 Backbone Ablation | CAB1 | `B0, B2, B3, B4` | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-17 | **B4 Best Backbone** (1.5401 val loss) |
| [`ablation_cab1_fold0_20260822-032911.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab1_fold0_20260822-032911.txt) | CAB1 Fold 0 Context & CAM Ablation | CAB1 | `EfficientNetB2` | `hhdc={0,3,7}`, `cam={0,1,5}` | 2026-08-22 | **`hhdc=7`** (1.5576 loss), **`cam=5`** (1.5552 loss) Best |
| [`ablation_cab2_fold0_20260818-175528.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab2_fold0_20260818-175528.txt) | CAB2 Fold 0 Backbone Ablation | CAB2 | `B0, B2, B3, B4` | `hhdc=5`, `cam=3`, `aaf=[2,4]` | 2026-08-18 | **B4 Best Backbone** (1.5328 val loss) |
| [`ablation_cab2_fold0_20260821-191722.txt`](file:///workspaces/multi-unit-floorplan/results/ablation_cab2_fold0_20260821-191722.txt) | CAB2 Fold 0 Context & CAM Ablation | CAB2 | `EfficientNetB2` | `hhdc={0,3,7}`, `cam={0,1,5}` | 2026-08-21 | **`no_hhdc`** (1.5462 loss), **`cam=3`** (1.5481 loss) Best |
| [`hyperparameter_recommendations.md`](file:///workspaces/multi-unit-floorplan/results/hyperparameter_recommendations.md) | V1/V2 Hyperparameter Analysis | All | `B2, B4, V2S` | All configurations & scripts | 2026-09-07 | Master guide for V1 & V2 setups |
| [`experiment_analysis_and_kfold_evaluation.md`](file:///workspaces/multi-unit-floorplan/results/experiment_analysis_and_kfold_evaluation.md) | 10-Fold Post-Mortem & Multi-Model Comparison | All | `B4, V2S, VGG16` | Full class-by-class comparison | 2026-09-07 | Comprehensive performance analysis |
| [`TEST_EVALUATION_GUIDE.md`](file:///workspaces/multi-unit-floorplan/results/TEST_EVALUATION_GUIDE.md) | Test Set Evaluation Runbook & Rebuild Guide | All | `B4, V2S, B2, VGG16` | Execution instructions on `test.txt` | 2026-09-07 | Rebuild instructions & post-rebuild runner |

---

## 2. Detailed Experiment Profiles & Resource Tracking

### 2.1 EfficientNetV1 Best Setup (B4) 10-Fold Cross-Validation Runs (August 30 – September 1, 2026)

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

### 2.2 Official EfficientNetB4 10-Fold Test Set Evaluation (September 7, 2026)

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

To eliminate ambiguity across future experiment runs, all evaluation outputs generated by `kfold_patch/evaluate_kfold.py` now adhere to the self-describing schema:

```
results/test_kfold_{model}_{backbone}_{YYYYMMDD-HHMMSS}.txt
```

Each generated file header explicitly includes:
* `Model`: Model architecture type (`cab1`, `cab2`, `cubicasa5k`, `zeng`)
* `Backbone`: Encoder network (`EfficientNetB4`, `EfficientNetV2S`, `EfficientNetB2`, `VGG16`)
* `Experiment Name`: Configured experiment label
* `Fold Checkpoint Sources`: Full paths to each fold model directory evaluated.
