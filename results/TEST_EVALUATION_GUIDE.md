# CubiCasa5k Test Set Evaluation Guide & Runbook
**Project:** Multi-Unit Floorplan Segmentation  
**Test Split Definition:** [`data/cubicasa5k/test.txt`](file:///workspaces/multi-unit-floorplan/data/cubicasa5k/test.txt) (400 floorplan samples)  
**Serialized Dataset:** [`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`](file:///workspaces/multi-unit-floorplan/data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords) (400 records)  
**Date:** September 8, 2026  

---

## 1. Executive Summary & Evaluation Status
 
### 1.1 GPU Passthrough & Evaluation Execution
NVIDIA GPU passthrough was verified across all four physical `NVIDIA A100-SXM4-40GB` GPUs (GPUs 0, 1, 2, 3) with CUDA / cuDNN acceleration. 

Following the implementation of evaluation and data pipeline fixes in [`CODEBASE_AUDIT.md`](file:///workspaces/multi-unit-floorplan/CODEBASE_AUDIT.md) (E-4, E-5, E-6, D-1, D-2), a full concurrent evaluation of **both Out-of-Fold Cross-Validation (4,600 floorplans)** and the **Official Test Set (400 floorplans)** across all six primary architectures was executed on September 8, 2026 via [`run_all_evaluations.sh`](file:///workspaces/multi-unit-floorplan/run_all_evaluations.sh):
* **GPU 0:** CAB1 EfficientNetB4 (Val & Test)
* **GPU 1:** CAB2 EfficientNetB4 (Val & Test)
* **GPU 2:** CubiCasa5k VGG16 → CAB1 EfficientNetV2S (Val & Test)
* **GPU 3:** Zeng VGG16 → CAB2 EfficientNetV2S (Val & Test)

All pipelines ran in parallel and completed in ~59 minutes total wall-clock time.

### 1.2 Evaluation Performance Summary (Test Set)
* **CAB1 EfficientNetB4:** Mean Test Accuracy = **94.52% ± 0.94%**, Non-Background Accuracy = **69.19%** (Best overall foreground segmentation accuracy).
* **CAB2 EfficientNetB4:** Mean Test Accuracy = **94.14% ± 0.86%**, Non-Background Accuracy = **66.45%**.
* **CubiCasa5k Reference (VGG16):** Mean Test Accuracy = **95.61% ± 0.28%**, Non-Background Accuracy = **61.65%**.
* **Zeng Reference (VGG16):** Mean Test Accuracy = **95.19% ± 0.22%**, Non-Background Accuracy = **58.08%**.

### 1.3 Out-of-Fold Validation Summary (10 Folds, 4,600 Floorplans)
* **CAB1 EfficientNetB4:** Mean Val Accuracy = **94.64% ± 0.88%**, Non-Background Accuracy = **69.55%**.
* **CAB2 EfficientNetB4:** Mean Val Accuracy = **94.26% ± 0.80%**, Non-Background Accuracy = **66.68%**.
* **CubiCasa5k Reference (VGG16):** Mean Val Accuracy = **96.42% ± 0.36%**, Non-Background Accuracy = **66.33%**.
* **Zeng Reference (VGG16):** Mean Val Accuracy = **95.79% ± 0.32%**, Non-Background Accuracy = **60.86%**.

> **Status:** All 10-fold validation and test set evaluations are complete and recorded in the repository.

---

## 2. Container Configuration & Verification

For future reproduction or re-running on fresh container instances, ensure GPU device access is configured:

### Option A: VS Code Dev Containers (Recommended)
Verify that `.devcontainer/devcontainer.json` includes GPU passthrough options:
```json
{
  "runArgs": [
    "--gpus", "all",
    "--ipc=host"
  ]
}
```
In VS Code, press `F1` (or `Ctrl+Shift+P`), type and select:
`Dev Containers: Rebuild Container` (or `Dev Containers: Rebuild and Reopen in Container`).

### Option B: Docker CLI
If rebuilding or starting the container via Docker CLI directly:
```bash
docker run --gpus all --ipc=host -it ...
```

### Verification
Verify GPU accessibility inside the environment with:
```bash
nvidia-smi
python -c "import tensorflow as tf; print('GPUs detected:', tf.config.list_physical_devices('GPU'))"
```

---

## 3. Test Dataset Overview & Model Audit

### 3.1 Test Dataset Split
The test split is defined in [`data/cubicasa5k/test.txt`](file:///workspaces/multi-unit-floorplan/data/cubicasa5k/test.txt) and contains exactly 400 floorplan directories from the CubiCasa5k dataset.
These samples are serialized in [`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`](file:///workspaces/multi-unit-floorplan/data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords) (verified 400 TFRecord entries).

#### 3.2 Status of Analyzed Models on the Test and Validation Datasets

| Model Family | Backbone | Folds / Scope | Evaluated on Val & Test? | Test Artifact File | Validation Artifact File |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **CAB1 B4 (Best V1)** | `EfficientNetB4` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_cab1_EfficientNetB4`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260908-075715.txt) | [`val_kfold_cab1_EfficientNetB4`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetB4_20260908-073505.txt) |
| **CAB2 B4 (Best V1)** | `EfficientNetB4` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_cab2_EfficientNetB4`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260908-075647.txt) | [`val_kfold_cab2_EfficientNetB4`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetB4_20260908-073515.txt) |
| **CAB1 V2S Final** | `EfficientNetV2S` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_cab1_EfficientNetV2S`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetV2S_20260908-080916.txt) | [`val_kfold_cab1_EfficientNetV2S`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab1_EfficientNetV2S_20260908-074843.txt) |
| **CAB2 V2S Final** | `EfficientNetV2S` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_cab2_EfficientNetV2S`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetV2S_20260908-075927.txt) | [`val_kfold_cab2_EfficientNetV2S`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cab2_EfficientNetV2S_20260908-073942.txt) |
| **CubiCasa5k Reference** | `VGG16` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_cubicasa5k_VGG16`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260908-072527.txt) | [`val_kfold_cubicasa5k_VGG16`](file:///workspaces/multi-unit-floorplan/results/val_kfold_cubicasa5k_VGG16_20260908-071805.txt) |
| **Zeng Reference** | `VGG16` | 10 Folds | **Yes (Audit Verified 2026-09-08)** | [`test_kfold_zeng_VGG16`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260908-071630.txt) | [`val_kfold_zeng_VGG16`](file:///workspaces/multi-unit-floorplan/results/val_kfold_zeng_VGG16_20260908-071329.txt) |
| **Fold 0 Ablation Models** | `B0, B3, CAM, HHDC` | Single Fold | N/A (HPO only) | Run with `save_model=False` during hyperparameter search. | N/A |

---

## 4. How to Run the Evaluation

An automated runner script [`run_all_evaluations.sh`](file:///workspaces/multi-unit-floorplan/run_all_evaluations.sh) and updated evaluation engine [`kfold_patch/evaluate_kfold.py`](file:///workspaces/multi-unit-floorplan/kfold_patch/evaluate_kfold.py) are prepared and verified.

### 4.1 Full 4-GPU Parallel Evaluation (Recommended)
Runs both validation (4,600 floorplans) and test (400 floorplans) across all 6 architectures in parallel:
```bash
./run_all_evaluations.sh
```
* **Runtime:** ~59 minutes total across 4 NVIDIA A100 GPUs.
* **Outputs:** 12 result files in `results/` (`test_kfold_*.txt` and `val_kfold_*.txt`).

### 4.2 Single-Target Evaluation via Runner Script
```bash
# Evaluate both CAB1 and CAB2 with EfficientNetB4 on GPU 0 (Test set):
./run_test_evaluation.sh b4 0

# Evaluate all primary architectures sequentially on GPU 0:
./run_test_evaluation.sh all 0
```

### 4.3 Direct Python CLI Invocation
`evaluate_kfold.py` supports evaluating the test split (`--split test`), out-of-fold validation split (`--split val`), or both (`--split both`):

```bash
# Evaluate CAB1 B4 on both validation and test sets:
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 \
    --backbone EfficientNetB4 \
    --k_fold 10 \
    --split both

# Evaluate CAB2 B4 on test set only:
CUDA_VISIBLE_DEVICES=1 python kfold_patch/evaluate_kfold.py \
    --models cab2 \
    --backbone EfficientNetB4 \
    --k_fold 10 \
    --split test
```

---

## 5. Output Verification & Result Metrics

### 5.1 Official Test Set Evaluation Results (CubiCasa5k 400 Test Images)

| Architecture | Backbone | Overall Test Acc | No-BG Acc | Walls IoU | Windows IoU | Doors IoU | Stairs IoU | Railings IoU | Macro IoU | Macro IoU (No-BG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAB1 B4** | `EfficientNetB4` | 94.52% ± 0.94% | **69.19%** | 60.63% | 53.92% | 27.43% | 14.69% | 9.16% | 43.47% | 33.17% |
| **CAB2 B4** | `EfficientNetB4` | 94.14% ± 0.86% | 66.45% | 59.18% | 47.80% | 19.35% | 9.87% | 7.47% | 39.72% | 28.73% |
| **CAB1 V2S** | `EfficientNetV2S` | 93.79% ± 0.99% | 63.88% | 56.11% | 46.57% | 6.86% | 6.03% | 3.81% | 35.63% | 23.88% |
| **CAB2 V2S** | `EfficientNetV2S` | 93.95% ± 1.69% | 63.05% | 57.04% | 42.91% | 26.24% | 15.82% | 10.65% | 41.17% | 30.53% |
| **CubiCasa5k** | `VGG16` | **95.61% ± 0.28%** | 61.65% | **63.46%** | **59.82%** | 41.95% | 36.56% | **13.92%** | **51.89%** | **43.14%** |
| **Zeng** | `VGG16` | 95.19% ± 0.22% | 58.08% | 59.33% | 56.47% | **43.34%** | **38.97%** | 8.97% | 50.37% | 41.42% |

### 5.2 Out-of-Fold Validation Results (10 Folds, 4,600 Floorplans)

| Architecture | Backbone | Overall Val Acc | No-BG Acc | Walls IoU | Windows IoU | Doors IoU | Stairs IoU | Railings IoU | Macro IoU | Macro IoU (No-BG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAB1 B4** | `EfficientNetB4` | 94.64% ± 0.88% | **69.55%** | 59.75% | 53.06% | 27.37% | 14.53% | 9.93% | 43.29% | 32.93% |
| **CAB2 B4** | `EfficientNetB4` | 94.26% ± 0.80% | 66.68% | 58.09% | 46.81% | 19.57% | 9.04% | 7.26% | 39.26% | 28.15% |
| **CAB1 V2S** | `EfficientNetV2S` | 93.99% ± 0.89% | 64.72% | 55.82% | 45.29% | 6.99% | 6.36% | 4.24% | 35.55% | 23.74% |
| **CAB2 V2S** | `EfficientNetV2S` | 94.10% ± 1.56% | 63.42% | 56.10% | 42.13% | 26.49% | 15.55% | 11.49% | 41.05% | 30.35% |
| **CubiCasa5k** | `VGG16` | **96.42% ± 0.36%** | 66.33% | **67.52%** | **66.07%** | **47.58%** | **57.13%** | **18.97%** | **58.95%** | **51.46%** |
| **Zeng** | `VGG16` | 95.79% ± 0.32% | 60.86% | 61.93% | 60.47% | 45.79% | 49.12% | 11.39% | 54.07% | 45.74% |

### 5.3 Documentation Integration Status
All newly generated validation and test metrics have been integrated across:
* [`results/EXPERIMENT_REGISTRY.md`](file:///workspaces/multi-unit-floorplan/results/EXPERIMENT_REGISTRY.md)
* [`results/experiment_analysis_and_kfold_evaluation.md`](file:///workspaces/multi-unit-floorplan/results/experiment_analysis_and_kfold_evaluation.md)
* [`results/hyperparameter_recommendations.md`](file:///workspaces/multi-unit-floorplan/results/hyperparameter_recommendations.md)
