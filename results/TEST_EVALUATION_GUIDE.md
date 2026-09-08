# CubiCasa5k Test Set Evaluation Guide & Runbook
**Project:** Multi-Unit Floorplan Segmentation  
**Test Split Definition:** [`data/cubicasa5k/test.txt`](file:///workspaces/multi-unit-floorplan/data/cubicasa5k/test.txt) (400 floorplan samples)  
**Serialized Dataset:** [`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`](file:///workspaces/multi-unit-floorplan/data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords) (400 records)  
**Date:** September 7, 2026  

---

## 1. Executive Summary & Evaluation Status
 
### 1.1 GPU Passthrough & Evaluation Execution
NVIDIA GPU passthrough was successfully verified on device `GPU:0` (`NVIDIA A100-SXM4-40GB`, 40GB VRAM) with CUDA / cuDNN acceleration. The full 10-fold test evaluation across all primary architectures (CAB1 B4, CAB2 B4, CubiCasa5k VGG16, and Zeng VGG16) was executed on September 7, 2026 via [`run_test_evaluation.sh all 0`](file:///workspaces/multi-unit-floorplan/run_test_evaluation.sh), completing in ~54 minutes total wall-clock time.

### 1.2 Evaluation Performance Summary
* **CAB1 EfficientNetB4:** Mean Test Accuracy = **94.52% ± 0.94%**, Non-Background Accuracy = **69.19%** (Best overall foreground segmentation accuracy).
* **CAB2 EfficientNetB4:** Mean Test Accuracy = **94.14% ± 0.86%**, Non-Background Accuracy = **66.46%**.
* **CubiCasa5k Reference (VGG16):** Mean Test Accuracy = **95.61% ± 0.28%**, Non-Background Accuracy = **61.65%**.
* **Zeng Reference (VGG16):** Mean Test Accuracy = **95.19% ± 0.22%**, Non-Background Accuracy = **58.09%**.

> **Status:** All 10-fold test set evaluations have completed and results are recorded in the repository.

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

### 3.2 Status of Analyzed Models on the Test Dataset

| Model Family | Backbone | Folds / Scope | Evaluated on Test Set? | Result Artifact File |
| :--- | :---: | :---: | :---: | :--- |
| **CAB1 B4 (Best V1)** | `EfficientNetB4` | 10 Folds | **Yes (Completed 2026-09-07)** | [`results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_EfficientNetB4_20260907-075825.txt) |
| **CAB2 B4 (Best V1)** | `EfficientNetB4` | 10 Folds | **Yes (Completed 2026-09-07)** | [`results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_EfficientNetB4_20260907-082006.txt) |
| **CubiCasa5k Reference** | `VGG16` | 10 Folds | **Yes (Verified 2026-09-07)** | [`results/test_kfold_cubicasa5k_VGG16_20260907-082742.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cubicasa5k_VGG16_20260907-082742.txt) |
| **Zeng Reference** | `VGG16` | 10 Folds | **Yes (Verified 2026-09-07)** | [`results/test_kfold_zeng_VGG16_20260907-083051.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_zeng_VGG16_20260907-083051.txt) |
| **CAB1 V2S Final** | `EfficientNetV2S` | 10 Folds | **Yes (Official V2S)** | [`results/test_kfold_cab1_20260830-211306.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260830-211306.txt) |
| **CAB2 V2S Final** | `EfficientNetV2S` | 10 Folds | **Yes (Official V2S)** | [`results/test_kfold_cab2_20260829-202421.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260829-202421.txt) |
| **CAB1 Baseline** | `EfficientNetB2` | 10 Folds | **Yes (Historical)** | [`results/test_kfold_cab1_20260804-105154.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab1_20260804-105154.txt) |
| **CAB2 Baseline** | `EfficientNetB2` | 10 Folds | **Yes (Historical)** | [`results/test_kfold_cab2_20260804-112730.txt`](file:///workspaces/multi-unit-floorplan/results/test_kfold_cab2_20260804-112730.txt) |
| **Fold 0 Ablation Models** | `B0, B3, CAM, HHDC` | Single Fold | N/A (HPO only) | Run with `save_model=False` during hyperparameter search; only validation metrics logged. |

---

## 4. How to Run the Evaluation Post-Rebuild

An automated runner script [`run_test_evaluation.sh`](file:///workspaces/multi-unit-floorplan/run_test_evaluation.sh) and updated evaluation engine [`kfold_patch/evaluate_kfold.py`](file:///workspaces/multi-unit-floorplan/kfold_patch/evaluate_kfold.py) are prepared and verified.

### 4.1 Quick Launch via Automated Runner Script

#### Option 1: Evaluate Both CAB1 B4 and CAB2 B4 Sequentially on a Single GPU (e.g. GPU 0)
```bash
./run_test_evaluation.sh b4 0
```
* Runtime: ~35–40 minutes on an A100 GPU.
* Output logs: `logs/test_eval_b4_<timestamp>.log`
* Output results:
  * `results/test_kfold_cab1_EfficientNetB4_<timestamp>.txt`
  * `results/test_kfold_cab2_EfficientNetB4_<timestamp>.txt`

#### Option 2: Run CAB1 and CAB2 in Parallel on Separate GPUs (Fastest: ~18–20 min total)
Open two terminal tabs:

* **Terminal 1 (CAB1 B4 on GPU 0):**
  ```bash
  ./run_test_evaluation.sh cab1_b4 0
  ```
* **Terminal 2 (CAB2 B4 on GPU 3):**
  ```bash
  ./run_test_evaluation.sh cab2_b4 3
  ```

#### Option 3: Evaluate All Models (B4 + CubiCasa5k + Zeng)
```bash
./run_test_evaluation.sh all 0
```

---

### 4.2 Direct Python CLI Invocation

You can also run the evaluation script directly:

```bash
# Evaluate CAB1 and CAB2 with EfficientNetB4:
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 cab2 \
    --backbone EfficientNetB4 \
    --k_fold 10

# Evaluate only CAB1 EfficientNetB4:
CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_kfold.py \
    --models cab1 \
    --backbone EfficientNetB4 \
    --k_fold 10

# Evaluate only CAB2 EfficientNetB4:
CUDA_VISIBLE_DEVICES=3 python kfold_patch/evaluate_kfold.py \
    --models cab2 \
    --backbone EfficientNetB4 \
    --k_fold 10
```

---

## 5. Output Verification & Result Metrics

Upon completion, each generated result file contains:
1. Header metadata: Model prefix, exact Backbone name, dataset, fold count, per-fold test accuracy list, and mean test accuracy ± standard deviation.
2. Full provenance list of the exact checkpoint directory evaluated for each fold (Folds 0 through 9).
3. Class-by-class confusion matrix breakdown:
   * Overall Accuracy
   * Accuracy without background
   * Per-class Class Accuracy, Recall, Precision, F1-Score, IoU, fwRecall, fwIoU, and raw TP/FP/TN/FN counts for:
     * `bg` (Background)
     * `walls`
     * `railings`
     * `doors`
     * `windows`
     * `stairs_all`
   * Macro Mean & Macro Mean without background.

### 5.1 Final Evaluation Results Summary (Evaluated on CubiCasa5k 400 Test Images)

| Architecture | Backbone | Overall Test Acc | No-BG Acc | Walls IoU | Windows IoU | Doors IoU | Stairs IoU | Railings IoU | Macro IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAB1 B4** | `EfficientNetB4` | 94.52% ± 0.94% | **69.19%** | 60.64% | 53.92% | 27.44% | 14.69% | 9.16% | 43.47% |
| **CAB2 B4** | `EfficientNetB4` | 94.14% ± 0.86% | 66.46% | 59.19% | 47.80% | 19.35% | 9.88% | 7.47% | 39.72% |
| **CAB1 V2S** | `EfficientNetV2S` | 93.79% ± 0.98% | 63.89% | 56.12% | 46.57% | 6.86% | 6.03% | 3.81% | 35.63% |
| **CAB2 V2S** | `EfficientNetV2S` | 93.95% ± 1.69% | 63.06% | 57.05% | 42.91% | 26.25% | 15.82% | 10.65% | 41.17% |
| **CubiCasa5k** | `VGG16` | **95.61% ± 0.28%** | 61.65% | **63.46%** | **59.83%** | 41.96% | 36.56% | **13.92%** | **51.89%** |
| **Zeng** | `VGG16` | 95.19% ± 0.22% | 58.09% | 59.33% | 56.48% | **43.35%** | **38.97%** | 8.97% | 50.37% |

### 5.2 Documentation Integration Status
All newly generated test metrics have been integrated across:
* [`results/EXPERIMENT_REGISTRY.md`](file:///workspaces/multi-unit-floorplan/results/EXPERIMENT_REGISTRY.md)
* [`results/experiment_analysis_and_kfold_evaluation.md`](file:///workspaces/multi-unit-floorplan/results/experiment_analysis_and_kfold_evaluation.md)
* [`results/hyperparameter_recommendations.md`](file:///workspaces/multi-unit-floorplan/results/hyperparameter_recommendations.md)
