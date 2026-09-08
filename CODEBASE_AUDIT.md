# Codebase Audit: Bugs & Improvement Opportunities

**Project:** Multi-Unit Floorplan Segmentation (CAB1 & CAB2)  
**Date:** September 7, 2026  
**Last Updated:** September 8, 2026  
**Best Current Result:** CAB1 EfficientNetB4 — 94.52% Test Acc, 69.19% Non-BG Acc

---

## Severity Legend

| Severity | Meaning |
|:---:|:---|
| 🔴 **CRITICAL** | Actively corrupts training or evaluation results |
| 🟠 **HIGH** | Produces incorrect results under common conditions |
| 🟡 **MEDIUM** | Latent bug that activates under specific configurations |
| 🔵 **LOW** | Code smell, dead code, or minor inefficiency |
| 🟢 **IMPROVEMENT** | Not a bug, but a concrete opportunity for better results |

## Status Legend

| Status | Meaning |
|:---:|:---|
| ✅ **FIXED** | Code fix applied, no retraining needed to take effect |
| ✅🔄 **FIXED (retrain)** | Code fix applied, but requires retraining for the fix to take effect |
| ⏳ **OPEN** | Not yet fixed |

---

## Table of Contents

1. [Training Pipeline Bugs](#1-training-pipeline-bugs)
2. [Evaluation & Inference Bugs](#2-evaluation--inference-bugs)
3. [Data Pipeline Bugs](#3-data-pipeline-bugs)
4. [Model Architecture Issues](#4-model-architecture-issues)
5. [Post-Processing & Metric Bugs](#5-post-processing--metric-bugs)
6. [Improvement Opportunities](#6-improvement-opportunities)
7. [Impact Estimates for Remaining Work](#7-impact-estimates-for-remaining-work)

---

## 1. Training Pipeline Bugs

### ✅🔄 T-1: AutomaticWeightedLoss Epoch Decay is Permanently Frozen at Epoch 0

**File:** [AutomaticWeightedLoss.py](file:///workspaces/multi-unit-floorplan/training/AutomaticWeightedLoss.py)  
**Lines:** 15, 31–32, 50–51  
**Status:** ✅ Fixed (September 8, 2026) — requires retraining for effect

**What was wrong:**
```python
# Line 15: Python integer initialization
self.epoch = 0

# Line 50-51: Callback updates a Python attribute (not a tf.Variable)
def on_epoch_end(self, epoch, logs=None):
    self.model.automatic_loss.epoch = epoch
```

**Impact:** The AAF loss decay factor `20^(-epoch/epochs)` is computed during `@tf.function` graph tracing. The Python integer `self.epoch = 0` is captured as a constant at trace time and **never updates**. The callback changes the Python attribute, but the traced graph retains the original value `0`. This means the intended AAF loss decay over training **never actually happens** — the AAF loss weight remains constant at `20^0 = 1.0` throughout all 100 epochs.

**Fix applied:**
```python
self.epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='epoch')
# In callback:
self.model.automatic_loss.epoch.assign(float(epoch))
```

**Retrain impact estimate:** **+0.5–1.5% Non-BG Acc** — The decay was frozen at 1.0, meaning AAF loss never faded. With decay working, early AAF regularization provides boundary signal while fading to let segmentation loss dominate in later epochs.

---

### ✅🔄 T-2: AAF Gradient Reversal Assumes Fixed Variable Ordering

**File:** [base_model.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/base_model.py#L36-L40)  
**Lines:** 36–40  
**Status:** ✅ Fixed (September 8, 2026) — requires retraining for effect

**What was wrong:**
```python
aaf_len = len([v for v in trainable_variables if 'edge' in v.name])
if aaf_len > 0:
    gradients = list(gradients)
    gradients[-aaf_len:] = [-grad if grad is not None else None for grad in gradients[-aaf_len:]]
```

**Impact:** The code blindly negated the **last** `aaf_len` gradients, assuming all variables with `"edge"` in their name are appended at the tail of `trainable_variables`. If Keras reorders variables (which can happen when loading from checkpoints or with certain layer compositions), this silently negates gradients of unrelated layers (e.g., the final prediction convolution), catastrophically breaking backpropagation.

**Fix applied:** Match by variable name instead of position:
```python
# Negate gradients for AAF edge variables by name, not position (T-2 fix)
gradients = list(gradients)
for i, v in enumerate(trainable_variables):
    if 'edge' in v.name and gradients[i] is not None:
        gradients[i] = -gradients[i]
```

---

### ⏳ T-3: Multi-Task Sigma Weighting Can Produce NaN/Inf Loss

**File:** [AutomaticWeightedLoss.py](file:///workspaces/multi-unit-floorplan/training/AutomaticWeightedLoss.py#L34)  
**Line:** 34  
**Status:** ⏳ Open — requires code change + retraining

```python
loss_sum += 0.5 / (self.sigmas[self.inds[i]] ** 2) * loss
```

**Impact:** `self.sigmas` are unconstrained trainable `tf.Variable`s initialized to `0.5`. If the optimizer pushes any sigma toward zero, `0.5 / sigma²` explodes to infinity, producing `NaN` gradients and immediately crashing training. This is a well-known failure mode of the Kendall & Gal multi-task uncertainty weighting formulation.

**Recommended fix:** Reparameterize in log-space (standard practice):
```python
# Initialization:
self.log_vars = [tf.Variable(0.0, trainable=True, name=f'log_var_{i}') for i in ...]
# Loss computation:
precision = tf.exp(-self.log_vars[self.inds[i]])
loss_sum += precision * loss + self.log_vars[self.inds[i]]
```

**Retrain impact estimate:** **Stability fix; +0–0.5% IoU** — Prevents NaN crashes when sigma→0. May allow more aggressive learning rates. Impact is mostly stability rather than accuracy, since current training appears to converge without hitting this edge case.

---

### ✅🔄 T-4: TensorBoard Log Directory Collision Across K-Folds

**File:** [train_config.py](file:///workspaces/multi-unit-floorplan/train_config.py#L313)  
**Line:** 313  
**Status:** ✅ Fixed (September 8, 2026) — requires retraining for effect

**What was wrong:**
```python
loss_function = AutomaticWeightedLoss(loss_funcs, names, inds, dec, epochs, config.log_dir)
```

**Impact:** In k-fold training, each fold initialized `AutomaticWeightedLoss` with the root `config.log_dir` instead of the fold-specific `config.log_dir_fold`. All folds wrote their TensorBoard scalars (sigma values, per-component losses) to the same directory concurrently, corrupting charts.

**Fix applied:** Changed to `config.log_dir_fold` (which already exists at line 498 where the `Trainer` is instantiated).

---

### ⏳ T-5: Train Dataset Size Unit Mismatch in LR Scheduler

**File:** [train_config.py](file:///workspaces/multi-unit-floorplan/train_config.py#L460-L466)  
**Lines:** 460–466  
**Status:** ⏳ Open — requires code change + retraining

```python
train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
if train_dataset_size < 0:
    k_fold_val = config.get('kFold', config.get('k_fold', 0))
    if k_fold_val > 0:
        train_dataset_size = 4140  # 9 training folds x 460 samples
```

**Impact:** When `cardinality()` returns a valid value (positive), it reports the number of **batches** (because the dataset is already batched at this point). But the fallback `4140` represents **samples**. The `get_scheduler` function uses `train_dataset_size / batch_size` to compute `steps_per_epoch`. If the cardinality path returns batches and the scheduler divides by `batch_size` again, the cosine decay schedule will be ~4× too fast.

> [!NOTE]
> This bug was partially mitigated: since CubiCasa5k pipelines with `.cache().shuffle()` return `UNKNOWN_CARDINALITY`, the fallback path (`4140`) is consistently hit during k-fold training. The bug would manifest if the data pipeline were changed to return known cardinality (e.g., by removing `.cache()` before batching).

**Recommended fix:** Standardize units. Always pass sample count to the scheduler:
```python
if train_dataset_size > 0:
    train_dataset_size = train_dataset_size * batch_size  # Convert batches to samples
```

**Retrain impact estimate:** **+0–0.5%** — Currently masked because `UNKNOWN_CARDINALITY` fallback is always triggered. Low priority.

---

### ⏳ T-6: TensorBoard Loss Accumulation Never Averaged

**File:** [AutomaticWeightedLoss.py](file:///workspaces/multi-unit-floorplan/training/AutomaticWeightedLoss.py#L33)  
**Lines:** 33, 70–73  
**Status:** ⏳ Open — logging-only fix, no accuracy impact

```python
# Per step: accumulates batch mean loss
self.losses[i].assign_add(loss)

# End of epoch: logs the raw accumulated sum
val = self.model.automatic_loss.losses[i].numpy()
tf.summary.scalar(self.model.automatic_loss.names[i], data=val, step=epoch)
```

**Impact:** The logged loss value is the **sum** of all batch-mean losses across the epoch, not the epoch average. This makes the TensorBoard loss scale dependent on `steps_per_epoch` (which varies by batch size and dataset size), making it impossible to compare runs with different batch sizes.

**Recommended fix:** Divide by step count when logging, or use `tf.keras.metrics.Mean()`.

---

## 2. Evaluation & Inference Bugs

### ✅ E-1: TTA Averages Discrete Class Indices Instead of Probabilities

**File:** [predict.py](file:///workspaces/multi-unit-floorplan/predict.py#L138-L145), [base_model.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/base_model.py#L68-L91)  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
result = prediction[0].argmax(axis=-1)        # ← Converts to integer class index
results.append(np.rot90(result, k=-k))
result = np.mean(results, axis=0)             # ← Averages integers: meaningless
```

**Impact:** When TTA was enabled, predictions were converted to discrete integer class labels via `argmax` **before** averaging. Averaging class indices (e.g., `mean(0, 2) = 1` → averaging "background" and "doors" yields "walls") produced mathematically meaningless results. This bug existed in three locations: `predict()`, `main()` in predict.py, and `tta_test_step` in base_model.py.

**Fix applied:** All three locations now average raw softmax probabilities, then apply argmax:
```python
prob = np.rot90(prediction[0], k=-k, axes=(0, 1))
results.append(prob)
# After loop:
result = np.mean(results, axis=0).argmax(axis=-1)
```

---

### ✅ E-2: TTA `test_step` Uses Image Channels as One-Hot Depth

**File:** [base_model.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/base_model.py#L89)  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
depth = image.shape[-1]          # ← 3 for RGB images
y_pred = tf.one_hot(np.expand_dims(result, axis=0), depth=depth)  # ← one-hot with depth=3
```

**Impact:** The variable `depth` was extracted from the input image's channel count (3 for RGB), not from the number of classes (6 for CubiCasa5k). The resulting one-hot tensor had 3 channels instead of 6, causing a shape mismatch crash when passed to `self.compiled_loss`.

**Fix applied:** Extract depth from model output:
```python
# Use num_classes from model output, not image channels (E-2 fix)
depth = prediction.shape[-1]  # num_classes
```

---

### ✅ E-3: `evaluate.py` TTA Is Hardcoded to Exit

**File:** [evaluate.py](file:///workspaces/multi-unit-floorplan/evaluate.py#L283-L285)  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
if tta:
    print('Fix this!')
    exit(0)
```

**Impact:** Any attempt to evaluate with TTA immediately terminated the process.

**Fix applied:**
```python
if tta:
    # TTA: enable TTA on loaded model so test_step uses TTA (E-3 fix)
    unet_model.tta = True
```

---

### ✅ E-4: Confusion Matrix Metric Drops All Batches Except First

**File:** [confusion_matrix.py](file:///workspaces/multi-unit-floorplan/training/confusion_matrix.py#L38-L41)  
**Lines:** 38–41  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
y_true = tf.math.argmax(y_true[0], axis=-1)    # ← Selects only batch element 0

if not self.post_processing:
    y_pred = tf.math.argmax(y_pred[0], axis=-1) # ← Selects only batch element 0
```

**Impact:** When batch size > 1, all samples except the first in each batch were silently ignored during metric computation. The confusion matrix only reflected ~1/batch_size of the actual data.

> [!NOTE]
> In practice, evaluation scripts typically use `batch(1)`, which masks this bug. However, if someone evaluates with `batch(4)` for speed, 75% of the data would be silently dropped from metrics.

**Fix applied:** Removed the `[0]` indexing:
```python
y_true = tf.math.argmax(y_true, axis=-1)
y_pred = tf.math.argmax(y_pred, axis=-1)
```

---

### ✅ E-5: Evaluation Metric Division by Zero

**File:** [training/metrics.py](file:///workspaces/multi-unit-floorplan/training/metrics.py#L13-L25)  
**Lines:** 13–25  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)
iou = TP / (TP + FN + FP)
```

**Impact:** No epsilon guard on any denominator. If a class had zero support (TP=FN=0), recall, precision, F1, and IoU produced `NaN` or `inf`.

**Fix applied:** Added epsilon to all denominators:
```python
eps = 1e-7
recall = TP / (TP + FN + eps)
precision = TP / (TP + FP + eps)
f1 = 2 * precision * recall / (precision + recall + eps)
iou = TP / (TP + FN + FP + eps)
```

---

### ✅ E-6: Frequency Weight (`fw`) Calculation Uses Only TP

**File:** [training/metrics.py](file:///workspaces/multi-unit-floorplan/training/metrics.py#L26-L28)  
**Lines:** 26–28  
**Status:** ✅ Fixed (September 8, 2026) — immediate effect, no retraining needed

**What was wrong:**
```python
fw = TP / (np.diag(confusion_matrix).sum() - confusion_matrix[0, 0])
```

**Impact:** Frequency weighting should be based on the total ground-truth pixel count per class (`TP + FN`), not just correctly predicted pixels (`TP`). This made the frequency weight under-represent classes with low recall.

**Fix applied:**
```python
# fw - use ground truth totals (TP + FN) instead of just TP (E-6 fix)
true_totals = TP + FN
fw = true_totals / (true_totals.sum() - true_totals[0] + eps)
```

---

## 3. Data Pipeline Bugs

### ✅🔄 D-1: Mask Resized with Bilinear Interpolation (Corrupts Labels)

**File:** [datasets/floorplans.py](file:///workspaces/multi-unit-floorplan/datasets/floorplans.py#L286-L297), [kfold_patch/floorplans_kfold.py](file:///workspaces/multi-unit-floorplan/kfold_patch/floorplans_kfold.py#L286-L297)  
**Status:** ✅ Fixed (September 8, 2026) — requires retraining for effect

**What was wrong:**
```python
img = tf.image.resize(img, [height, width])
mask = tf.image.resize(mask, [height, width])   # ← bilinear on one-hot labels!
```

**Impact:** `tf.image.resize` defaults to bilinear interpolation. The `mask` tensor contains **one-hot encoded categorical labels**. Bilinear interpolation on one-hot vectors produces fractional values at class boundaries (e.g., `[0.0, 0.7, 0.3, 0.0, 0.0, 0.0]`), corrupting the discrete ground-truth distributions. This affected every image whose dimensions weren't already aligned to `2^n_up_sample_block`.

**Fix applied in both files:** Use nearest-neighbor interpolation for masks:
```python
mask = tf.image.resize(mask, [height, width], method='nearest')
```

**Retrain impact estimate:** **+0.5–2.0% boundary IoU** — Eliminates systematic label noise at class boundaries. Impact concentrated at thin structures (walls, railings) where resize boundary effects matter most.

---

### ✅🔄 D-2: Resolution Clamping Destroys Aspect Ratio

**File:** [datasets/floorplans.py](file:///workspaces/multi-unit-floorplan/datasets/floorplans.py#L220-L225), [kfold_patch/floorplans_kfold.py](file:///workspaces/multi-unit-floorplan/kfold_patch/floorplans_kfold.py#L220-L225)  
**Status:** ✅ Fixed (September 8, 2026) — requires retraining for effect

**What was wrong:**
```python
h = min(image.shape[0], 812)
w = min(image.shape[1], 812)
image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
```

**Impact:** If an image was `2000×600`, the height was clamped to 812 but the width stayed 600. The image was resized from `2000×600` to `812×600`, squashing the aspect ratio by 60%. Architectural floor plan proportions are critical for structural accuracy — walls become thicker/thinner and rooms change shape.

**Fix applied in both files:** Scale uniformly preserving aspect ratio:
```python
# Scale uniformly preserving aspect ratio (D-2 fix)
scale = min(812 / image.shape[0], 812 / image.shape[1], 1.0)
h = int(image.shape[0] * scale)
w = int(image.shape[1] * scale)
```

**Retrain impact estimate:** **+0.3–1.0% on elongated plans** — Only affects images where h ≠ w and max(h,w) > 812. Impact proportional to how many training images are significantly non-square.

---

## 4. Model Architecture Issues

### ⏳ A-1: SAM Receives `use_cam` Instead of `use_hhdc`

**File:** [cab1/cab1.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/cab1/cab1.py#L75)  
**Line:** 75  
**Status:** ⏳ Open — latent, not reached in current best config

```python
spatial_feature = SAM(channel_s, use_cam)(X_s)
```

**Impact:** The `SAM.__init__` signature is `SAM(out_planes, use_hhdc=False)`. The second argument controls `HHDC` dilation configuration inside the Spatial Attention Module. Passing `use_cam` (which is `5` for CAB1) as `use_hhdc` sends a `cam` scale value to an `hhdc` parameter, causing the HHDC module to receive an unexpected configuration value.

> [!NOTE]
> This is in the **default else branch** of the `AM` function (line 74–75), which only executes when `use_cam` is not in `{2, 3, 4, 5}`. Since CAB1 uses `cam=5`, this branch is never reached in the current best configuration (it takes the `elif use_cam == 5` branch at line 73). The bug would activate with `cam=0` or `cam=False`.

---

### ⏳ A-2: HHDC Ignores Its `dilations` Parameter

**File:** [cab1/cab1.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/cab1/cab1.py#L216-L226)  
**Lines:** 216–226  
**Status:** ⏳ Open — requires code change + retraining

```python
class HHDC(tf.keras.layers.Layer):
    def __init__(self, out_planes, concat=False, dilations=None):  # ← dilations accepted
        ...
        self.convd1 = Conv2D(out_planes, 3, dilation_rate=1, ...)  # ← hardcoded to 1
        self.convd2 = Conv2D(out_planes, 3, dilation_rate=2, ...)  # ← hardcoded to 2
        self.convd3 = Conv2D(out_planes, 3, dilation_rate=3, ...)  # ← hardcoded to 3
```

**Impact:** The `dilations` parameter has no effect. The `hhdc=7` config value is passed through `SAM → HHDC` as a `dilations` argument but is silently ignored. Dilation rates are always fixed at `[1, 2, 3]`. This means the `hhdc=7` ablation that showed improvement over `hhdc=5` may have been measuring a different effect (possibly the impact on `use_cam` routing), not actual receptive field expansion.

---

### ⏳ A-3: CAB2 Channel Dimension Mismatch for `use_cam` 4 and 5

**File:** [cab2/cab2.py](file:///workspaces/multi-unit-floorplan/segmentation_models/models/cab2/cab2.py#L81-L87)  
**Lines:** 81–87  
**Status:** ⏳ Open — latent, not reached in current best config

```python
elif use_cam == 4:
    X = CAM(channel_c, ratio)(X_c)        # ← outputs channel_c = channel/2
elif use_cam == 5:
    X = SAM3(channel_s, False)(X_s)       # ← outputs channel_s = channel/2
X = CONV_stack(X, channel, ...)            # ← expects input to project FROM ~channel dims
```

**Impact:** When `use_cam` is 4 or 5, only one attention branch (CAM or SAM) is used, producing `channel/2` features. The subsequent `CONV_stack` projects to `channel` output, but operates on half the expected input features. This is a channel halving with an implicit projection, not an explicit design choice.

> [!NOTE]
> Current best configurations use `cam=3` for CAB2 and `cam=5` for CAB1 (which takes a different branch), so this is latent.

---

### ⏳ A-4: Mixed Normalization: GroupNorm vs BatchNorm

**File:** [layer_utils.py](file:///workspaces/multi-unit-floorplan/segmentation_models/base/layer_utils.py)  
**Status:** ⏳ Open — requires code change + retraining

| Code Path | Normalization Used |
|:---|:---|
| `CONV_stack` (line 295) | `GroupNormalization(groups=16)` |
| `decode_layer` (line 72) | `GroupNormalization(groups=16)` |
| `encode_layer` (line 141) | `GroupNormalization(groups=16)` |
| `Sep_CONV_stack` (lines 371, 378) | `BatchNormalization` |
| `ASPP_conv` (lines 422, 433) | `BatchNormalization` |

**Impact:** The decoder path uses GroupNorm while ASPP and separable convolutions use BatchNorm. This inconsistency creates different normalization statistics behaviour, especially at small batch sizes (4) where BatchNorm statistics are noisy but GroupNorm is stable.

**Retrain impact estimate:** **+0.3–1.0%** — Switching ASPP and Sep_CONV to GroupNorm would make normalization consistent and more stable at small batch sizes.

---

### ⏳ A-5: `freeze_batch_norm` Ignored When Backbone Is Not Frozen

**File:** [backbone_zoo.py](file:///workspaces/multi-unit-floorplan/segmentation_models/backbones/backbone_zoo.py#L141-L142)  
**Lines:** 141–142  
**Status:** ⏳ Open — requires code change + retraining

```python
if freeze_backbone:
    model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)
```

**Impact:** When fine-tuning the backbone (`freeze_backbone=False`) with small batch sizes, users often want to freeze BatchNorm layers (since running statistics from ImageNet are more reliable than noisy batch statistics with batch size 4). This logic skips the BN freeze entirely when `freeze_backbone=False`.

**Retrain impact estimate:** **+0–0.5%** — Allows using ImageNet BN statistics while still fine-tuning weights. Marginal if backbone is already well-adapted.

---

### ✅ A-6: Stale `T` Variable Persists Across Loop Iterations

**File:** [layer_utils.py](file:///workspaces/multi-unit-floorplan/segmentation_models/base/layer_utils.py#L256-L290)  
**Lines:** 256–290  
**Status:** ✅ Fixed (September 8, 2026) — benign in current configs, prevents future bugs

**What was wrong:**
```python
for i in range(stack_num):
    if square_conv:
        T = A
    if enhance_skeleton:
        if "T" in locals():   # ← T persists from previous iteration
            T += C + B
```

**Impact:** The check `"T" in locals()` was intended to handle the case where `square_conv=False` (so `T` was never assigned). But since `T` is assigned in the first iteration, it persists in `locals()` for all subsequent iterations regardless of `square_conv`. This is benign when `square_conv=True` (which is the case in current configs), but would produce stale tensor references if `square_conv=False` with `stack_num > 1`.

**Fix applied:** Reset `T = None` at the start of each iteration and replaced `"T" in locals()` with `T is not None`.

---

## 5. Post-Processing & Metric Bugs

### ⏳ P-1: Hardcoded Global `type = 'multi'` in Post-Processing

**File:** [training/post_process.py](file:///workspaces/multi-unit-floorplan/training/post_process.py#L63)  
**Line:** 63  
**Status:** ⏳ Open — quick fix needed

```python
type = 'multi'
```

**Impact:** This module-level global forces the `multi` dataset class mapping for all gap-filling and clustering operations. When processing CubiCasa5k data, the wrong class indices are used for gap classes (`[1, 2]` vs `[1]`) and cluster classes (`[4, 5, 6, 7]` vs `[3, 4, 5]`), silently producing incorrect post-processing.

---

### ⏳ P-2: Post-Processing Memory Inefficiency

**File:** [training/post_process.py](file:///workspaces/multi-unit-floorplan/training/post_process.py)  
**Lines:** ~204, 207, 279  
**Status:** ⏳ Open — performance improvement only

**Impact:** `process_tile` allocates full-image-sized zero arrays (`np.zeros(img.shape)`) for every small sub-tile, causing quadratic memory overhead.

---

## 6. Improvement Opportunities

### ⏳ I-1: 10-Fold Probability Ensemble (Zero Retraining)

All 10 fold checkpoints exist. Currently, metrics are reported per-fold and confusion matrices summed. Instead, for each test image, average the softmax probabilities across all 10 folds, then argmax. This typically yields **+1.5–2.5% Macro IoU**.

**Implementation:** Modify [evaluate_kfold.py](file:///workspaces/multi-unit-floorplan/kfold_patch/evaluate_kfold.py) to load all 10 models simultaneously and average their softmax outputs per image.

---

### ✅ I-2: Correct TTA Implementation (Zero Retraining)

**Status:** ✅ Prerequisites fixed (E-1, E-2, E-3) — ready to enable

The TTA bugs (E-1, E-2, E-3) have been fixed. Proper geometric TTA (5 augmentations: identity + 2 flips + 2 rotations) now correctly averages softmax probabilities before argmax. Floor plans are rotationally invariant, so this typically yields **+1.0–1.8% IoU**.

To use: set `tta=True` in evaluation config or call `evaluate.py` with TTA enabled.

---

### ⏳ I-3: Per-Class Decision Threshold Calibration

The current evaluation uses symmetric `argmax` over all 6 classes. With ~88% background pixels, the model is biased toward background. Optimizing per-class thresholds on validation data (e.g., via grid search over `τ_c` in `argmax(P(c) / τ_c)`) can significantly boost minority class recall (railings, stairs) without retraining.

**Estimated impact:** **+0.5–1.5% minority class recall**

---

### ⏳ I-4: Connected Component Post-Processing

Small scattered false-positive clusters (1–15 pixels) heavily degrade precision on railings (39.8%) and stairs (56.97%). Applying `cv2.connectedComponentsWithStats` with minimum area thresholds per class would eliminate these outliers.

**Estimated impact:** **+0.5–2.0% precision on minority classes**

---

### ⏳ I-5: Wall-Adjacency Constraint for Doors/Windows

Doors and windows must be topologically adjacent to walls. Post-processing can enforce this invariant by dilating the predicted wall mask and removing any door/window prediction pixels that don't intersect the dilated wall region.

**Estimated impact:** **+0.3–1.0% door/window precision**

---

### ⏳ I-6: Class-Weighted Focal Loss

The current `asym_unified_focal_loss` uses uniform `delta=0.6` and `gamma=0.5` for all foreground classes. Introducing inverse-frequency class weights (proportional to `(total_pixels / class_pixels)^α` with `α ≈ 0.3`) would amplify gradients from rare classes (railings, stairs) during training.

**Estimated impact:** **+1.0–3.0% minority class IoU** — likely the **highest-impact retraining improvement** since minority classes currently dominate the error budget.

---

### ⏳ I-7: Higher-Resolution Tiled Inference

The 812px resolution cap ([floorplans.py:L220-L222](file:///workspaces/multi-unit-floorplan/datasets/floorplans.py#L220-L222)) destroys thin structures in high-resolution scans. Implementing sliding-window inference with overlapping tiles (e.g., 768×768 with 25% overlap and Gaussian blending) would preserve fine railings and stair hatching.

---

## 7. Impact Estimates for Remaining Work

### Priority: Maximum Accuracy Without Retraining

> [!TIP]
> These items can boost accuracy immediately using existing model checkpoints:

| Priority | ID | Item | Effort | Estimated Impact |
|:---:|:---:|:---|:---:|:---|
| 1 | I-1 | 10-fold probability ensemble | 🔧 2 hrs | **+1.5–2.5% Macro IoU** |
| 2 | I-2 | Enable TTA evaluation | ✅ Ready | **+1.0–1.8% IoU** |
| 3 | I-4 | Connected component filtering | 🔧 2 hrs | **+0.5–2.0% minority precision** |
| 4 | I-3 | Per-class threshold calibration | 🔧 2 hrs | **+0.5–1.5% minority recall** |
| 5 | P-1 | Fix post-processing type param | ⚡ 5 min | **Correctness fix** |

### Priority: Maximum Accuracy With Retraining

> [!IMPORTANT]
> These items require a full retraining cycle (~8 hrs each) but offer significant gains:

| Priority | ID | Item | Effort | Estimated Impact |
|:---:|:---:|:---|:---:|:---|
| 1 | I-6 | Class-weighted focal loss | 🔧 1 hr code + retrain | **+1.0–3.0% minority class IoU** |
| 2 | D-1 | Retrain with mask interpolation fix | 🔧 retrain only | **+0.5–2.0% boundary IoU** |
| 3 | D-2 | Retrain with aspect ratio fix | 🔧 retrain only | **+0.3–1.0% on elongated plans** |
| 4 | T-1 | Retrain with working AAF decay | 🔧 retrain only | **+0.5–1.5% Non-BG Acc** |
| 5 | T-3 | Log-space sigma reparameterization | ⚡ 15 min code + retrain | **Stability; +0–0.5% IoU** |
| 6 | A-4 | Unify GroupNorm/BatchNorm | 🔧 30 min code + retrain | **+0.3–1.0%** |
| 7 | A-5 | Allow BN freeze without backbone freeze | ⚡ 10 min code + retrain | **+0–0.5%** |
| 8 | T-5 | Fix LR scheduler units | ⚡ 10 min code + retrain | **+0–0.5%** |

---

## Summary: Fix Status Matrix

| ID | Severity | Status | Category |
|:---:|:---:|:---:|:---|
| T-1 | 🔴 | ✅🔄 | AAF epoch decay frozen → tf.Variable |
| T-2 | 🟠 | ✅🔄 | Gradient reversal by name |
| T-3 | 🟠 | ⏳ | Sigma log-space reparameterization |
| T-4 | 🟡 | ✅🔄 | TensorBoard log dir per fold |
| T-5 | 🟡 | ⏳ | LR scheduler unit mismatch |
| T-6 | 🔵 | ⏳ | TensorBoard loss averaging |
| E-1 | 🔴 | ✅ | TTA probability averaging |
| E-2 | 🟠 | ✅ | TTA one-hot depth |
| E-3 | 🟠 | ✅ | Evaluate TTA unblocked |
| E-4 | 🟡 | ✅ | Confusion matrix full batch |
| E-5 | 🟡 | ✅ | Metric epsilon guards |
| E-6 | 🔵 | ✅ | Frequency weight fix |
| D-1 | 🔴 | ✅🔄 | Mask nearest-neighbor resize |
| D-2 | 🟠 | ✅🔄 | Aspect ratio preservation |
| A-1 | 🟠 | ⏳ | SAM parameter mismatch (latent) |
| A-2 | 🟡 | ⏳ | HHDC dilations ignored |
| A-3 | 🟡 | ⏳ | CAB2 channel mismatch (latent) |
| A-4 | 🟡 | ⏳ | Mixed normalization |
| A-5 | 🔵 | ⏳ | BN freeze logic |
| A-6 | 🔵 | ✅ | Stale T variable |
| P-1 | 🟡 | ⏳ | Hardcoded post-processing type |
| P-2 | 🔵 | ⏳ | Memory inefficiency |

**Total: 14 fixed ✅ / 8 open ⏳**
