# MLSTRUCT-FP results (bg / wall)

Protocol: 5-fold CV over 934 plan images (seed 0; split k: test = fold k, val = fold k+1), tiled inference (768 px tiles, 25% overlap, Hann-weighted softmax averaging), predictions upsampled and scored at **native** plan resolution. `s40` = plans resampled to 40 px/m (0.2 m walls ≈ 8 px, matching CubiCasa training); `native` = original pixels (as Paek et al.). CubiCasa walls → wall, every other class → bg. Values in %, mean ± std over the 5 CV test folds.

## 1. Zero-shot, all 10 CubiCasa fold models (s40)

Per CV fold, the confusion matrices of the 10 fold models are pooled.

| Model | Wall Rec | Wall Prec | Wall F1 | Wall IoU | mIoU (bg+wall) | bg acc | Overall acc | Wall IoU pooled |
|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 54.5 ± 1.5 | 35.2 ± 1.4 | 42.7 ± 1.5 | 27.2 ± 1.2 | 61.6 ± 0.6 | 97.3 ± 0.1 | 96.1 ± 0.1 | 27.2 |
| CAB1 V2M | 70.1 ± 1.1 | 42.6 ± 1.5 | 53.0 ± 1.5 | 36.0 ± 1.4 | 66.3 ± 0.7 | 97.4 ± 0.1 | 96.7 ± 0.1 | 36.0 |
| CAB2 B4 | 46.4 ± 1.5 | 34.5 ± 1.6 | 39.6 ± 1.5 | 24.7 ± 1.2 | 60.5 ± 0.6 | 97.6 ± 0.1 | 96.3 ± 0.1 | 24.7 |
| CAB2 V2M | 53.5 ± 1.7 | 40.9 ± 1.7 | 46.4 ± 1.7 | 30.2 ± 1.4 | 63.4 ± 0.7 | 97.9 ± 0.0 | 96.7 ± 0.1 | 30.2 |

## 2. Zero-shot scale sensitivity (fine-tune checkpoint, best-val CubiCasa fold)

| Model | Wall Rec | Wall Prec | Wall F1 | Wall IoU | mIoU (bg+wall) | bg acc | Overall acc | Wall IoU pooled |
|---|---|---|---|---|---|---|---|---|
| CAB1 B4 @ s30 | 60.2 ± 1.4 | 30.4 ± 1.4 | 40.4 ± 1.5 | 25.3 ± 1.2 | 60.3 ± 0.6 | 96.3 ± 0.1 | 95.3 ± 0.1 | 25.3 |
| CAB1 V2M @ s30 | 74.2 ± 1.0 | 41.4 ± 1.6 | 53.1 ± 1.5 | 36.2 ± 1.4 | 66.3 ± 0.8 | 97.2 ± 0.1 | 96.5 ± 0.1 | 36.2 |
| CAB2 B4 @ s30 | 61.7 ± 1.0 | 29.7 ± 1.6 | 40.1 ± 1.6 | 25.1 ± 1.3 | 60.1 ± 0.7 | 96.0 ± 0.2 | 95.1 ± 0.2 | 25.1 |
| CAB2 V2M @ s30 | 56.9 ± 1.8 | 40.7 ± 1.9 | 47.5 ± 1.9 | 31.1 ± 1.6 | 63.9 ± 0.9 | 97.8 ± 0.1 | 96.7 ± 0.1 | 31.1 |
| CAB1 B4 @ s40 | 60.8 ± 1.3 | 32.2 ± 1.2 | 42.1 ± 1.3 | 26.7 ± 1.1 | 61.1 ± 0.6 | 96.5 ± 0.1 | 95.6 ± 0.1 | 26.7 |
| CAB1 V2M @ s40 | 73.6 ± 0.9 | 43.0 ± 1.6 | 54.3 ± 1.5 | 37.2 ± 1.4 | 66.9 ± 0.8 | 97.4 ± 0.1 | 96.7 ± 0.1 | 37.2 |
| CAB2 B4 @ s40 | 63.6 ± 1.3 | 32.9 ± 1.7 | 43.4 ± 1.7 | 27.7 ± 1.4 | 61.6 ± 0.8 | 96.5 ± 0.1 | 95.6 ± 0.1 | 27.7 |
| CAB2 V2M @ s40 | 55.7 ± 1.7 | 40.8 ± 1.8 | 47.1 ± 1.8 | 30.8 ± 1.5 | 63.7 ± 0.8 | 97.8 ± 0.1 | 96.7 ± 0.1 | 30.8 |
| CAB1 B4 @ s50 | 62.9 ± 1.5 | 33.4 ± 1.3 | 43.6 ± 1.4 | 27.9 ± 1.2 | 61.8 ± 0.6 | 96.6 ± 0.1 | 95.7 ± 0.1 | 27.9 |
| CAB1 V2M @ s50 | 74.5 ± 0.7 | 43.5 ± 1.5 | 54.9 ± 1.4 | 37.8 ± 1.3 | 67.3 ± 0.7 | 97.4 ± 0.1 | 96.8 ± 0.1 | 37.8 |
| CAB2 B4 @ s50 | 66.4 ± 1.0 | 33.9 ± 1.3 | 44.9 ± 1.4 | 29.0 ± 1.1 | 62.3 ± 0.6 | 96.5 ± 0.1 | 95.7 ± 0.1 | 28.9 |
| CAB2 V2M @ s50 | 56.7 ± 1.6 | 41.1 ± 1.6 | 47.7 ± 1.7 | 31.3 ± 1.4 | 64.0 ± 0.7 | 97.8 ± 0.1 | 96.7 ± 0.1 | 31.3 |
| CAB1 B4 @ s60 | 63.3 ± 1.7 | 34.1 ± 1.3 | 44.3 ± 1.5 | 28.5 ± 1.3 | 62.1 ± 0.7 | 96.7 ± 0.1 | 95.8 ± 0.1 | 28.5 |
| CAB1 V2M @ s60 | 75.4 ± 0.8 | 43.6 ± 1.6 | 55.2 ± 1.5 | 38.1 ± 1.4 | 67.4 ± 0.8 | 97.3 ± 0.1 | 96.8 ± 0.1 | 38.1 |
| CAB2 B4 @ s60 | 66.4 ± 0.9 | 33.7 ± 1.1 | 44.7 ± 1.2 | 28.8 ± 1.0 | 62.2 ± 0.5 | 96.5 ± 0.1 | 95.7 ± 0.1 | 28.8 |
| CAB2 V2M @ s60 | 57.3 ± 1.4 | 41.2 ± 1.5 | 48.0 ± 1.5 | 31.6 ± 1.3 | 64.1 ± 0.7 | 97.8 ± 0.0 | 96.7 ± 0.1 | 31.6 |
| CAB1 B4 @ native | 59.6 ± 1.3 | 32.4 ± 1.1 | 42.0 ± 1.2 | 26.6 ± 1.0 | 61.1 ± 0.5 | 96.6 ± 0.1 | 95.6 ± 0.1 | 26.6 |
| CAB1 V2M @ native | 57.7 ± 0.6 | 37.1 ± 1.1 | 45.1 ± 0.9 | 29.2 ± 0.8 | 62.7 ± 0.4 | 97.3 ± 0.1 | 96.3 ± 0.1 | 29.2 |
| CAB2 B4 @ native | 45.7 ± 0.8 | 22.6 ± 0.8 | 30.2 ± 0.9 | 17.8 ± 0.6 | 56.1 ± 0.3 | 95.8 ± 0.0 | 94.4 ± 0.0 | 17.8 |
| CAB2 V2M @ native | 46.8 ± 0.8 | 36.3 ± 1.0 | 40.9 ± 1.0 | 25.7 ± 0.8 | 61.0 ± 0.4 | 97.8 ± 0.0 | 96.4 ± 0.1 | 25.7 |

## 3a. Fine-tuned, backbone unfrozen (s40)

Same checkpoint as §2; LR 3e-5, ≤300 epochs × 500 steps of random 512 crops, early stopping on val loss (patience 25). Zero-shot row = the same checkpoint on the same test folds.

| Model | Wall Rec | Wall Prec | Wall F1 | Wall IoU | mIoU (bg+wall) | bg acc | Overall acc | Wall IoU pooled |
|---|---|---|---|---|---|---|---|---|
| CAB1 B4 zeroshot | 60.8 ± 1.3 | 32.2 ± 1.2 | 42.1 ± 1.3 | 26.7 ± 1.1 | 61.1 ± 0.6 | 96.5 ± 0.1 | 95.6 ± 0.1 | 26.7 |
| CAB1 B4 adabn | 52.0 ± 1.2 | 32.0 ± 1.3 | 39.6 ± 1.4 | 24.7 ± 1.1 | 60.2 ± 0.6 | 97.0 ± 0.1 | 95.8 ± 0.1 | 24.7 |
| CAB1 B4 finetuned | 95.1 ± 1.3 | 79.0 ± 1.6 | 86.3 ± 0.9 | 75.9 ± 1.4 | 87.5 ± 0.7 | 99.3 ± 0.1 | 99.2 ± 0.0 | 75.9 |
| CAB1 V2M zeroshot | 73.6 ± 0.9 | 43.0 ± 1.6 | 54.3 ± 1.5 | 37.2 ± 1.4 | 66.9 ± 0.8 | 97.4 ± 0.1 | 96.7 ± 0.1 | 37.2 |
| CAB1 V2M adabn | 68.0 ± 1.5 | 30.6 ± 1.8 | 42.2 ± 2.0 | 26.8 ± 1.6 | 60.9 ± 0.9 | 95.8 ± 0.2 | 95.1 ± 0.2 | 26.7 |
| CAB1 V2M finetuned | 95.8 ± 0.8 | 79.3 ± 1.6 | 86.7 ± 1.1 | 76.6 ± 1.7 | 87.9 ± 0.9 | 99.3 ± 0.1 | 99.2 ± 0.1 | 76.6 |
| CAB2 B4 zeroshot | 63.6 ± 1.3 | 32.9 ± 1.7 | 43.4 ± 1.7 | 27.7 ± 1.4 | 61.6 ± 0.8 | 96.5 ± 0.1 | 95.6 ± 0.1 | 27.7 |
| CAB2 B4 adabn | 59.1 ± 0.8 | 32.5 ± 1.7 | 41.9 ± 1.6 | 26.5 ± 1.3 | 61.1 ± 0.7 | 96.7 ± 0.2 | 95.7 ± 0.2 | 26.5 |
| CAB2 B4 finetuned | 88.1 ± 2.0 | 72.3 ± 1.7 | 79.4 ± 1.7 | 65.9 ± 2.3 | 82.3 ± 1.2 | 99.1 ± 0.1 | 98.8 ± 0.1 | 65.8 |
| CAB2 V2M zeroshot | 55.7 ± 1.7 | 40.8 ± 1.8 | 47.1 ± 1.8 | 30.8 ± 1.5 | 63.7 ± 0.8 | 97.8 ± 0.1 | 96.7 ± 0.1 | 30.8 |
| CAB2 V2M adabn | 44.2 ± 1.6 | 33.1 ± 2.3 | 37.9 ± 2.1 | 23.4 ± 1.6 | 59.8 ± 0.9 | 97.6 ± 0.1 | 96.2 ± 0.1 | 23.4 |
| CAB2 V2M finetuned | 94.2 ± 1.5 | 75.3 ± 2.2 | 83.7 ± 1.4 | 71.9 ± 2.1 | 85.5 ± 1.1 | 99.2 ± 0.1 | 99.0 ± 0.1 | 71.9 |

## 3b. Fine-tuned, backbone unfrozen (native)

Same checkpoint as §2; LR 3e-5, ≤300 epochs × 500 steps of random 512 crops, early stopping on val loss (patience 25). Zero-shot row = the same checkpoint on the same test folds.

| Model | Wall Rec | Wall Prec | Wall F1 | Wall IoU | mIoU (bg+wall) | bg acc | Overall acc | Wall IoU pooled |
|---|---|---|---|---|---|---|---|---|
| CAB1 B4 zeroshot | 59.6 ± 1.3 | 32.4 ± 1.1 | 42.0 ± 1.2 | 26.6 ± 1.0 | 61.1 ± 0.5 | 96.6 ± 0.1 | 95.6 ± 0.1 | 26.6 |
| CAB1 B4 finetuned | 90.7 ± 3.3 | 78.2 ± 4.3 | 83.9 ± 3.0 | 72.3 ± 4.5 | 85.7 ± 2.4 | 99.3 ± 0.2 | 99.1 ± 0.2 | 72.2 |
| CAB1 V2M zeroshot | 57.7 ± 0.6 | 37.1 ± 1.1 | 45.1 ± 0.9 | 29.2 ± 0.8 | 62.7 ± 0.4 | 97.3 ± 0.1 | 96.3 ± 0.1 | 29.2 |
| CAB1 V2M finetuned | 91.6 ± 2.2 | 73.0 ± 1.8 | 81.2 ± 0.6 | 68.3 ± 0.9 | 83.6 ± 0.4 | 99.1 ± 0.1 | 98.9 ± 0.1 | 68.3 |
| CAB2 B4 zeroshot | 45.7 ± 0.8 | 22.6 ± 0.8 | 30.2 ± 0.9 | 17.8 ± 0.6 | 56.1 ± 0.3 | 95.8 ± 0.0 | 94.4 ± 0.0 | 17.8 |
| CAB2 B4 finetuned | 80.0 ± 4.7 | 61.3 ± 5.9 | 69.0 ± 2.2 | 52.7 ± 2.6 | 75.4 ± 1.4 | 98.6 ± 0.4 | 98.1 ± 0.3 | 52.4 |
| CAB2 V2M zeroshot | 46.8 ± 0.8 | 36.3 ± 1.0 | 40.9 ± 1.0 | 25.7 ± 0.8 | 61.0 ± 0.4 | 97.8 ± 0.0 | 96.4 ± 0.1 | 25.7 |
| CAB2 V2M finetuned | 88.4 ± 2.8 | 65.8 ± 3.1 | 75.4 ± 2.2 | 60.5 ± 2.9 | 79.5 ± 1.5 | 98.7 ± 0.2 | 98.5 ± 0.2 | 60.5 |

## 4. Comparison with published MLSTRUCT-FP results

| Method | Wall Rec | Wall Prec | Wall F1 | Wall IoU | mIoU (bg+wall) | bg acc | Overall acc |
|---|---|---|---|---|---|---|---|
| Paek et al. 2025, SegFormer-B5 patch-wise (Table 3) | 85 | - | - | - | 92 | 99 | 94 |
| R2V, retrained by Paek et al. | - | - | - | - | 84 | - | - |
| DeepFP, retrained by Paek et al. | - | - | - | - | 84 | - | - |
| Pizarro et al. 2023 U-Net (via Paek et al.) | - | - | - | - | 77 | - | - |
| Kratochvila et al. v1 CAB1 B2 (Table 9, 60/20/20) | 92 | - | 89 | 81 | - | - | - |
| Kratochvila et al. v1 CAB2 B2 (Table 9, 60/20/20) | 92 | - | 90 | 83 | - | - | - |
| **Ours CAB1 B4, fine-tuned (s40, 5-fold CV)** | 95.1 ± 1.3 | 79.0 ± 1.6 | 86.3 ± 0.9 | 75.9 ± 1.4 | 87.5 ± 0.7 | 99.3 ± 0.1 | 99.2 ± 0.0 |
| **Ours CAB1 V2M, fine-tuned (s40, 5-fold CV)** | 95.8 ± 0.8 | 79.3 ± 1.6 | 86.7 ± 1.1 | 76.6 ± 1.7 | 87.9 ± 0.9 | 99.3 ± 0.1 | 99.2 ± 0.1 |
| **Ours CAB2 B4, fine-tuned (s40, 5-fold CV)** | 88.1 ± 2.0 | 72.3 ± 1.7 | 79.4 ± 1.7 | 65.9 ± 2.3 | 82.3 ± 1.2 | 99.1 ± 0.1 | 98.8 ± 0.1 |
| **Ours CAB2 V2M, fine-tuned (s40, 5-fold CV)** | 94.2 ± 1.5 | 75.3 ± 2.2 | 83.7 ± 1.4 | 71.9 ± 2.1 | 85.5 ± 1.1 | 99.2 ± 0.1 | 99.0 ± 0.1 |
| **Ours CAB1 B4, fine-tuned (native, 5-fold CV)** | 90.7 ± 3.3 | 78.2 ± 4.3 | 83.9 ± 3.0 | 72.3 ± 4.5 | 85.7 ± 2.4 | 99.3 ± 0.2 | 99.1 ± 0.2 |
| **Ours CAB1 V2M, fine-tuned (native, 5-fold CV)** | 91.6 ± 2.2 | 73.0 ± 1.8 | 81.2 ± 0.6 | 68.3 ± 0.9 | 83.6 ± 0.4 | 99.1 ± 0.1 | 98.9 ± 0.1 |
| **Ours CAB2 B4, fine-tuned (native, 5-fold CV)** | 80.0 ± 4.7 | 61.3 ± 5.9 | 69.0 ± 2.2 | 52.7 ± 2.6 | 75.4 ± 1.4 | 98.6 ± 0.4 | 98.1 ± 0.3 |
| **Ours CAB2 V2M, fine-tuned (native, 5-fold CV)** | 88.4 ± 2.8 | 65.8 ± 3.1 | 75.4 ± 2.2 | 60.5 ± 2.9 | 79.5 ± 1.5 | 98.7 ± 0.2 | 98.5 ± 0.2 |

Notes: Paek et al. report a single run on an "original split" that MLSTRUCT-FP does not define; their per-class accuracy is recall and their mIoU includes background. v1 Table 9 used a random 60/20/20 split and an undocumented tiling. `fp.json` has no building ID, so floors of one building can fall into different folds (same limitation as both references). MLSTRUCT-FP labels structural walls only, so thin partition walls that CubiCasa models segment count as false positives zero-shot. All numbers use plain argmax (p_wall > 0.5). A validation-tuned cutoff (kfold_patch/tune_threshold.py; tested on CAB1 V2M s40 split 0 only) picks p_wall > 0.73. On test it gives wall IoU 79.0 -> 80.8, mIoU 89.2 -> 90.1, precision 82.0 -> 86.1 and recall 95.6 -> 92.9. On R3D / CVC-FP / WAFFLE the same tuning changes mIoU by only -0.6 to +0.6, so it is not applied anywhere (results/threshold_tuning/SUMMARY.md).
