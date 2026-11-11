
### r3d_test

| Model | Backbone | Mode | Acc | Acc no-bg | Walls IoU | Openings IoU | mIoU (no bg) | Walls class acc | Openings class acc |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 | EfficientNetB4 | pooled (10 folds) | 97.94 | 80.09 | 80.69 | 48.73 | 64.71 | 84.04 | 64.09 |
| | | mean ± std | 97.94 ± 0.55 | 80.09 ± 8.23 | 80.78 ± 5.81 | 47.47 ± 15.86 | 64.12 ± 10.79 | 84.04 ± 5.09 | 64.09 ± 21.40 |
| | | single: fold 9 (val 96.30%) | 98.07 | 82.82 | 82.70 | 50.99 | 66.85 | 86.29 | 68.76 |
| CAB1 | EfficientNetV2M | pooled (10 folds) | 97.98 | 82.02 | 81.55 | 50.23 | 65.89 | 84.90 | 70.33 |
| | | mean ± std | 97.98 ± 0.10 | 82.02 ± 1.74 | 81.54 ± 1.45 | 50.23 ± 1.14 | 65.89 ± 1.13 | 84.90 ± 1.74 | 70.33 ± 2.74 |
| | | single: fold 1 (val 96.54%) | 98.01 | 83.54 | 82.76 | 49.15 | 65.95 | 86.61 | 71.08 |
| CAB2 | EfficientNetB4 | pooled (10 folds) | 97.17 | 79.71 | 76.42 | 34.73 | 55.58 | 86.85 | 50.76 |
| | | mean ± std | 97.17 ± 0.45 | 79.71 ± 4.05 | 76.56 ± 3.96 | 34.78 ± 8.07 | 55.67 ± 5.31 | 86.85 ± 2.93 | 50.76 ± 11.64 |
| | | single: fold 3 (val 96.40%) | 97.99 | 86.33 | 80.77 | 56.92 | 68.85 | 88.59 | 77.17 |
| CAB2 | EfficientNetV2M | pooled (10 folds) | 98.19 | 81.91 | 82.07 | 55.37 | 68.72 | 85.11 | 68.94 |
| | | mean ± std | 98.19 ± 0.05 | 81.91 ± 0.76 | 82.07 ± 0.82 | 55.37 ± 0.74 | 68.72 ± 0.39 | 85.11 ± 1.01 | 68.94 ± 1.13 |
| | | single: fold 4 (val 96.38%) | 98.24 | 81.85 | 82.80 | 55.00 | 68.90 | 85.44 | 67.31 |
| CUBICASA5K | VGG16 | pooled (10 folds) | 96.29 | 50.70 | 54.30 | 32.76 | 43.53 | 54.72 | 34.42 |
| | | mean ± std | 96.29 ± 0.33 | 50.70 ± 4.75 | 54.29 ± 4.95 | 32.71 ± 3.83 | 43.50 ± 4.02 | 54.72 ± 5.14 | 34.42 ± 4.42 |
| | | single: fold 1 (val 97.03%) | 96.56 | 54.68 | 57.55 | 38.05 | 47.80 | 58.05 | 41.02 |
| ZENG | VGG16 | pooled (10 folds) | 96.07 | 47.62 | 50.28 | 34.27 | 42.27 | 50.56 | 35.71 |
| | | mean ± std | 96.07 ± 0.27 | 47.62 ± 3.90 | 50.27 ± 4.28 | 34.26 ± 2.30 | 42.27 ± 3.03 | 50.56 ± 4.43 | 35.71 ± 2.51 |
| | | single: fold 1 (val 96.36%) | 96.59 | 55.03 | 58.62 | 36.40 | 47.51 | 59.22 | 38.00 |

### r3d_all

| Model | Backbone | Mode | Acc | Acc no-bg | Walls IoU | Openings IoU | mIoU (no bg) | Walls class acc | Openings class acc |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 | EfficientNetB4 | pooled (10 folds) | 97.87 | 81.34 | 80.94 | 50.31 | 65.62 | 84.76 | 66.41 |
| | | mean ± std | 97.87 ± 0.62 | 81.34 ± 8.41 | 81.05 ± 6.12 | 49.00 ± 16.36 | 65.02 ± 11.23 | 84.76 ± 5.28 | 66.41 ± 22.16 |
| | | single: fold 9 (val 96.30%) | 98.03 | 85.20 | 83.40 | 53.08 | 68.24 | 87.98 | 73.11 |
| CAB1 | EfficientNetV2M | pooled (10 folds) | 97.89 | 83.25 | 81.24 | 52.48 | 66.86 | 85.47 | 73.60 |
| | | mean ± std | 97.89 ± 0.06 | 83.25 ± 1.14 | 81.24 ± 0.76 | 52.49 ± 0.95 | 66.87 ± 0.68 | 85.47 ± 1.17 | 73.60 ± 1.83 |
| | | single: fold 1 (val 96.54%) | 97.92 | 84.66 | 82.33 | 51.50 | 66.92 | 87.08 | 74.09 |
| CAB2 | EfficientNetB4 | pooled (10 folds) | 96.87 | 78.09 | 74.67 | 33.74 | 54.20 | 84.66 | 49.43 |
| | | mean ± std | 96.87 ± 0.50 | 78.09 ± 4.56 | 74.77 ± 4.01 | 33.84 ± 8.73 | 54.31 ± 5.80 | 84.66 ± 3.44 | 49.43 ± 12.15 |
| | | single: fold 3 (val 96.40%) | 97.84 | 86.46 | 79.99 | 58.23 | 69.11 | 88.28 | 78.57 |
| CAB2 | EfficientNetV2M | pooled (10 folds) | 98.24 | 84.18 | 83.51 | 58.47 | 70.99 | 86.85 | 72.54 |
| | | mean ± std | 98.24 ± 0.04 | 84.18 ± 0.69 | 83.51 ± 0.62 | 58.47 ± 0.50 | 70.99 ± 0.25 | 86.85 ± 0.84 | 72.54 ± 0.93 |
| | | single: fold 4 (val 96.38%) | 98.27 | 83.80 | 83.80 | 58.59 | 71.20 | 86.65 | 71.36 |
| CUBICASA5K | VGG16 | pooled (10 folds) | 96.17 | 53.35 | 55.90 | 36.12 | 46.01 | 56.84 | 38.16 |
| | | mean ± std | 96.17 ± 0.33 | 53.35 ± 4.53 | 55.88 ± 4.52 | 36.08 ± 3.97 | 45.98 ± 3.94 | 56.84 ± 4.76 | 38.16 ± 4.59 |
| | | single: fold 1 (val 97.03%) | 96.41 | 56.43 | 58.49 | 40.65 | 49.57 | 59.36 | 43.69 |
| ZENG | VGG16 | pooled (10 folds) | 95.96 | 50.14 | 52.40 | 36.32 | 44.36 | 52.95 | 37.91 |
| | | mean ± std | 95.96 ± 0.27 | 50.14 ± 3.70 | 52.39 ± 3.97 | 36.31 ± 2.27 | 44.35 ± 2.84 | 52.95 ± 4.17 | 37.91 ± 2.49 |
| | | single: fold 1 (val 96.36%) | 96.49 | 57.34 | 60.26 | 38.46 | 49.36 | 61.28 | 40.19 |

### cvc_all

| Model | Backbone | Mode | Acc | Acc no-bg | Walls IoU | Openings IoU | mIoU (no bg) | Walls class acc | Openings class acc |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 | EfficientNetB4 | pooled (10 folds) | 97.39 | 55.94 | 64.59 | 24.74 | 44.67 | 74.89 | 26.97 |
| | | mean ± std | 97.39 ± 0.22 | 55.94 ± 4.53 | 64.66 ± 2.74 | 24.51 ± 8.23 | 44.59 ± 5.40 | 74.89 ± 1.68 | 26.97 ± 9.08 |
| | | single: fold 9 (val 96.30%) | 97.44 | 56.91 | 65.02 | 26.87 | 45.94 | 75.04 | 29.19 |
| CAB1 | EfficientNetV2M | pooled (10 folds) | 97.37 | 58.14 | 64.21 | 27.10 | 45.66 | 76.18 | 30.55 |
| | | mean ± std | 97.37 ± 0.05 | 58.14 ± 1.11 | 64.22 ± 0.76 | 27.10 ± 0.93 | 45.66 ± 0.69 | 76.18 ± 1.31 | 30.55 ± 1.21 |
| | | single: fold 1 (val 96.54%) | 97.43 | 59.22 | 65.32 | 28.01 | 46.67 | 77.32 | 31.55 |
| CAB2 | EfficientNetB4 | pooled (10 folds) | 96.25 | 53.85 | 52.70 | 13.98 | 33.34 | 78.01 | 16.89 |
| | | mean ± std | 96.25 ± 0.38 | 53.85 ± 3.02 | 53.00 ± 3.64 | 13.94 ± 5.43 | 33.47 ± 4.08 | 78.01 ± 1.88 | 16.89 ± 6.44 |
| | | single: fold 3 (val 96.40%) | 97.18 | 59.13 | 60.79 | 28.02 | 44.40 | 77.31 | 31.31 |
| CAB2 | EfficientNetV2M | pooled (10 folds) | 97.50 | 58.41 | 66.48 | 27.32 | 46.90 | 76.95 | 30.05 |
| | | mean ± std | 97.50 ± 0.02 | 58.41 ± 0.86 | 66.48 ± 0.47 | 27.32 ± 0.36 | 46.90 ± 0.32 | 76.95 ± 1.39 | 30.05 ± 0.59 |
| | | single: fold 4 (val 96.38%) | 97.55 | 58.78 | 67.51 | 27.55 | 47.53 | 77.46 | 30.21 |
| CUBICASA5K | VGG16 | pooled (10 folds) | 96.91 | 36.83 | 48.79 | 15.66 | 32.22 | 50.40 | 16.09 |
| | | mean ± std | 96.91 ± 0.11 | 36.83 ± 2.75 | 48.77 ± 3.01 | 15.65 ± 2.40 | 32.21 ± 2.41 | 50.40 ± 3.40 | 16.09 ± 2.53 |
| | | single: fold 1 (val 97.03%) | 97.00 | 39.09 | 50.78 | 17.84 | 34.31 | 52.59 | 18.43 |
| ZENG | VGG16 | pooled (10 folds) | 96.76 | 33.32 | 45.85 | 11.91 | 28.88 | 47.22 | 12.07 |
| | | mean ± std | 96.76 ± 0.09 | 33.32 ± 2.08 | 45.84 ± 2.62 | 11.91 ± 1.46 | 28.88 ± 1.74 | 47.22 ± 2.90 | 12.07 ± 1.50 |
| | | single: fold 1 (val 96.36%) | 96.93 | 37.54 | 51.46 | 13.08 | 32.27 | 53.40 | 13.27 |

CVC-FP: 122 plans from the original release (data/cvc/ImagesGT), GT rasterized from the SVG polygons (Wall -> walls; Door, Window -> openings; rest -> bg). CVC door polygons include the swing arc, so openings IoU is not directly comparable with R3D. Earlier colour-mask CVC results (2026-11-05, invalid GT) are archived in results/archive/cross_dataset_cvc_colorY/.

### waffle_all

| Model | Backbone | Mode | Acc | Acc no-bg | Walls IoU | Openings IoU | mIoU (no bg) | Walls class acc | Openings class acc |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 | EfficientNetB4 | pooled (10 folds) | 93.18 | 61.36 | 55.75 | 20.77 | 38.26 | 62.97 | 43.22 |
| | | mean ± std | 93.18 ± 1.18 | 61.36 ± 13.34 | 55.60 ± 11.34 | 19.67 ± 6.60 | 37.63 ± 8.95 | 62.97 ± 13.24 | 43.22 ± 14.58 |
| | | single: fold 9 (val 96.30%) | 93.30 | 68.54 | 59.93 | 20.26 | 40.10 | 70.14 | 50.45 |
| CAB1 | EfficientNetV2M | pooled (10 folds) | 93.19 | 65.40 | 57.63 | 20.82 | 39.22 | 67.08 | 46.50 |
| | | mean ± std | 93.19 ± 0.24 | 65.40 ± 2.14 | 57.63 ± 1.29 | 20.82 ± 0.67 | 39.22 ± 0.54 | 67.08 ± 2.23 | 46.50 ± 3.56 |
| | | single: fold 1 (val 96.54%) | 93.12 | 65.75 | 57.23 | 20.68 | 38.95 | 67.66 | 44.26 |
| CAB2 | EfficientNetB4 | pooled (10 folds) | 93.63 | 63.40 | 58.14 | 18.52 | 38.33 | 66.22 | 31.62 |
| | | mean ± std | 93.63 ± 0.72 | 63.40 ± 6.68 | 58.14 ± 5.76 | 18.11 ± 3.04 | 38.13 ± 3.34 | 66.22 ± 6.85 | 31.62 ± 9.95 |
| | | single: fold 3 (val 96.40%) | 93.44 | 70.78 | 60.22 | 22.20 | 41.21 | 72.87 | 47.25 |
| CAB2 | EfficientNetV2M | pooled (10 folds) | 93.23 | 59.77 | 54.50 | 24.02 | 39.26 | 60.94 | 46.49 |
| | | mean ± std | 93.23 ± 0.16 | 59.77 ± 3.10 | 54.46 ± 2.08 | 24.06 ± 0.84 | 39.26 ± 0.92 | 60.94 ± 3.35 | 46.49 ± 2.04 |
| | | single: fold 4 (val 96.38%) | 93.27 | 59.14 | 54.59 | 23.12 | 38.85 | 60.27 | 46.39 |
| CUBICASA5K | VGG16 | pooled (10 folds) | 91.84 | 35.41 | 35.90 | 18.21 | 27.06 | 36.59 | 22.10 |
| | | mean ± std | 91.84 ± 0.41 | 35.41 ± 3.74 | 35.89 ± 3.77 | 18.16 ± 1.97 | 27.03 ± 2.55 | 36.59 ± 3.94 | 22.10 ± 3.04 |
| | | single: fold 1 (val 97.03%) | 91.82 | 35.55 | 35.67 | 19.57 | 27.62 | 36.53 | 24.49 |
| ZENG | VGG16 | pooled (10 folds) | 91.00 | 28.48 | 28.43 | 18.99 | 23.71 | 28.97 | 22.97 |
| | | mean ± std | 91.00 ± 0.35 | 28.48 ± 3.63 | 28.41 ± 3.59 | 18.98 ± 1.24 | 23.70 ± 2.15 | 28.97 ± 3.84 | 22.97 ± 1.91 |
| | | single: fold 1 (val 96.36%) | 91.61 | 34.84 | 34.55 | 20.70 | 27.63 | 35.63 | 25.97 |

WAFFLE: benchmark subset of the WAFFLE dataset (data/waffle/benchmark, 110 hand-segmented plans, mostly historic churches, castles and public buildings). GT from segmented_descrete_pngs (red wall -> walls; blue door, cyan window -> openings; black background, white interior -> bg); RGBA images composited on white. StephansdomBauHistorie is excluded (mask 660x455 vs image 648x424, visibly misaligned), leaving 109 plans. Pixel share bg/walls/openings = 87.9 / 11.1 / 1.0 %. Zero-shot run 2026-11-10, same checkpoint sets as CVC-FP (scripts/run_waffle_all.sh).

## R3D test (53 images): comparison with published results

Class acc = per-class recall (DeepFloorplan definition). Our rows: pooled over 10 fold models, mean ± std across folds in brackets.
Published rows are trained on R3D train (179 images), from Zeng et al., ICCV 2019, Table 2 (arXiv 1908.11025). Their mIoU includes room classes, so it is omitted.

| Method | Backbone | Trained on | Walls class acc | Openings class acc | Walls IoU | Openings IoU | mIoU (walls, openings) |
|---|---|---|---|---|---|---|---|
| DeepFloorplan | VGG16 | R3D | 98 | 83 | – | – | – |
| DeepLabV3+ | ResNet | R3D | 93 | 60 | – | – | – |
| PSPNet | ResNet | R3D | 91 | 54 | – | – | – |
| **CAB2** | EfficientNetV2M | CubiCasa5k (zero-shot) | 85.1 (85.1 ± 1.0) | 68.9 (68.9 ± 1.1) | **82.1** (82.1 ± 0.8) | **55.4** (55.4 ± 0.7) | **68.7** (68.7 ± 0.4) |
| CAB1 | EfficientNetV2M | CubiCasa5k (zero-shot) | 84.9 (84.9 ± 1.7) | **70.3** (70.3 ± 2.7) | 81.6 (81.5 ± 1.5) | 50.2 (50.2 ± 1.1) | 65.9 (65.9 ± 1.1) |
| CAB1 | EfficientNetB4 | CubiCasa5k (zero-shot) | 84.0 (84.0 ± 5.1) | 64.1 (64.1 ± 21.4) | 80.7 (80.8 ± 5.8) | 48.7 (47.5 ± 15.9) | 64.7 (64.1 ± 10.8) |
| CAB2 | EfficientNetB4 | CubiCasa5k (zero-shot) | **86.9** (86.9 ± 2.9) | 50.8 (50.8 ± 11.6) | 76.4 (76.6 ± 4.0) | 34.7 (34.8 ± 8.1) | 55.6 (55.7 ± 5.3) |
| CubiCasa5k | VGG16 | CubiCasa5k (zero-shot) | 54.7 (54.7 ± 5.1) | 34.4 (34.4 ± 4.4) | 54.3 (54.3 ± 5.0) | 32.8 (32.7 ± 3.8) | 43.5 (43.5 ± 4.0) |
| Zeng | VGG16 | CubiCasa5k (zero-shot) | 50.6 (50.6 ± 4.4) | 35.7 (35.7 ± 2.5) | 50.3 (50.3 ± 4.3) | 34.3 (34.3 ± 2.3) | 42.3 (42.3 ± 3.0) |

## WAFFLE benchmark (109 plans): per-class comparison with published results

From `kfold_patch/eval_waffle_perclass.py` (`waffle5_kfold_*` files): doors and windows scored separately, pooled over the 10 fold models, "±" = IoU std across folds. Published rows are from Ganon et al., WACV 2025 (arXiv:2412.00955), Tables 4 and 7, on all 110 plans; their resolution and aggregation are not stated. Interior is merged into bg for our models (no interior class). See `results/COMPARISON_paper_sota.md` §4b.

| Method | Trained on | Walls P / R / IoU | Doors P / R / IoU | Windows P / R / IoU |
|---|---|---|---|---|
| CubiCasa5K official model (ResNet-152), WAFFLE paper Table 4 | CubiCasa5k (zero-shot) | 73.7 / 59.0 / 48.8 | 20.1 / 16.3 / 9.9 | **33.9** / 33.4 / **20.2** |
| ControlNet wall model (SD 1.4), WAFFLE paper Table 7 | CubiCasa5k walls (zero-shot) | 74.6 / **80.5** / **63.2** | – | – |
| **CAB1 B4** (ours) | CubiCasa5k (zero-shot) | 83.0 / 63.0 / 55.8 (± 11.3) | 22.7 / 22.6 / 12.8 (± 4.2) | 17.0 / 35.6 / 13.0 (± 4.2) |
| **CAB1 V2M** (ours) | CubiCasa5k (zero-shot) | 80.4 / 67.1 / 57.6 (± 1.3) | 20.4 / 23.9 / 12.4 (± 1.3) | 17.4 / 40.0 / 13.8 (± 1.2) |
| **CAB2 B4** (ours) | CubiCasa5k (zero-shot) | 82.7 / 66.2 / 58.1 (± 5.8) | 18.4 / 6.2 / 4.9 (± 4.9) | 20.3 / 36.5 / 15.0 (± 1.5) |
| **CAB2 V2M** (ours) | CubiCasa5k (zero-shot) | 83.7 / 60.9 / 54.5 (± 2.1) | 24.9 / **24.7 / 14.2** (± 0.6) | 21.6 / **40.2** / 16.3 (± 1.2) |
| CubiCasa5k VGG16 (our k-fold retrain) | CubiCasa5k (zero-shot) | **95.0** / 36.6 / 35.9 (± 3.8) | **28.1** / 4.9 / 4.4 (± 1.7) | 28.3 / 20.6 / 13.6 (± 1.7) |
| Zeng VGG16 (our k-fold retrain) | CubiCasa5k (zero-shot) | 93.9 / 29.0 / 28.4 (± 3.6) | 23.5 / 5.1 / 4.4 (± 1.1) | 34.7 / 24.0 / 16.6 (± 1.4) |
