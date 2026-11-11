# CAB1/CAB2 (EfficientNet B4 / V2M) vs. the original paper and published methods

Compiled 2026-11-09; fine-tuning sections (§2c, §3c) added 2026-11-09/10; WAFFLE (§4, §4b) added 2026-11-11. Original paper: Kratochvila et al., *Recognizing and Reconstructing a Multi-Unit Floor Plan*, arXiv:2408.01526 (v2, 2026), Tables III–IV.

For one comparable 3-class view of all datasets, see [`OVERVIEW_all_datasets.md`](OVERVIEW_all_datasets.md).

All numbers are pixel-wise and given in %. Rec = recall (equal to DeepFloorplan's "class accuracy"), F1, IoU.

## Read this first: what is comparable

| | This repo (k-fold, EfficientNet backbones) | Original paper (CAB1/CAB2) |
|---|---|---|
| Training data | CubiCasa5k only: 4200 train + val, 10-fold CV, one model per fold | A separate model per dataset (60/20/20 split) |
| CubiCasa test set | CubiCasa5k official test (400 plans), pooled over the 10 fold models | "Cubicasa": a refined 560-plan subset of CubiCasa5k |
| R3D / CVC-FP / WAFFLE | **Zero-shot** (never trained on them), plus **fine-tuned** rows (§2c, §3c, §4): one CubiCasa fold model per combination, decoder-only fine-tuning (backbone frozen) | Trained and tested in-domain |
| Opening classes on R3D/CVC/WAFFLE | doors + windows → *openings* | not stated; "Mean" class set not defined |

So on R3D and CVC-FP the paper's numbers (and the published results below) are an **in-domain upper bound** for our zero-shot rows, which measure **generalization**. The fine-tuned rows are in-domain and directly comparable in setting (R3D: 143 train / 36 val from the official 179 train plans, official 53-plan test; CVC-FP: 5-fold CV, 60/20/20, test confusion matrices pooled over folds). On CubiCasa the subsets differ. The paper itself notes that the full CubiCasa5k annotations are noisier, which lowers every metric on the full test set.

Our "Mean" on CubiCasa = mean over the 5 foreground classes (walls, railings, doors, windows, stairs), matching the paper's Table III. On R3D/CVC/WAFFLE it is the mean over walls + openings.

---

## 1. CubiCasa5k

### 1a. Ours vs. the paper (Table III)

| Class | **Paper CAB1** Rec / F1 / IoU | **Paper CAB2** | Ours CAB1 B4 | Ours CAB1 V2M | Ours CAB2 B4 | **Ours CAB2 V2M** |
|---|---|---|---|---|---|---|
| Walls | 90 / 90 / 83 | 90 / 90 / 83 | 79.0 / 80.8 / 67.8 | **83.7 / 84.3 / 72.8** | 78.1 / 74.4 / 59.2 | 78.6 / 81.4 / 68.7 |
| Railings | 74 / 77 / 63 | 76 / 77 / 63 | 36.5 / 43.1 / 27.5 | 37.8 / 43.1 / 27.5 | 8.5 / 13.9 / 7.5 | 37.2 / **43.8 / 28.0** |
| Doors | 82 / 82 / 70 | 82 / 82 / 70 | 68.2 / 69.7 / 53.5 | **75.9** / 71.9 / 56.2 | 22.1 / 32.4 / 19.4 | 73.5 / **74.4 / 59.2** |
| Windows | 91 / 90 / 82 | 92 / 90 / 82 | 76.1 / 77.3 / 63.0 | **84.5** / 80.4 / 67.3 | 67.7 / 64.7 / 47.8 | 82.9 / **82.3 / 69.9** |
| Stairs | 87 / 89 / 81 | 86 / 89 / 81 | 54.6 / 63.1 / 46.1 | 56.3 / 64.8 / 47.9 | 11.1 / 18.0 / 9.9 | **58.4 / 66.6 / 49.9** |
| **Mean** | 85 / 86 / 76 | 85 / 86 / 76 | 62.9 / 66.8 / 51.6 | **67.6** / 68.9 / 54.3 | 37.5 / 40.7 / 28.7 | 66.1 / **69.7 / 55.1** |

Our baselines, re-implemented and trained on the same k-fold protocol (mean Rec / F1 / IoU): CubiCasa5k VGG16 46.4 / 57.9 / 43.1, DeepFloorplan (Zeng) VGG16 44.2 / 55.9 / 41.4.
Paper baselines on its 560-plan subset: DFPRβ 65 / 74 / 61, CubiCasa5kβ 65 / 74 / 60.

### 1b. Published results on CubiCasa5k (official test split, 400 plans)

| Method | Venue | Walls IoU | Windows IoU | Doors IoU | Railings IoU | Note |
|---|---|---|---|---|---|---|
| CubiCasa5k multi-task (Kalervo et al.) | SCIA 2019 | **73.0** | 66.8 | 53.6 | 23.6 | raw segmentation; 12 room + 11 icon classes |
| FGSSNet (Norrby et al.) | arXiv 2025 | **78.3** | – | – | – | walls only (binary), 64×64 crops, wall width normalized |
| **Ours CAB2 V2M** | – | 68.7 | **69.9** | **59.2** | **28.0** | 6 classes, full images, 10-fold pooled |
| Ours CAB1 V2M | – | 72.8 | 67.3 | 56.2 | 27.5 | |

On CubiCasa5k our best models beat the original CubiCasa5k model on windows (+3.1 IoU), doors (+5.6) and railings (+4.4). On walls they are 0.2 IoU (CAB1 V2M) to 4.3 IoU (CAB2 V2M) below it. FGSSNet's wall IoU is higher, but it solves an easier task: walls only, on crops with normalized wall width.

---

## 2. R3D (official 53-image test split)

### 2a. Ours (zero-shot) vs. the paper (in-domain, Table IV)

| Model | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU |
|---|---|---|---|
| Paper CAB1 (trained on R3D) | 94 / 94 / 90 | – | 89 / 90 / 82 |
| Paper CAB2 (trained on R3D) | 94 / 94 / 90 | – | 89 / 90 / 82 |
| Ours CAB1 B4 (zero-shot) | 84.0 / 89.3 / 80.7 | 64.1 / 65.5 / 48.7 | 74.1 / 77.4 / 64.7 |
| Ours CAB1 V2M (zero-shot) | 84.9 / 89.8 / 81.5 | **70.3** / 66.9 / 50.2 | **77.6** / 78.4 / 65.9 |
| Ours CAB2 B4 (zero-shot) | **86.8** / 86.6 / 76.4 | 50.8 / 51.6 / 34.7 | 68.8 / 69.1 / 55.6 |
| **Ours CAB2 V2M (zero-shot)** | 85.1 / **90.2 / 82.1** | 68.9 / **71.3 / 55.4** | 77.0 / **80.7 / 68.7** |

On all 232 R3D images (train + test), CAB2 V2M gets: walls 86.9 / 91.0 / 83.5, mean 79.7 / 82.4 / 71.0.

### 2b. Published results on R3D (all trained on R3D train, 179 images)

The class accuracy below is per-class recall. mIoU in these papers also includes the room classes, so it is not comparable with ours.

| Method | Venue | Walls class acc | Door & window class acc | Overall acc | mIoU (incl. rooms) |
|---|---|---|---|---|---|
| DeepFloorplan (Zeng et al.) | ICCV 2019 | **98** | **83** | 89 | 63 |
| Offset-Guided Attention Net (Wang & Sun) | arXiv 2022 | 95 | **83** | 91 | **74** |
| Direction-aware additive kernels + GAN (Zhang et al.) | arXiv 2020 | 95 | 82 | **94** | **74** |
| Paper CAB1 / CAB2 (wall recall) | arXiv 2024/26 | 94 | – | – | – |
| Yang et al. (wall recall, as quoted in the paper) | 2022 | 89 | – | – | – |
| DeepLabV3+ (as reported by Zeng et al.) | ECCV 2018 | 93 | 60 | 85 | 50 |
| PSPNet (as reported by Zeng et al.) | CVPR 2017 | 91 | 54 | 84 | 50 |
| **Ours CAB2 V2M, zero-shot** | – | 85.1 | 68.9 | – | – |
| **Ours CAB1 V2M, zero-shot** | – | 84.9 | 70.3 | – | – |
| **Ours CAB1 B4, fine-tuned (§2c)** | – | 97.4 | 83.4 | – | – |
| **Ours CAB2 V2M, fine-tuned (§2c)** | – | 96.6 | 81.5 | – | – |

Without ever seeing R3D, our V2M models beat in-domain DeepLabV3+ and PSPNet on door & window class accuracy (68.9–70.3 vs 54–60). They trail in-domain SOTA by about 10–13 points on walls and about 13 points on openings. After fine-tuning (§2c), walls recall is 96.6–97.4 and openings recall 81.5–83.4, close to DeepFloorplan (98 / 83).

### 2c. Fine-tuned on R3D (in-domain, official 53-plan test)

From `results/finetune/SUMMARY.md`. Decoder-only fine-tuning from the CubiCasa fold model with the best validation score, LR 1e-5, early stopping (patience 25, max 150 epochs), geometric + photometric/scan augmentation. Zero-shot here is that single fold model, so it differs slightly from §2a (pooled over 10 folds). AdaBN = BatchNorm statistics recomputed on the unlabelled R3D train images, no weight updates.

| Model | Setting | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU |
|---|---|---|---|---|
| Paper CAB1 / CAB2 (trained on R3D) | in-domain | 94 / 94 / 90 | – | 89 / 90 / 82 |
| Ours CAB1 B4 | zero-shot | 86.3 / 90.5 / 82.7 | 68.7 / 67.5 / 50.9 | 77.5 / 79.0 / 66.8 |
| | AdaBN | 64.2 / 67.9 / 51.4 | 54.2 / 50.5 / 33.7 | 59.2 / 59.2 / 42.6 |
| | **fine-tuned** | **97.4 / 96.6 / 93.4** | 83.4 / 81.3 / 68.5 | **90.4 / 88.9 / 80.9** |
| Ours CAB1 V2M | zero-shot | 86.6 / 90.6 / 82.8 | 71.1 / 65.9 / 49.1 | 78.8 / 78.2 / 65.9 |
| | AdaBN | 72.6 / 76.2 / 61.6 | 46.4 / 51.3 / 34.5 | 59.5 / 63.8 / 48.1 |
| | **fine-tuned** | 97.1 / 96.2 / 92.7 | 83.2 / 81.5 / 68.8 | 90.1 / 88.9 / 80.8 |
| Ours CAB2 B4 | zero-shot | 88.6 / 89.4 / 80.8 | 77.2 / 72.5 / 56.9 | 82.9 / 81.0 / 68.8 |
| | AdaBN | 63.3 / 67.1 / 50.5 | 57.2 / 57.8 / 40.6 | 60.2 / 62.4 / 45.6 |
| | **fine-tuned** | 96.8 / 95.4 / 91.2 | 83.2 / **81.6 / 68.9** | 90.0 / 88.5 / 80.1 |
| Ours CAB2 V2M | zero-shot | 85.4 / 90.6 / 82.8 | 67.2 / 70.9 / 55.0 | 76.3 / 80.8 / 68.9 |
| | AdaBN | 75.4 / 82.0 / 69.5 | 53.6 / 58.6 / 41.4 | 64.5 / 70.3 / 55.5 |
| | **fine-tuned** | 96.6 / 95.8 / 91.9 | 81.5 / 81.2 / 68.4 | 89.0 / 88.5 / 80.1 |

After fine-tuning, all four models beat the paper on R3D walls (IoU 91.2–93.4 vs 90) and the four are within 0.8 mean-IoU points of each other. That is a single test split, so the ranking between them is not meaningful. AdaBN hurts every model (walls IoU −13 to −31 points).

---

## 3. CVC-FP (122 plans)

### 3a. Ours (zero-shot) vs. the paper (in-domain, Table IV)

| Model | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU |
|---|---|---|---|
| Paper CAB1 (trained on CVC-FP) | 95 / 94 / 88 | – | 87 / 86 / 77 |
| Paper CAB2 (trained on CVC-FP) | 96 / 95 / 91 | – | 91 / 90 / 82 |
| Ours CAB1 B4 (zero-shot) | 74.9 / 78.5 / 64.6 | 27.0 / 39.7 / 24.7 | 50.9 / 59.1 / 44.7 |
| Ours CAB1 V2M (zero-shot) | 76.2 / 78.2 / 64.2 | **30.5** / 42.6 / 27.1 | 53.4 / 60.4 / 45.7 |
| Ours CAB2 B4 (zero-shot) | **78.0** / 69.0 / 52.7 | 16.9 / 24.5 / 14.0 | 47.5 / 46.8 / 33.3 |
| **Ours CAB2 V2M (zero-shot)** | 76.9 / **79.9 / 66.5** | 30.1 / **42.9 / 27.3** | **53.5 / 61.4 / 46.9** |

### 3b. Published results on CVC-FP

| Method | Venue | Setting | Walls | Doors | Windows |
|---|---|---|---|---|---|
| Dodge et al. (FCN-2s) | MVA 2017 | in-domain | mean IoU 94.4 ¹ | – | – |
| U-Net (as reported in the Versailles-FP paper) | 2021 | in-domain | mean IoU 95.01, Dice 95.65 ¹ | – | – |
| Paper CAB2 | arXiv 2024/26 | in-domain | IoU 91, F1 95 | – | – |
| Zernike-moment GNN (Graumann et al.) | arXiv 2025 | **zero-shot from CubiCasa** | IoU 20.5–29.6, F1 34.0–45.7 | F1 1.9–3.0 | F1 3.7–12.7 |
| **Ours CAB2 V2M** | – | **zero-shot from CubiCasa5k** | **IoU 66.5, F1 79.9** | openings F1 42.9 | (merged) |
| **Ours CAB1 B4, fine-tuned (§3c)** | – | in-domain, 5-fold CV | IoU 80.9, F1 89.4 | openings F1 79.9 | (merged) |

¹ Cited from secondary sources (a survey and the Versailles-FP paper). I could not read the original Dodge et al. PDF, so I have not checked these figures against it. "Mean IoU" there is likely the wall/background average for binary wall segmentation, which is not directly comparable with our walls-class IoU.

The only other published zero-shot CubiCasa → CVC-FP result I found (Graumann et al. 2025) is far below ours: walls IoU 29.6 vs 66.5. CVC door labels include the area the door sweeps when it opens. That partly explains the low openings scores for every zero-shot model.

### 3c. Fine-tuned on CVC-FP (in-domain, 5-fold CV, all 122 plans tested once)

Same procedure as §2c, but with **LR 3e-5** and up to 300 epochs (patience 25). A first pass with LR 1e-5 / 150 epochs was still improving at the cap (see below). Each fold uses 60/20/20 train/val/test, and the test confusion matrices are pooled over the 5 folds. "±" is the std of mean IoU across folds.

| Model | Setting | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU |
|---|---|---|---|---|
| Paper CAB1 (trained on CVC-FP) | in-domain | 95 / 94 / 88 | – | 87 / 86 / 77 |
| Paper CAB2 (trained on CVC-FP) | in-domain | 96 / 95 / 91 | – | 91 / 90 / 82 |
| Ours CAB1 B4 | zero-shot | 75.0 / 78.8 / 65.0 | 29.2 / 42.4 / 26.9 | 52.1 / 60.6 / 46.0 |
| | AdaBN | 62.8 / 52.7 / 35.8 | 19.8 / 26.8 / 15.5 | 41.3 / 39.8 / 25.6 |
| | **fine-tuned** | **90.8 / 89.4 / 80.9** | **82.7** / 79.9 / 66.5 | **86.8** / 84.7 / 73.7 (± 2.0) |
| Ours CAB1 V2M | zero-shot | 77.3 / 79.0 / 65.3 | 31.6 / 43.8 / 28.0 | 54.4 / 61.4 / 46.7 |
| | AdaBN | 63.2 / 56.3 / 39.1 | 14.6 / 20.7 / 11.5 | 38.9 / 38.5 / 25.3 |
| | **fine-tuned** | 90.6 / 89.1 / 80.3 | 82.3 / **80.5 / 67.4** | 86.4 / **84.8 / 73.8** (± 1.7) |
| Ours CAB2 B4 | zero-shot | 77.3 / 75.6 / 60.8 | 31.3 / 43.8 / 28.0 | 54.3 / 59.7 / 44.4 |
| | AdaBN | 58.6 / 57.7 / 40.5 | 19.4 / 26.8 / 15.5 | 39.0 / 42.3 / 28.0 |
| | **fine-tuned** | 88.4 / 85.0 / 73.9 | 73.4 / 70.9 / 55.0 | 80.9 / 78.0 / 64.4 (± 4.4) |
| Ours CAB2 V2M | zero-shot | 77.4 / 80.6 / 67.5 | 30.2 / 43.2 / 27.6 | 53.8 / 61.9 / 47.6 |
| | AdaBN | 66.2 / 65.6 / 48.8 | 17.2 / 25.6 / 14.7 | 41.7 / 45.6 / 31.7 |
| | **fine-tuned** | **90.8** / 88.4 / 79.1 | 80.8 / 78.7 / 64.9 | 85.8 / 83.6 / 72.0 (± 2.9) |

Fine-tuning more than doubles openings IoU (≈28 → 65–67) and raises walls IoU by 12–16 points. A clear gap to the paper's in-domain CAB2 remains (walls IoU 80.9 vs 91; mean F1 84.8 vs 90). CAB1 B4, CAB1 V2M and CAB2 V2M are within the fold spread of each other (mean IoU 72.0–73.8, std 1.7–2.9). CAB2 B4 is clearly weaker: in 4 of 5 folds its validation loss is lowest at epoch 4, so early stopping keeps an almost un-adapted model.

**Learning-rate history.** All runs converged (longest run 158 epochs, best epochs 43–133, with CAB2 B4 at 4).

| Setting (pooled over 5 folds) | CAB1 B4 | CAB1 V2M | CAB2 B4 | CAB2 V2M |
|---|---|---|---|---|
| LR 1e-5, max 150 ep. — mean IoU (walls / openings) | 71.9 (79.9 / 63.9) | 71.2 (78.7 / 63.7) | 62.4 (72.4 / 52.5) | 70.8 (78.6 / 63.0) |
| **LR 3e-5, max 300 ep.** — mean IoU (walls / openings) | **73.7** (80.9 / 66.5) | **73.8** (80.3 / 67.4) | **64.4** (73.9 / 55.0) | **72.0** (79.1 / 64.9) |

An earlier partial run with LR 1e-4 and geometric-only augmentation exists only for CVC split 2 (and CAB2 B4 split 3). On split 2 it is still 2–3 points higher on openings IoU (72.0–74.0 vs 69.8–71.1), with walls within ±1. That is one fold, so it doesn't decide the question, but it suggests the remaining CVC gap is not only under-training. A higher LR, or the photometric augmentation hurting the openings class, are both possible. The runs are kept in `results/finetune/cvc_lr1e-5/` and `results/finetune/lr1e-4_geomaug/`. R3D was fine-tuned only with LR 1e-5. There the 1e-4 run was not better (CAB2 B4: walls 91.4 vs 91.2, openings 66.9 vs 68.9).

---

## 4. WAFFLE benchmark (109 plans)

WAFFLE (Ganon et al., *WAFFLE: Multimodal Floorplan Understanding in the Wild*, WACV 2025) is a set of floorplans collected from Wikimedia Commons. Its benchmark subset has 110 hand-segmented plans: mostly churches, castles, palaces and other historic or public buildings, drawn in many styles and often with filled ("poché") walls. That is far from CubiCasa's Finnish apartment plans. The CAB paper does not evaluate on WAFFLE. The published results (zero-shot only) are compared per class in §4b.

Setup: identical to CVC-FP. Labels come from `segmented_descrete_pngs` (wall → walls; door + window → openings; background + interior → bg). One plan (`StephansdomBauHistorie`) is excluded because its mask is misaligned with the image. Zero-shot uses all 10 fold models, pooled. Fine-tuning uses the same checkpoint per combination as §2c/§3c, 5-fold CV 60/20/20, LR 3e-5, max 300 epochs, patience 25. Openings are only 1.0% of pixels (CVC-FP 1.9%, R3D test 1.5%), so openings scores are noisy. Walls, by contrast, are 11.1% (CVC-FP 2.9%) because of the thick filled walls.

| Model | Setting | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU |
|---|---|---|---|---|
| Ours CAB1 B4 | zero-shot (10 folds) | – / – / 55.8 | – / – / 20.8 | – / – / 38.3 |
| | zero-shot (fine-tune checkpoint) | 70.1 / 74.9 / 59.9 | 50.5 / 33.7 / 20.3 | 60.3 / 54.3 / 40.1 |
| | AdaBN | 43.8 / 52.7 / 35.7 | 37.5 / 23.9 / 13.6 | 40.7 / 38.3 / 24.6 |
| | **fine-tuned** | 89.2 / 85.1 / 74.0 | 55.9 / 52.2 / 35.3 | 72.5 / 68.7 / 54.7 (± 2.4) |
| Ours CAB1 V2M | zero-shot (10 folds) | – / – / 57.6 | – / – / 20.8 | – / – / 39.2 |
| | zero-shot (fine-tune checkpoint) | 67.6 / 72.8 / 57.2 | 44.3 / 34.3 / 20.7 | 56.0 / 53.5 / 38.9 |
| | AdaBN | 60.4 / 66.6 / 49.9 | 31.7 / 24.0 / 13.6 | 46.0 / 45.3 / 31.7 |
| | **fine-tuned** | 88.9 / 85.2 / 74.3 | 58.3 / **55.4 / 38.3** | 73.6 / 70.3 / 56.3 (± 1.5) |
| Ours CAB2 B4 | zero-shot (10 folds) | – / – / 58.1 | – / – / 18.5 | – / – / 38.3 |
| | zero-shot (fine-tune checkpoint) | 72.8 / 75.2 / 60.2 | 47.3 / 36.4 / 22.2 | 60.1 / 55.8 / 41.2 |
| | AdaBN | 49.2 / 57.9 / 40.8 | 28.5 / 25.0 / 14.3 | 38.8 / 41.5 / 27.5 |
| | **fine-tuned** | 88.6 / **86.5 / 76.2** | **59.0** / 55.0 / 37.9 | 73.8 / **70.7 / 57.1** (± 2.7) |
| Ours CAB2 V2M | zero-shot (10 folds) | – / – / 54.5 | – / – / 24.0 | – / – / 39.3 |
| | zero-shot (fine-tune checkpoint) | 60.2 / 70.6 / 54.6 | 46.4 / 37.6 / 23.1 | 53.3 / 54.1 / 38.9 |
| | AdaBN | 54.9 / 65.2 / 48.4 | 37.8 / 30.1 / 17.7 | 46.3 / 47.7 / 33.1 |
| | **fine-tuned** | **90.4** / 86.2 / 75.7 | **59.0** / 54.2 / 37.2 | **74.7** / 70.2 / 56.5 (± 2.6) |

"Zero-shot (10 folds)" is from `cross_dataset/SUMMARY.md`, pooled over all fold models. The other rows are from `finetune/SUMMARY.md`, for the one checkpoint that is fine-tuned, pooled over the 5 test folds.

- WAFFLE is the hardest target so far. Zero-shot mean IoU is 38–39 for every combination (R3D 56–69, CVC-FP 44–48). The fold spread is small for V2M (± 0.5–0.9) and larger for B4 (± 3–9), as on the other datasets.
- Fine-tuning lifts walls IoU by about 15–21 points (to 74–76) and openings IoU by about 15–18 points (to 35–38), giving mean IoU 54.7–57.1. All four combinations are within the fold spread of each other (std 1.5–2.7). Unlike on CVC-FP, CAB2 B4 is not weaker here.
- Runs stop early: best epochs 3–47, and every run ends by epoch 72. With only about 65 training plans per fold and very diverse drawing styles, the frozen backbone looks like the limit rather than training length.
- AdaBN hurts every model again.

### 4b. Published results on the WAFFLE benchmark (zero-shot, per class)

The WAFFLE paper evaluates only models trained on CubiCasa5k, with no training on WAFFLE: the official CubiCasa5K model (Table 4: walls, doors, windows, interior, background) and a ControlNet diffusion model trained to draw CubiCasa5k walls (Table 7, walls only). To match its classes, `kfold_patch/eval_waffle_perclass.py` re-scores all 10 fold models with doors and windows kept separate (`data/tfrecords/waffle/waffle5_all`, `scripts/run_waffle_perclass.sh`). It also scores our CubiCasa5k and Zeng VGG16 baselines, which were not run on WAFFLE before. Our rows are confusion matrices pooled over the 10 fold models; "±" is the IoU std across folds.

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

Comparability caveats:
- The paper uses all 110 plans; we use 109 (one misaligned mask dropped). It does not state the evaluation resolution or whether metrics are pooled over pixels or averaged per image. We cap the longest side at 812 px and pool.
- The CubiCasa5k model rows are different checkpoints. The paper uses the official ResNet-152 model; ours is a VGG16 k-fold retrain (same as on R3D). Ours scores lower on walls (IoU 35.9 vs 48.8), with very high precision but low recall, so the two baselines are not interchangeable.
- The paper's background and interior columns cannot be compared. CubiCasa has no interior class in our 6-class head, so our models predict interior as background. Merged bg + interior IoU is 93.0–93.4 for CAB and 90.7–91.5 for the VGG16 baselines.
- Merging doors + windows into openings and interior into bg reproduces the 3-class `waffle_all` results to within a few dozen pixels per fold (GPU nondeterminism).

Findings:
- **Walls:** every CAB model beats the paper's CubiCasa5K model zero-shot (IoU 54.5–58.1 vs 48.8, precision 80–84 vs 74, recall 61–67 vs 59). The diffusion wall model is still better (63.2), almost entirely through recall (80.5). Our models are more precise but miss about a third of the wall pixels. Which wall types are missed has not been analysed; the paper's own examples point to pillars and curved walls.
- **Doors:** CAB IoU 12.4–14.2 (CAB2 B4: 4.9) vs 9.9. That is slightly better, but every model is poor.
- **Windows:** CAB IoU 13.0–16.3 vs 20.2, so lower than the official model. Recall is higher (36–40 vs 33) but precision is about half (17–22 vs 34).
- **After fine-tuning (§4)**, walls IoU is 74.0–76.2, above every zero-shot model including the diffusion one. That is an in-domain result (5-fold CV on the benchmark itself), and no published method is fine-tuned on WAFFLE, so there is no like-for-like published comparison.

**Other published WAFFLE numbers.** Raster2Seq (Phung & Averbuch-Elor, arXiv:2602.09016, Table 6) reports zero-shot **interior** segmentation on the WAFFLE test set. Its models are trained on CubiCasa5K, with IoU / P / R: Raster2Seq 73.9 / 81.6 / 88.6, RoomFormer 60.5 / 65.7 / 88.3, FRI-Net 56.7 / 63.4 / 84.2. Its CubiCasa5K row (46.1 / 79.9 / 52.1) is the interior column of the WAFFLE paper's Table 4. That is room/interior polygon prediction; our models do not predict interior, so it is not comparable. A search (Nov 11, 2026) found no other wall, door or window results on the WAFFLE benchmark. FGSSNet, MitUNet and FloorplanVLM do not use WAFFLE.

---

## Summary

- **Best combination zero-shot: CAB2 + EfficientNetV2M.** It has the best mean IoU on every dataset (CubiCasa5k 55.1, R3D test 68.7, CVC-FP 46.9) and by far the smallest variance between folds on R3D and CVC-FP (mIoU std ≤ 0.4 points, against 4–11 for B4). CAB1 V2M is close and slightly better on CubiCasa walls. CAB2 B4 is consistently the weakest.
- **Against the paper:** our numbers are lower everywhere. On CubiCasa this mostly reflects the full, noisier test set (400 plans) against the paper's curated 560-plan subset. On R3D and CVC-FP it is zero-shot against in-domain training. The R3D gap is small (walls IoU 82 vs 90). The CVC-FP gap is larger (66 vs 91), as expected given its scanned, stylistically different plans.
- **Against SOTA:** on CubiCasa5k, the openings and railing classes improve over the original CubiCasa5k model. On R3D, zero-shot performance beats in-domain DeepLabV3+/PSPNet on openings but stays below DeepFloorplan-family methods. On CVC-FP, ours is the strongest zero-shot result found, far above the only other one.
- **Fine-tuned (like-for-like, §2c/§3c):** decoder-only fine-tuning (frozen backbone) closes the R3D gap completely. Walls IoU is 91.2–93.4 against the paper's 90, and walls/openings recall (96.6–97.4 / 81.5–83.4) is close to DeepFloorplan (98 / 83). On CVC-FP, mean IoU rises from about 46 to 72–74 and openings IoU from about 28 to 65–67, but walls IoU (79.1–80.9) is still clearly below the paper's 88–91. After fine-tuning, CAB1 B4, CAB1 V2M and CAB2 V2M are within noise of each other, so the CAB2 V2M advantage seen zero-shot disappears. CAB2 B4 stays the weakest. AdaBN alone hurts every model on both datasets.
- **WAFFLE (§4):** the hardest dataset. Zero-shot mean IoU is 38–39, and decoder-only fine-tuning reaches 54.7–57.1 (walls IoU 74–76, openings 35–38). The four combinations are tied after fine-tuning. Against the WAFFLE paper (§4b, zero-shot), every CAB model beats the official CubiCasa5K model on walls (IoU 54.5–58.1 vs 48.8) and doors (12–14 vs 10, except CAB2 B4), but not on windows (13–16 vs 20). The ControlNet wall model (63.2) is still ahead zero-shot.
- **Open follow-ups:** a full CVC run at LR 1e-4 with and without photometric augmentation (one fold hints it is better on openings), unfreezing the backbone on CVC and WAFFLE (likely needed to reach the paper's walls IoU), an ImageNet-only baseline (decoder trained from scratch on the target data), and a check of how much CubiCasa accuracy the fine-tuned models lose.


## Sources
- Kratochvila et al., arXiv:2408.01526 — https://arxiv.org/abs/2408.01526
- Kalervo et al., CubiCasa5K, SCIA 2019 — https://arxiv.org/abs/1904.01920
- Zeng et al., Deep Floor Plan Recognition Using a Multi-Task Network, ICCV 2019 — https://arxiv.org/abs/1908.11025
- Wang & Sun, Offset-Guided Attention Network, arXiv 2022 — https://arxiv.org/abs/2210.17411
- Zhang et al., Direction-Aware Learnable Additive Kernels, arXiv 2020 — https://arxiv.org/abs/2001.11194
- Ganon et al., WAFFLE: Multimodal Floorplan Understanding in the Wild, WACV 2025 — https://arxiv.org/abs/2412.00955
- Phung & Averbuch-Elor, Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction, arXiv 2026 — https://arxiv.org/abs/2602.09016
- Norrby et al., FGSSNet, arXiv 2025 — https://arxiv.org/abs/2507.10343
- Graumann et al., Rotation Invariance in Floor Plan Digitization using Zernike Moments, arXiv 2025 — https://arxiv.org/abs/2504.03241
- Dodge, Xu, Stenger, Parsing Floor Plan Images, MVA 2017 — https://bjornstenger.github.io/papers/dodge_mva2017.pdf
- Versailles-FP dataset paper (U-Net on CVC-FP) — https://www.researchgate.net/publication/350088184_Versailles-FP_dataset_Wall_Detection_in_Ancient
