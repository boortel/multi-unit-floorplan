# CAB1 / CAB2 across all datasets: overview and SOTA comparison

Compiled 2026-11-11. One comparable view of everything in `official/`, `cross_dataset/` and `finetune/`. Details and per-dataset discussion are in [`COMPARISON_paper_sota.md`](COMPARISON_paper_sota.md); the full generated tables (all models, Rec / F1 / IoU) are in [`ALL_DATASETS_tables.md`](ALL_DATASETS_tables.md), from `kfold_patch/summarize_all_datasets.py`.

## Summary

- **Best models:** the two **EfficientNetV2M** variants. **CAB2 V2M** is the best zero-shot model: mean IoU 51.6 over R3D, CVC-FP and WAFFLE, with the smallest fold-to-fold spread. **CAB1 V2M** is the best in-domain on CubiCasa5k (mean IoU 69.2) and after fine-tuning (70.3 over the three datasets). After fine-tuning, all four CAB variants are within about 3 points of each other.
- **Against our own baselines** (the CubiCasa5k and DeepFloorplan/Zeng architectures, trained on the same folds): on CubiCasa5k the V2M models are +8 to +12 mean IoU ahead. CAB2 B4 (51.9) is below both baselines there. Zero-shot, every CAB model is ahead, by +8 to +20 averaged over the three datasets (V2M: +16 to +20), so the CAB models generalize much better.
- **CubiCasa5k:** CAB matches the original CubiCasa5k multi-task model on walls (IoU 72.8 vs 73.0) and beats it on doors and windows. It remains below the CAB paper's numbers, which come from a curated 560-plan subset rather than the full noisy test set.
- **R3D:** fine-tuning only the decoder reaches walls IoU 91–93, above the CAB paper (90), and recall close to DeepFloorplan (walls 97 vs 98, openings 83 vs 83). Zero-shot, walls IoU is already 82.
- **CVC-FP:** CAB2 V2M is the strongest zero-shot result published (walls IoU 66.5 vs 29.6). After fine-tuning, walls IoU is 79–81, about 10 below the CAB paper's in-domain 91.
- **WAFFLE (historic and public buildings):** the hardest dataset. Zero-shot, CAB beats the official CubiCasa5k model on walls (IoU 54.5–58.1 vs 48.8) but not the diffusion wall model (63.2). After fine-tuning, walls IoU is 74–76; nobody has published a fine-tuned result to compare against.
- **Main gap:** openings on stylistically different plans. Zero-shot openings IoU drops from 68 on CubiCasa to 24–27 on CVC-FP and WAFFLE. The frozen backbone limits fine-tuning on CVC-FP and WAFFLE. Unfreezing it is the main open experiment.

## 1. Our models on all datasets (3-class, directly comparable)

All cells use the same classes, the same metric code and the same pooling. Classes are bg / walls / openings (doors + windows); railings, stairs and WAFFLE interior count as bg. The mean is over walls and openings. CubiCasa5k and zero-shot numbers come from 10 fold models; fine-tuned numbers come from one fold model per combination (R3D: official test split; CVC-FP and WAFFLE: 5-fold CV). The "±" is the std across folds or splits.

**Mean IoU (walls + openings)**

| Model | CubiCasa5k (in-domain) | R3D zero-shot | CVC-FP zero-shot | WAFFLE zero-shot | R3D fine-tuned | CVC-FP fine-tuned | WAFFLE fine-tuned | Avg zero-shot | Avg fine-tuned |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 64.7 ± 14.0 | 64.7 ± 10.8 | 44.7 ± 5.4 | 38.3 ± 9.0 | **80.9** | 73.7 ± 2.0 | 54.7 ± 2.4 | 49.2 | 69.8 |
| **CAB1 V2M** | **69.2 ± 0.3** | 65.9 ± 1.1 | 45.7 ± 0.7 | 39.2 ± 0.5 | 80.8 | **73.8 ± 1.7** | 56.3 ± 1.5 | 50.3 | **70.3** |
| CAB2 B4 | 51.9 ± 7.7 | 55.6 ± 5.3 | 33.3 ± 4.1 | 38.3 ± 3.3 | 80.1 | 64.4 ± 4.4 | **57.1 ± 2.7** | 42.4 | 67.2 |
| **CAB2 V2M** | 68.5 ± 0.8 | **68.7 ± 0.4** | **46.9 ± 0.3** | **39.3 ± 0.9** | 80.1 | 72.0 ± 2.9 | 56.5 ± 2.6 | **51.6** | 69.5 |
| CubiCasa5k VGG16 | 60.7 ± 2.7 | 43.5 ± 4.0 | 32.2 ± 2.4 | 27.1 ± 2.6 | – | – | – | 34.3 | – |
| DeepFloorplan (Zeng) VGG16 | 56.7 ± 2.1 | 42.3 ± 3.0 | 28.9 ± 1.7 | 23.7 ± 2.1 | – | – | – | 31.6 | – |

**Walls IoU / openings IoU**

| Model | CubiCasa5k | R3D zero-shot | CVC-FP zero-shot | WAFFLE zero-shot | R3D fine-tuned | CVC-FP fine-tuned | WAFFLE fine-tuned |
|---|---|---|---|---|---|---|---|
| CAB1 B4 | 67.8 / 61.6 | 80.7 / 48.7 | 64.6 / 24.7 | 55.8 / 20.8 | **93.4** / 68.5 | **80.9** / 66.5 | 74.0 / 35.3 |
| CAB1 V2M | **72.8** / 65.5 | 81.5 / 50.2 | 64.2 / 27.1 | 57.6 / 20.8 | 92.7 / 68.8 | 80.3 / **67.4** | 74.3 / **38.3** |
| CAB2 B4 | 59.2 / 44.6 | 76.4 / 34.7 | 52.7 / 14.0 | **58.1** / 18.5 | 91.2 / **68.9** | 73.9 / 55.0 | **76.2** / 37.9 |
| CAB2 V2M | 68.7 / **68.3** | **82.1 / 55.4** | **66.5 / 27.3** | 54.5 / **24.0** | 91.9 / 68.4 | 79.1 / 64.9 | 75.7 / 37.2 |
| CubiCasa5k VGG16 | 63.5 / 58.0 | 54.3 / 32.8 | 48.8 / 15.7 | 35.9 / 18.2 | – | – | – |
| DeepFloorplan (Zeng) VGG16 | 59.3 / 54.1 | 50.3 / 34.3 | 45.9 / 11.9 | 28.4 / 19.0 | – | – | – |

Notes:
- **CubiCasa5k:** these numbers are 3-class, so they differ from the 6-class tables in `COMPARISON_paper_sota.md` §1. Doors and windows are merged, and railings and stairs count as bg. The matrices were re-exported with `kfold_patch/export_cubicasa_cms.py` and reproduce the official 6-class IoUs to within 0.01 points.
- **Fine-tuning:** it starts from one CubiCasa fold model per combination, so its zero-shot baseline differs slightly from the 10-fold pooled column. The single-model zero-shot rows are in `finetune/SUMMARY.md`.
- **Baselines:** the VGG16 baselines were not fine-tuned. Their CVC-FP and WAFFLE zero-shot runs were added on 2026-11-11.
- **B4 spread:** the B4 models have a large fold-to-fold spread (CAB1 B4 ± 14 mean IoU on CubiCasa), so a single B4 fold is a gamble. The V2M models are stable (± 0.3–1.1).

## 2. Best models vs. SOTA

Only like-for-like metrics are compared; the setting column says whether a method was trained on the target dataset. All values are %. Rec = per-class recall (DeepFloorplan's "class accuracy"). Bold marks the best value within each setting.

### CubiCasa5k (official test, 400 plans, in-domain)

| Method | Walls IoU | Doors IoU | Windows IoU |
|---|---|---|---|
| CubiCasa5k multi-task (Kalervo et al., SCIA 2019) | **73.0** | 53.6 | 66.8 |
| **Ours CAB1 V2M** | 72.8 | 56.2 | 67.3 |
| **Ours CAB2 V2M** | 68.7 | **59.2** | **69.9** |
| *CAB paper (560-plan curated subset, not the same test set)* | *83* | *70* | *82* |
| *FGSSNet (walls only, 64×64 crops, normalized wall width)* | *78.3* | – | – |

### R3D (official test, 53 plans)

| Method | Setting | Walls Rec | Openings Rec | Walls IoU | Mean IoU |
|---|---|---|---|---|---|
| DeepFloorplan (Zeng et al., ICCV 2019) | trained on R3D | **98** | 83 | – | – ¹ |
| Offset-guided attention (Wang & Sun, 2022) | trained on R3D | 95 | 83 | – | – ¹ |
| CAB paper, CAB1 / CAB2 | trained on R3D | 94 | – | 90 | 82 ² |
| **Ours CAB1 B4, fine-tuned** | CubiCasa → R3D | 97.4 | **83.4** | **93.4** | **80.9** |
| **Ours CAB1 V2M, fine-tuned** | CubiCasa → R3D | 97.1 | 83.2 | 92.7 | 80.8 |
| **Ours CAB2 V2M, zero-shot** | never saw R3D | 85.1 | 68.9 | 82.1 | 68.7 |
| DeepLabV3+ / PSPNet (as reported by Zeng et al.) | trained on R3D | 93 / 91 | 60 / 54 | – | – ¹ |

¹ Their mIoU includes the room classes, so it is not comparable. ² The CAB paper's "Mean" class set is not defined.

### CVC-FP (122 plans)

| Method | Setting | Walls F1 | Walls IoU | Openings F1 | Mean IoU |
|---|---|---|---|---|---|
| CAB paper, CAB2 | trained on CVC-FP | **95** | **91** | – | 82 ² |
| CAB paper, CAB1 | trained on CVC-FP | 94 | 88 | – | 77 ² |
| Dodge et al. (MVA 2017), U-Net (Versailles-FP paper) | trained on CVC-FP | – | 94.4 / 95.0 ³ | – | – |
| **Ours CAB1 V2M, fine-tuned** | CubiCasa → CVC-FP | 89.1 | 80.3 | **80.5** | **73.8** |
| **Ours CAB2 V2M, zero-shot** | never saw CVC-FP | **79.9** | **66.5** | **42.9** | **46.9** |
| Zernike-moment GNN (Graumann et al., 2025) | zero-shot from CubiCasa | 45.7 | 29.6 | doors 3.0, windows 12.7 | – |

³ From secondary sources, and probably the mean of the wall and background IoU in binary segmentation, so higher than a walls-class IoU.

### WAFFLE benchmark (109–110 plans)

| Method | Setting | Walls Rec / IoU | Doors IoU | Windows IoU |
|---|---|---|---|---|
| ControlNet wall model (WAFFLE paper, Table 7) | zero-shot from CubiCasa | **80.5 / 63.2** | – | – |
| CubiCasa5k official model (WAFFLE paper, Table 4) | zero-shot from CubiCasa | 59.0 / 48.8 | 9.9 | **20.2** |
| **Ours CAB1 V2M** | zero-shot from CubiCasa | 67.1 / 57.6 | 12.4 | 13.8 |
| **Ours CAB2 V2M** | zero-shot from CubiCasa | 60.9 / 54.5 | **14.2** | 16.3 |
| **Ours CAB2 V2M, fine-tuned** | CubiCasa → WAFFLE (5-fold CV) | 90.4 / 75.7 | openings 37.2 | (merged) |
| **Ours CAB1 V2M, fine-tuned** | CubiCasa → WAFFLE (5-fold CV) | 88.9 / 74.3 | openings 38.3 | (merged) |

The WAFFLE paper does not state its evaluation resolution or pooling. Raster2Seq (arXiv:2602.09016) reports WAFFLE *interior* segmentation (IoU 73.9), which is a different task. There is no published fine-tuned WAFFLE result.

## 3. Caveats

- **Settings differ:** the published in-domain methods train a separate model per dataset on 60–80% of it. Our fine-tuned rows start from CubiCasa5k and fine-tune only the decoder with the backbone frozen. Our zero-shot rows never see the target dataset.
- **Openings:** the published R3D and CVC-FP tables do not report openings IoU. CVC-FP door labels include the door swing area, which lowers every openings score.
- **Test sets:** R3D fine-tuned results come from a single test split, so differences under about 1 point are noise. CVC-FP and WAFFLE results are pooled over 5 folds.
