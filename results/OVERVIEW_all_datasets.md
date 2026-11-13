# CAB1 / CAB2 across all datasets: overview and SOTA comparison

Compiled 2026-11-11; fine-tuning with the backbone unfrozen added 2026-11-13. One comparable view of everything in `official/`, `cross_dataset/`, `finetune/` (decoder only, backbone frozen) and `finetune_unfrozen/` (encoder trained too). Details and per-dataset discussion are in [`COMPARISON_paper_sota.md`](COMPARISON_paper_sota.md); the full generated tables (all models, Rec / F1 / IoU) are in [`ALL_DATASETS_tables.md`](ALL_DATASETS_tables.md), from `kfold_patch/summarize_all_datasets.py`.

## Summary

- **Best models:** the two **EfficientNetV2M** variants. **CAB2 V2M** is the best zero-shot model: mean IoU 51.6 over R3D, CVC-FP and WAFFLE, with the smallest fold-to-fold spread. **CAB1 V2M** is the best in-domain on CubiCasa5k (mean IoU 69.2). After fine-tuning with the backbone unfrozen, CAB2 V2M (73.0 over the three datasets) and CAB1 V2M (72.6) are the best. CAB2 B4 (68.8) is the weakest, mostly because of CVC-FP.
- **Against our own baselines** (the CubiCasa5k and DeepFloorplan/Zeng architectures, trained on the same folds): on CubiCasa5k the V2M models are +8 to +12 mean IoU ahead. CAB2 B4 (51.9) is below both baselines there. Zero-shot, every CAB model is ahead, by +8 to +20 averaged over the three datasets (V2M: +16 to +20), so the CAB models generalize much better.
- **CubiCasa5k:** CAB matches the original CubiCasa5k multi-task model on walls (IoU 72.8 vs 73.0) and beats it on doors and windows. It remains below the CAB paper's numbers, which come from a curated 560-plan subset rather than the full noisy test set.
- **R3D:** fine-tuning reaches walls IoU 91–94, above the CAB paper (90). With the backbone unfrozen, openings recall is 84–87, above DeepFloorplan's 83 (walls recall 97 vs 98), and CAB2 V2M reaches mean IoU 83.2 (decoder only: 80.1–80.9). Zero-shot, walls IoU is already 82.
- **CVC-FP:** CAB2 V2M is the strongest zero-shot result published (walls IoU 66.5 vs 29.6). After fine-tuning with the backbone unfrozen, walls IoU is 82–83 (CAB2 B4: 74), still about 8 below the CAB paper's in-domain 91. The best model, CAB1 V2M, reaches mean IoU 79.8 and openings IoU 77.0 (decoder only: 73.8 and 67.4).
- **WAFFLE (historic and public buildings):** the hardest dataset. Zero-shot, CAB beats the official CubiCasa5k model on walls (IoU 54.5–58.1 vs 48.8) but not the diffusion wall model (63.2). After fine-tuning, walls IoU is 74–77 and mean IoU 56–59. Unfreezing the backbone changes WAFFLE by only −0.5 to +2.2 mean IoU. Nobody has published a fine-tuned result to compare against.
- **Unfreezing the backbone** (Nov 13) helps every model on R3D and CVC-FP: mean IoU +1.0 to +3.1 on R3D and +1.9 to +6.0 on CVC-FP, mostly through openings IoU (CVC-FP: +3.4 to +9.6). Walls IoU changes less (R3D +0.3 to +1.4, CVC-FP +0.4 to +2.7).
- **Main gap:** openings on WAFFLE. Zero-shot openings IoU drops from 68 on CubiCasa to 24–27 on CVC-FP and WAFFLE. Fine-tuning brings CVC-FP openings to 72–77, but WAFFLE openings stay at 37–42 even with the backbone unfrozen, and WAFFLE runs still stop early (best epochs 3–70). So the frozen backbone was not WAFFLE's main limit; the small training set (about 65 plans per fold) and the very diverse drawing styles are more likely.

## 1. Our models on all datasets (3-class, directly comparable)

All cells use the same classes, the same metric code and the same pooling. Classes are bg / walls / openings (doors + windows); railings, stairs and WAFFLE interior count as bg. The mean is over walls and openings. CubiCasa5k and zero-shot numbers come from 10 fold models; fine-tuned numbers come from one fold model per combination (R3D: official test split; CVC-FP and WAFFLE: 5-fold CV). The "±" is the std across folds or splits. The fine-tuned columns in these first two tables are decoder-only (backbone frozen); the backbone-unfrozen results are in the third table.

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

**Fine-tuning: backbone frozen vs. unfrozen**

Same checkpoints, splits, augmentation and schedule; "unfrozen" also trains all encoder Conv weights (BatchNorm stays frozen). Mean IoU is frozen → unfrozen; walls and openings IoU are for the unfrozen runs.

| Model | R3D mean IoU | CVC-FP mean IoU | WAFFLE mean IoU | Avg of 3 | R3D walls / openings (unfrozen) | CVC-FP walls / openings (unfrozen) | WAFFLE walls / openings (unfrozen) |
|---|---|---|---|---|---|---|---|
| CAB1 B4 | 80.9 → 82.5 | 73.7 → 76.7 ± 1.9 | 54.7 → 56.2 ± 2.1 | 69.8 → 71.8 | **93.7** / 71.2 | 81.9 / 71.6 | 73.6 / 38.8 |
| CAB1 V2M | 80.8 → 82.2 | 73.8 → **79.8 ± 1.7** | 56.3 → 55.8 ± 2.2 | 70.3 → 72.6 | 93.1 / 71.2 | **82.6 / 77.0** | 74.7 / 36.9 |
| CAB2 B4 | 80.1 → 81.1 | 64.4 → 66.3 ± 3.4 | 57.1 → **59.0 ± 3.6** | 67.2 → 68.8 | 91.7 / 70.6 | 74.3 / 58.4 | **77.4** / 40.6 |
| CAB2 V2M | 80.1 → **83.2** | 72.0 → 77.1 ± 1.3 | 56.5 → 58.7 ± 2.6 | 69.5 → **73.0** | 93.3 / **73.1** | 81.8 / 72.4 | 75.6 / **41.7** |

Notes:
- **CubiCasa5k:** these numbers are 3-class, so they differ from the 6-class tables in `COMPARISON_paper_sota.md` §1. Doors and windows are merged, and railings and stairs count as bg. The matrices were re-exported with `kfold_patch/export_cubicasa_cms.py` and reproduce the official 6-class IoUs to within 0.01 points.
- **Fine-tuning:** it starts from one CubiCasa fold model per combination, so its zero-shot baseline differs slightly from the 10-fold pooled column. The single-model zero-shot rows are in `finetune/SUMMARY.md`; the backbone-unfrozen runs are in `finetune_unfrozen/SUMMARY.md`.
- **CAB2 B4 on CVC-FP:** in 4 of 5 folds its best validation score comes at epoch 4–6, with the backbone frozen or not, so early stopping keeps an almost un-adapted model. That is why it trails the other three there by 8–9 mean IoU (frozen) and 10–14 (unfrozen).
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
| **Ours CAB2 V2M, fine-tuned, unfrozen** | CubiCasa → R3D | 97.2 | **86.8** | 93.3 | **83.2** |
| **Ours CAB1 B4, fine-tuned, unfrozen** | CubiCasa → R3D | 97.4 | 85.6 | **93.7** | 82.5 |
| Ours CAB1 B4, fine-tuned, frozen | CubiCasa → R3D | 97.4 | 83.4 | 93.4 | 80.9 |
| **Ours CAB2 V2M, zero-shot** | never saw R3D | 85.1 | 68.9 | 82.1 | 68.7 |
| DeepLabV3+ / PSPNet (as reported by Zeng et al.) | trained on R3D | 93 / 91 | 60 / 54 | – | – ¹ |

¹ Their mIoU includes the room classes, so it is not comparable. ² The CAB paper's "Mean" class set is not defined.

### CVC-FP (122 plans)

| Method | Setting | Walls F1 | Walls IoU | Openings F1 | Mean IoU |
|---|---|---|---|---|---|
| CAB paper, CAB2 | trained on CVC-FP | **95** | **91** | – | 82 ² |
| CAB paper, CAB1 | trained on CVC-FP | 94 | 88 | – | 77 ² |
| Dodge et al. (MVA 2017), U-Net (Versailles-FP paper) | trained on CVC-FP | – | 94.4 / 95.0 ³ | – | – |
| **Ours CAB1 V2M, fine-tuned, unfrozen** | CubiCasa → CVC-FP | 90.5 | 82.6 | **87.0** | **79.8** |
| Ours CAB1 V2M, fine-tuned, frozen | CubiCasa → CVC-FP | 89.1 | 80.3 | 80.5 | 73.8 |
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
| **Ours CAB2 V2M, fine-tuned, unfrozen** | CubiCasa → WAFFLE (5-fold CV) | 90.2 / 75.6 | openings 41.7 | (merged) |
| **Ours CAB2 B4, fine-tuned, unfrozen** | CubiCasa → WAFFLE (5-fold CV) | 89.4 / 77.4 | openings 40.6 | (merged) |
| Ours CAB2 V2M, fine-tuned, frozen | CubiCasa → WAFFLE (5-fold CV) | 90.4 / 75.7 | openings 37.2 | (merged) |

The WAFFLE paper does not state its evaluation resolution or pooling. Raster2Seq (arXiv:2602.09016) reports WAFFLE *interior* segmentation (IoU 73.9), which is a different task. There is no published fine-tuned WAFFLE result.

## 3. Caveats

- **Settings differ:** the published in-domain methods train a separate model per dataset on 60–80% of it. Our fine-tuned rows start from CubiCasa5k and fine-tune either only the decoder ("frozen") or the decoder and the encoder Conv weights ("unfrozen"; BatchNorm stays frozen because the batch size is 1). Our zero-shot rows never see the target dataset.
- **Openings:** the published R3D and CVC-FP tables do not report openings IoU. CVC-FP door labels include the door swing area, which lowers every openings score.
- **Test sets:** R3D fine-tuned results come from a single test split, so differences under about 1 point are noise. CVC-FP and WAFFLE have no official test split, so their results come from 5-fold cross-validation (60/20/20), pooled so that each plan is tested once. See "Evaluation protocol and data splits" in [`COMPARISON_paper_sota.md`](COMPARISON_paper_sota.md).
