# Related work not covered by Kratochvila et al.: review, comparison and suggested experiments

Compiled 2026-11-19. Baseline: the 46 references of Kratochvila et al., *Recognizing and Reconstructing a Multi-Unit Floor Plan* (arXiv:2408.01526 v2). Papers already compared in [`COMPARISON_paper_sota.md`](COMPARISON_paper_sota.md) (OGA, DLAK-GAN, WAFFLE, Raster2Seq, FGSSNet, Graumann et al., Paek et al.) are only cross-referenced here.

"Ours" means the k-fold CubiCasa5k models from this repo (10 folds, pooled confusion matrices, official 400-plan test set) unless stated otherwise. Our zero-shot and fine-tuned numbers (backbone frozen and unfrozen) are the current best results in [`OVERVIEW_all_datasets.md`](OVERVIEW_all_datasets.md) (state of 2026-11-13). All numbers are in %. A number marked † was read from a model summary of the page, not from the verbatim table, and should be checked before it is cited.

---

## 0. A finding about our own paper first

**v1 of our arXiv paper evaluated MLSTRUCT-FP; v2 does not.** arXiv:2408.01526**v1**, Table 9 (walls only), reports MLSTRUCT-FP results that are missing from v2:

| Model (v1, EfficientNetB2) | MLSTRUCT-FP walls Rec / F1 / IoU † |
|---|---|
| CAB1 | 92 / 89 / 81 |
| CAB2 | 92 / 90 / 83 |

The column order (Rec / F1 / IoU) is inferred from the matching CubiCasa and R3D columns. The split is the generic 60/20/20 of Kratochvila et al., and v1 does not describe the tiling. These are the only published CAB numbers on a multi-unit dataset. They are the reference point for the MLSTRUCT-FP experiment proposed in §3.

---

## 1. Papers reviewed

### 1.1 Pixel segmentation of walls and openings (directly comparable task)

**Knechtel et al., *Semantic floorplan segmentation using self-constructing graph networks*, Automation in Construction 166, 2024.**
A ResNet-101 encoder shares two decoders, WallNet (walls and openings) and RoomNet (room types). A self-constructing graph module feeds a GCN, which captures long-range dependencies. The paper evaluates on R2V (715 train / 100 test) and R3D (179 / 53, Zeng's split). Its mIoU includes the room classes. Read verbatim from the open-access PDF.

| R3D (Table 4, per-class accuracy = recall) | Walls | Doors & windows |
|---|---|---|
| DeepLab v3 | 93 | 60 |
| DeepFP (Zeng) | 98 | 83 |
| DLAK-GAN | 95 | 82 |
| Knechtel (multi-task) | 97 | **88** |
| Knechtel WallNet alone | **98** | 87 |
| **Ours CAB2 V2M, fine-tuned, backbone unfrozen** | 97.2 | 86.8 |
| Ours CAB1 B4, fine-tuned, backbone unfrozen | 97.4 | 85.6 |
| Ours CAB2 V2M, zero-shot | 85.1 | 68.9 |

R3D mIoU incl. rooms: 78 without and 80 with post-processing; on R2V: 85 / 84. In their Table 2 survey, the best previous R3D mIoU is 74 (DLAK-GAN, OGA), and FPNet scores 73.
→ **Comparison:** after fine-tuning, our models are on par with the best published R3D wall and opening recall (within about 1 point), without room supervision or post-processing. We cannot compare mIoU because theirs includes rooms.

**MuraNet** (Huang et al., ICDAR Workshops 2023, arXiv:2309.00348). A multi-task network: wall segmentation plus door/window *detection* with a YOLO-style head. It is trained on CubiCasa5k with a 4000 / 500 / 500 split (not the official one) at 1536×1536, from scratch. Read verbatim from the arXiv HTML.

| CubiCasa5k, wall IoU (binary) | |
|---|---|
| U-Net base / 5-stage / 6-stage | 65.5 / 74.4 / 75.8 |
| SegNeXt base / large | 75.8 / 76.3 |
| **MuraNet base** | **78.4** |
| Ours CAB1 V2M (6-class model, wall IoU) | 72.8 |
| Ours CAB2 V2M | 68.7 |

Door/window detection: AP50 91.7, AP[.5:.95] 53.8.
→ **Comparison:** MuraNet's wall IoU is 5.6 points above our best. Different splits, a binary task and a larger input resolution all favour it, so the gap is an upper estimate. Together with FGSSNet (78.3) and MiT-UNet (77.2, below), walls-only models consistently reach 77–78 on CubiCasa5k. That is 5–6 points above our multi-class models. This is the largest consistent gap we see in the recent literature.

**MitUNet** (arXiv:2512.02413; Machine Vision and Applications 2026). A Mix-Transformer (MiT-B4) encoder with a U-Net decoder and scSE attention, trained with Tversky loss for walls only. It is pre-trained on CubiCasa5k and fine-tuned on the new **Floor Plan CIS** dataset: 500 plans from Russian and CIS listings, open on Zenodo (doi:10.5281/zenodo.17871079). Results are on one 80/20 validation split, averaged over 3 runs, and its mIoU includes background.

| Floor Plan CIS (bg + wall) | Recall | Precision | mIoU | Boundary IoU |
|---|---|---|---|---|
| DeepLabV3+ R50 (best loss) | 91.4 | 92.6 | 85.2 | 81.6 |
| SegFormer MiT-B4 (best) | 93.9 | 91.8 | 86.6 | 83.3 |
| UNet++ R50 (best) | 93.7 | 92.7 | 87.3 | 84.7 |
| MitUNet (from scratch, Tversky 0.6/0.4) | 92.4 | 94.8 | 87.9 | 85.0 |
| MitUNet (CubiCasa5k pre-trained, Lovász) | 94.5 | 93.6 | **88.8** | **85.9** |

The SALI-FP paper (below) also scores MiT-UNet on the CubiCasa5k test set: visible-wall IoU **77.2**, wall clDice 93.0.
→ **Not comparable yet.** We have not run anything on Floor Plan CIS. It is an open, small, out-of-region wall dataset, which makes it a cheap generalization test (§3). CubiCasa5k pre-training gives them about +0.9 mIoU, the same "pre-train on CubiCasa, fine-tune on target" recipe we use for R3D, CVC-FP and WAFFLE.

**Paek et al., IEEE Access 2025: patch-wise SegFormer-B5.** See [`COMPARISON_paper_sota.md` §5](COMPARISON_paper_sota.md). On MLSTRUCT-FP they report mIoU (bg + wall) of 92, wall recall 85 and overall accuracy 94. Their retrained baselines R2V and DeepFP score mIoU 84, and the MLSTRUCT-FP U-Net of Pizarro et al. 77. Their implied wall IoU is about 85–86. Our v1 CAB numbers on MLSTRUCT-FP (wall recall 92, IoU 81–83 †) are in the same range. Their CubiCasa5k numbers are not credible (see §5 of the comparison doc).

**Xu et al., *Automatic floor plan analysis using a boundary attention-based deep network* (FloorNet), IJDAR 2024.** Multi-task: a room-type decoder, a room-boundary decoder and multi-scale boundary attention, evaluated on R2V and R3D. I could not get the result tables (Springer paywall). Like Knechtel, it reports mIoU including rooms, so at most the wall and opening recall would be comparable.

### 1.2 CubiCasa5k checkpoints under an independent protocol

**SALI-FP** (Chen, Wang et al., Tsinghua, arXiv:2609.25615, Sep 2026). A multimodal-LLM floor-plan parser (gpt-image-2 / gpt-5.5) with deterministic evidence gates and vectorization. Its main value for us is the **independent re-evaluation of the official CubiCasa5K checkpoint** on the 400 official test plans: a 1024² aspect-preserving canvas, four classes (Boundary = walls + railings, door, window, room; stairs mapped to background), pooled confusion counts and IoU excluding background. Read verbatim from the arXiv HTML.

| CubiCasa5k test (pooled IoU) | Boundary (walls + railings) | Door | Window | Room |
|---|---|---|---|---|
| CubiCasa5K official checkpoint, raw | 71.7 | **61.8** | **72.0** | 90.1 |
| SALI-FP (LLM parser), raw | 10.9 | 4.0 | 5.2 | 62.8 |
| SALI-FP, after registration | 31.8 | 14.5 | 23.2 | 74.3 |
| **Ours CAB1 V2M** (same class mapping) | **72.4** | 56.2 | 67.3 | – |
| **Ours CAB2 V2M** (same class mapping) | 67.9 | 59.2 | 69.9 | – |
| Ours CAB1 B4 / CAB2 B4 | 67.3 / 59.6 | 53.5 / 19.3 | 63.0 / 47.8 | – |

Our rows were recomputed from the pooled 10-fold test confusion matrices with SALI-FP's class mapping (walls + railings → Boundary, stairs → bg). Merging railings lowers our wall IoU slightly (CAB1 V2M 72.8 → 72.4). Our evaluation resolution differs from their 1024² canvas.

The official checkpoint's other numbers: Boundary IoU (the boundary-band metric) 55.5, boundary F1 at 0.5% tolerance 88.9, wall clDice 85.8.
→ **Comparison:** under SALI-FP's protocol, the official CubiCasa5K model scores **higher** than its own paper reported (doors 61.8 vs 53.6, windows 72.0 vs 66.8). With the same class mapping, our best model is **0.7 points ahead on walls** (CAB1 V2M 72.4 vs 71.7) but **2.6 points behind on doors and 2.1 on windows** (CAB2 V2M 59.2 / 69.9 vs 61.8 / 72.0). Our claim "we beat CubiCasa5K on doors and windows" therefore holds only against the numbers in the CubiCasa5K paper. The remaining differences (canvas size, rasterization) call for running the checkpoint through **our** evaluation (§3, experiment 2). The table also shows that multimodal LLM parsers are not yet competitive for pixel segmentation.

### 1.3 Datasets we do not use

| Dataset | Content | Size | Access | Used by | Relevance |
|---|---|---|---|---|---|
| **MLSTRUCT-FP** (Pizarro et al., Autom. Constr. 2023) | Multi-unit, whole-floor plans (Chile); walls + unit slabs | 954 plans, 6300–9650 px | Open (GitHub) | Our v1, Paek 2025, Pizarro U-Net | **High:** multi-unit, prior CAB numbers exist, loader already in repo |
| **R2V / LIFULL HOME'S** (Liu et al. 2017) | Japanese residential; walls (single-line) + openings + rooms | 815–870 plans | Application to NII-IDR | Zeng, OGA, DLAK, Knechtel, Paek, VectorFloorSeg | **High** for comparability: the most-used benchmark we lack |
| **Floor Plan CIS** (MitUNet, 2025) | Russian/CIS listings; walls | 500 | Open (Zenodo) | MitUNet | Medium: cheap out-of-region wall test |
| **MSD** (van Engelenburg et al., ECCV 2024) | Multi-apartment building complexes from Swiss Dwellings; structural masks incl. load-bearing walls | 5.3K floors, 18.9K apartments | Open | Generation methods (HouseDiffusion) | Low–medium: rasterized from vectors, not real drawings; good for multi-unit structure, not drawing style |
| **ArchP10k** (SALI-FP, 2026) | Mixed architectural plans | 11.5K | Not stated as public | SALI-FP | Low until released |
| FloorPlanCAD (Fan et al., ICCV 2021) | Vector CAD symbol spotting | 15K | Open | GAT-CADNet, CADSpotting | Low: vector input, different task (already cited by Kratochvila et al.) |

### 1.4 Adjacent tasks (not directly comparable)

- **Room-polygon reconstruction:** RoomFormer, FRI-Net, PolyRoom, Raster2Seq and Raster-to-Graph (Hu et al., CGF 2024). They predict room polygons, not wall/opening masks, and are evaluated with room/corner/angle F1. On CubiCasa5k, room / corner / angle F1 is: Raster2Seq 76.3 / 54.6 / 36.8 †; official CubiCasa5K polygons 63.0 / 32.7 / 23.6 † (SALI-FP Appendix). These only become relevant if we evaluate the reconstruction stage of our pipeline (§3, experiment 7).
- **Vectorization without segmentation:** Xing et al. (sparse point sets, Autom. Constr. 2025) and Song & Yu (GNN on floor-plan graphs). These assume a Manhattan world; there is no shared metric with us.
- **Patch-based processing of large buildings:** Kim, Kim & Yu (ISPRS IJGI 2021) and Paek et al. 2025. Both are directly relevant to how we would handle MLSTRUCT-FP-sized plans.

---

## 2. Where we stand: summary

Our column uses the current best results from [`OVERVIEW_all_datasets.md`](OVERVIEW_all_datasets.md).

| Setting | Best published (incl. Kratochvila et al. where relevant) | Ours, current best | Verdict |
|---|---|---|---|
| R3D walls / openings recall, in-domain | Knechtel 97–98 / 87–88; DeepFP 98 / 83 | 97.2 / 86.8 (CAB2 V2M) and 97.4 / 85.6 (CAB1 B4), fine-tuned, unfrozen | **On par** with Knechtel, above DeepFP on openings |
| R3D walls IoU / mean IoU (walls + openings), in-domain | Kratochvila et al. 90 / 82 | 93.7 (CAB1 B4) / 83.2 (CAB2 V2M), fine-tuned, unfrozen | **Ahead** of Kratochvila et al.; Knechtel reports no class IoU |
| CubiCasa5k wall IoU | MuraNet 78.4, FGSSNet 78.3, MiT-UNet 77.2 (walls-only models) | 72.8 (CAB1 V2M) | **5–6 points behind** walls-only models |
| CubiCasa5k Boundary / door / window IoU, SALI-FP class mapping | Official CubiCasa5K checkpoint 71.7 / 61.8 / 72.0 | 72.4 (CAB1 V2M) / 59.2 / 69.9 (CAB2 V2M) | **Walls +0.7; doors −2.6; windows −2.1** (resolution still differs) |
| CubiCasa5k, Paek et al. | 0.96 4-class mIoU | 0.736 | Not credible / not comparable (comparison doc §5) |
| MLSTRUCT-FP walls | Paek ≈ 85–86 IoU (implied), recall 85 | v1 CAB: IoU 81–83 †, recall 92 † | Roughly on par; **no current-repo numbers** |
| CVC-FP zero-shot walls IoU | Graumann 2025: 29.6 | 66.5 (CAB2 V2M) | **Far ahead** |
| CVC-FP in-domain walls IoU | Kratochvila et al. 91 (CAB2); Dodge / U-Net 94–95 ³ | 82.6 (CAB1 V2M, fine-tuned, unfrozen); openings IoU 77.0, mean IoU 79.8 | **About 8 points behind** Kratochvila et al.; no new published CVC results to add |
| WAFFLE zero-shot walls IoU | ControlNet wall model 63.2; CubiCasa5K official 48.8 | 58.1 (CAB2 B4), 57.6 (CAB1 V2M) | Behind the diffusion model, ahead of CubiCasa5K |
| WAFFLE in-domain walls / openings IoU | No published fine-tuned result | 77.4 / 40.6 (CAB2 B4), 75.6 / 41.7 (CAB2 V2M), fine-tuned, unfrozen | No comparison available |

³ Secondary sources; probably the wall/background mean IoU, so not a walls-class IoU.

None of the newly reviewed papers report CVC-FP or WAFFLE results. The CVC-FP and WAFFLE rows come from the comparison doc and are listed here for completeness.

---

## 3. Suggested experiments, by priority

1. **MLSTRUCT-FP, zero-shot and fine-tuned (highest priority).**
   - *Why:* it is the project's target domain (multi-unit), it puts back the v1 Table 9 numbers that v2 dropped, and it is the one dataset where Paek et al.'s numbers are plausible enough to compare against.
   - *How:*
     - Use the official MLSTRUCT-FP split, as Paek et al. do, with walls only.
     - Use tiled inference with overlap and softmax averaging. Whole-plan resizing of 6000–9000 px images would erase walls.
     - Report wall IoU / recall / F1, plus bg+wall mIoU and overall accuracy, to match Paek's table.
     - Run zero-shot from the CubiCasa5k folds, then fine-tune with the same protocol as R3D/CVC-FP (unfrozen).
   - *Effort:* data download, `datasets/mlstructfp_extract.py` (needs `MLStructFP` + `fp.json`), and a tiling wrapper in the evaluation.
2. **Run the official CubiCasa5K checkpoint through our evaluation (cheap, high value).** SALI-FP shows it scores 61.8 / 72.0 on doors / windows under another protocol, above our models. Running the public PyTorch checkpoint on our 400-plan test with our class mapping would settle whether our "beats CubiCasa5K on openings" claim holds. Also report it with our 3-class mapping on R3D, CVC-FP and WAFFLE (zero-shot), where we currently only have the WAFFLE paper's numbers.
3. **Close the wall gap on CubiCasa5k (diagnostic).** Walls-only models reach 77–78 IoU and ours 69–73. Three checks would locate the cause:
   - **Input resolution:** MuraNet uses 1536², and FGSSNet uses crops. Evaluate our models at a higher resolution or tiled.
   - **Class competition:** wall IoU of the same model with railings merged into walls, which is also SALI-FP's "Boundary" definition and directly comparable to 71.7.
   - **A walls-only CAB head trained on the same folds.**

   If resolution explains most of it, tiled inference (needed for MLSTRUCT-FP anyway) also improves CubiCasa.
4. **Add boundary-aware metrics.** Boundary IoU, boundary F1 at fixed tolerances and wall clDice are now reported by SALI-FP and MitUNet. They matter more than pixel IoU for vectorization and 3D reconstruction, which is the goal of our paper. They need predicted masks, not just confusion matrices, so add them to `evaluate.py` and re-run the saved fold models.
5. **R2V / LIFULL.** This is the most-used benchmark we lack (Zeng, OGA, DLAK, Knechtel, Paek, VectorFloorSeg), and walls + openings recall is comparable across all of them. It requires an NII-IDR data application, so start the request early. Report fine-tuned wall and opening recall to match Knechtel Table 4.
6. **Floor Plan CIS (Zenodo, 500 plans).** A cheap zero-shot and fine-tuned wall test in a new drawing style and region. It is directly comparable with MitUNet (bg+wall mIoU 88.8, Boundary IoU 85.9). Note their numbers are on a validation split; we would use 5-fold CV as for CVC-FP.
7. **Evaluate the reconstruction stage, not only segmentation.** The title of Kratochvila et al. promises reconstruction, but every number we report is pixel segmentation. Options:
   - room / corner / angle F1 with the Raster2Seq evaluator on CubiCasa5k, to compare with Raster2Seq, RoomFormer and the CubiCasa5K polygons;
   - wall-graph or vector-level metrics on R3D and MLSTRUCT-FP.
8. **DeepFP-style post-processing on R3D** (closing, hole filling, one class per room region). Knechtel reports +1 to +6 points per class from it. It is cheap, and it is the standard in R3D comparisons.
9. **MSD (lower priority).** It is useful mainly as a large multi-unit *structural* benchmark (load-bearing walls). The images are rasterized from vector data, so it measures layout understanding, not robustness to drawing style.

Experiments 1–3 can share one engineering step (tiled inference) and need no new data applications. Together they would answer the main open questions: multi-unit performance, the opening baseline and the wall gap.

---

## Sources

- Kratochvila et al., arXiv:2408.01526 v1 (MLSTRUCT-FP Table 9) — https://arxiv.org/html/2408.01526v1 ; v2 — https://arxiv.org/html/2408.01526v2
- Knechtel, Rottmann, Haunert, Dehbi, Automation in Construction 166 (2024) 105649 — https://repos.hcu-hamburg.de/bitstream/hcu/1063/1/1-s2.0-S0926580524003856-main.pdf
- Huang et al., MuraNet, ICDAR-W 2023 — https://arxiv.org/abs/2309.00348
- MitUNet, arXiv:2512.02413 / Machine Vision and Applications 2026 — https://arxiv.org/abs/2512.02413 , https://dl.acm.org/doi/10.1007/s00138-026-01815-y ; Floor Plan CIS dataset — https://doi.org/10.5281/zenodo.17871079
- Chen, Wang et al., SALI-FP, arXiv:2609.25615 — https://arxiv.org/abs/2609.25615
- Paek et al., IEEE Access 13 (2025) — https://doi.org/10.1109/ACCESS.2025.3642914
- Xu et al., FloorNet, IJDAR 2024 — https://doi.org/10.1007/s10032-024-00487-6
- van Engelenburg et al., MSD, ECCV 2024 — https://arxiv.org/abs/2407.10121
- Pizarro, Hitschfeld, Sipiran, MLSTRUCT-FP, Autom. Constr. 156 (2023) — https://github.com/MLSTRUCT/MLSTRUCT-FP
- Hu et al., Raster-to-Graph, CGF 2024 — https://diglib.eg.org/handle/10.1111/cgf15007
- Compliance-check survey (notes that no study segments individual apartments in multi-unit plans), arXiv:2607.00015 — https://arxiv.org/abs/2607.00015
