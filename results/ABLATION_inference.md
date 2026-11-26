# Ablation study: CAB1 and CAB2 V2M champions on CubiCasa5k

Written 2026-11-26. Plan, method and run log: [`docs/ABLATION_PLAN.md`](../docs/ABLATION_PLAN.md). Generated tables: [`ablation_inference/SUMMARY_main.md`](ablation_inference/SUMMARY_main.md) (10 folds), [`ablation_inference/SUMMARY_screen.md`](ablation_inference/SUMMARY_screen.md) (fold 0, 100 images, includes single edges and AC probes) and [`loss_ablation/SUMMARY.md`](loss_ablation/SUMMARY.md) (retraining).

## Summary

- **Every architectural component the trained champions contain is used.** With mean-ablation at inference:
  - CAB1 loses 4.1 mIoU (fg) without the channel attention, 9.4 without spatial attention, 14.2 without both, and 19.2 with single-dilation spatial attention.
  - CAB2 loses 17.2 when reduced to plain U-Net skips and 13.4 when reduced to CAB1's wiring.
  - Every drop holds in 10 of 10 folds.
- **The extra loss terms do not help.** Retrained on folds 0–4, focal loss + AWL alone is **+0.70 (CAB1) and +1.65 (CAB2, better in 5/5 folds)** mIoU (fg) above the full champion objective (focal + AAF + heatmap).
- **The asymmetric conv (AC) block is exactly one 3×3 conv at inference** (re-parameterisation error < 1e-6). Any benefit it has comes from training dynamics, which this study cannot measure.
- **No component costs measurable inference time.** All variants run at 41–42 ms (CAB1) and 43 ms (CAB2) per 768² image on an idle A100, because mean-ablation changes weights, not the graph. The champion latencies are CAB1 41.8 ms and CAB2 43.0 ms.
- **What this does and does not show:**
  - An inference-time ablation measures how much the *trained* network relies on a component, not how good a network *trained without it* would be. Drops are therefore upper bounds on the training-time value of a component.
  - The loss ablation is a real retraining ablation. It is the more reliable of the two for the claims "component X improves the model".

## 1. Setup

| | CAB1 | CAB2 |
|---|---|---|
| Model | EfficientNetV2M, HPO trial 21 | EfficientNetV2M, HPO trial 32 |
| Components in the checkpoint | AC (h + v + 2 diagonals), CAM, SAM (dilations 1, 2, 3), regular UNet3+-style skips | AC (square + h + v), **no attention** (HPO chose `cam=1`), fully connected skips |
| Reference (S0) on the 400-plan test set, 10 folds pooled | mIoU (fg) 54.33, mIoU (all) 61.37 | mIoU (fg) 55.14, mIoU (all) 61.99 |

- **Mean-ablation:** every decoder branch ends in GroupNorm → ReLU. Setting GN γ = 0 and β = μ replaces a branch by its per-channel mean μ. The mean is measured on 100 images of each fold model's own **validation** split, never the test set. Attention gates are replaced by their validation means the same way.
- **Step 0:** the unedited models reproduce the official fold-0 confusion matrices (relative difference 8e-6 for CAB1, 4e-6 for CAB2).
- **Metrics:** pooled confusion matrix over the 10 fold models × 400 test plans (screen: fold 0, 100 plans). mIoU (fg) is the mean over the 5 foreground classes. mIoU (all) includes background and is the headline metric in `results/official`. Δ is in percentage points against S0 on the same folds and images. "Worse in" counts folds.

## 2. Attention module (CAB1 only)

10 folds × 400 test plans.

| Variant | mIoU (fg) | Δ | Δ fold std | worse in | Mean class acc | walls | railings | doors | windows | stairs |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 full | 54.33 | – | – | – | 67.63 | 72.8 | 27.5 | 56.2 | 67.3 | 47.9 |
| A1 CAM → mean | 50.23 | −4.10 | 0.62 | 10/10 | 62.43 | 70.4 | 24.3 | 55.0 | 64.2 | 37.2 |
| A2 SAM → mean | 44.93 | −9.40 | 8.17 | 10/10 | 53.53 | 70.7 | 22.9 | 42.1 | 52.9 | 36.0 |
| A3 CAM + SAM → mean | 40.10 | −14.23 | 8.36 | 10/10 | 47.81 | 68.7 | 20.7 | 41.7 | 47.8 | 21.6 |
| A4 SAM single dilation | 35.17 | −19.16 | 15.37 | 10/10 | 43.31 | 66.8 | 20.3 | 27.9 | 33.3 | 27.5 |

- **Spatial attention matters most, mainly for openings:** door IoU drops 14 points and window IoU 14 points. Walls barely change (−2).
- **Removing the dilation-2 and dilation-3 terms of SAM is worse than removing SAM entirely.** The network depends on the multi-dilation response, and replacing two of the three terms by constants shifts the gate far from its trained operating point. The large fold std (15.4) shows that this depends strongly on the fold model.
- **Against the original ablation** (supplement, CAB2 variants, each trained once, CubiCasa + MURF):

  | Pixel acc / mean class acc / mIoU | AC only | +CAM | +SAM | +CAM+SAM |
  |---|---|---|---|---|
  | | 0.85 / 0.66 / 0.54 | 0.85 / 0.66 / 0.55 | 0.85 / 0.66 / 0.54 | 0.85 / 0.70 / 0.54 |

  - The original training-time effects are ≤ 0.01 mIoU and single runs.
  - Our CAB2 HPO champion has no attention at all, which agrees with them: attention did not pay off for CAB2.
  - For CAB1, attention was never ablated before. The trained CAB1 relies on it heavily.

## 3. Skip connections (both models)

10 folds × 400 test plans, mIoU (fg).

| Variant | What is removed | CAB1 | Δ | CAB2 | Δ |
|---|---|---|---|---|---|
| S0 full | – | 54.33 | – | 55.14 | – |
| S1 U-Net only | every input except the same-level encoder skip and the previous decoder stage | 49.84 | −4.49 | 37.91 | **−17.23** |
| S2 no shallower encoder | encoder maps from shallower levels | 49.84 | −4.49 | 51.23 | −3.91 |
| S3 no same-level skip | the same-level encoder skip | 34.84 | −19.49 | 43.26 | −11.88 |
| S4 no previous decoder | the input from the deeper decoder stage | 1.76 | −52.57 | 2.65 | −52.49 |
| S5 CAB2 → CAB1 wiring | the edges CAB2 has and CAB1 lacks | n/a | | 41.77 | −13.37 |
| S6 no extra deeper | deeper decoder inputs beyond the previous stage | n/a | | 41.99 | −13.15 |
| S7 no far shallower | shallower encoder inputs beyond one level | n/a | | 54.98 | −0.16 (8/10) |

- **For CAB1, S1 and S2 are the same edit.** In CAB1's regular wiring, the shallower encoder maps are the only cross-scale inputs.
- **CAB2's full connectivity matters through the extra *deeper* decoder inputs.** S6 alone costs 13.2, about the same as S5 (13.4). The extra far-shallower encoder inputs are nearly unused (S7 −0.16). So the closest inference-time answer to "does the fully connected wiring matter" is: yes, through deep-to-shallow decoder links, not through extra encoder links.
- **S4 collapses both models.** At the last stage, the previous-decoder input carries the whole deep path; once it is constant, nothing from the earlier decoder stages reaches the output. It is an expected near-zero, not a finding about wiring quality.

**Single edges (screen, fold 0, 100 plans; Δ mIoU (fg)).** Decoder stage 0 is the deepest stage and stage 2 the last.

| Stage | CAB1 previous decoder | CAB1 same-level | CAB1 +1 shallower | CAB2 previous decoder | CAB2 same-level | CAB2 +1 shallower | CAB2 extra deeper |
|---|---|---|---|---|---|---|---|
| 0 | −32.1 | −10.7 | −5.3 | −29.4 | −7.1 | −4.3 (+2: −0.4) | – |
| 1 | −52.2 | −3.2 | −1.0 | −43.5 | −1.9 | +0.1 | −1.7 |
| 2 | −51.1 | −1.4 | – | −51.5 | −1.2 | – | −5.4 (−2), −2.2 (−3) |

Same-level and shallower encoder skips matter most at the deepest decoder stage. CAB2's extra deeper inputs matter most at the last stage. No single extra edge accounts for the 13-point S5 drop on its own; the effect is cumulative.

## 4. Asymmetric conv block (both models)

- **Re-parameterisation:** in every AC block of all 10 fold models of both champions, the fused 3×3 kernel equals the sum of the branch convolutions (max relative error 9e-7; `ablation_inference/ac_kernels.json`). AC adds no inference-time capacity over a plain 3×3 conv.
- **Energy share of each branch,** relative to the fused kernel:
  - CAB1: horizontal 0.26, vertical 0.27, each diagonal 0.18.
  - CAB2: square 0.42, horizontal 0.19, vertical 0.20.
  - The centre cross holds 71% (CAB1) and 78% (CAB2) of the energy.
- **Sensitivity probes** (screen, fold 0): zeroing h + v gives −46.2 (CAB1) and −32.7 (CAB2); zeroing the diagonals (CAB1) or the square kernel (CAB2) gives all-background output (−53). These zero half of every fused 3×3 conv, so they show only that the fused kernels are not redundant. **They must not be reported as "AC removed".** A real AC ablation needs retraining with plain 3×3 convs.

## 5. Loss components (retrained, folds 0–4)

Each variant was retrained from scratch with the champion's config and only the loss set changed (100 epochs, early stopping on val loss, patience 30). Pooled over folds 0–4 × 400 test plans; Δ against the champion folds 0–4.

| Model | Variant | Epochs | mIoU (fg) | mIoU (all) | Δ fg | Δ fold std | worse in | Mean class acc | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| CAB1 | L1 focal (aUF) + AWL | 57–65 (early stop) | 54.79 | 61.73 | +0.70 | 0.73 | 0/5 | 70.81 | 41.7 |
| CAB1 | L2 focal + AAF + AWL | 100 | 55.02 | 61.97 | +0.94 | 1.30 | 1/5 | 65.69 | 41.8 |
| CAB1 | L3 focal + AAF + heatmap (champion) | 100 | 54.09 | 61.17 | – | – | – | 67.32 | 41.8 |
| CAB2 | L1 focal (aUF) + AWL | 57–82 (early stop) | **56.53** | **63.21** | **+1.65** | 0.66 | 0/5 | 71.92 | 43.1 |
| CAB2 | L2 focal + AAF + AWL | 100 | 54.43 | 61.39 | −0.45 | 0.63 | 4/5 | 63.38 | 43.1 |
| CAB2 | L3 focal + AAF + heatmap (champion) | 100 | 54.88 | 61.77 | – | – | – | 65.38 | 43.2 |

- **Neither AAF nor heatmap regression improves test mIoU.** For CAB2, the plain focal objective is better in all 5 folds, by more than the champion's fold std (0.5).
- **L1 has a clearly higher mean class accuracy** (+3.5 CAB1, +6.5 CAB2), at slightly lower precision on CAB1.
- **This is the opposite of the original table,** which reports a small gain from the extra terms: L_aUF 0.53 → +AAF 0.54 → +AAF +MHR 0.54 mIoU, with mean class accuracy 0.66 → 0.70. Their effects are within single-run noise, and ours are measured over 5 folds.
- **Caveats:**
  - The champion folds come from a different training run, so each Δ also contains seed noise.
  - Every L1 run stopped early, while all L2 and champion runs used the full 100 epochs. The saved weights are the last epoch, without restoring the best one.
  - The loss set changes the val loss that drives LR reduction and early stopping. That is part of the ablated objective, not a confound.

## 6. Paper vs code differences found during the study

- **CAB2 attention:** the paper's CAB2 has attention; the HPO champion has none (`cam=1`).
- **CAM bottleneck:** the supplement compresses pooled channel features to 1/16, while the code uses `CAM(..., ratio=2)`, a 1/2 bottleneck (`cab1.py:424`, `cab2.py:453`).
- **SAM:** C_s = Σ_{i=1..3}(1×1 conv + 3×3 conv with dilation i) matches the code (`HHDC`, dilations 1, 2, 3).
- **Heatmaps:** B = {2, 10} matches `datasets/create_heatmap.py`. The code also adds the opening contour to each heatmap, which the supplement does not describe.
- **The champions differ in more than skip wiring** (AC kernels, attention, AAF dilations, LR schedule). Comparing CAB1 and CAB2 headline numbers is not a skip-connection ablation; S5 is the controlled substitute.

## 7. Not covered

- **A training-time AC ablation and a training-time attention ablation:** both need retraining.
- **Learned AWL loss weights from the TensorBoard events** (plan §5, optional): not extracted.
- **Test-time scale robustness** (plan A4, optional): not run.
