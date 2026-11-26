# Ablation plan: CAB1 + CAB2 champions, no retraining

Status (2026-11-26): **everything has run.** The inference-time ablation (screen and main phase), the loss ablation by retraining (§9, §13) and the idle-latency benchmark are done and committed. Results are in `results/ablation_inference/SUMMARY_main.md` and `results/loss_ablation/SUMMARY.md`; the latest status is §12–13. The final write-up (§8.3 item 4) is [`results/ABLATION_inference.md`](../results/ABLATION_inference.md) (2026-11-26). Sections 1–11 are kept as the plan and log at the time they were written.

Original status: **plan only, nothing run yet.** Revised 2026-11-17. Scope agreed with the user: study **only the current best models**, with **the setup that produced the current best results**, and **no retraining**.

## 1. Scope and setup (unchanged from the reported results)

| | CAB1 | CAB2 |
|---|---|---|
| Model | V2M, HPO trial 21 (`kfold_patch/eval_cab1_hpo_v2m.py`) | V2M, HPO trial 32 (`kfold_patch/eval_cab2_hpo_v2m.py`) |
| Checkpoints | `models/cab1_cab1_hpo_trial21_v2m_*` (10 fold models) | `models/cab2_cab2_hpo_trial32_v2m_*` (fold dirs picked as in `evaluate_kfold.find_fold_model_dirs`) |
| Reference result | `results/official/test/test_kfold_cab1_EfficientNetV2M_20261014-122132_fold_cms.npy` | `…cab2_EfficientNetV2M_20261014-120852…` |
| Test data | CubiCasa5k official test, 400 plans, 768×768 (`data/tfrecords/cubicasa5k/cubicasa5k_test.tfrecords`) | same |
| Metric | per-class Rec / F1 / IoU, mean foreground IoU, pooled confusion matrix over the 10 fold models (same code path as `evaluate_kfold.evaluate_fold`) | same |

**Step 0 (sanity check):** the unmodified pipeline must reproduce the reference confusion matrices exactly before any ablation is trusted.

**Metrics for comparability with the original ablation:** report the supplement's three metrics (pixel accuracy, mean class accuracy = mean recall, mean class IoU) next to our usual per-class Rec/F1/IoU. All are computed from the same pooled confusion matrix.

## 1b. The original ablation (`docs/suppl.tex`, section "Ablation Study")

| | Original (supplement) | This plan |
|---|---|---|
| Model | **CAB2 only** (variants of it) | CAB1 and CAB2 V2M champions |
| Data | "combined dataset" = CubiCasa (refined 560) + **MURF** (private, 8 plans; not in this repo) | CubiCasa5k official test (400) |
| Method | each variant **trained**, one run each, no variance reported | frozen checkpoints, inference-time ablation, 10 fold models → fold std |
| Attention (Tab. "ablation_am") | baseline = CAB2 with a single AC conv per decoder block; +CAM, +SAM, +CAM+SAM | CAB1: CAM / SAM / both / single dilation removed. CAB2: none to ablate |
| Losses (Tab. "ablation_losses") | cumulative: L_aUF → +L_AAF → +L_MHR (on CAB2) | **cannot be reproduced without retraining**; cite the original |
| Not covered | AC block, CAB1 vs CAB2 skip wiring, SAM dilations, learned loss weighting (AWL), augmentation, post-processing | AC block, skip wiring, SAM dilations |

Original numbers (Pixel Acc / Mean Class Acc / Mean Class IoU):

| Attention | AC only | +CAM | +SAM | +CAM+SAM |
|---|---|---|---|---|
| | 0.85 / 0.66 / 0.54 | 0.85 / 0.66 / **0.55** | 0.85 / 0.66 / 0.54 | 0.85 / **0.70** / 0.54 |

| Losses | L_aUF | +L_AAF | +L_AAF +L_MHR |
|---|---|---|---|
| | 0.85 / 0.66 / 0.53 | 0.85 / 0.68 / 0.54 | 0.85 / **0.70** / 0.54 |

What this means for the new study:
- **The original effects are small and untested for noise.** Single runs, IoU differences ≤ 0.01, pixel accuracy identical everywhere. The text says CAM+SAM is best, but on mean IoU it equals the AC-only baseline (0.54) and CAM alone is higher (0.55). Only mean class accuracy improves clearly (+0.04).
- **It agrees with our HPO result:** the CAB2 V2M champion ended up with **no attention** (`cam=1`), consistent with attention giving no IoU gain for CAB2 in the original table. CAB1's attention was never ablated before, so A2 below is new.
- **Skip wiring (CAB1 regular vs CAB2 full) and the AC block were never ablated.** A1 and A3 fill that gap.
- **Loss ablation stays as the original table.** Our run can only add the indirect evidence of §5.

Paper vs code differences found while checking the supplement (to mention in the write-up, no change to the champions):
- **CAM bottleneck:** the supplement compresses the pooled channel features to **1/16**; the code calls `CAM(..., ratio=2)` (`cab1.py:424`, `cab2.py:453`), i.e. a 1/2 bottleneck.
- **SAM:** C_s = Σ_{i=1..3} (1×1 conv + 3×3 conv with dilation i) matches `HHDC` (dilations 1, 2, 3).
- **Heatmaps:** B = {2, 10} matches `datasets/create_heatmap.py` (`sigmas = [2, 10]`). The code also adds the opening contour (`contours_map / 255`) to each β-heatmap, which the supplement doesn't describe.

## 2. Which paper components can be ablated without retraining

The paper (arXiv:2408.01526, §III-A) describes: asymmetric conv block (AC), attention module (AM = CAM ‖ SAM with multi-dilation spatial branch), multi-scale skip connections (regular in CAB1, fully connected in CAB2), and a multi-task objective (asymmetric unified focal loss + adaptive affinity fields + heatmap regression, with learned loss weights).

What the champion checkpoints actually contain (checked by loading them):

| Component | CAB1 champion | CAB2 champion | Inference-time ablation possible? |
|---|---|---|---|
| AC block | h + v + 2 diagonal kernels, no square kernel | square + h + v | **Partly.** See A3 |
| CAM | yes (`cam`, `cam_1`, `cam_2`) | **absent** (`cam=1` = no attention in cab2.py) | CAB1 only |
| SAM, dilations [1,2,3] | yes (`sam5*`, per-scale variant) | **absent** | CAB1 only |
| Multi-scale skips | regular (`unet3plus_down_from{i}_to{j}`) | fully connected | **Yes**, both |
| Focal loss, AAF, AWL | training only | training only | **No** |
| Heatmap regression | training only (no separate head: the model outputs 6 channels = bg + 5 classes; the heatmap loss supervises the door/window probability maps) | same | **No** |

**Consequence to state in the write-up:** a no-retraining ablation measures how much the *trained network relies on* a component. It does not measure how much worse a network *trained without it* would be. The four loss components cannot be ablated this way at all. Section 5 lists the cheap indirect evidence for them.

## 3. Ablations

All ablations are **weight edits on the loaded SavedModel**, so no graph surgery or retraining is needed. Every decoder branch ends in GroupNorm → ReLU, which allows exact *mean-ablation*: set GN γ = 0 and β = μ, where μ is the per-channel mean of that branch's output, measured on the fold model's own **validation** fold (never the test set). Then ReLU(μ) = μ. Mean-ablation is preferred over zeroing because zeroing also shifts the input distribution of the next layer.

### A1. Multi-scale skip connections (CAB1 and CAB2): the CAB1-vs-CAB2 question
Mean-ablate skip branches `unet3plus_down_from{i}_to{j}` (and the `up_*` decoder→decoder branches) in groups:

| Variant | What is removed | CAB1 | CAB2 |
|---|---|---|---|
| S0 | nothing (reference) | ✓ | ✓ |
| S1 "U-Net only" | every cross-scale input; keep same-level encoder + previous decoder | ✓ | ✓ |
| S2 shallower encoder inputs | encoder maps from shallower levels | ✓ | ✓ |
| S3 deeper encoder inputs | encoder maps from deeper levels | ✓ | ✓ |
| S4 extra decoder inputs | decoder maps other than the immediately deeper one | n/a (CAB1 has none) | ✓ |
| S5 "CAB2 → CAB1 wiring" | the edges CAB2 has but CAB1 does not | n/a | ✓ |
| S-edge (screening only) | each single edge, one at a time → **edge-importance matrix** per model | ✓ | ✓ |

S5 is the closest inference-time answer to "does the fully connected wiring matter".

### A2. Attention module (CAB1 only)
| Variant | Edit |
|---|---|
| AM-CAM | replace the CAM gate with its validation-mean value (fold the constant into CAM's 1×1 conv kernel; the gate logits weights become 0) |
| AM-SAM | same for each SAM5 spatial gate |
| AM-both | both |
| AM-dil | SAM dilated branches: mean-ablate the dilation-2 and dilation-3 kernels, keep dilation 1 ("single dilation") |

For CAB2 the HPO already removed attention. This is reported as a finding, not ablated.

### A3. Asymmetric conv block (both)
1. **Re-parameterization check.** The parallel kernels are summed before normalization (`CONV_stack`), so each AC block collapses exactly into one 3×3 kernel. For CAB1, h + v + both diagonals cover all 9 taps. Fuse them and confirm the outputs are identical (tolerance 1e-4). This shows AC adds **no inference-time capacity**, so any benefit comes from training dynamics, which cannot be tested without retraining.
2. **Branch sensitivity** (reported as sensitivity only, not as "training without"): drop the h+v kernels; drop the diagonal kernels (CAB1); drop the square kernel (CAB2).

### A4. (Optional, cheap) test-time scale
The paper trains multi-scale. Evaluate at 0.75× / 1.0× / 1.25× input to show scale robustness. This is an evaluation choice, not a component ablation.

## 4. Do we need a GPU? **No** (but it is ~50× faster; see §8 for the GPU resume)

Measured in this container (CPU only, 128 cores): one CAB1 V2M fold model takes **~2.1 s per 768×768 test image** single-process; in practice ~2.5 s per image per job with 32 threads, i.e. ~17 min per variant on 400 images (after warm-up, mixed_float16 policy on CPU; float32 may be faster). Model load takes ~16 s.

| Item | Count | CPU time, one process |
|---|---|---|
| One variant, 10 folds × 400 images | 4,000 forward passes | ~2.3 h |
| Validation-mean statistics (once per fold model, 1 val fold) | 20 models | ~2–3 h total |
| CAB1 variants (S0–S3, AM ×4, AC ×3) | ~11 | ~25 h |
| CAB2 variants (S0–S5, AC ×2) | ~8 | ~18 h |
| Edge-importance screening (fold 0, 100 test images) | ~15 + ~25 edges | ~3 h |

That is about **45–50 CPU-hours serial**. With 4–6 parallel processes on 128 cores (to be measured) it is roughly **10–15 h wall-clock**. A GPU would cut it to ~1–2 h but isn't required. Suggested order: screening on fold 0 first, then all 10 folds only for the main table.

## 5. Loss components: evidence available without retraining

| Component | Retrain-free evidence | Limitation |
|---|---|---|
| AWL (learned weights) | Read the final learned task weights/uncertainties for each loss term from the fold models' TensorBoard event files (`models/*/{fold}/events.out.tfevents*`) | Shows how much each term was weighted, not whether it helped |
| AAF | Boundary-quality metric (boundary IoU / boundary F1) of the champions | Descriptive only |
| Heatmap regression | Opening endpoint localization quality of door/window maps | Descriptive only |
| Focal loss | none | needs retraining |
| Optuna studies (`optuna_cab1.db`, `optuna_cab2_v2m.db`) | Trials differ in `aaf_preset` and `cam` | **Not usable as ablation:** the objective is total val loss, which depends on the loss set and on AWL (plateau ≈ −16.6 vs cosine ≈ −12.8 for otherwise similar configs), and in CAB1 `hhdc ∈ {False,5,7}` build identical networks |

If loss ablations are needed later, they require retraining. That is out of scope here; the previous version of this plan (git history of this file) has the retraining design.

## 6. Implementation (all CPU)

1. `kfold_patch/ablation_inference.py`:
   - load each fold model with `tf.keras.models.load_model`
   - apply a named edit set (layer-name regex → γ/β/kernel edits)
   - run the test set and save the pooled confusion matrix as `results/ablation_inference/{model}_{variant}_fold_cms.npy` plus a JSON with the edited layers
   - resumable; one variant × fold per job; `--folds`, `--max-images` for screening
2. `kfold_patch/ablation_stats.py`: per fold model, collect per-channel means of every skip-branch output and every attention gate on its validation fold; cache them as `.npz`.
3. Step 0 check (unedited = reference CMs) and A3.1 check (fused kernels = identical output).
4. `kfold_patch/summarize_ablation.py` writes `results/ABLATION_inference.md`:
   - Δ mean-fg IoU and per-class Δ IoU vs S0
   - fold std and a paired test over the 10 folds
   - edge-importance heatmaps for both models

## 7. Known findings to carry into the write-up
- The CAB2 champion has no attention module, unlike the paper's CAB2.
- CAB1 and CAB2 champions differ in more than skip wiring (AC kernels, attention, AAF dilations, LR schedule). Comparing their headline numbers is not a skip-connection ablation; A1/S5 is the controlled substitute.
- The Sept 2026 `hhdc`/`cam` ablation rows compared identical or mislabeled architectures (`hhdc ∈ {False,5,7}` are the same network; `cam` selects SAM variants in CAB1 and switches attention off in CAB2).

## 8. Status (2026-11-18) and how to resume on GPU

CPU work was stopped so the container can be rebuilt with GPUs. Nothing is committed yet. All files listed below are in the working tree (untracked).

### 8.1 Code (ready)

| File | Purpose |
|---|---|
| `kfold_patch/ablation_inference.py` | Main runner. Loads one fold model, collects validation statistics, applies each variant as an in-place weight edit (or a `clone_model` layer swap for SAM single-dilation), evaluates on the test set, writes one JSON per variant. It is resumable: finished variants are skipped. |
| `kfold_patch/summarize_ablation.py` | Pools the per-variant JSONs and writes `results/ablation_inference/SUMMARY_<phase>.md`. |
| `scripts/run_ablation_inference.sh` | Runs one job per (model, fold) and then the summarizer. The device is set with `DEVICES` (`-1` = CPU, default; `0` or `0,1` = GPU ids, round-robin). |
| `kfold_patch/ac_kernel_analysis.py` | AC re-parameterisation and branch-energy analysis. It uses only the weights and is already done. |

Runner CLI:

```
python kfold_patch/ablation_inference.py --model {cab1,cab2} --fold K --phase {check,screen,main} \
    [--max_test N] [--max_val 100] [--threads T] [--variants a,b] [--out_dir DIR]
```

- `--max_test` defaults to 100 test images in `screen` and all 400 otherwise.
- Validation statistics come from 100 images of that fold's validation split. They are cached in `results/ablation_inference/stats/{model}_fold{K}_val100.npz`.
- **Device:** `ablation_inference.py` no longer forces CPU; the caller sets `CUDA_VISIBLE_DEVICES`. `ac_kernel_analysis.py` still defaults to CPU, which is fine because it uses only the weights.

### 8.2 Done

**Step 0 check, fold 0, 400 test images, CPU** (`results/ablation_inference/check/*.json`):

| Model | GPU reference mIoU(fg), fold 0 | CPU S0_full | CPU X_manual_sam_identity |
|---|---|---|---|
| CAB1 | 0.5287 | 0.5126 (−1.6 pp) | 0.5127 |
| CAB2 | 0.5440 | 0.5400 (−0.4 pp) | – |

- **Manual SAM matches the model.** For CAB1, the manual SAM5 re-implementation reproduces the model (X ≈ S0), so the attention ablations are trustworthy.
  - The manual attention max abs error on val is 2e-2. That is fp16 rounding.
  - The CAM must be computed through its own sublayers (`cam.conv`, `avg_pool`, `fc11`…), because the saved model pools in float16.
- **CPU vs GPU gap.** The gap to the GPU reference is CPU mixed_float16 numerics, mostly in the CAB1 attention path. Each pixel-level confusion matrix differs by 1.45 % (CAB1) and 0.22 % (CAB2) relative to the reference.
  - All Δ values are measured against S0 on the same device, so the gap cancels.
  - **On GPU, rerun the `check` phase first.** S0 should then match `results/official/test/*_fold_cms.npy` almost exactly (`reference_cm_rel_diff` ≈ 0).
- **Validation stats exist for fold 0 only.** They were computed on CPU and are fine to reuse on GPU: `stats/cab1_fold0_val100.npz`, `stats/cab2_fold0_val100.npz`.

**AC kernel analysis, all 10 folds × both models, weights only** (`results/ablation_inference/ac_kernels.json`, log `logs/ac_kernels.log`):
- The fused 3×3 kernel equals the sum of the branch convs (max relative error 9e-7). At inference, the AC block is exactly one 3×3 conv, so AC can only matter through training.
- Energy share of each branch (the branch alone, relative to the fused kernel):
  - CAB1: horizontal 0.26, vertical 0.27, diagonal 0.18.
  - CAB2: square 0.42, horizontal 0.19, vertical 0.20.
- The centre cross holds 0.71 (CAB1) and 0.78 (CAB2) of the fused-kernel energy.
- The CAB1 diagonal kernels are exactly diagonal: the off-diagonal weights are 0.

### 8.3 Not done yet

1. **Screening (fold 0, 100 test images).** It was started on CPU and stopped before the first variant finished, so there are no results; `results/ablation_inference/screen/` is empty or absent.
2. **Investigate A3 (CAM+SAM → mean) = 0.0 mIoU.** In the 3-image smoke test, A3 and `AC_zero_diagonal` both gave exactly 0 (everything predicted as bg). A1 alone gave 0.35, A2 0.38 and S0 0.45.
   - Check the screening result. If it is still ≈ 0, inspect the edited layer outputs on one image.
   - The likely suspects are the combined CAM gate × SAM gate edit producing a large constant shift that later GroupNorms don't absorb, or a genuine collapse.
   - The weights are restored with `set_weights(original)` before every variant, so edits don't compound.
3. **Main phase:** 10 folds × 400 test images, all non-edge variants.
4. **Final write-up:** `results/ABLATION_inference.md`. It should compare against the supplement tables in §1b, list the paper-vs-code differences from §7, add the AC analysis above, and optionally read the AWL loss weights from the TensorBoard events (§5).

### 8.4 Resume commands (GPU container)

```bash
# 0) sanity: GPU visible, S0 reproduces the reference (~20 s/variant on GPU instead of ~17 min)
CUDA_VISIBLE_DEVICES=0 python kfold_patch/ablation_inference.py --model cab1 --fold 0 --phase check
CUDA_VISIBLE_DEVICES=0 python kfold_patch/ablation_inference.py --model cab2 --fold 0 --phase check
python -c "import json,glob;[print(p,json.load(open(p))['reference_cm_rel_diff']) for p in sorted(glob.glob('results/ablation_inference/check/*.json'))]"
#    (the old CPU check JSONs are skipped as 'already done'; delete results/ablation_inference/check/ first to rerun on GPU)

# 1) screening, fold 0  -> results/ablation_inference/SUMMARY_screen.md
DEVICES=0 JOBS=1 scripts/run_ablation_inference.sh screen

# 2) main, 10 folds     -> results/ablation_inference/SUMMARY_main.md
DEVICES=0 JOBS=1 scripts/run_ablation_inference.sh main      # or DEVICES=0,1 JOBS=2 with two GPUs
```

- **Memory:** one V2M model at 768×768 needs only a few GB, so 2 jobs per GPU (`DEVICES=0,0 JOBS=2`) should also fit.
- **Device consistency:** keep everything on one device type so the Δ values compare like with like. If the GPU check matches the reference, re-collect nothing; the stats from CPU are fine.
- **Logs:** `results/ablation_inference/logs/{phase}_{model}_f{fold}.log`.

## 9. Loss ablation by retraining (started 2026-11-19, option C)

Scope agreed with the user: **CAB1 + CAB2 champions, folds 0–4, retrained**, cumulative design as in the supplement. The full loss (L_aUF + L_AAF + L_MHR) is the existing champion fold models, so it is not retrained.

| Variant | `loss_functions` | Change vs champion | Config |
|---|---|---|---|
| L1 `auf` | focal + AWL | no heatmap, no AAF (`aaf=[]`; the model is then a plain Keras `Model`, AWL log-var still trained) | `kfold_patch/loss_ablation/{cab1,cab2}_auf.py` |
| L2 `auf_aaf` | focal + AAF + AWL | no heatmap regression | `kfold_patch/loss_ablation/{cab1,cab2}_auf_aaf.py` |
| L3 full | champion | – | existing `models/cab*_hpo_trial*_v2m_*` |

Everything else (backbone, architecture, LR schedule, 100 epochs, batch 4, folds) is inherited from `eval_{cab1,cab2}_hpo_v2m.py`.

- **Runner:** `scripts/run_loss_ablation.sh`. Queue A = CAB1 on GPU 0, queue B = CAB2 on GPU 2 (GPU 1 and 3 belong to other users). The order is fold-major, so fold 0 of both variants finishes first. Resumable: a fold counts as done when its `training_cfg.py` has `epochs=100` and a `saved_model.pb`. Restart with `GPU_A=0 GPU_B=2 FOLDS="0 1 2 3 4" setsid nohup scripts/run_loss_ablation.sh all > logs/loss_ablation/runner.log 2>&1 &`.
- **Outputs:** `models/{cab1,cab2}_lossabl_{model}_{variant}_v2m_*/{fold}` (one dir per fold run, ~75–90 MB). Logs: `logs/loss_ablation/{model}_{variant}_f{fold}.log`, `queue_{model}.log`.
- **Expected time:** CAB1 ~7.5 h/run, CAB2 ~8.6 h/run, 10 runs per queue, so **~3.6 days**, finishing about 2026-11-22/23. Fold 0 of both variants is done after ~17 h.
- **Smoke test (1 epoch, fold 0, L1, both models):** trains and saves fine. The first epoch takes ~400 s including XLA compilation, and the AWL log-var moves (0 → −0.11). The smoke dirs were deleted.
- **Caveats for the write-up:**
  - The champion fold models come from a different training run, so the paired Δ per fold also includes seed noise.
  - `reduce-lr-on-plateau` monitors val loss, and that changes with the loss set. This is part of the ablated "objective", not a confound to remove.
  - Fold-to-fold std of test mIoU(fg) for the champions is 0.006 (CAB1) and 0.005 (CAB2). With 5 folds, effects of ≥ ~0.01 are detectable.
- **Evaluation (to do once runs finish):** test-set CMs per fold through the same path as `evaluate_kfold.evaluate_fold`. Report pixel acc / mean class acc / mean class IoU plus per-class IoU and the paired Δ vs the champion on folds 0–4.

## 10. Inference-time ablation on GPU (2026-11-19)

- **Step 0 on GPU passes.** S0 reproduces the reference fold-0 CMs: CAB1 mIoU(fg) 0.5287, rel. CM diff 8e-6; CAB2 0.5440, rel. diff 4e-6. The CPU check results were moved to `results/ablation_inference/check_cpu/`.
- **Inference timing added.** Every variant JSON now has a `timing` block: per-image forward pass + argmax, batch 1, 768×768, the first 5 images excluded, synced through `.numpy()`; median, mean and p90 in ms, plus the device name. `SUMMARY_<phase>.md` has an `ms/img` column.
  - These runs share GPU 0/2 with the loss-ablation training, so the timings are only indicative. One inference job next to training measured ~42 ms/img; two jobs, one per GPU, measured ~85–89 ms/img.
  - **A clean latency benchmark (idle GPU) is still to do** after the training finishes. *(Done: §12 and §13.)*
- **Running:** `DEVICES=0,2 JOBS=2`, screen phase and then main phase. Log: `results/ablation_inference/logs/runner_gpu.log`.

## 11. Stopped 2026-11-19 08:35 UTC (moving the session to tmux)

All jobs were stopped by hand. GPU 0 and 2 are free again.

- **Loss ablation:** only `auf` fold 0 had started, for both models; it was stopped after 6 of 100 epochs. Measured speed is ~190–220 s/epoch, so ~5.5–6 h per run instead of 7.5–8.6 h; the full option C would take ~2.5 days.
  - Partial dirs remain: `models/{cab1,cab2}_lossabl_*_auf_v2m_*_20261119-080712/0`. They have a 6-epoch checkpoint and no `training_cfg.py`.
  - The runner retrains such folds from scratch. **Delete these dirs before restarting**, so that evaluation can't pick them up.
  - Restart with the command in §9.
- **Inference ablation:** screening (fold 0, 100 images) was partly done: 18 of ~60 variants (`results/ablation_inference/screen/`, `SUMMARY_screen.md`). It is resumable; finished variants are skipped. Restart with `TF_FORCE_GPU_ALLOW_GROWTH=true DEVICES=0,2 JOBS=2 scripts/run_ablation_inference.sh screen`.
- **Blocking issue found in screening:** many variants collapse to "everything is background" (mIoU 0, pixel acc 89.54 % = bg share):
  - CAB1: A1 (CAM only), A2, A3, A4 (single dilation), S4.
  - CAB2: S1, S4, S5, S6, and the single edge E_stage0_src0_ps-1 (previous decoder).
  - Every variant that mean-ablates a **decoder→decoder** branch or an **attention gate** collapses. Encoder-skip ablations give graded drops (S2 −4.8/−5.9 pp, S3 −13/−19 pp, S7 −0.4 pp).
  - In the earlier CPU smoke test, A1 alone gave 0.35, not 0. So this is most likely a bug in those edits (e.g. the stats for decoder-input branches or the CAM fold under float16), not a real finding. **Debug it on one image (compare edited-layer outputs vs. the expected mean) before running the main phase.**

## 12. All-background bug fixed, jobs restarted (2026-11-19)

- **Cause of the collapse in §11:** the cached fold-0 validation stats (`stats/{cab1,cab2}_fold0_val100.npz`, computed on **CPU** on 11-17) contained **NaN** in exactly the collapsing entries: every `mu/` of a decoder-input branch (ps ≤ −1), all `cam_g/*`, and `sam_*/<sam>/0`. Mean-ablating with NaN makes every logit NaN, and `argmax` then returns class 0 → all background. The encoder-skip means were finite, so S2/S3/S7 looked fine. Most likely the CPU mixed_float16 forward pass produced non-finite activations for some validation images. In addition, `reduce_sum` was done in float16: the per-image sums of the stage-2 branches reach ~63 k, close to the 65 504 limit.
- **Fix in `ablation_inference.py`:** activations are cast to float32 and summed in float64. Validation images with non-finite activations are skipped and counted (`n_val_skipped_nonfinite`). `check_stats` refuses non-finite stats at load and after collection, `apply_variant` asserts finite means, and the stats record the `device`.
- Stats were recomputed on GPU (A100): 0 of 100 images skipped, manual-attention max abs err 2.1e-2 (CAB1) / 0 (CAB2). The NaN stats are kept in `stats/cpu_nan/`. The old screen JSONs and summary are in `screen_nanstats/` (do not use); the whole screen is rerun with the new stats.
- **Check after the fix (fold 0, 100 images):** CAB1 A1 0.482 (S0 0.530), CAB2 S6 0.454 (S0 0.546). S4 ("no previous decoder") still nearly collapses (CAB1 0.019, CAB2 0.008), but the predictions are no longer all background (pixel acc 87.0 / 89.7 %). It looks like a real effect, because the previous decoder stage carries the deep path. Sanity-check it before the write-up.
- **Clean latency (idle A100-SXM4-40GB, GPU 0, nothing else on it; fold 0, 400 test images, batch 1, 768×768, 5 warm-up images excluded):** CAB1 median **40.7 ms** (p90 41.2), CAB2 median **42.1 ms** (p90 42.8). The mean (~160 ms) is inflated by `tf.function` retraces on some images; report median or p90. The S0 CMs again match the reference (rel. diff 8e-6 / 3e-6). Files: `results/ablation_inference/timing_idle/check/`. The `ms/img` values in the screen/main summaries come from GPUs shared with training and stay indicative only.
- **Restarted:** the partial `*_auf_*_20261119-080712` dirs were deleted. The loss ablation was restarted with the §9 command (log `logs/loss_ablation/runner.log`). The inference ablation runs `screen` and then `main` on `DEVICES=0,2 JOBS=2` (log `results/ablation_inference/logs/runner_gpu.log`).
- **Screen rerun done (41 variants, GPU stats).** No variant shows the NaN signature any more. Two near-collapses are real:
  - **S4 / ps−1 edges:** CAB1 S4 equals the single edge `E_stage2_src2_ps-1` (0.0185). That is expected: once the last stage's previous-decoder input is constant, nothing from the earlier decoder stages reaches the output. In CAB2 the extra deeper branches keep some signal (S4 0.008 vs. single edge 0.032).
  - **CAB1 `AC_zero_diagonal`:** mIoU 0 / acc 89.54 %, i.e. all background, but the outputs are finite (checked on one image: 26 kernels zeroed, 0 NaN/Inf). It is a genuine collapse from zeroing half of every fused 3×3 conv. It is a sensitivity probe, not a component ablation; do not report it as "AC removed".

## 13. Loss-ablation evaluation, idle latency for all steps (2026-11-23)

- **All 20 retraining runs finished** (no errors). Evaluation: `kfold_patch/eval_loss_ablation.py`, run alone on idle GPU 2. Output: `results/loss_ablation/*.json` and `SUMMARY.md`. The champion folds 0–4 were re-evaluated in the same job (they match the reference fold CMs), so all three loss variants have latency measured the same way.
- **Early stopping:** both setups use `EarlyStopping(val_loss, patience=30)` without `restore_best_weights`. The champion folds (CAB1 and CAB2, folds 0–4) all trained the full 100 epochs; so did every L2 run. **Every L1 (focal only) run stopped early**: CAB1 at 57–65 epochs, CAB2 at 57–82. The saved weights are the last epoch, 30 epochs after the best val loss. State this in the write-up. It does not explain the result below, because L1 is not worse than the champion despite fewer epochs.
- **Result (pooled over folds 0–4, Δ vs champion L3):**

| Model | Variant | mIoU fg | mIoU all | Δ fg (pp) | fold std | worse in | ms/img idle |
|---|---|---|---|---|---|---|---|
| CAB1 | L1 focal + AWL | 54.79 | 61.73 | +0.70 | 0.73 | 0/5 | 41.7 |
| CAB1 | L2 focal + AAF + AWL | 55.02 | 61.97 | +0.94 | 1.30 | 1/5 | 41.8 |
| CAB1 | L3 champion | 54.09 | 61.17 | – | – | – | 41.8 |
| CAB2 | L1 focal + AWL | 56.53 | 63.21 | +1.65 | 0.66 | 0/5 | 43.1 |
| CAB2 | L2 focal + AAF + AWL | 54.43 | 61.39 | −0.45 | 0.63 | 4/5 | 43.1 |
| CAB2 | L3 champion | 54.88 | 61.77 | – | – | – | 43.2 |

- **Interpretation:** the extra loss terms do not improve test mIoU. For CAB2, focal loss alone is better in 5 of 5 folds (+1.65 pp, larger than the champion's fold std of 0.5). L1 also has a clearly higher mean class accuracy (+3.5 / +6.5 pp), at slightly lower precision on CAB1. Caveat (§9): the champions come from a different training run, so each Δ also contains seed noise. HPO selected the champion configuration on val loss, which, if anything, favours L3. **This contradicts the original paper's loss table**; consider one more seed of L1 vs L3 before claiming it strongly.
- **Latency:** the loss variants do not change inference cost (41.7–41.8 ms CAB1, 43.1–43.2 ms CAB2). Any difference is within measurement noise.
- **Idle latency for every inference-ablation step:** fold 0, 400 images, each model alone on idle GPU 2 (`results/ablation_inference/timing_idle/main/`). All variants are 41.4–42.1 ms (CAB1) and 42.7–43.5 ms (CAB2). The spread is noise: the mean-ablation edits change weights, not the graph, so these variants cost the same as the full model. The idle CMs reproduce the main-phase fold-0 values (S0 rel. diff vs reference 9e-6 / 3e-6).
- **Summaries** now include **mean IoU (all)** (with background, = the `results/official` headline; S0 = 61.37 / 61.99 matches) with its Δ, and both latency columns ("shared" = indicative, "idle" = report this).
