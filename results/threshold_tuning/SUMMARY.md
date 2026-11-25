# Validation-tuned decision thresholds (fine-tuned models)

Decision rule `argmax_c(log p_c + b_c)`, `b_bg = 0`: one log-odds offset per foreground class, chosen on each run's **validation** split (the split used for early stopping) to maximise mean IoU over all classes, grid [-3, 3] step 0.1, then applied once to the test split. `b = 0` is the plain argmax of the reported results (reproduced to < 0.01% of pixels). Inference only, no retraining. Values in %, mean ± std over CV test splits (R3D: one run, official test split); Δ = tuned − argmax, (n/N↑) = splits that improved. **Stopped 2026-11-25 after 72/128 runs:** R3D and CVC-FP complete, WAFFLE unfrozen partial, MLSTRUCT-FP one split only. Only MLSTRUCT-FP showed a real gain, so the tuned numbers are noted but not used in the result tables. Note: mIoU here averages all classes including bg, while COMPARISON_paper_sota.md averages walls and openings for R3D / CVC-FP / WAFFLE.

## R3D, decoder-only fine-tuning (frozen backbone)

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 1 | 87.0 | 87.1 | +0.2 | 93.4 | 93.5 | +0.2 | 68.5 | 68.8 | +0.3 | -0.6, -0.8 |
| CAB1 V2M | 1 | 86.9 | 87.0 | +0.1 | 92.7 | 92.9 | +0.2 | 68.8 | 69.0 | +0.2 | -1.5, -1.5 |
| CAB2 B4 | 1 | 86.3 | 86.5 | +0.1 | 91.2 | 91.3 | +0.1 | 68.9 | 69.1 | +0.2 | -0.6, -0.5 |
| CAB2 V2M | 1 | 86.4 | 86.5 | +0.1 | 91.9 | 92.1 | +0.1 | 68.4 | 68.4 | +0.0 | -0.5, -0.4 |

## R3D, backbone unfrozen

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 1 | 88.0 | 88.1 | +0.1 | 93.7 | 93.9 | +0.2 | 71.2 | 71.2 | -0.1 | -1.0, -1.4 |
| CAB1 V2M | 1 | 87.8 | 87.9 | +0.1 | 93.1 | 93.1 | +0.1 | 71.2 | 71.5 | +0.2 | -1.1, -1.8 |
| CAB2 B4 | 1 | 87.1 | 87.1 | +0.0 | 91.7 | 91.8 | +0.0 | 70.6 | 70.6 | +0.0 | -0.6, -0.6 |
| CAB2 V2M | 1 | 88.5 | 88.5 | -0.0 | 93.3 | 93.3 | +0.0 | 73.1 | 73.0 | -0.1 | -0.2, -1.6 |

## CVC-FP, decoder-only fine-tuning (frozen backbone)

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 5 | 82.0 ± 1.3 | 82.5 ± 1.4 | +0.5 (5/5↑) | 80.9 ± 1.7 | 81.4 ± 1.8 | +0.5 (4/5↑) | 66.5 ± 3.4 | 67.4 ± 3.7 | +0.9 (5/5↑) | -1.6, -1.4 |
| CAB1 V2M | 5 | 82.1 ± 1.1 | 82.5 ± 1.2 | +0.3 (4/5↑) | 80.3 ± 1.6 | 80.8 ± 1.9 | +0.5 (4/5↑) | 67.4 ± 2.8 | 67.9 ± 2.6 | +0.5 (4/5↑) | -1.5, -1.4 |
| CAB2 B4 | 5 | 75.8 ± 3.0 | 76.1 ± 2.8 | +0.3 (5/5↑) | 74.0 ± 3.3 | 74.7 ± 3.2 | +0.7 (4/5↑) | 55.3 ± 6.2 | 55.5 ± 6.0 | +0.2 (4/5↑) | -1.1, -0.6 |
| CAB2 V2M | 5 | 80.9 ± 2.0 | 81.5 ± 1.8 | +0.6 (5/5↑) | 79.2 ± 1.9 | 80.2 ± 1.8 | +1.0 (5/5↑) | 65.1 ± 4.6 | 65.7 ± 4.4 | +0.6 (5/5↑) | -1.9, -1.2 |

## CVC-FP, backbone unfrozen

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 5 | 84.1 ± 1.3 | 84.5 ± 1.4 | +0.4 (4/5↑) | 81.9 ± 1.7 | 82.4 ± 2.2 | +0.4 (3/5↑) | 71.6 ± 2.8 | 72.2 ± 2.8 | +0.6 (5/5↑) | -1.5, -1.5 |
| CAB1 V2M | 5 | 86.2 ± 1.2 | 86.5 ± 1.2 | +0.2 (4/5↑) | 82.7 ± 1.6 | 83.2 ± 1.8 | +0.5 (4/5↑) | 77.1 ± 2.7 | 77.2 ± 2.6 | +0.1 (2/5↑) | -1.8, -1.4 |
| CAB2 B4 | 5 | 77.0 ± 2.4 | 77.6 ± 2.4 | +0.6 (4/5↑) | 74.3 ± 3.3 | 75.4 ± 3.4 | +1.1 (4/5↑) | 58.4 ± 4.4 | 59.1 ± 4.8 | +0.6 (3/5↑) | -1.6, -1.2 |
| CAB2 V2M | 5 | 84.4 ± 0.9 | 84.9 ± 1.2 | +0.5 (5/5↑) | 81.8 ± 1.6 | 82.5 ± 2.0 | +0.7 (4/5↑) | 72.4 ± 1.9 | 73.2 ± 2.3 | +0.7 (5/5↑) | -2.1, -1.9 |

## WAFFLE, decoder-only fine-tuning (frozen backbone)

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 5 | 68.4 ± 1.9 | 68.6 ± 2.0 | +0.2 (3/5↑) | 74.3 ± 5.1 | 74.2 ± 4.8 | -0.0 (2/5↑) | 35.6 ± 3.9 | 36.1 ± 5.1 | +0.5 (4/5↑) | -1.1, -1.2 |
| CAB1 V2M | 5 | 69.4 ± 1.0 | 69.2 ± 0.8 | -0.3 (2/5↑) | 74.3 ± 2.9 | 73.9 ± 3.3 | -0.4 (2/5↑) | 38.5 ± 2.6 | 38.1 ± 2.3 | -0.4 (3/5↑) | -0.9, -1.4 |
| CAB2 B4 | 5 | 70.1 ± 2.0 | 69.9 ± 2.5 | -0.2 (2/5↑) | 76.4 ± 2.7 | 76.0 ± 3.8 | -0.4 (2/5↑) | 38.1 ± 4.3 | 38.1 ± 4.5 | -0.1 (3/5↑) | -0.5, -0.6 |
| CAB2 V2M | 5 | 69.6 ± 1.9 | 69.8 ± 1.3 | +0.2 (2/5↑) | 75.8 ± 2.3 | 75.4 ± 2.2 | -0.4 (1/5↑) | 37.5 ± 3.4 | 38.4 ± 3.1 | +0.9 (4/5↑) | -1.0, -1.0 |

## WAFFLE, backbone unfrozen

| Model | n | mIoU argmax | mIoU tuned | Δ | Walls IoU argmax | Walls IoU tuned | Δ | Openings IoU argmax | Openings IoU tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 B4 | 4 | 69.6 ± 1.7 | 69.0 ± 1.7 | -0.6 (1/4↑) | 73.7 ± 4.8 | 72.5 ± 4.2 | -1.3 (0/4↑) | 39.7 ± 3.6 | 39.4 ± 4.7 | -0.3 (1/4↑) | -0.2, -0.0 |

## MLSTRUCT-FP s40, backbone unfrozen

| Model | n | Wall IoU argmax | Wall IoU tuned | Δ | mIoU (bg+wall) argmax | mIoU (bg+wall) tuned | Δ | Wall Prec argmax | Wall Prec tuned | Δ | Wall Rec argmax | Wall Rec tuned | Δ | offsets (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAB1 V2M | 1 | 79.0 | 80.8 | +1.8 | 89.2 | 90.1 | +0.9 | 82.0 | 86.1 | +4.1 | 95.6 | 92.9 | -2.7 | -1.0 (p_wall > 0.73) |
