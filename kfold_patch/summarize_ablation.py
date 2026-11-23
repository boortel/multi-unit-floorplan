"""
Summarize results/ablation_inference/<phase>/*.json into results/ablation_inference/SUMMARY_<phase>.md.

Metrics come from the confusion matrix pooled over all evaluated folds (as in evaluate_kfold.py):
pixel accuracy, mean class accuracy (= mean recall over the 5 foreground classes), mean IoU over the
5 foreground classes, per-class IoU. Δ values are against S0_full of the same model. With several
folds, the fold std of the per-fold Δ mIoU and the number of folds where the variant is worse are given.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

CLASSES = ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs']
OUT_DIR = 'results/ablation_inference'

DESCRIPTIONS = {
    'S0_full': 'unedited model (reference)',
    'S1_unet_only': 'keep only same-level skip + previous decoder (plain U-Net wiring)',
    'S2_no_shallower_encoder': 'remove inputs from shallower encoder levels',
    'S3_no_same_level_skip': 'remove the same-level encoder skip',
    'S4_no_previous_decoder': 'remove the input from the previous (deeper) decoder stage',
    'S5_cab1_wiring': 'CAB2 reduced to CAB1 wiring (remove extra deeper + far shallower inputs)',
    'S6_no_extra_deeper': 'remove deeper inputs beyond the previous decoder stage',
    'S7_no_far_shallower': 'remove shallower encoder inputs beyond one level',
    'A1_cam_mean': 'channel attention gate -> its mean',
    'A2_sam_mean': 'spatial attention gates -> their mean',
    'A3_cam_sam_mean': 'both attention gates -> their means (no attention)',
    'A4_sam_single_dilation': 'spatial attention: dilation 2 and 3 terms -> their mean (single dilation)',
    'AC_zero_h_v': 'AC block: horizontal + vertical kernels zeroed (sensitivity)',
    'AC_zero_diagonal': 'AC block: diagonal kernels zeroed (sensitivity)',
    'AC_zero_square': 'AC block: square kernels zeroed (sensitivity)',
    'X_manual_sam_identity': 'check: SAM5 replaced by the manual re-implementation, nothing ablated',
}


def metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return dict(acc=tp.sum() / cm.sum(), mca=rec[1:].mean(), miou=iou[1:].mean(), miou_all=iou.mean(), iou=iou)


def load(phase):
    rows = defaultdict(dict)  # (model, variant) -> {fold: record}
    for p in glob.glob(os.path.join(OUT_DIR, phase, '*.json')):
        r = json.load(open(p))
        rows[(r['model'], r['variant'])][r['fold']] = r
    return rows


def idle_timing():
    """(model, variant) -> median ms/img measured alone on an idle GPU (timing_idle/{main,check}, fold 0, 400 images)."""
    out = {}
    for p in glob.glob(os.path.join(OUT_DIR, 'timing_idle', '*', '*.json')):
        r = json.load(open(p))
        if (r.get('timing') or {}).get('ms_median') and r.get('n_test_images') == 400:
            out[(r['model'], r['variant'])] = r['timing']['ms_median']
    return out


def table(rows, model, variants, idle=None):
    idle = idle or {}
    ref = rows.get((model, 'S0_full'), {})
    lines = ['| Variant | Folds | Pixel acc | Mean class acc | Mean IoU (fg) | Mean IoU (all) | Δ mIoU fg (pp) | '
             'Δ mIoU all (pp) | Δ fold std | worse in | ms/img (shared) | ms/img (idle) | '
             + ' | '.join(f'{c} IoU' for c in CLASSES[1:]) + ' |',
             '|' + '---|' * (12 + len(CLASSES) - 1)]
    for v in variants:
        recs = rows.get((model, v))
        if not recs:
            continue
        folds = sorted(recs)
        cm = sum(np.asarray(recs[f]['cm']) for f in folds)
        m = metrics(cm)
        d_mean = d_all = d_std = worse = ''
        common = [f for f in folds if f in ref]
        if v != 'S0_full' and common:
            ref_cm = sum(np.asarray(ref[f]['cm']) for f in common)
            cm_c = sum(np.asarray(recs[f]['cm']) for f in common)
            d_mean = f'{100 * (metrics(cm_c)["miou"] - metrics(ref_cm)["miou"]):+.2f}'
            d_all = f'{100 * (metrics(cm_c)["miou_all"] - metrics(ref_cm)["miou_all"]):+.2f}'
            per = [metrics(recs[f]['cm'])['miou'] - metrics(ref[f]['cm'])['miou'] for f in common]
            d_std = f'{100 * np.std(per):.2f}' if len(per) > 1 else '–'
            worse = f'{sum(x < 0 for x in per)}/{len(per)}'
        ms = [recs[f]['timing']['ms_median'] for f in folds if (recs[f].get('timing') or {}).get('ms_median')]
        ms = f'{np.median(ms):.1f}' if ms else '–'
        ms_idle = f'{idle[(model, v)]:.1f}' if (model, v) in idle else '–'
        lines.append(f'| `{v}` | {len(folds)} | {100 * m["acc"]:.2f} | {100 * m["mca"]:.2f} | {100 * m["miou"]:.2f} | '
                     f'{100 * m["miou_all"]:.2f} | {d_mean} | {d_all} | {d_std} | {worse} | {ms} | {ms_idle} | ' + ' | '.join(f'{100 * x:.1f}' for x in m['iou'][1:]) + ' |')
    return '\n'.join(lines)


def edge_matrix(rows, model):
    ref = rows.get((model, 'S0_full'), {})
    edges = sorted(v for (m, v) in rows if m == model and v.startswith('E_stage'))
    if not edges or not ref:
        return ''
    lines = ['| Decoder stage \\ source level | ' + ' | '.join(f'src {l}' for l in range(4)) + ' |',
             '|---|' + '---|' * 4]
    grid = {}
    for v in edges:
        parts = v.split('_')  # E stageI srcL psX
        i, lev = int(parts[1][5:]), int(parts[2][3:])
        recs = rows[(model, v)]
        common = [f for f in recs if f in ref]
        cm = sum(np.asarray(recs[f]['cm']) for f in common)
        rcm = sum(np.asarray(ref[f]['cm']) for f in common)
        grid[(i, lev)] = (100 * (metrics(cm)['miou'] - metrics(rcm)['miou']), parts[3][2:])
    for i in range(3):
        cells = []
        for lev in range(4):
            if (i, lev) in grid:
                d, ps = grid[(i, lev)]
                cells.append(f'{d:+.2f} (ps {ps})')
            else:
                cells.append('–')
        lines.append(f'| stage {i} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='screen')
    args = ap.parse_args()
    rows = load(args.phase)
    out = [f'# Inference-time ablation: phase `{args.phase}`', '',
           'Generated by `kfold_patch/summarize_ablation.py` from `results/ablation_inference/'
           f'{args.phase}/*.json`. Plan and method: `docs/ABLATION_PLAN.md`.', '',
           'Mean class acc and mean IoU (fg) are over the 5 foreground classes; mean IoU (all) also includes '
           'background (the headline metric of `results/official`). Δ is in percentage points vs `S0_full` on '
           'the same folds/images; Δ fold std and "worse in" refer to mIoU (fg). Latency = per-image forward '
           'pass + argmax (768×768, batch 1, 5 warm-up images excluded, synced via .numpy()), median. '
           '"ms/img (shared)": median over folds, measured while the GPU was shared with training jobs, '
           'indicative only. "ms/img (idle)": fold 0, 400 test images, the job alone on an idle A100 '
           '(`timing_idle/`); use this one. Mean-ablation edits change weights, not the graph, so latency '
           'differences between S*/A1–A3 variants are noise; only A4 (replacement SAM layer) changes the graph.', '']
    for model in ('cab1', 'cab2'):
        variants = sorted({v for (m, v) in rows if m == model and not v.startswith('E_stage')},
                          key=lambda v: (v != 'S0_full', v))
        if not variants:
            continue
        out += [f'## {model.upper()}', '', table(rows, model, variants, idle_timing()), '']
        em = edge_matrix(rows, model)
        if em:
            out += [f'### {model.upper()}: single skip-branch ablation (Δ mIoU, pp)', '',
                    'ps = pool scale: −1 previous decoder stage, <−1 deeper decoder/bottleneck, 0 same-level '
                    'encoder skip, >0 shallower encoder.', '', em, '']
        out += ['Variants:', ''] + [f'- `{v}`: {DESCRIPTIONS.get(v, "")}' for v in variants
                                    if v in DESCRIPTIONS] + ['']
    path = os.path.join(OUT_DIR, f'SUMMARY_{args.phase}.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out))
    print('\n'.join(out))
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
