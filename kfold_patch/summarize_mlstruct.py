"""
Summarise the MLSTRUCT-FP runs into results/mlstruct/SUMMARY.md (also printed).

Inputs: results/mlstruct/zeroshot/*.npz (evaluate_mlstruct.py) and results/mlstruct/finetune/*.json
(finetune_cross_dataset.py --dataset mlstruct). Latest file wins per (scale, model, backbone, checkpoint set).
All numbers are bg / wall at native resolution; "+/-" = std over the 5 CV test folds.
"""
import glob
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

os.chdir(_ROOT)
import mlstruct_data as D  # noqa: E402
from tiled_inference import wall_metrics  # noqa: E402

COLS = [('wall_rec', 'Wall Rec'), ('wall_prec', 'Wall Prec'), ('wall_f1', 'Wall F1'), ('wall_iou', 'Wall IoU'),
        ('miou', 'mIoU (bg+wall)'), ('bg_acc', 'bg acc'), ('overall_acc', 'Overall acc')]
ORDER = [('cab1', 'EfficientNetB4'), ('cab1', 'EfficientNetV2M'), ('cab2', 'EfficientNetB4'), ('cab2', 'EfficientNetV2M')]
SHORT = {'EfficientNetB4': 'B4', 'EfficientNetV2M': 'V2M'}
LIT = [  # (method, {metric: value}); MLSTRUCT-FP, bg / wall; '-' where not reported
    ('Paek et al. 2025, SegFormer-B5 patch-wise (Table 3)', {'wall_rec': .85, 'miou': .92, 'bg_acc': .99, 'overall_acc': .94}),
    ('R2V, retrained by Paek et al.', {'miou': .84}),
    ('DeepFP, retrained by Paek et al.', {'miou': .84}),
    ('Pizarro et al. 2023 U-Net (via Paek et al.)', {'miou': .77}),
    ('Kratochvila et al. v1 CAB1 B2 (Table 9, 60/20/20)', {'wall_rec': .92, 'wall_f1': .89, 'wall_iou': .81}),
    ('Kratochvila et al. v1 CAB2 B2 (Table 9, 60/20/20)', {'wall_rec': .92, 'wall_f1': .90, 'wall_iou': .83}),
]


def latest(files, key):
    out = {}
    for f in sorted(files):
        out[key(f)] = f
    return out


def cell(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return '-'
    return f'{100 * np.mean(vals):.1f} ± {100 * np.std(vals):.1f}' if len(vals) > 1 else f'{100 * vals[0]:.1f}'


def row(name, per_fold, pooled=None):
    cells = [cell([m[k] for m in per_fold]) for k, _ in COLS]
    return f'| {name} | ' + ' | '.join(cells) + ' |' + (f' {100 * pooled["wall_iou"]:.1f} |' if pooled else ' - |')


HEADER = '| Model | ' + ' | '.join(c for _, c in COLS) + ' | Wall IoU pooled |\n' + '|---' * (len(COLS) + 2) + '|'


def zeroshot():
    pat = re.compile(r'(\w+?)_(cab\d)_(\w+?)_(allfolds|single)_\d')
    files = latest(glob.glob('results/mlstruct/zeroshot/*.npz'),
                   lambda f: pat.match(os.path.basename(f)).groups() if '_n' not in os.path.basename(f) else None)
    files.pop(None, None)
    folds = D.folds()
    res = {}
    for k, f in files.items():
        z = np.load(f)
        pos = {i: j for j, i in enumerate(z['ids'])}
        per_fold = [wall_metrics(z['cms'][:, [pos[i] for i in fo]].sum((0, 1))) for fo in folds]
        res[k] = (per_fold, wall_metrics(z['cms'].sum((0, 1))), z['cms'].shape[0])
    return res


def finetune():
    runs = {}
    for f in sorted(glob.glob('results/mlstruct/finetune/*.json')):
        r = json.load(open(f))
        runs[(r['scale'], r['model'], r['backbone'], r['split'])] = r
    return runs


def main():
    zs, ft = zeroshot(), finetune()
    out = ['# MLSTRUCT-FP results (bg / wall)', '',
           'Protocol: 5-fold CV over 934 plan images (seed 0; split k: test = fold k, val = fold k+1), '
           'tiled inference (768 px tiles, 25% overlap, Hann-weighted softmax averaging), predictions upsampled '
           'and scored at **native** plan resolution. `s40` = plans resampled to 40 px/m (0.2 m walls ≈ 8 px, '
           'matching CubiCasa training); `native` = original pixels (as Paek et al.). CubiCasa walls → wall, '
           'every other class → bg. Values in %, mean ± std over the 5 CV test folds.', '']

    out += ['## 1. Zero-shot, all 10 CubiCasa fold models (s40)', '',
            'Per CV fold, the confusion matrices of the 10 fold models are pooled.', '', HEADER]
    for m, b in ORDER:
        k = ('s40', m, b, 'allfolds')
        if k in zs:
            out.append(row(f'{m.upper()} {SHORT[b]}', zs[k][0], zs[k][1]))
    out += ['', '## 2. Zero-shot scale sensitivity (fine-tune checkpoint, best-val CubiCasa fold)', '', HEADER]
    for s in ['s30', 's40', 's50', 's60', 'native']:
        for m, b in ORDER:
            k = (s, m, b, 'single')
            if k in zs:
                out.append(row(f'{m.upper()} {SHORT[b]} @ {s}', zs[k][0], zs[k][1]))

    for s in ['s40', 'native']:
        out += ['', f'## 3{"a" if s == "s40" else "b"}. Fine-tuned, backbone unfrozen ({s})', '',
                'Same checkpoint as §2; LR 3e-5, ≤300 epochs × 500 steps of random 512 crops, early stopping on '
                'val loss (patience 25). Zero-shot row = the same checkpoint on the same test folds.', '', HEADER]
        for m, b in ORDER:
            rs = [ft[(s, m, b, k)] for k in range(D.N_FOLDS) if (s, m, b, k) in ft]
            if not rs:
                continue
            n = f' ({len(rs)}/5 splits)' if len(rs) < D.N_FOLDS else ''
            for phase in ['zeroshot', 'adabn', 'finetuned']:
                cms = [np.array(r[f'cm_{phase}']) for r in rs]
                if phase == 'adabn' and not np.sum(cms):
                    continue
                out.append(row(f'{m.upper()} {SHORT[b]} {phase}{n}', [wall_metrics(c) for c in cms],
                               wall_metrics(np.sum(cms, 0))))

    out += ['', '## 4. Comparison with published MLSTRUCT-FP results', '',
            '| Method | ' + ' | '.join(c for _, c in COLS) + ' |', '|---' * (len(COLS) + 1) + '|']
    for name, d in LIT:
        out.append(f'| {name} | ' + ' | '.join(f'{100 * d[k]:.0f}' if k in d else '-' for k, _ in COLS) + ' |')
    best = []
    for s in ['s40', 'native']:
        for m, b in ORDER:
            rs = [ft[(s, m, b, k)] for k in range(D.N_FOLDS) if (s, m, b, k) in ft]
            if len(rs) == D.N_FOLDS:
                pf = [wall_metrics(np.array(r['cm_finetuned'])) for r in rs]
                best.append(f'| **Ours {m.upper()} {SHORT[b]}, fine-tuned ({s}, 5-fold CV)** | '
                            + ' | '.join(cell([p[k] for p in pf]) for k, _ in COLS) + ' |')
    out += best
    out += ['', 'Notes: Paek et al. report a single run on an "original split" that MLSTRUCT-FP does not define; '
            'their per-class accuracy is recall and their mIoU includes background. v1 Table 9 used a random '
            '60/20/20 split and an undocumented tiling. `fp.json` has no building ID, so floors of one building '
            'can fall into different folds (same limitation as both references). MLSTRUCT-FP labels structural '
            'walls only, so thin partition walls that CubiCasa models segment count as false positives zero-shot. '
            'All numbers use plain argmax (p_wall > 0.5). A validation-tuned cutoff (kfold_patch/tune_threshold.py; '
            'tested on CAB1 V2M s40 split 0 only) picks p_wall > 0.73. On test it gives wall IoU 79.0 -> 80.8, mIoU 89.2 -> 90.1, '
            'precision 82.0 -> 86.1 and recall 95.6 -> 92.9. On R3D / CVC-FP / WAFFLE the same tuning changes mIoU by only '
            '-0.6 to +0.6, so it is not applied anywhere (results/threshold_tuning/SUMMARY.md).']
    text = '\n'.join(out) + '\n'
    os.makedirs('results/mlstruct', exist_ok=True)
    open('results/mlstruct/SUMMARY.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
