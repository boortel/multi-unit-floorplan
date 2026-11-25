"""
Summarise results/threshold_tuning/*/*.json (tune_threshold.py) into results/threshold_tuning/SUMMARY.md.

Per dataset / backbone mode / model: test metrics with argmax (= the reported fine-tuned numbers) and with
the validation-tuned offsets, mean +/- std over CV splits (R3D: one run on the official test split).
"""
import glob
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
OUT = 'results/threshold_tuning'
ORDER = [('cab1', 'EfficientNetB4'), ('cab1', 'EfficientNetV2M'), ('cab2', 'EfficientNetB4'), ('cab2', 'EfficientNetV2M')]
SHORT = {'EfficientNetB4': 'B4', 'EfficientNetV2M': 'V2M'}
GROUPS = [  # (title, sub-directory, dataset, scale)
    ('R3D, decoder-only fine-tuning (frozen backbone)', 'finetune', 'r3d', None),
    ('R3D, backbone unfrozen', 'finetune_unfrozen', 'r3d', None),
    ('CVC-FP, decoder-only fine-tuning (frozen backbone)', 'finetune', 'cvc', None),
    ('CVC-FP, backbone unfrozen', 'finetune_unfrozen', 'cvc', None),
    ('WAFFLE, decoder-only fine-tuning (frozen backbone)', 'finetune', 'waffle', None),
    ('WAFFLE, backbone unfrozen', 'finetune_unfrozen', 'waffle', None),
    ('MLSTRUCT-FP s40, backbone unfrozen', 'mlstruct', 'mlstruct', 's40'),
    ('MLSTRUCT-FP native, backbone unfrozen', 'mlstruct', 'mlstruct', 'native'),
]


def per_class(cm):
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return {'iou': tp / np.maximum(tp + fp + fn, 1), 'rec': tp / np.maximum(tp + fn, 1),
            'prec': tp / np.maximum(tp + fp, 1)}


def cols(k):
    if k == 3:
        return [('mIoU', lambda m: m['iou'].mean()), ('Walls IoU', lambda m: m['iou'][1]),
                ('Openings IoU', lambda m: m['iou'][2])]
    return [('Wall IoU', lambda m: m['iou'][1]), ('mIoU (bg+wall)', lambda m: m['iou'].mean()),
            ('Wall Prec', lambda m: m['prec'][1]), ('Wall Rec', lambda m: m['rec'][1])]


def ms(v):
    v = 100 * np.asarray(v)
    return f'{v.mean():.1f} ± {v.std():.1f}' if len(v) > 1 else f'{v[0]:.1f}'


def dlt(a, b):
    d = 100 * (np.asarray(b) - np.asarray(a))
    return f'{d.mean():+.1f}' + (f' ({(d > 0).sum()}/{len(d)}↑)' if len(d) > 1 else '')


def main():
    runs = [json.load(open(f)) | {'_sub': os.path.basename(os.path.dirname(f))}
            for f in sorted(glob.glob(f'{OUT}/*/*.json'))]
    lines = ['# Validation-tuned decision thresholds (fine-tuned models)', '',
             'Decision rule `argmax_c(log p_c + b_c)`, `b_bg = 0`: one log-odds offset per foreground class, chosen on '
             "each run's **validation** split (the split used for early stopping) to maximise mean IoU over all "
             'classes, grid [-3, 3] step 0.1, then applied once to the test split. `b = 0` is the plain argmax of the '
             'reported results (reproduced to < 0.01% of pixels). Inference only, no retraining. Values in %, '
             'mean ± std over CV test splits (R3D: one run, official test split); Δ = tuned − argmax, (n/N↑) = '
             'splits that improved. **Stopped 2026-11-25 after 72/128 runs:** R3D and CVC-FP complete, WAFFLE unfrozen partial, '
             'MLSTRUCT-FP one split only. Only MLSTRUCT-FP showed a real gain, so the tuned numbers are noted but not used '
             'in the result tables. Note: mIoU here averages all classes including bg, while COMPARISON_paper_sota.md '
             'averages walls and openings for R3D / CVC-FP / WAFFLE.', '']
    for title, sub, ds, scale in GROUPS:
        sel = [r for r in runs if r['_sub'] == sub and r['dataset'] == ds and r.get('scale') == scale]
        if not sel:
            continue
        k = len(sel[0]['classes'])
        cs = cols(k)
        lines += [f'## {title}', '', '| Model | n | ' + ' | '.join(f'{c} argmax | {c} tuned | Δ' for c, _ in cs)
                  + ' | offsets (mean) |', '|---' * (2 + 3 * len(cs) + 1) + '|']
        for m, b in ORDER:
            rs = sorted([r for r in sel if r['model'] == m and r['backbone'] == b], key=lambda r: r['split'])
            if not rs:
                continue
            a = [per_class(r['cm_argmax']) for r in rs]
            t = [per_class(r['cm_tuned']) for r in rs]
            cells = []
            for _, f in cs:
                va, vt = [f(x) for x in a], [f(x) for x in t]
                cells += [ms(va), ms(vt), dlt(va, vt)]
            off = np.mean([r['offsets'] for r in rs], 0)
            off_s = ', '.join(f'{o:+.1f}' for o in off)
            if k == 2:
                off_s += f' (p_wall > {np.mean([r["p_wall_threshold"] for r in rs]):.2f})'
            lines.append(f'| {m.upper()} {SHORT[b]} | {len(rs)} | ' + ' | '.join(cells) + f' | {off_s} |')
        lines.append('')
    bad = [r['source'] for r in runs if r['repro_diff_frac'] > 1e-4]
    if bad:
        lines += ['**Warning:** b = 0 did not reproduce the stored confusion matrix for: ' + ', '.join(bad), '']
    text = '\n'.join(lines)
    open(f'{OUT}/SUMMARY.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
