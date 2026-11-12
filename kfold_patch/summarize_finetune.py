"""
Summarize fine-tuning runs from finetune_cross_dataset.py (results/finetune/*.json; R3D, CVC-FP, WAFFLE).

Per model x backbone x dataset: confusion matrices are pooled over splits (CVC/WAFFLE: 5-fold CV,
every plan tested once; R3D: official test split). Reports Rec / F1 / IoU (%) for walls,
openings and their mean: zero-shot, AdaBN (BN statistics recomputed on target train images) and fine-tuned, in the format of
arXiv:2408.01526 Table IV.

Usage:
    python kfold_patch/summarize_finetune.py > results/finetune/SUMMARY.md
    python kfold_patch/summarize_finetune.py results/finetune_unfrozen > results/finetune_unfrozen/SUMMARY.md
"""
import glob
import sys
import json
from collections import defaultdict

import numpy as np


def metrics(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    rec, f1, iou = tp / (tp + fn), 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)
    return rec * 100, f1 * 100, iou * 100


def fmt(cm):
    r, f, i = metrics(cm)
    cells = [f'{r[c]:.1f} / {f[c]:.1f} / {i[c]:.1f}' for c in (1, 2)]
    cells.append(f'{r[1:].mean():.1f} / {f[1:].mean():.1f} / {i[1:].mean():.1f}')
    return cells


def main():
    runs = defaultdict(dict)
    root = sys.argv[1] if len(sys.argv) > 1 else 'results/finetune'
    for p in sorted(glob.glob(f'{root}/*.json')):  # top level only (old runs live in subfolders)
        r = json.load(open(p))
        runs[(r['dataset'], r['model'], r['backbone'])][r['split']] = r  # sorted -> latest per split wins

    names = {'r3d': 'R3D (official test, 53 plans)', 'cvc': 'CVC-FP (5-fold CV, all 122 plans tested once)',
             'waffle': 'WAFFLE benchmark (5-fold CV, all 109 usable plans tested once)'}
    for ds in ['r3d', 'cvc', 'waffle']:
        print(f'\n### {names[ds]}\n')
        print('| Model | Backbone | Setting | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU | Splits | LR | Best epochs |')
        print('|---|---|---|---|---|---|---|---|---|')
        for (d, m, b), splits in sorted(runs.items()):
            if d != ds:
                continue
            zs = sum(np.array(r['cm_zeroshot']) for r in splits.values())
            ft = sum(np.array(r['cm_finetuned']) for r in splits.values())
            ep = ', '.join(str(splits[k]['best_epoch']) for k in sorted(splits))
            lr = ', '.join(sorted({f"{r.get('lr', 1e-4):.0e}" for r in splits.values()}))
            print(f'| {m.upper()} | {b} | zero-shot | ' + ' | '.join(fmt(zs)) + f' | {len(splits)} | – | – |')
            if all('cm_adabn' in r for r in splits.values()):
                ab = sum(np.array(r['cm_adabn']) for r in splits.values())
                print(f'| | | AdaBN | ' + ' | '.join(fmt(ab)) + f' | {len(splits)} | – | – |')
            print(f'| | | **fine-tuned** | ' + ' | '.join(fmt(ft)) + f' | {len(splits)} | {lr} | {ep} |')
            if len(splits) > 1:
                per = np.array([[metrics(np.array(r['cm_finetuned']))[2][1:].mean()] for r in splits.values()])
                print(f'| | | fine-tuned mIoU across splits | | | {per.mean():.1f} ± {per.std():.1f} | | | |')


if __name__ == '__main__':
    main()
