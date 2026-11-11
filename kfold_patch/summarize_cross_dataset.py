"""
Summarize zero-shot cross-dataset results (R3D / CVC-FP / WAFFLE) from evaluate_cross_dataset.py.

For each model x backbone x dataset (latest *_fold_cms.npy):
  - pooled:  metrics from the confusion matrix summed over all folds
  - mean±std: per-fold metrics, mean and std over the 10 folds
  - single:  one checkpoint = fold with the best CubiCasa5k out-of-fold val accuracy
             (from the matching results/official/val file; never selected on R3D/CVC)

Usage:
    python kfold_patch/summarize_cross_dataset.py > results/cross_dataset/SUMMARY.md
"""
import glob
import os
import re

import numpy as np

RES = 'results/cross_dataset'
DATASETS = ['r3d_test', 'r3d_all', 'cvc_all', 'waffle_all']


def metrics(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    nb = cm[1:]
    return {'acc': tp.sum() / cm.sum(),
            'acc_nobg': np.diag(cm)[1:].sum() / max(nb.sum(), 1),
            'walls': iou[1], 'openings': iou[2],
            'rec_walls': tp[1] / max(cm[1].sum(), 1), 'rec_open': tp[2] / max(cm[2].sum(), 1),
            'miou_nobg': iou[1:].mean()}


def best_val_fold(model, txt_path):
    """Fold index with the highest CubiCasa val accuracy for the same checkpoint set."""
    text = open(txt_path).read()
    src = re.search(r'Checkpoint source: (\S+)', text).group(1)
    fold0 = re.search(r'Fold 0: (\S+)', open(src).read()).group(1)
    for v in glob.glob(f'results/official/val/val_kfold_{model}_*.txt'):
        vt = open(v).read()
        if fold0 in vt:
            accs = [float(a) for a in re.findall(r"'([0-9.]+)'", re.search(r'Per-fold accuracy: (.*)', vt).group(1))]
            return int(np.argmax(accs)), accs[int(np.argmax(accs))]
    return None, None


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, '*_fold_cms.npy'))):
        m = re.match(r'(.+?)_kfold_([a-z0-9]+)_(\S+?)_(\d{8}-\d{6})_fold_cms\.npy', os.path.basename(f))
        ds, model, bb, ts = m.groups()
        runs[(model, bb, ds)] = f  # sorted -> latest wins

    keys = ['acc', 'acc_nobg', 'walls', 'openings', 'miou_nobg', 'rec_walls', 'rec_open']
    hdr = '| Model | Backbone | Mode | Acc | Acc no-bg | Walls IoU | Openings IoU | mIoU (no bg) | Walls class acc | Openings class acc |'
    sep = '|' + '---|' * 10
    for ds in DATASETS:
        print(f'\n### {ds}\n\n{hdr}\n{sep}')
        for (model, bb, d), f in sorted(runs.items()):
            if d != ds:
                continue
            cms = np.load(f)
            pooled = metrics(cms.sum(0))
            per = np.array([[metrics(c)[k] for k in keys] for c in cms]) * 100
            fold, vacc = best_val_fold(model, f[:-len('_fold_cms.npy')] + '.txt')
            print(f'| {model.upper()} | {bb} | pooled (10 folds) | ' +
                  ' | '.join(f'{pooled[k] * 100:.2f}' for k in keys) + ' |')
            print(f'| | | mean ± std | ' +
                  ' | '.join(f'{mu:.2f} ± {sd:.2f}' for mu, sd in zip(per.mean(0), per.std(0))) + ' |')
            if fold is not None:
                s = metrics(cms[fold])
                print(f'| | | single: fold {fold} (val {vacc * 100:.2f}%) | ' +
                      ' | '.join(f'{s[k] * 100:.2f}' for k in keys) + ' |')


if __name__ == '__main__':
    main()
