"""
One comparable view of CAB1/CAB2 (B4, V2M) and the VGG16 baselines over all four datasets.

Every number uses the same 3-class scheme and the same metric code:
    bg (incl. railings, stairs, WAFFLE interior) / walls / openings (doors + windows)
    Rec / F1 / IoU per class, "mean" = average over walls and openings, pixels pooled.

Sources (latest file wins):
    CubiCasa5k test (in-domain, 400 plans, 10 fold models pooled):
        results/official/test/<result>_fold_cms.npy  (6-class, from export_cubicasa_cms.py)
    Zero-shot (10 fold models pooled):
        results/cross_dataset/{r3d_test,cvc_all,waffle_all}_kfold_<model>_<backbone>_<ts>_fold_cms.npy
    Fine-tuned (one CubiCasa fold model per combination, test CMs pooled over splits):
        results/finetune/{r3d,cvc,waffle}_<model>_<backbone>_split<k>_<ts>.json

Usage:
    python kfold_patch/summarize_all_datasets.py > results/ALL_DATASETS_tables.md
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

MODELS = [('cab1', 'EfficientNetB4'), ('cab1', 'EfficientNetV2M'), ('cab2', 'EfficientNetB4'),
          ('cab2', 'EfficientNetV2M'), ('cubicasa5k', 'VGG16'), ('zeng', 'VGG16')]
NAMES = {'cab1': 'CAB1', 'cab2': 'CAB2', 'cubicasa5k': 'CubiCasa5k', 'zeng': 'Zeng (DeepFloorplan)'}
BB = {'EfficientNetB4': 'B4', 'EfficientNetV2M': 'V2M', 'VGG16': 'VGG16'}
CUBI_SOURCES = {
    ('cab1', 'EfficientNetB4'): 'test_kfold_cab1_EfficientNetB4_20261014-140050',
    ('cab1', 'EfficientNetV2M'): 'test_kfold_cab1_EfficientNetV2M_20261014-122132',
    ('cab2', 'EfficientNetB4'): 'test_kfold_cab2_EfficientNetB4_20260908-075647',
    ('cab2', 'EfficientNetV2M'): 'test_kfold_cab2_EfficientNetV2M_20261014-120852',
    ('cubicasa5k', 'VGG16'): 'test_kfold_cubicasa5k_VGG16_20260908-072527',
    ('zeng', 'VGG16'): 'test_kfold_zeng_VGG16_20260908-071630',
}
TO3 = np.array([0, 1, 0, 2, 2, 0])  # CubiCasa 6 classes -> bg / walls / openings
COLS = [('cubi', 'CubiCasa5k test (in-domain)'), ('r3d_zs', 'R3D test zero-shot'), ('cvc_zs', 'CVC-FP zero-shot'),
        ('waffle_zs', 'WAFFLE zero-shot'), ('r3d_ft', 'R3D fine-tuned'), ('cvc_ft', 'CVC-FP fine-tuned'),
        ('waffle_ft', 'WAFFLE fine-tuned')]


def merge6(cm):
    out = np.zeros((3, 3), np.int64)
    for i in range(6):
        for j in range(6):
            out[TO3[i], TO3[j]] += cm[i, j]
    return out


def metrics(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    rec, f1, iou = tp / np.maximum(tp + fn, 1), 2 * tp / np.maximum(2 * tp + fp + fn, 1), tp / np.maximum(tp + fp + fn, 1)
    r = {}
    for k, v in (('rec', rec), ('f1', f1), ('iou', iou)):
        r[f'walls_{k}'], r[f'open_{k}'], r[f'mean_{k}'] = v[1] * 100, v[2] * 100, v[1:].mean() * 100
    return r


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def collect():
    """{(model, backbone): {col: (pooled_cm, [per-fold/split cms])}}"""
    data = defaultdict(dict)
    for key, src in CUBI_SOURCES.items():
        f = f'results/official/test/{src}_fold_cms.npy'
        if glob.glob(f):
            cms = [merge6(c) for c in np.load(f)]
            data[key]['cubi'] = (sum(cms), cms)
    for m, b in MODELS:
        for ds, col in (('r3d_test', 'r3d_zs'), ('cvc_all', 'cvc_zs'), ('waffle_all', 'waffle_zs')):
            f = latest(f'results/cross_dataset/{ds}_kfold_{m}_{b}_*_fold_cms.npy')
            if f:
                cms = list(np.load(f))
                data[(m, b)][col] = (sum(cms), cms)
    ft = defaultdict(dict)
    for p in sorted(glob.glob('results/finetune/*.json')):
        r = json.load(open(p))
        ft[(r['model'], r['backbone'], r['dataset'])][r['split']] = np.array(r['cm_finetuned'])
    for (m, b, ds), splits in ft.items():
        cms = [splits[k] for k in sorted(splits)]
        data[(m, b)][f'{ds}_ft'] = (sum(cms), cms)
    return data


def compact(data, key, title, with_std=False):
    hdr = '| Model | ' + ' | '.join(c[1] for c in COLS) + ' | Mean of 3 zero-shot | Mean of 3 fine-tuned |'
    print(f'\n### {title}\n\n{hdr}\n|' + '---|' * (len(COLS) + 3))
    for mb in MODELS:
        cells, zs, ftv = [], [], []
        for col, _ in COLS:
            if col not in data[mb]:
                cells.append('–')
                continue
            cm, parts = data[mb][col]
            v = metrics(cm)[key]
            s = f'{v:.1f}'
            if with_std and len(parts) > 1:
                s += f' ± {np.std([metrics(p)[key] for p in parts]):.1f}'
            cells.append(s)
            (zs if col.endswith('_zs') else ftv if col.endswith('_ft') else []).append(v)
        mz = f'{np.mean(zs):.1f}' if len(zs) == 3 else '–'
        mf = f'{np.mean(ftv):.1f}' if len(ftv) == 3 else '–'
        print(f'| {NAMES[mb[0]]} {BB[mb[1]]} | ' + ' | '.join(cells) + f' | {mz} | {mf} |')


def full(data):
    print('\n### All results, Rec / F1 / IoU (%)\n')
    print('| Dataset | Setting | Model | Walls Rec / F1 / IoU | Openings Rec / F1 / IoU | Mean Rec / F1 / IoU | mIoU std |')
    print('|---|---|---|---|---|---|---|')
    for col, title in COLS:
        ds, setting = (title.rsplit(' ', 1) if col != 'cubi' else ('CubiCasa5k test', 'in-domain'))
        for mb in MODELS:
            if col not in data[mb]:
                continue
            cm, parts = data[mb][col]
            r = metrics(cm)
            std = np.std([metrics(p)['mean_iou'] for p in parts]) if len(parts) > 1 else float('nan')
            cell = lambda c: f"{r[c + '_rec']:.1f} / {r[c + '_f1']:.1f} / {r[c + '_iou']:.1f}"  # noqa: E731
            print(f'| {ds} | {setting} | {NAMES[mb[0]]} {BB[mb[1]]} | {cell("walls")} | {cell("open")} | '
                  f'{cell("mean")} | {"–" if np.isnan(std) else f"{std:.1f}"} |')


def main():
    data = collect()
    print('Generated by `kfold_patch/summarize_all_datasets.py`. 3 classes everywhere: bg / walls / openings '
          '(doors + windows); railings, stairs and WAFFLE interior count as bg. Pixels pooled over fold models '
          '(CubiCasa, zero-shot: 10 folds) or test splits (fine-tuned: R3D 1 split, CVC-FP and WAFFLE 5-fold CV). '
          '"±" = std across folds/splits.')
    compact(data, 'mean_iou', 'Mean IoU (walls + openings)', with_std=True)
    compact(data, 'walls_iou', 'Walls IoU')
    compact(data, 'open_iou', 'Openings IoU')
    full(data)


if __name__ == '__main__':
    main()
