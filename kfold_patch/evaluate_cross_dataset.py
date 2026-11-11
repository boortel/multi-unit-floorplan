"""
Zero-shot cross-dataset evaluation of the CubiCasa5k k-fold models on R3D, CVC-FP and WAFFLE.

Uses exactly the fold checkpoints listed in the official k-fold test result files
(results/official/test/test_kfold_<model>_EfficientNetV2M_*.txt), i.e. CAB1 V2M HPO t21
and CAB2 V2M HPO t32. No fine-tuning, no augmentation.

Each fold model is evaluated on every requested TFRecord set. Predictions on the 6
CubiCasa classes are mapped to the 3-class target scheme:
    bg -> bg, walls -> walls, doors/windows -> openings, railings/stairs -> bg
Confusion matrices are summed over folds (same as evaluate_kfold.py).
"R3D all" is reported as R3D test + R3D train.

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_cross_dataset.py --models cab1
    CUDA_VISIBLE_DEVICES=2 python kfold_patch/evaluate_cross_dataset.py --models cab2
"""
import argparse
import glob
import os
import re
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tabulate import tabulate

import evaluate_kfold  # noqa: E402  (preloads GPU libs, imports TF)
import tensorflow as tf  # noqa: E402
import segmentation_models as sm  # noqa: E402
from datasets import floorplans  # noqa: E402
from training.metrics import cm_metrics  # noqa: E402

TARGET_CLASSES = ['bg', 'walls', 'openings']
# CubiCasa model output order: ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all']
PRED_MAP = np.array([0, 1, 0, 2, 2, 0], dtype=np.int64)

DATASETS = {
    'r3d_test': 'data/tfrecords/r3d/r3d_test',
    'r3d_train': 'data/tfrecords/r3d/r3d_train',
    'cvc_all': 'data/tfrecords/cvc/cvc_all',
    'waffle_all': 'data/tfrecords/waffle/waffle_all',
}
COMBINED = {'r3d_all': ['r3d_test', 'r3d_train']}


def official_checkpoints(model, source=None):
    """Fold checkpoints from an official k-fold test result file (default: latest V2M)."""
    if source is None:
        files = sorted(glob.glob(f'results/official/test/test_kfold_{model}_EfficientNetV2M_*.txt'))
        assert files, f'No official result file for {model}'
        source = files[-1]
    text = open(source).read()
    folds = dict((int(i), p) for i, p in re.findall(r'Fold (\d+): (models/\S+)', text))
    backbone = re.search(r'Backbone: (\S+)', text).group(1)
    return source, folds, backbone


def eval_model_on(model, path, normalize=False):
    ds = floorplans.load_dataset(path, normalize=normalize, classes=TARGET_CLASSES, n_upsample=6)
    k = len(TARGET_CLASSES)
    cm = np.zeros((k, k), np.int64)
    for img, mask in ds.batch(1):
        y = model(img, training=False)
        if isinstance(y, (tuple, list)):
            y = y[0]
        t = tf.math.argmax(mask[0, :, :, :k], axis=-1).numpy().ravel()
        p = PRED_MAP[tf.math.argmax(y[0, :, :, :6], axis=-1).numpy().ravel()]
        cm += np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    return cm


def table_for(cm):
    cols = cm_metrics(cm)
    rows = [['Accuracy', np.diag(cm).sum() / cm.sum()]]
    nb = cm.copy(); nb[0] = 0
    rows.append(['Accuracy no bg', np.diag(nb).sum() / nb.sum()])
    for i, c in enumerate(TARGET_CLASSES):
        rows.append([c] + [col[i] for col in cols[:-1]])
    rows.append(['Mean'] + [np.nanmean(col) for col in cols[:-1]])
    rows.append(['Mean no bg'] + [np.nanmean(col[1:]) for col in cols[:-1]])
    return tabulate(rows, headers=['Class', 'Class Acc', 'Recall', 'Precision', 'F1', 'IoU', 'fwRecall', 'fwIoU'],
                    floatfmt='.4f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['cab1', 'cab2'])
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--source', default=None,
                    help='Official test result file listing the fold checkpoints (one model only).')
    args = ap.parse_args()
    sm.set_framework('tf.keras')
    datasets = [d for d in args.datasets if os.path.exists(DATASETS[d] + '.tfrecords')]
    print('Datasets:', datasets, flush=True)

    for m in args.models:
        src, folds, backbone = official_checkpoints(m, args.source)
        print(f'\n=== {m} (checkpoints from {src}): {len(folds)} folds', flush=True)
        cms = {d: [] for d in datasets}
        for f in sorted(folds):
            tic = time.time()
            model = tf.keras.models.load_model(folds[f], compile=False)
            for d in datasets:
                cm = eval_model_on(model, DATASETS[d], normalize=evaluate_kfold.MODEL_CONFIGS[m]['normalize'])
                cms[d].append(cm)
                print(f'  fold {f} {d}: acc={np.diag(cm).sum() / cm.sum():.4f}', flush=True)
            del model
            tf.keras.backend.clear_session()
            print(f'  fold {f} done in {(time.time() - tic) / 60:.1f} min', flush=True)

        for name, parts in COMBINED.items():
            if all(p in cms for p in parts):
                cms[name] = [sum(x) for x in zip(*(cms[p] for p in parts))]

        os.makedirs('results/cross_dataset', exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        for d, lst in cms.items():
            cm = np.sum(lst, axis=0)
            accs = [np.diag(c).sum() / c.sum() for c in lst]
            out = os.path.join('results/cross_dataset', f'{d}_kfold_{m}_{backbone}_{ts}.txt')
            np.save(out[:-4] + '_fold_cms.npy', np.stack(lst))  # per-fold CMs, fold order = sorted(folds)
            with open(out, 'w') as fh:
                fh.write(f'Model: {m} ({backbone}, zero-shot, trained on CubiCasa5k only)\n')
                fh.write(f'Dataset: {d}\nClasses: {TARGET_CLASSES} '
                         f'(pred map: walls->walls, doors+windows->openings, railings+stairs->bg)\n')
                fh.write(f'Folds evaluated: {len(lst)}\nPer-fold accuracy: {[f"{a:.4f}" for a in accs]}\n')
                fh.write(f'Mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n')
                fh.write(f'Checkpoint source: {src}\n\nConfusion matrix (rows=GT, cols=pred):\n{cm}\n\n')
                fh.write(table_for(cm))
            print(f'\n--- {m} on {d} ---\n{table_for(cm)}\n  saved: {out}', flush=True)


if __name__ == '__main__':
    main()
