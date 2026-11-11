"""
Zero-shot per-class evaluation on the WAFFLE benchmark, in the format of the WAFFLE paper
(Ganon et al., WACV 2025, arXiv:2412.00955, Table 4: precision / recall / IoU for walls, doors,
windows, interior, background).

GT: data/tfrecords/waffle/waffle5_all (0 bg, 1 walls, 2 doors, 3 windows, 4 interior; 109 plans).
Prediction: argmax over the 6 CubiCasa classes, mapped bg/railings/stairs -> bg, walls -> walls,
doors -> doors, windows -> windows. Interior is never predicted (CubiCasa has no interior class), so
it is reported only merged with bg ("bg + interior").

All 10 fold models of each combination are evaluated, using the same checkpoint sets as
evaluate_cross_dataset.py. Confusion matrices (5x5, rows = GT, cols = pred) are pooled over folds.
Merging doors + windows -> openings and interior -> bg reproduces the 3-class waffle_all result.

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/eval_waffle_perclass.py --model cab2 \
        --source results/official/test/test_kfold_cab2_EfficientNetV2M_20261014-120852.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import evaluate_kfold  # noqa: E402  (preloads GPU libs, imports TF)
import tensorflow as tf  # noqa: E402
import segmentation_models as sm  # noqa: E402
from datasets import floorplans  # noqa: E402
from evaluate_cross_dataset import official_checkpoints  # noqa: E402

GT_CLASSES = ['bg', 'walls', 'doors', 'windows', 'interior']
# CubiCasa output order: ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all']
PRED_MAP = np.array([0, 1, 0, 2, 3, 0], dtype=np.int64)
RECORDS = 'data/tfrecords/waffle/waffle5_all'


def evaluate(model, normalize):
    k = len(GT_CLASSES)
    cm = np.zeros((k, k), np.int64)
    ds = floorplans.load_dataset(RECORDS, normalize=normalize, classes=GT_CLASSES, n_upsample=6)
    for img, mask in ds.batch(1):
        y = model(img, training=False)
        if isinstance(y, (tuple, list)):
            y = y[0]
        t = tf.math.argmax(mask[0, :, :, :k], axis=-1).numpy().ravel()
        p = PRED_MAP[tf.math.argmax(y[0, :, :, :6], axis=-1).numpy().ravel()]
        cm += np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    return cm


def table4(cm):
    """Precision / recall / IoU for walls, doors, windows and bg+interior (interior merged into bg)."""
    m = cm.copy()
    m[0] += m[4]
    m[:, 0] += m[:, 4]
    m = m[:4, :4]
    tp = np.diag(m).astype(float)
    fp, fn = m.sum(0) - tp, m.sum(1) - tp
    prec, rec, iou = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1), tp / np.maximum(tp + fp + fn, 1)
    names = ['bg + interior', 'walls', 'doors', 'windows']
    return {n: (prec[i], rec[i], iou[i]) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', required=True, help='Official test result file listing the fold checkpoints.')
    args = ap.parse_args()
    sm.set_framework('tf.keras')
    src, folds, backbone = official_checkpoints(args.model, args.source)
    normalize = evaluate_kfold.MODEL_CONFIGS[args.model]['normalize']
    cms = []
    for f in sorted(folds):
        tic = time.time()
        model = tf.keras.models.load_model(folds[f], compile=False)
        cms.append(evaluate(model, normalize))
        del model
        tf.keras.backend.clear_session()
        print(f'fold {f} done in {(time.time() - tic) / 60:.1f} min', flush=True)
    cms = np.stack(cms)
    out = f'results/cross_dataset/waffle5_kfold_{args.model}_{backbone}_{time.strftime("%Y%m%d-%H%M%S")}'
    np.save(out + '_fold_cms.npy', cms)
    with open(out + '.txt', 'w') as fh:
        fh.write(f'Model: {args.model} ({backbone}, zero-shot, trained on CubiCasa5k only)\n')
        fh.write(f'Dataset: waffle5_all (109 plans), GT classes {GT_CLASSES}\n')
        fh.write(f'Checkpoint source: {src}\nFolds: {len(cms)} (pooled)\n\n')
        fh.write(f'Pooled confusion matrix (rows=GT, cols=pred):\n{cms.sum(0)}\n\n')
        fh.write('Class | Precision | Recall | IoU\n')
        for n, (p, r, i) in table4(cms.sum(0)).items():
            fh.write(f'{n} | {p:.3f} | {r:.3f} | {i:.3f}\n')
    print(open(out + '.txt').read(), f'saved: {out}.txt', flush=True)


if __name__ == '__main__':
    main()
