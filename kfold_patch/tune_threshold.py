"""
Validation-tuned decision thresholds for one fine-tuning run (finetune_cross_dataset.py result JSON).

The reported fine-tuned numbers use argmax over the merged class probabilities. Here the decision rule is
    pred = argmax_c (log p_c + b_c),   b_bg = 0
i.e. one log-odds offset per foreground class (for bg / wall this is the probability threshold
p_wall > 1 / (1 + exp(b_wall)); b = 0 reproduces argmax / p > 0.5). The offsets are chosen on the run's
own **validation** split (the split used for early stopping; never the test split), maximising mean IoU
over all classes (bg / walls / openings for R3D, CVC-FP, WAFFLE; bg + wall for MLSTRUCT-FP), then applied
once to the test split. No retraining: the run's saved best.weights.h5 is loaded into the same wrapped model.

Search: per-pixel log-odds d_c = log p_c - log p_bg are binned (bin width = grid step, offsets on the same
grid, so the val search is exact up to pixels exactly on a bin edge) into one histogram per true class;
every offset combination on the grid is scored from the histograms. The test split is then scored exactly
at b = 0 (must reproduce cm_finetuned of the JSON; checked) and at the tuned b.

Output: <out>/<run tag>.json with cm_argmax, cm_tuned, offsets, the val objective at b = 0 and tuned, and
the equivalent wall probability threshold for 2-class runs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/tune_threshold.py results/finetune_unfrozen/cvc_..._split0_....json
"""
import argparse
import itertools
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import evaluate_kfold  # noqa: E402,F401  (preloads GPU libs)
import tensorflow as tf  # noqa: E402
import segmentation_models as sm  # noqa: E402
import finetune_cross_dataset as F  # noqa: E402
import mlstruct_data  # noqa: E402
from tiled_inference import _resize_prob, predict_tiled, tile_batch  # noqa: E402

EPS = 1e-7
D_MAX = 12.  # |log-odds| range kept exactly; beyond it the decision does not depend on offsets within RANGE


def miou(cm):
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm)
    return float(np.mean(tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)))


def log_odds(prob):
    lp = np.log(np.clip(prob, EPS, 1.))
    return lp[..., 1:] - lp[..., :1]  # (..., K-1)


def decide(d, b):
    """d: (..., K-1) log-odds vs bg, b: (K-1,) offsets -> class index (0 = bg)."""
    s = d + b
    best = s.argmax(-1)
    return np.where(s.max(-1) > 0, best + 1, 0)


class Hist:
    """Per-true-class histogram of the clipped log-odds on a grid aligned with the offsets."""

    def __init__(self, k, step):
        self.k, self.step = k, step
        self.n = int(round(2 * D_MAX / step))
        self.h = np.zeros((k,) + (self.n,) * (k - 1), np.int64)

    def add(self, d, t):
        idx = np.clip(np.floor((d + D_MAX) / self.step).astype(np.int64), 0, self.n - 1)
        flat = np.ravel_multi_index(tuple(idx.reshape(-1, self.k - 1).T), (self.n,) * (self.k - 1))
        tt = t.ravel()
        for c in range(self.k):
            self.h[c] += np.bincount(flat[tt == c], minlength=self.n ** (self.k - 1)).reshape(self.h.shape[1:])

    def centers(self):
        c = -D_MAX + (np.arange(self.n) + 0.5) * self.step
        return np.stack(np.meshgrid(*([c] * (self.k - 1)), indexing='ij'), -1)

    def cm(self, b, cen=None):
        p = decide(self.centers() if cen is None else cen, np.asarray(b))
        return np.stack([np.bincount(p.ravel(), weights=self.h[c].ravel(), minlength=self.k) for c in range(self.k)])


def search(hist, rng, step):
    grid = np.round(np.arange(-rng, rng + step / 2, step), 6)
    cen = hist.centers()
    best = None
    for b in itertools.product(grid, repeat=hist.k - 1):
        v = miou(hist.cm(b, cen))
        key = (v, -float(np.abs(b).sum()))  # ties -> smallest offsets
        if best is None or key > best[0]:
            best = (key, list(map(float, b)))
    return best[1]


def build_model(r):
    base = tf.keras.models.load_model(r['checkpoint'], compile=False)
    agg = F.AGG_WALL if r['dataset'] == 'mlstruct' else F.AGG
    model = F.wrap(base, agg)
    if not r.get('backbone_frozen', True):
        F.unfreeze_backbone(model)  # same trainable split as at save time -> same weight order in the .h5
    model.load_weights(r['weights'])
    return model


def iterate(model, r, split_name):
    """Yields (prob (H, W, K), true label (H, W)) for every plan of the run's `split_name` split."""
    if r['dataset'] == 'mlstruct':
        ids = mlstruct_data.split_ids(r['split'])[split_name]
        bs = tile_batch(r['model'], r['backbone'])
        for i in ids:
            p = predict_tiled(model, mlstruct_data.load_input(r['scale'], i).astype(np.float32),
                              agg=np.eye(2, dtype=np.float32), batch=bs)[..., 1]
            gt = mlstruct_data.load_mask('native', i)
            if p.shape != gt.shape:
                p = _resize_prob(p, gt.shape)
            yield np.stack([1. - p, p], -1), (gt > 0).astype(np.int64)
    else:
        recs = F.records(r['dataset'])
        src, idx = F.split_indices(r['dataset'], r['split'])[split_name]
        for img, mask in F.subset(*recs[src], idx, F.N_UP).batch(1):
            yield model(img, training=False)[0].numpy(), tf.argmax(mask[0], -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result', help='finetune_cross_dataset.py result JSON')
    ap.add_argument('--out', default='results/threshold_tuning')
    ap.add_argument('--step', type=float, default=0.1, help='offset grid step (log-odds)')
    ap.add_argument('--range', type=float, default=3., help='offsets searched in [-range, range]')
    args = ap.parse_args()
    sm.set_framework('tf.keras')
    r = json.load(open(args.result))
    k = 2 if r['dataset'] == 'mlstruct' else 3
    model = build_model(r)

    hist = Hist(k, args.step)
    for prob, t in iterate(model, r, 'val'):
        hist.add(log_odds(prob), t)
    b = search(hist, args.range, args.step)
    val0, val_t = miou(hist.cm([0.] * (k - 1))), miou(hist.cm(b))
    print(f'val mIoU argmax {val0:.4f} -> tuned {val_t:.4f}, offsets {b}', flush=True)

    cm0, cmt = np.zeros((k, k), np.int64), np.zeros((k, k), np.int64)
    for prob, t in iterate(model, r, 'test'):
        d = log_odds(prob)
        for cm, off in ((cm0, np.zeros(k - 1)), (cmt, np.array(b))):
            cm += np.bincount(t.ravel() * k + decide(d, off).ravel(), minlength=k * k).reshape(k, k)
    ref = np.array(r['cm_finetuned'])
    diff = float(np.abs(cm0 - ref).sum() / ref.sum())
    print(f'test b=0 vs stored cm_finetuned: {100 * diff:.4f}% of pixels differ', flush=True)

    tag = os.path.splitext(os.path.basename(args.result))[0]
    res = {'source': args.result, 'dataset': r['dataset'], 'scale': r.get('scale'), 'model': r['model'],
           'backbone': r['backbone'], 'split': r['split'], 'backbone_frozen': r.get('backbone_frozen', True),
           'classes': ['bg', 'wall'] if k == 2 else F.CLASSES, 'step': args.step, 'range': args.range,
           'offsets': b, 'p_wall_threshold': float(1 / (1 + np.exp(b[0]))) if k == 2 else None,
           'val_miou_argmax': val0, 'val_miou_tuned': val_t, 'repro_diff_frac': diff,
           'cm_argmax': cm0.tolist(), 'cm_tuned': cmt.tolist(), 'cm_finetuned_stored': ref.tolist(),
           'test_miou_argmax': miou(cm0), 'test_miou_tuned': miou(cmt)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, tag + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f'test mIoU argmax {res["test_miou_argmax"]:.4f} -> tuned {res["test_miou_tuned"]:.4f}')


if __name__ == '__main__':
    main()
