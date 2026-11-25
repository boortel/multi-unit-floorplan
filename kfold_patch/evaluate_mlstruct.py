"""
Zero-shot evaluation of CubiCasa5k-trained CAB models on MLSTRUCT-FP (bg / wall), with tiled inference.

Predictions (6 CubiCasa classes) are merged to wall = walls, bg = everything else (railings, doors,
windows, stairs), because MLSTRUCT-FP annotates only (structural) walls. Inference runs at `--scale`
(sN = N px/m, or native); the wall probability is upsampled to the native plan size and scored against the
native-resolution mask, so every scale is scored on identical pixels (as Paek et al. 2025 score full plans).

Checkpoints: all fold models from an official k-fold result file (default; `--source` for B4), or one
explicit `--checkpoint` (+ `--backbone`).

Output: results/mlstruct/zeroshot/<scale>_<model>_<backbone>_<ckpt>_<ts>.{txt,npz}
    npz: cms (n_ckpt, n_img, 2, 2) per checkpoint and plan, ids, checkpoints
    txt: pooled metrics, mean +/- std over checkpoints, and over the 5 CV test folds

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_mlstruct.py --model cab2 --scale s40
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/evaluate_mlstruct.py --model cab2 --scale native \
        --checkpoint models/.../4 --backbone EfficientNetV2M
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import evaluate_kfold  # noqa: E402,F401  (preloads GPU libs)
import tensorflow as tf  # noqa: E402
import segmentation_models as sm  # noqa: E402
from evaluate_cross_dataset import official_checkpoints  # noqa: E402
import mlstruct_data as D  # noqa: E402
from tiled_inference import predict_tiled, tile_batch, wall_cm, wall_metrics  # noqa: E402

OUT = 'results/mlstruct/zeroshot'


def eval_checkpoint(model, scale, ids, tile, overlap, normalize=False, batch=8):
    cms = np.zeros((len(ids), 2, 2), np.int64)
    for j, iid in enumerate(ids):
        img = D.load_input(scale, iid).astype(np.float32)
        if normalize:
            img = img / 255.
        prob = predict_tiled(model, img, tile=tile, overlap=overlap, batch=batch)[..., 1]
        cms[j] = wall_cm(prob, D.load_mask('native', iid))
    return cms


def fmt(m):
    return ' '.join(f'{k}={v:.4f}' for k, v in m.items())


def report(cms, ids, ckpts):
    """Text block: pooled, mean +/- std over checkpoints, and mean +/- std over the 5 CV test folds."""
    lines = []
    pooled = cms.sum((0, 1))
    lines.append(f'Pooled over {cms.shape[0]} checkpoint(s) x {cms.shape[1]} plans, CM (rows=GT bg/wall):\n{pooled}')
    lines.append('Pooled: ' + fmt(wall_metrics(pooled)))
    per_ck = [wall_metrics(c.sum(0)) for c in cms]
    for k in per_ck[0]:
        v = [m[k] for m in per_ck]
        lines.append(f'  over checkpoints  {k:12s} {np.mean(v):.4f} +/- {np.std(v):.4f}')
    pos = {iid: j for j, iid in enumerate(ids)}
    fold_m = []
    for f in D.folds():
        idx = [pos[i] for i in f if i in pos]
        if idx:
            fold_m.append(wall_metrics(cms[:, idx].sum((0, 1))))
    if len(fold_m) == D.N_FOLDS:
        for k in fold_m[0]:
            v = [m[k] for m in fold_m]
            lines.append(f'  over CV folds     {k:12s} {np.mean(v):.4f} +/- {np.std(v):.4f}')
    lines.append('Checkpoints:\n' + '\n'.join(f'  {c}' for c in ckpts))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['cab1', 'cab2'])
    ap.add_argument('--scale', default=D.PRIMARY_SCALE, choices=D.SCALES)
    ap.add_argument('--source', default=None, help='official k-fold result file (all folds); default latest V2M')
    ap.add_argument('--checkpoint', default=None, help='evaluate this single checkpoint instead')
    ap.add_argument('--backbone', default=None, help='backbone name for --checkpoint')
    ap.add_argument('--tile', type=int, default=768)
    ap.add_argument('--overlap', type=float, default=0.25)
    ap.add_argument('--limit', type=int, default=None, help='first N plans only (smoke test)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    sm.set_framework('tf.keras')

    if args.checkpoint:
        assert args.backbone, '--backbone is required with --checkpoint'
        ckpts, backbone, tag = [args.checkpoint], args.backbone, 'single'
    else:
        src, folds, backbone = official_checkpoints(args.model, args.source)
        ckpts, tag = [folds[f] for f in sorted(folds)], 'allfolds'
    ids = D.all_ids()[:args.limit]
    normalize = evaluate_kfold.MODEL_CONFIGS[args.model]['normalize']
    print(f'{args.model} {backbone} scale={args.scale} plans={len(ids)} checkpoints={len(ckpts)}', flush=True)

    cms = []
    for c in ckpts:
        tic = time.time()
        model = tf.keras.models.load_model(c, compile=False)
        cm = eval_checkpoint(model, args.scale, ids, args.tile, args.overlap, normalize,
                             tile_batch(args.model, backbone))
        cms.append(cm)
        print(f'  {c}: {fmt(wall_metrics(cm.sum(0)))}  ({(time.time() - tic) / 60:.1f} min)', flush=True)
        del model
        tf.keras.backend.clear_session()
    cms = np.stack(cms)

    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    name = f'{args.scale}_{args.model}_{backbone}_{tag}' + (f'_n{args.limit}' if args.limit else '') + f'_{ts}'
    np.savez(os.path.join(args.out, name + '.npz'), cms=cms, ids=np.array(ids), checkpoints=np.array(ckpts))
    text = (f'Model: {args.model} ({backbone}), zero-shot (CubiCasa5k only)\nDataset: MLSTRUCT-FP, bg/wall '
            f'(pred: walls -> wall, all else -> bg)\nScale: {args.scale} (scored at native resolution), '
            f'tile {args.tile}, overlap {args.overlap}\n\n' + report(cms, ids, ckpts))
    with open(os.path.join(args.out, name + '.txt'), 'w') as fh:
        fh.write(text + '\n')
    print(text, flush=True)
    with open(os.path.join(args.out, name + '.json'), 'w') as fh:
        json.dump({'model': args.model, 'backbone': backbone, 'scale': args.scale, 'tag': tag,
                   'pooled': wall_metrics(cms.sum((0, 1)))}, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
