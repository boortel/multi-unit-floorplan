"""
Fine-tune a CubiCasa5k-trained CAB1/CAB2 fold checkpoint on R3D, CVC-FP or WAFFLE (in-domain
comparison with arXiv:2408.01526, Table IV) and evaluate on a held-out test split.

The 6-class CubiCasa head is kept unchanged. Its softmax output is merged into the
3 target classes by summing probabilities:
    bg = bg + railings + stairs,  walls = walls,  openings = doors + windows
and trained with the asymmetric unified focal loss (as in the original CAB training).
By default the backbone stays frozen as in the saved checkpoint. With --unfreeze, all backbone
Conv/DepthwiseConv weights become trainable too (BatchNorm layers stay frozen, i.e. in inference mode
with their CubiCasa statistics, since batch size is 1); same LR for backbone and decoder.

Splits:
    r3d: official test (53) as test; official train (179) -> 80/20 train/val (seeded).
    cvc: 5-fold CV over all 122 plans; fold k = test, the next fold = val, rest = train
         (i.e. 60/20/20 as in the paper); pool test CMs over the 5 folds.
    waffle: same 5-fold CV protocol as cvc over the 109 WAFFLE benchmark plans.

The test split is evaluated zero-shot, after AdaBN (BatchNorm statistics recomputed on the
unlabelled train-split images, no weight updates) and after fine-tuning (LR 1e-5 by default,
~1/10 of the original CubiCasa training LR; geometric + photometric/scan augmentation).

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/finetune_cross_dataset.py \
        --model cab2 --checkpoint models/.../4 --backbone EfficientNetV2M --dataset cvc --split 0
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
from datasets import floorplans  # noqa: E402
from training import loss_functions  # noqa: E402

CLASSES = ['bg', 'walls', 'openings']
# rows: 6 CubiCasa outputs -> 3 targets
AGG = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]], np.float32)
N_UP = 6
CROP = 512
CVC_FOLDS = 5


def records(dataset):
    if dataset == 'r3d':
        return {'train': ('data/tfrecords/r3d/r3d_train', 179), 'test': ('data/tfrecords/r3d/r3d_test', 53)}
    if dataset == 'waffle':
        return {'all': ('data/tfrecords/waffle/waffle_all', 109)}
    return {'all': ('data/tfrecords/cvc/cvc_all', 122)}


def split_indices(dataset, split, seed=0):
    rng = np.random.RandomState(seed)
    if dataset == 'r3d':
        perm = rng.permutation(179)
        n_val = int(round(0.2 * 179))
        return {'train': ('train', np.sort(perm[n_val:])), 'val': ('train', np.sort(perm[:n_val])),
                'test': ('test', np.arange(53))}
    perm = rng.permutation(records(dataset)['all'][1])
    folds = np.array_split(perm, CVC_FOLDS)
    test, val = folds[split], folds[(split + 1) % CVC_FOLDS]
    train = np.concatenate([folds[i] for i in range(CVC_FOLDS) if i not in (split, (split + 1) % CVC_FOLDS)])
    return {'train': ('all', np.sort(train)), 'val': ('all', np.sort(val)), 'test': ('all', np.sort(test))}


def subset(path, n_total, idx, n_up):
    keep = tf.constant(np.isin(np.arange(n_total), idx))
    ds = floorplans.load_dataset(path, normalize=False, classes=CLASSES, n_upsample=n_up)
    ds = ds.enumerate().filter(lambda i, x: tf.gather(keep, i))
    return ds.map(lambda i, x: (x[0], x[1][..., :3]))


def augment(img, mask):
    # flips + 90-degree rotations
    k = tf.random.uniform([], 0, 4, tf.int32)
    img, mask = tf.image.rot90(img, k), tf.image.rot90(mask, k)
    if tf.random.uniform([]) < 0.5:
        img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
    # random scale 0.8-1.2
    s = tf.random.uniform([], 0.8, 1.2)
    hw = tf.cast(tf.cast(tf.shape(img)[:2], tf.float32) * s, tf.int32)
    img = tf.image.resize(img, hw)
    mask = tf.image.resize(mask, hw, method='nearest')
    # fixed-size random crop (constant shapes -> no retracing); pad with white paper / bg label
    ph, pw = tf.maximum(CROP - hw[0], 0), tf.maximum(CROP - hw[1], 0)
    pad = [[ph // 2, ph - ph // 2], [pw // 2, pw - pw // 2], [0, 0]]
    img = tf.pad(img, pad, constant_values=255.)
    bg = tf.pad(mask[..., :1], pad, constant_values=1.)
    mask = tf.concat([bg, tf.pad(mask[..., 1:], pad)], -1)
    both = tf.image.random_crop(tf.concat([img, mask], -1), [CROP, CROP, 6])
    return photometric(both[..., :3]), both[..., 3:]


def _gauss_blur(img, sigma):
    x = tf.range(-3., 4.)
    k = tf.exp(-x ** 2 / (2. * sigma ** 2))
    k = k / tf.reduce_sum(k)
    kh = tf.tile(tf.reshape(k, [1, 7, 1, 1]), [1, 1, 3, 1])
    kv = tf.tile(tf.reshape(k, [7, 1, 1, 1]), [1, 1, 3, 1])
    y = tf.pad(img[None], [[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
    y = tf.nn.depthwise_conv2d(y, kh, [1, 1, 1, 1], 'VALID')
    return tf.nn.depthwise_conv2d(y, kv, [1, 1, 1, 1], 'VALID')[0]


def photometric(img):
    # scan-like appearance changes on a 0-255 float image (mask untouched)
    u = lambda: tf.random.uniform([])  # noqa: E731
    img = img + tf.random.uniform([], -0.15, 0.15) * 255.
    m = tf.reduce_mean(img)
    img = (img - m) * tf.random.uniform([], 0.7, 1.3) + m
    img = tf.where(u() < 0.3, tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img)), img)
    img = tf.where(u() < 0.3, _gauss_blur(img, tf.random.uniform([], 0.3, 1.2)), img)
    img = tf.where(u() < 0.5, img + tf.random.normal(tf.shape(img)) * tf.random.uniform([], 0., 8.), img)
    img = tf.clip_by_value(img, 0., 255.)
    jpg = tf.image.adjust_jpeg_quality(img / 255., tf.random.uniform([], 40, 96, tf.int32)) * 255.
    img = tf.where(u() < 0.3, tf.reshape(jpg, tf.shape(img)), img)
    return img


def wrap(base):
    inp = tf.keras.Input((None, None, 3))
    y = base(inp)
    if isinstance(y, (list, tuple)):
        y = y[0]
    y = tf.cast(y, tf.float32)
    out = tf.keras.layers.Lambda(lambda t: tf.einsum('bhwc,ck->bhwk', t, tf.constant(AGG)), name='merge3')(y)
    return tf.keras.Model(inp, out)


def unfreeze_backbone(model):
    """Make every frozen non-BN, non-Normalization layer trainable (the encoder). Returns #layers unfrozen."""
    n = 0
    for m in model.submodules:
        if isinstance(m, tf.keras.Model) and not m.trainable:
            m.trainable = True
    for m in model.submodules:
        if isinstance(m, tf.keras.Model) or not isinstance(m, tf.keras.layers.Layer) or not m.weights:
            continue
        name = type(m).__name__
        if 'BatchNormalization' in name or name == 'Normalization':
            m.trainable = False
        elif not m.trainable:
            m.trainable = True
            n += 1
    return n


def adabn(model, ds):
    """Recompute every BatchNorm layer's moving mean/var as the cumulative average over `ds` images."""
    bns = [m for m in model.submodules if 'BatchNormalization' in type(m).__name__]
    saved = [(b, b.trainable, b.momentum) for b in bns]
    for b in bns:
        b.trainable = True
    for i, (img, _) in enumerate(ds.batch(1)):
        for b in bns:
            b.momentum = i / (i + 1.)   # i=0 -> take batch stats; then running mean over images
        model(img, training=True)       # eager call; only BN statistics change, no gradients
    for b, t, mo in saved:
        b.trainable, b.momentum = t, mo
    return len(bns)


def confusion(model, ds):
    cm = np.zeros((3, 3), np.int64)
    for img, mask in ds.batch(1):
        p = tf.argmax(model(img, training=False)[0], -1).numpy().ravel()
        t = tf.argmax(mask[0], -1).numpy().ravel()
        cm += np.bincount(t * 3 + p, minlength=9).reshape(3, 3)
    return cm


def metrics(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return {'rec': (tp / np.maximum(tp + fn, 1)).tolist(), 'f1': (2 * tp / np.maximum(2 * tp + fp + fn, 1)).tolist(),
            'iou': (tp / np.maximum(tp + fp + fn, 1)).tolist(), 'acc': float(tp.sum() / cm.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', choices=['r3d', 'cvc', 'waffle'], required=True)
    ap.add_argument('--split', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--out', default='results/finetune')
    ap.add_argument('--models_out', default='models/finetune')
    ap.add_argument('--unfreeze', action='store_true', help='also fine-tune the backbone (BN stays frozen)')
    args = ap.parse_args()
    sm.set_framework('tf.keras')
    tf.random.set_seed(args.split)

    recs = records(args.dataset)
    sp = split_indices(args.dataset, args.split)
    tr = subset(*recs[sp['train'][0]], sp['train'][1], None).map(augment, tf.data.AUTOTUNE)
    tr = tr.shuffle(256, reshuffle_each_iteration=True).batch(1).prefetch(tf.data.AUTOTUNE)
    va = subset(*recs[sp['val'][0]], sp['val'][1], N_UP).cache()
    te = subset(*recs[sp['test'][0]], sp['test'][1], N_UP).cache()

    base = tf.keras.models.load_model(args.checkpoint, compile=False)
    model = wrap(base)
    cm_zero = confusion(model, te)

    w0 = model.get_weights()
    n_bn = adabn(model, subset(*recs[sp['train'][0]], sp['train'][1], N_UP))
    cm_ada = confusion(model, te)
    model.set_weights(w0)

    n_unfrozen = unfreeze_backbone(model) if args.unfreeze else 0
    n_trainable = int(sum(v.shape.num_elements() for v in model.trainable_variables))
    print(f'unfrozen backbone layers: {n_unfrozen}, trainable params: {n_trainable}', flush=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=loss_functions.asym_unified_focal_loss(3))
    tag = f'{args.dataset}_{args.model}_{args.backbone}_split{args.split}_{time.strftime("%Y%m%d-%H%M%S")}'
    ckpt = os.path.join(args.models_out, tag, 'best.weights.h5')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    cbs = [tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
           tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)]
    tic = time.time()
    hist = model.fit(tr, validation_data=va.batch(1), epochs=args.epochs, callbacks=cbs, verbose=2)
    model.load_weights(ckpt)
    cm_ft = confusion(model, te)

    os.makedirs(args.out, exist_ok=True)
    res = {'model': args.model, 'backbone': args.backbone, 'checkpoint': args.checkpoint, 'dataset': args.dataset,
           'split': args.split, 'n_train': len(sp['train'][1]), 'n_val': len(sp['val'][1]), 'n_test': len(sp['test'][1]),
           'epochs_run': len(hist.history['loss']), 'best_epoch': int(np.argmin(hist.history['val_loss'])) + 1,
           'minutes': (time.time() - tic) / 60, 'lr': args.lr, 'aug': 'geom+photo', 'n_bn_adabn': n_bn,
           'backbone_frozen': not args.unfreeze, 'n_unfrozen_layers': n_unfrozen, 'n_trainable_params': n_trainable,
           'cm_zeroshot': cm_zero.tolist(), 'cm_adabn': cm_ada.tolist(), 'cm_finetuned': cm_ft.tolist(),
           'zeroshot': metrics(cm_zero), 'adabn': metrics(cm_ada), 'finetuned': metrics(cm_ft), 'weights': ckpt}
    with open(os.path.join(args.out, tag + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ['zeroshot', 'adabn', 'finetuned', 'epochs_run', 'best_epoch', 'minutes']}, indent=1))


if __name__ == '__main__':
    main()
