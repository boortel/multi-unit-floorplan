"""
Inference-time ablation of the CAB1 / CAB2 champion models (no retraining).

Each paper component is "switched off" in the trained fold models by editing
weights, and the edited model is re-evaluated on the CubiCasa5k test set with
the same pipeline as kfold_patch/evaluate_kfold.py (pooled confusion matrix).

Ablation = mean-ablation: the component's output is replaced by its average
value measured on the fold model's own validation fold (never the test set).

  * Skip branch  unet3plus_down_from{i}_to{lev}_0 (CONV_stack -> GN -> ReLU):
        GN gamma = 0, beta = mean post-ReLU activation  =>  ReLU(beta) = beta.
  * CAM gate (CAB1): gate replaced by its per-channel mean g_c:
        fc12 = fc22 = 0 (gate = 0.5) and conv kernel *= 2 g_c.
  * SAM5 gate (CAB1): spatial gate of each block replaced by its mean g_j:
        HHDC kernels = 0 (gate = 0.5) and conv_j kernel *= 2 g_j.
  * SAM single dilation (CAB1): the dilation-2/3 terms of the gate logits are
        replaced by their mean (needs a replacement layer, built with clone_model).
  * AC branches (screening only, sensitivity): kernels of a branch type set to 0.

Usage:
    python kfold_patch/ablation_inference.py --model cab1 --fold 0 --phase screen
    python kfold_patch/ablation_inference.py --model cab2 --fold 3 --phase main
    python kfold_patch/ablation_inference.py --model cab1 --fold 0 --phase check
"""
import argparse
import json
import os
import re
import sys
import time

# Device is chosen by the caller (CUDA_VISIBLE_DEVICES); scripts/run_ablation_inference.sh sets it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import segmentation_models as sm  # noqa: E402
from datasets import floorplans  # noqa: E402

sm.set_framework("tf.keras")

CLASSES = ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all']
NUM_CLASSES = len(CLASSES)
DATA_ROOT = 'data/tfrecords/cubicasa5k'
N_UPSAMPLE = 6
REFERENCE = {
    'cab1': 'results/official/test/test_kfold_cab1_EfficientNetV2M_20261014-122132',
    'cab2': 'results/official/test/test_kfold_cab2_EfficientNetV2M_20261014-120852',
}
OUT_DIR = 'results/ablation_inference'
BRANCH_RE = re.compile(r'^unet3plus_down_from(\d+)_to(\d+)_0_gn$')


# ── data ─────────────────────────────────────────────────────────────────────
def fold_dirs(model):
    dirs = {}
    with open(REFERENCE[model] + '.txt') as f:
        for line in f:
            m = re.match(r'\s+Fold (\d+): (\S+)', line)
            if m:
                dirs[int(m.group(1))] = m.group(2)
    return dirs


def load_split(split, fold, max_images=None):
    """Returns a list of (image[1,H,W,3] float32, label[H*W] int64), same pipeline as evaluate_kfold."""
    if split == 'test':
        ds = floorplans.load_test_data(CLASSES, 'cubicasa5k', normalize=False,
                                       base_dir=DATA_ROOT, n_upsample=N_UPSAMPLE)
    else:
        ds = floorplans.load_dataset(os.path.join(DATA_ROOT, f'cubicasa5k_fold_{fold}'), normalize=False,
                                     classes=CLASSES, n_upsample=N_UPSAMPLE)
    out = []
    for img, mask in ds.batch(1):
        lab = tf.math.argmax(mask[0, :, :, :NUM_CLASSES], axis=-1).numpy().astype(np.int64).ravel()
        out.append((img.numpy(), lab))
        if max_images and len(out) >= max_images:
            break
    return out


# ── model structure ─────────────────────────────────────────────────────────
def branches(model):
    """{gn_layer_name: (decoder_stage i, source level lev, pool_scale)}; pool_scale<0: decoder/deeper,
    0: same-level encoder skip, >0: shallower encoder."""
    out = {}
    for l in model.layers:
        m = BRANCH_RE.match(l.name)
        if m:
            i, lev = int(m.group(1)), int(m.group(2))
            out[l.name] = (i, lev, lev - i - 1)
    return out


def attention_layers(model):
    cams = [l for l in model.layers if type(l).__name__ == 'CAM']
    sams = [l for l in model.layers if type(l).__name__ == 'SAM5']
    return cams, sams


def _k(layer):
    return layer.kernel.numpy()


def sam_layers(sam):
    """Per block j: dict(conv=layer, c=[1x1 layers], d=[dilated 3x3 layers])."""
    return [dict(conv=conv, c=[h.conv1, h.conv2, h.conv3], d=[h.convd1, h.convd2, h.convd3])
            for conv, h in zip(sam.conv, sam.hhdc)]


def sam_parts(sam):
    """Same as sam_layers but with float32 numpy kernels and the dilation rates (graph-safe)."""
    out = []
    for L in sam_layers(sam):
        out.append(dict(conv=L['conv'].kernel.numpy().astype(np.float32),
                        c=[l.kernel.numpy().astype(np.float32) for l in L['c']],
                        d=[l.kernel.numpy().astype(np.float32) for l in L['d']],
                        dil=[int(l.dilation_rate[0]) for l in L['d']]))
    return out


def _w(layer):
    """Kernel as float32 constant (variables may be AutocastVariables under mixed precision)."""
    return tf.constant(layer.kernel.numpy(), dtype=tf.float32)


def _conv(x, k, dil=1):
    if not isinstance(k, tf.Tensor) or k.dtype != tf.float32:
        k = _w(k) if hasattr(k, 'kernel') else tf.constant(np.asarray(k), tf.float32)
    return tf.nn.conv2d(tf.cast(x, tf.float32), k, strides=1, padding='SAME', dilations=dil)


def _dil(layer):
    d = getattr(layer, 'dilation_rate', None)
    return int(d[0]) if d is not None else None


def sam_block(x, p, mode='normal', r_mean=None, gate_mean=None):
    """Manual SAM5 block. Returns (output, gate, r) where r = dilation-2/3 part of the logits."""
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx = tf.reduce_max(x, axis=-1, keepdims=True)
    s = tf.concat([avg, mx], axis=-1)
    base = sum(_conv(s, c) for c in p['c']) + _conv(s, p['d'][0], p['dil'][0])
    r = _conv(s, p['d'][1], p['dil'][1]) + _conv(s, p['d'][2], p['dil'][2])
    if mode == 'single_dilation':
        z = base + r_mean
    else:
        z = base + r
    g = tf.sigmoid(z)
    if mode == 'gate_mean':
        g = tf.ones_like(g) * gate_mean
    return _conv(x, p['conv']) * g, g, r


def cam_gate(x, cam):
    """CAM via the layer's own sublayers (exactly as the saved model computes it, incl. float16 pooling)."""
    xc = cam.conv(x)
    z = cam.fc12(cam.relu1(cam.fc11(cam.avg_pool(xc)))) + cam.fc22(cam.relu1(cam.fc21(cam.max_pool(xc))))
    return tf.cast(xc, tf.float32), tf.cast(cam.sigmoid(z), tf.float32)


class SAM5SingleDilation(tf.keras.layers.Layer):
    """Drop-in replacement for a trained SAM5 layer whose dilation-2/3 logit terms are mean-ablated."""

    def __init__(self, sam, r_means, mode='single_dilation', **kw):
        super().__init__(name=sam.name + '_' + mode, dtype='float32', **kw)
        self.mode = mode
        self.out_dtype = sam.compute_dtype
        self.parts = sam_parts(sam)
        self.r_means = [float(v) for v in r_means]

    def call(self, inputs):
        outs = [sam_block(x, p, self.mode, r_mean=rm)[0]
                for x, p, rm in zip(inputs, self.parts, self.r_means)]
        return tf.cast(tf.concat(outs, axis=-1), self.out_dtype)


# ── statistics on the validation fold ───────────────────────────────────────
def collect_stats(model, val_data, path):
    """Validation-fold means used for mean-ablation (cached as .npz).

    mu/<branch_gn>      per-channel mean of the branch's post-ReLU output
    cam_g/<cam>         per-channel mean CAM gate
    sam_g/<sam>/<j>     mean spatial gate of SAM5 block j
    sam_r/<sam>/<j>     mean of the dilation-2/3 part of block j's gate logits
    manual_attention_max_abs_err   check: manual CAM/SAM5 re-implementation vs. the saved layers
    """
    if os.path.exists(path):
        stats = dict(np.load(path, allow_pickle=True))
        check_stats(stats, path)
        return stats
    names = sorted(branches(model))
    cams, sams = attention_layers(model)
    act_names = [n.replace('_gn', '_activation') for n in names]
    cam_in_names = [c.input.node.layer.name for c in cams]
    sam_in_names = [[t.node.layer.name for t in s.input] for s in sams]
    tensors = ([model.get_layer(n).output for n in act_names + cam_in_names]
               + [c.output for c in cams] + [s.output for s in sams])
    probe = tf.keras.Model(model.input, tensors)

    tot = {n: 0.0 for n in names}
    cnt = {n: 0 for n in names}
    att = {}
    max_err = 0.0
    nimg = 0
    n_skipped = 0
    for img, _ in val_data:
        outs = [tf.cast(o, tf.float32) for o in probe(img, training=False)]
        # float16 activations: skip images whose activations are non-finite (seen with mixed_float16 on CPU)
        # instead of letting one NaN/Inf poison the means.
        if not all(bool(tf.reduce_all(tf.math.is_finite(o))) for o in outs):
            n_skipped += 1
            continue
        k = len(act_names)
        acts = dict(zip(act_names, outs[:k]))
        cam_in = outs[k:k + len(cams)]
        cam_out = outs[k + len(cams):k + 2 * len(cams)]
        sam_out = outs[k + 2 * len(cams):]
        for n, an in zip(names, act_names):
            a = acts[an]
            tot[n] = tot[n] + a.numpy().astype(np.float64).sum(axis=(0, 1, 2))  # float64: no fp16 overflow
            cnt[n] += int(a.shape[1] * a.shape[2])
        for cam, x, y in zip(cams, cam_in, cam_out):
            xc, g = cam_gate(x, cam)
            key = f'cam_g/{cam.name}'
            att[key] = att.get(key, 0.0) + g.numpy().reshape(-1).astype(np.float64)
            max_err = max(max_err, float(tf.reduce_max(tf.abs(xc * g - tf.cast(y, tf.float32)))))
        for sam, ins, y in zip(sams, sam_in_names, sam_out):
            outs_j = []
            for j, (p, an) in enumerate(zip(sam_parts(sam), ins)):
                o, g, r = sam_block(acts[an], p)
                outs_j.append(o)
                att[f'sam_g/{sam.name}/{j}'] = att.get(f'sam_g/{sam.name}/{j}', 0.0) + float(tf.reduce_mean(g))
                att[f'sam_r/{sam.name}/{j}'] = att.get(f'sam_r/{sam.name}/{j}', 0.0) + float(tf.reduce_mean(r))
            max_err = max(max_err, float(tf.reduce_max(tf.abs(tf.concat(outs_j, -1) - tf.cast(y, tf.float32)))))
        nimg += 1
    stats = {'mu/' + n: (tot[n] / cnt[n]).astype(np.float32) for n in names}
    stats.update({k: np.asarray(v) / nimg for k, v in att.items()})
    stats['manual_attention_max_abs_err'] = np.array(max_err)
    stats['n_val_images'] = np.array(nimg)
    stats['n_val_skipped_nonfinite'] = np.array(n_skipped)
    stats['device'] = np.array(device_info()['device'])
    check_stats(stats, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **stats)
    return stats


def check_stats(stats, path):
    """Mean-ablation with a NaN mean makes every logit NaN, so argmax -> class 0 (all background). Fail loudly."""
    bad = [k for k, v in stats.items()
           if np.asarray(v).dtype.kind == 'f' and not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise ValueError(f'non-finite ablation stats in {path}: {bad}; delete the file to recompute it')


# ── variants ────────────────────────────────────────────────────────────────
def variant_specs(model_name, model, phase):
    br = branches(model)

    def sel(pred):
        return sorted(n for n, (i, lev, ps) in br.items() if pred(i, lev, ps))

    v = {
        'S0_full': {},
        'S1_unet_only': {'branches': sel(lambda i, l, ps: ps > 0 or ps < -1)},
        'S2_no_shallower_encoder': {'branches': sel(lambda i, l, ps: ps > 0)},
        'S3_no_same_level_skip': {'branches': sel(lambda i, l, ps: ps == 0)},
        'S4_no_previous_decoder': {'branches': sel(lambda i, l, ps: ps == -1)},
    }
    if model_name == 'cab2':
        v['S5_cab1_wiring'] = {'branches': sel(lambda i, l, ps: ps > 1 or ps < -1)}
        v['S6_no_extra_deeper'] = {'branches': sel(lambda i, l, ps: ps < -1)}
        v['S7_no_far_shallower'] = {'branches': sel(lambda i, l, ps: ps > 1)}
    if model_name == 'cab1':
        v['A1_cam_mean'] = {'cam_mean': True}
        v['A2_sam_mean'] = {'sam_mean': True}
        v['A3_cam_sam_mean'] = {'cam_mean': True, 'sam_mean': True}
        v['A4_sam_single_dilation'] = {'sam_single_dilation': True}
    if phase == 'screen':
        for n, (i, lev, ps) in sorted(br.items()):
            v[f'E_stage{i}_src{lev}_ps{ps:+d}'] = {'branches': [n]}
        v['AC_zero_h_v'] = {'zero_kernels': r'^[hv]_unet3plus_'}
        if model_name == 'cab1':
            v['AC_zero_diagonal'] = {'zero_kernels': r'^d[12]_unet3plus_'}
        else:
            v['AC_zero_square'] = {'zero_kernels': r'^unet3plus_(down_from\d_to\d|am_\d_am_conv|plain_up\d_conv_\w+)_0$'}
    return v


def apply_variant(model, spec, stats):
    """In-place weight edits; returns the model to evaluate (possibly a clone with replaced layers)."""
    for n in spec.get('branches', []):
        gn = model.get_layer(n)
        gamma, beta = gn.get_weights()
        mu = np.asarray(stats['mu/' + n], dtype=beta.dtype)
        assert np.all(np.isfinite(mu)), n
        gn.set_weights([np.zeros_like(gamma), mu])
    cams, sams = attention_layers(model)
    if spec.get('cam_mean'):
        for cam in cams:
            g = np.asarray(stats[f'cam_g/{cam.name}'])
            assert np.all(np.isfinite(g)), cam.name
            for fc in (cam.fc12, cam.fc22):
                fc.kernel.assign(np.zeros(fc.kernel.shape, np.float32))
            cam.conv.kernel.assign((cam.conv.kernel.numpy() * (2.0 * g)[None, None, None, :]).astype(np.float32))
    if spec.get('sam_mean'):
        for sam in sams:
            for j, p in enumerate(sam_layers(sam)):
                g = float(stats[f'sam_g/{sam.name}/{j}'])
                assert np.isfinite(g), (sam.name, j)
                for c in p['c'] + p['d']:
                    c.kernel.assign(np.zeros(c.kernel.shape, np.float32))
                p['conv'].kernel.assign((p['conv'].kernel.numpy() * (2.0 * g)).astype(np.float32))
    if spec.get('zero_kernels'):
        rx = re.compile(spec['zero_kernels'])
        hit = 0
        for l in model.layers:
            if rx.search(l.name) and isinstance(getattr(l, 'kernel', None), tf.Variable):
                l.kernel.assign(np.zeros(l.kernel.shape, np.float32))
                hit += 1
        assert hit > 0, spec
    if spec.get('sam_single_dilation'):
        assert all(np.isfinite(float(stats[f'sam_r/{s.name}/{j}'])) for s in sams for j in range(len(s.conv)))
    if spec.get('sam_single_dilation') or spec.get('sam_manual_identity'):
        mode = 'normal' if spec.get('sam_manual_identity') else 'single_dilation'
        repl = {s.name: SAM5SingleDilation(
            s, [stats[f'sam_r/{s.name}/{j}'] for j in range(len(s.conv))], mode=mode) for s in sams}
        return tf.keras.models.clone_model(model, clone_function=lambda l: repl.get(l.name, l))
    return model


# ── evaluation ──────────────────────────────────────────────────────────────
N_WARMUP = 5  # images excluded from timing (tf.function tracing, cuDNN autotune)


def evaluate(model, data):
    """Returns (cm, timing). timing = per-image forward pass + argmax, synced via .numpy(), warm-up excluded."""
    f = tf.function(lambda x: model(x, training=False))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    times = []
    for k, (img, lab) in enumerate(data):
        img = tf.convert_to_tensor(img)
        t = time.perf_counter()
        y = f(img)
        if isinstance(y, (list, tuple)):
            y = y[0]
        pred = tf.math.argmax(y[0, :, :, :NUM_CLASSES], axis=-1).numpy()
        if k >= N_WARMUP:
            times.append(time.perf_counter() - t)
        pred = pred.astype(np.int64).ravel()
        cm += np.bincount(lab * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    ms = np.array(times) * 1e3
    timing = dict(n_timed=int(ms.size), warmup=N_WARMUP,
                  ms_median=float(np.median(ms)) if ms.size else None,
                  ms_mean=float(ms.mean()) if ms.size else None,
                  ms_p90=float(np.percentile(ms, 90)) if ms.size else None)
    return cm, timing


def device_info():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return dict(device='CPU')
    name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')
    return dict(device=name, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                note='GPU may be shared with training jobs; timings are indicative')


def summary(cm):
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return dict(pixel_acc=tp.sum() / cm.sum(), mean_fg_iou=iou[1:].mean(), mean_fg_recall=rec[1:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['cab1', 'cab2'])
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--phase', default='screen', choices=['check', 'screen', 'main'])
    ap.add_argument('--max_test', type=int, default=None, help='default: 100 for screen, all for main')
    ap.add_argument('--max_val', type=int, default=100)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--out_dir', default=OUT_DIR)
    args = ap.parse_args()

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    d = fold_dirs(args.model)[args.fold]
    max_test = args.max_test if args.max_test is not None else (100 if args.phase == 'screen' else None)
    out_dir = os.path.join(args.out_dir, args.phase)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    model = tf.keras.models.load_model(d, compile=False)
    original = model.get_weights()
    print(f'[{args.model} f{args.fold}] loaded {d} in {time.time() - t0:.0f}s', flush=True)

    val = load_split('val', args.fold, args.max_val)
    stats = collect_stats(model, val, os.path.join(args.out_dir, 'stats', f'{args.model}_fold{args.fold}_val{args.max_val}.npz'))
    del val
    print(f'[{args.model} f{args.fold}] stats ready, manual attention max abs err = '
          f'{float(stats["manual_attention_max_abs_err"]):.2e}', flush=True)

    test = load_split('test', args.fold, max_test)
    specs = variant_specs(args.model, model, args.phase)
    if args.phase == 'check':
        specs = {'S0_full': {}}
        if args.model == 'cab1':
            specs['X_manual_sam_identity'] = {'sam_manual_identity': True}
    if args.variants:
        specs = {k: v for k, v in specs.items() if k in args.variants}

    for name, spec in specs.items():
        path = os.path.join(out_dir, f'{args.model}_fold{args.fold}_{name}.json')
        if os.path.exists(path):
            continue
        t = time.time()
        model.set_weights(original)
        m = apply_variant(model, spec, stats)
        cm, timing = evaluate(m, test)
        model.set_weights(original)
        rec = dict(model=args.model, fold=args.fold, variant=name, phase=args.phase, checkpoint=d,
                   n_test_images=len(test), spec={k: v for k, v in spec.items()},
                   cm=cm.tolist(), seconds=round(time.time() - t, 1), timing=timing,
                   **device_info(), **summary(cm))
        if name in ('S0_full', 'X_manual_sam_identity'):
            ref = np.load(REFERENCE[args.model] + '_fold_cms.npy')[args.fold]
            if len(test) == 400:
                rec['reference_cm_max_abs_diff'] = int(np.abs(ref - cm).max())
                rec['reference_cm_rel_diff'] = float(np.abs(ref - cm).sum() / ref.sum())
        with open(path + '.tmp', 'w') as f:
            json.dump(rec, f)
        os.replace(path + '.tmp', path)
        print(f'[{args.model} f{args.fold}] {name}: mIoU(fg)={rec["mean_fg_iou"]:.4f} '
              f'acc={rec["pixel_acc"]:.4f} ({rec["seconds"]:.0f}s, {timing["ms_median"] or 0:.1f} ms/img)'
              + (f' ref_diff={rec.get("reference_cm_rel_diff")}' if 'reference_cm_rel_diff' in rec else ''),
              flush=True)


if __name__ == '__main__':
    main()
