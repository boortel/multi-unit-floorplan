"""
Asymmetric-convolution (AC) block analysis of the CAB1/CAB2 champions, weights only (no data, no retraining).

For every decoder AC block (parallel square / horizontal 1x3 / vertical 3x1 / diagonal 3x3 kernels summed
before GroupNorm):
  1. Structural re-parameterisation: the branches are fused into one 3x3 kernel and the fused conv is
     checked against the sum of branch convs on a random input. If they agree, AC adds no capacity at
     inference time; its effect can only come from training.
  2. Branch energy: share of the fused kernel's squared norm contributed by each branch, and the share of
     the fused kernel's energy on the centre cross vs. the corners.

Writes results/ablation_inference/ac_kernels.json and prints a summary.
"""
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from kfold_patch.ablation_inference import fold_dirs  # noqa: E402

PREFIXES = {'': 'square', 'h_': 'horizontal', 'v_': 'vertical', 'd1_': 'diagonal', 'd2_': 'anti-diagonal'}
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def pad3(k):
    """Place a (kh, kw, cin, cout) kernel into a 3x3 kernel (same padding, centred)."""
    out = np.zeros((3, 3) + k.shape[2:], k.dtype)
    oh, ow = (3 - k.shape[0]) // 2, (3 - k.shape[1]) // 2
    out[oh:oh + k.shape[0], ow:ow + k.shape[1]] = k
    return out


def blocks(model):
    names = {l.name for l in model.layers}
    bases = sorted({re.sub(r'^(h_|v_|d1_|d2_)', '', n) for n in names
                    if re.match(r'^(h_|v_)unet3plus_', n)})
    for b in bases:
        yield b, {PREFIXES[p]: model.get_layer(p + b) for p in PREFIXES if (p + b) in names}


def analyse(model):
    res = []
    rng = np.random.default_rng(0)
    for base, br in blocks(model):
        ks = {t: l.kernel.numpy().astype(np.float64) for t, l in br.items()}
        fused = sum(pad3(k) for k in ks.values())
        x = rng.standard_normal((1, 24, 24, fused.shape[2])).astype(np.float32)
        y_sum = sum(tf.nn.conv2d(x, k.astype(np.float32), 1, 'SAME') for k in ks.values())
        y_fused = tf.nn.conv2d(x, fused.astype(np.float32), 1, 'SAME')
        err = float(tf.reduce_max(tf.abs(y_sum - y_fused)) / (tf.reduce_max(tf.abs(y_sum)) + 1e-12))
        tot = float((fused ** 2).sum())
        diag_offdiag = 0.0
        for t in ('diagonal', 'anti-diagonal'):
            if t in ks:
                mask = np.eye(3, dtype=bool) if t == 'diagonal' else np.fliplr(np.eye(3, dtype=bool))
                diag_offdiag = max(diag_offdiag, float(np.abs(ks[t][~mask]).max()))
        res.append(dict(block=base, branches=sorted(ks), fused_rel_err=err,
                        branch_energy={t: float((pad3(k) ** 2).sum()) / tot for t, k in ks.items()},
                        cross_energy=float((fused[CROSS] ** 2).sum()) / tot,
                        diag_offdiag_max=diag_offdiag))
    return res


def main():
    out = {}
    for model_name in ('cab1', 'cab2'):
        out[model_name] = {}
        for fold, d in sorted(fold_dirs(model_name).items()):
            m = tf.keras.models.load_model(d, compile=False)
            out[model_name][fold] = analyse(m)
            tf.keras.backend.clear_session()
            r = out[model_name][fold]
            print(f'{model_name} fold {fold}: {len(r)} AC blocks, max fused rel err '
                  f'{max(b["fused_rel_err"] for b in r):.1e}', flush=True)
    os.makedirs('results/ablation_inference', exist_ok=True)
    with open('results/ablation_inference/ac_kernels.json', 'w') as f:
        json.dump(out, f, indent=1)
    for model_name, folds in out.items():
        allb = [b for r in folds.values() for b in r]
        types = sorted({t for b in allb for t in b['branches']})
        print(f'\n{model_name}: {len(allb)} block instances over {len(folds)} folds')
        print('  max fused-vs-sum relative error:', f'{max(b["fused_rel_err"] for b in allb):.1e}')
        for t in types:
            v = [b['branch_energy'][t] for b in allb if t in b['branch_energy']]
            print(f'  {t:14s} energy share (alone / fused): mean {np.mean(v):.3f}  min {np.min(v):.3f}  max {np.max(v):.3f}')
        print(f'  centre-cross share of fused kernel energy: mean {np.mean([b["cross_energy"] for b in allb]):.3f}')
        if any('diagonal' in b['branches'] for b in allb):
            print(f'  max |off-diagonal weight| in diagonal kernels: {max(b["diag_offdiag_max"] for b in allb):.1e}')


if __name__ == '__main__':
    main()
