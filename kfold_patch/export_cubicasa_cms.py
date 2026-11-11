"""
Re-run the official CubiCasa5k k-fold test evaluation for one result file and save the per-fold
6x6 confusion matrices next to it (results/official/test/<name>_fold_cms.npy).

The original evaluate_kfold.py run kept only the per-class metric table. The matrices are needed to
merge doors + windows into "openings" for the 3-class cross-dataset tables
(kfold_patch/summarize_all_datasets.py). Same checkpoints (taken from the result file), same
data and preprocessing (evaluate_kfold.evaluate_fold).

Usage:
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/export_cubicasa_cms.py \
        --source results/official/test/test_kfold_cab2_EfficientNetV2M_20261014-120852.txt
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import evaluate_kfold  # noqa: E402
from evaluate_cross_dataset import official_checkpoints  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    args = ap.parse_args()
    model = re.search(r'Model: (\S+)', open(args.source).read()).group(1)
    src, folds, backbone = official_checkpoints(model, args.source)
    cfg = evaluate_kfold.MODEL_CONFIGS[model]
    cms = []
    for f in sorted(folds):
        cms.append(evaluate_kfold.evaluate_fold(folds[f], cfg, fold_idx=f, split='test'))
        print(f'fold {f}: acc {np.diag(cms[-1]).sum() / cms[-1].sum():.4f}', flush=True)
    out = args.source[:-4] + '_fold_cms.npy'
    np.save(out, np.stack(cms))
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
