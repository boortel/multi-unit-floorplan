"""
MLSTRUCT-FP data layout and 5-fold CV split, shared by prepare / evaluate / fine-tune.

    data/mlstruct_proc/<scale>/<image_id>/{input.png, mask.png}   scale in {native, s30, s40, s50, s60}
    data/mlstruct_proc/meta.json    per image: scale (px/m), native size, floors
    data/mlstruct_proc/folds.json   5 folds of image ids (seed 0), same protocol as CVC/WAFFLE:
                                    split k -> test = fold k, val = fold k+1, train = the other 3 (60/20/20)
"""
import json
import os

import cv2
import numpy as np

ROOT = 'data/mlstruct_proc'
RAW = 'data/mlstruct'
N_FOLDS = 5
SCALES = ['native', 's30', 's40', 's50', 's60']
PRIMARY_SCALE = 's40'  # 40 px/m: 0.2 m structural walls -> 8 px, the median CubiCasa wall thickness at 812 px


def ppm(scale):
    return None if scale == 'native' else float(scale[1:])


def scale_dir(scale):
    return os.path.join(ROOT, scale)


def make_folds(ids, seed=0):
    ids = sorted(ids)
    perm = np.random.RandomState(seed).permutation(len(ids))
    return [sorted(ids[i] for i in f) for f in np.array_split(perm, N_FOLDS)]


def folds():
    with open(os.path.join(ROOT, 'folds.json')) as fh:
        return json.load(fh)['folds']


def meta():
    with open(os.path.join(ROOT, 'meta.json')) as fh:
        return json.load(fh)


def split_ids(split):
    f = folds()
    test, val = f[split], f[(split + 1) % N_FOLDS]
    train = sorted(i for k in range(N_FOLDS) if k not in (split, (split + 1) % N_FOLDS) for i in f[k])
    return {'train': train, 'val': val, 'test': test}


def all_ids():
    return sorted(i for f in folds() for i in f)


def load_input(scale, iid):
    """BGR uint8 (as cv2.imread; the CubiCasa models were trained on cv2-loaded BGR images)."""
    return cv2.imread(os.path.join(scale_dir(scale), iid, 'input.png'), cv2.IMREAD_COLOR)


def load_mask(scale, iid):
    return cv2.imread(os.path.join(scale_dir(scale), iid, 'mask.png'), cv2.IMREAD_GRAYSCALE)
