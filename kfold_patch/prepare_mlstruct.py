"""
Build data/mlstruct_proc/ from the raw MLSTRUCT-FP folder (data/mlstruct: fp.json + plan PNGs):
per-scale sample folders, meta.json and the 5-fold split (folds.json).

Usage:
    python kfold_patch/prepare_mlstruct.py                       # all scales
    python kfold_patch/prepare_mlstruct.py --scales s40 --overlay
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.mlstructfp_extract import create_folders_and_masks, load_index  # noqa: E402
import mlstruct_data as D  # noqa: E402


def _job(a):
    scale, ids, overlay, recreate = a
    return create_folders_and_masks(D.RAW, overlay, recreate, D.scale_dir(scale), D.ppm(scale), set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', nargs='+', default=D.SCALES)
    ap.add_argument('--overlay', action='store_true')
    ap.add_argument('--recreate', action='store_true')
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()

    ids = [os.path.splitext(k)[0] for k in load_index(D.RAW)]
    os.makedirs(D.ROOT, exist_ok=True)
    fpath = os.path.join(D.ROOT, 'folds.json')
    if not os.path.exists(fpath):
        with open(fpath, 'w') as fh:
            json.dump({'seed': 0, 'n_images': len(ids), 'protocol': 'split k: test=fold k, val=fold k+1, train=rest',
                       'folds': D.make_folds(ids)}, fh, indent=1)

    meta = {}
    for scale in args.scales:
        chunks = [ids[i::args.workers] for i in range(args.workers)]
        with ProcessPoolExecutor(args.workers) as ex:
            for m in ex.map(_job, [(scale, c, args.overlay, args.recreate) for c in chunks]):
                for iid, v in m.items():
                    meta.setdefault(iid, {k: v[k] for k in ('image', 'floors', 'scale', 'native_hw', 'n_rects')})
                    meta[iid].setdefault('hw', {})[scale] = v['hw']
        print(f'{scale}: {len(ids)} plans -> {D.scale_dir(scale)}', flush=True)

    mpath = os.path.join(D.ROOT, 'meta.json')
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        for iid, v in old.items():
            for s, hw in v.get('hw', {}).items():
                meta.setdefault(iid, v)['hw'].setdefault(s, hw)
    with open(mpath, 'w') as fh:
        json.dump(meta, fh)


if __name__ == '__main__':
    main()
