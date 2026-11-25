"""
MLSTRUCT-FP (Pizarro et al. 2023) extraction: one sample folder per unique plan image.

    <out_dir>/<image_id>/input.png  RGB plan, drawn strokes composited onto white paper (no inversion)
    <out_dir>/<image_id>/mask.png   uint8 label, 0 = bg, 1 = wall (all wall rects of every floor using this image)

The raw dataset is `fp.json` + PNGs in one folder. Images are RGBA/LA/P, with the drawing in the alpha
channel over a transparent background. 954 floors map onto 934 images (19 images are shared by two floors);
the rects of all floors that share an image are merged.

`target_ppm` resamples every plan to a fixed pixel density (px per metre, using the floor's `scale`).
The image uses INTER_AREA; the mask is rasterised directly at the target resolution (not resized),
so wall geometry stays exact. `target_ppm=None` keeps the native resolution.

Wall rects are in metres with y up: pixel = (x * scale, -y * scale).
"""
import argparse
import collections
import json
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
SHIFT = 4  # fillPoly sub-pixel precision (1/16 px)


def load_index(root_path):
    """{image_file: {'scale': px/m, 'floors': [floor ids], 'rects': [(xs, ys), ...]}} for every image."""
    with open(os.path.join(root_path, 'fp.json')) as fh:
        db = json.load(fh)
    index = collections.OrderedDict()
    floor_img = {}
    for fid, f in sorted(db['floor'].items(), key=lambda kv: kv[1]['image']):
        e = index.setdefault(f['image'], {'scale': f['scale'], 'floors': [], 'rects': []})
        assert abs(e['scale'] - f['scale']) < 1e-6, f'floors sharing {f["image"]} have different scales'
        e['floors'].append(int(fid))
        floor_img[int(fid)] = f['image']
    for r in db['rect'].values():
        index[floor_img[r['floorID']]]['rects'].append((r['x'], r['y']))
    return index


def load_plan(path):
    """RGB uint8 plan with the alpha-drawn strokes composited onto white."""
    im = Image.open(path)
    im = im.convert('RGBA') if im.mode != 'LA' else im.convert('LA')
    a = np.asarray(im).astype(np.float32)
    alpha = a[..., -1:] / 255.
    rgb = a[..., :1].repeat(3, -1) if a.shape[-1] == 2 else a[..., :3]
    out = rgb * alpha + 255. * (1. - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def rasterise(rects, scale, shape, factor=1.0):
    """Wall mask (0/1) of `shape` (h, w); `factor` = output px per native px."""
    mask = np.zeros(shape, np.uint8)
    s = scale * factor * (1 << SHIFT)
    polys = [np.round(np.stack([np.asarray(xs) * s, -np.asarray(ys) * s], -1)).astype(np.int32) for xs, ys in rects]
    cv2.fillPoly(mask, polys, 1, lineType=cv2.LINE_8, shift=SHIFT)
    return mask


def create_folders_and_masks(root_path, overlay=False, recreate=False, out_dir=None, target_ppm=None, ids=None):
    """Write <out_dir>/<image_id>/{input.png, mask.png}; returns {image_id: meta}."""
    index = load_index(root_path)
    out_dir = out_dir or os.path.join(root_path, 'MLSTRUCT-FP_v1')
    meta = {}
    for img_file, e in tqdm(index.items(), 'MLSTRUCT-FP'):
        iid = os.path.splitext(img_file)[0]
        if ids is not None and iid not in ids:
            continue
        sub = os.path.join(out_dir, iid)
        os.makedirs(sub, exist_ok=True)
        f_in, f_mask = os.path.join(sub, 'input.png'), os.path.join(sub, 'mask.png')
        w0, h0 = Image.open(os.path.join(root_path, img_file)).size
        factor = 1.0 if target_ppm is None else target_ppm / e['scale']
        w, h = int(round(w0 * factor)), int(round(h0 * factor))
        meta[iid] = {'image': img_file, 'floors': e['floors'], 'scale': e['scale'], 'native_hw': [h0, w0],
                     'hw': [h, w], 'factor': factor, 'n_rects': len(e['rects'])}
        if os.path.exists(f_in) and os.path.exists(f_mask) and not recreate:
            continue
        img = load_plan(os.path.join(root_path, img_file))
        if factor != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mask = rasterise(e['rects'], e['scale'], (h, w), factor)
        cv2.imwrite(f_in, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f_mask, mask)
        if overlay:
            ov = img.copy()
            ov[mask == 1] = (0.4 * ov[mask == 1] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            cv2.imwrite(os.path.join(sub, 'overlay.jpg'), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MLSTRUCT-FP into per-plan folders with input and wall mask')
    parser.add_argument('--root_path', default='data/mlstruct', help='Folder with fp.json and the plan PNGs')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--target_ppm', type=float, default=None, help='Resample to this many px per metre')
    parser.add_argument('--overlay', action='store_true', help='Create overlay to show annotation')
    parser.add_argument('--recreate', action='store_true', help='Recreate flag.')
    args = parser.parse_args()
    create_folders_and_masks(args.root_path, args.overlay, args.recreate, args.out_dir, args.target_ppm)
