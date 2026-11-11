"""
Build TFRecords for cross-dataset (zero-shot) evaluation on R3D, CVC-FP and WAFFLE.

No augmentation. Same serialization/preprocessing as datasets/floorplans.py
(BGR uint8 image via cv2, longest side capped at 812 px, INTER_NEAREST), so the
records are read by floorplans.load_dataset() unchanged.

Label scheme (3 classes): 0 = bg, 1 = walls, 2 = openings (doors + windows).
Heatmaps: a single all-zero uint8 channel (unused for evaluation).

Outputs:
    data/tfrecords/r3d/r3d_test.tfrecords    (53 samples, official test split)
    data/tfrecords/r3d/r3d_train.tfrecords   (179 samples, official train split)
    data/tfrecords/cvc/cvc_all.tfrecords     (122 CVC-FP samples with SVG GT, data/cvc/ImagesGT)
    data/tfrecords/waffle/waffle_all.tfrecords (109 of the 110 WAFFLE benchmark plans, data/waffle/benchmark)
    data/tfrecords/waffle/waffle5_all.tfrecords (same plans, 5 classes bg/walls/doors/windows/interior,
        for the per-class comparison with the WAFFLE paper; see eval_waffle_perclass.py)

Usage:
    python kfold_patch/create_cross_tfrecords.py --dataset r3d
    python kfold_patch/create_cross_tfrecords.py --dataset cvc
    python kfold_patch/create_cross_tfrecords.py --dataset waffle
    python kfold_patch/create_cross_tfrecords.py --dataset waffle5
"""
import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

MAX_SIDE = 812


def scale(image, mask):
    s = min(MAX_SIDE / image.shape[0], MAX_SIDE / image.shape[1], 1.0)
    h, w = int(image.shape[0] * s), int(image.shape[1] * s)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def write_records(samples, out_path, n_classes=3):
    """samples: iterable of (name, loader) where loader() -> (bgr_image uint8, mask uint8)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts = np.zeros(n_classes, np.int64)
    with tf.io.TFRecordWriter(out_path) as writer:
        for name, loader in tqdm(samples, os.path.basename(out_path)):
            image, mask = loader()
            image, mask = scale(image, mask)
            h, w = mask.shape
            heat = np.zeros((h, w, 1), np.uint8)
            counts += np.bincount(mask.ravel(), minlength=n_classes)[:n_classes]
            feat = {'image': _bytes(image.astype(np.uint8).tobytes()),
                    'mask': _bytes(mask.astype(np.uint8).tobytes()),
                    'heatmaps': _bytes(heat.tobytes()),
                    'width': _int(w), 'height': _int(h)}
            writer.write(tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString())
    print(f'{out_path}: {len(samples)} samples, {os.path.getsize(out_path) / 1e6:.1f} MB, '
          f'pixel share per class = {np.round(counts / counts.sum() * 100, 2).tolist()} %')


def r3d_loader(base):
    def load():
        image = cv2.imread(base + '.jpg', cv2.IMREAD_COLOR)
        wall = cv2.imread(base + '_wall.png', cv2.IMREAD_GRAYSCALE)
        close = cv2.imread(base + '_close.png', cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(wall.shape, np.uint8)
        mask[wall > 127] = 1
        mask[close > 127] = 2
        assert image.shape[:2] == mask.shape, base
        return image, mask
    return load


def build_r3d(root, out_dir):
    for split in ['test', 'train']:
        bases = sorted(p[:-4] for p in glob.glob(os.path.join(root, split, '*.jpg')))
        write_records([(os.path.basename(b), r3d_loader(b)) for b in bases],
                      os.path.join(out_dir, f'r3d_{split}.tfrecords'))


# CVC-FP original release (data/cvc/ImagesGT): <stem>.png|jpg + <stem>_gt_<n>.svg with
# class-labelled polygons. Wall -> walls; Door, Window -> openings (drawn last, overriding
# walls as in R3D); Room, Separation, Parking, Text -> bg.
CVC_CLASSES = {'Wall': 1, 'Door': 2, 'Window': 2}
_SHIFT = 4  # sub-pixel precision for cv2.fillPoly


def cvc_svg_mask(svg_path, shape):
    import xml.etree.ElementTree as ET
    root = ET.parse(svg_path).getroot()
    polys = {1: [], 2: []}
    for el in root.iter():
        if el.tag.split('}')[-1] != 'polygon':
            continue
        cls = CVC_CLASSES.get(el.get('class', ''))
        if cls is None:
            continue
        pts = np.array([[float(v) for v in xy.split(',')] for xy in el.get('points', '').split()])
        if len(pts) >= 3:
            polys[cls].append(np.round(pts * (1 << _SHIFT)).astype(np.int32))
    mask = np.zeros(shape, np.uint8)
    for cls in (1, 2):
        if polys[cls]:
            cv2.fillPoly(mask, polys[cls], cls, lineType=cv2.LINE_8, shift=_SHIFT)
    return mask


def cvc_loader(img_path, svg_path):
    def load():
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # grayscale scans -> 3-channel BGR
        return image, cvc_svg_mask(svg_path, image.shape[:2])
    return load


def build_cvc(root, out_dir):
    samples, skipped = [], []
    for svg in sorted(glob.glob(os.path.join(root, 'ImagesGT', '*_gt_*.svg'))):
        stem = os.path.basename(svg).rsplit('_gt_', 1)[0]
        imgs = [p for ext in ('png', 'jpg') for p in glob.glob(os.path.join(root, 'ImagesGT', f'{stem}.{ext}'))]
        assert len(imgs) == 1, (svg, imgs)
        samples.append((stem, cvc_loader(imgs[0], svg)))
    for img in glob.glob(os.path.join(root, 'ImagesGT', '*.[pj][np]g')):
        stem = os.path.splitext(os.path.basename(img))[0]
        if not glob.glob(os.path.join(root, 'ImagesGT', f'{stem}_gt_*.svg')):
            skipped.append(stem)
    write_records(samples, os.path.join(out_dir, 'cvc_all.tfrecords'))
    print(f'Skipped (no SVG ground truth): {sorted(skipped)}')


# WAFFLE benchmark (data/waffle/benchmark): pngs/<stem>.png (RGBA, composited on white paper) +
# segmented_descrete_pngs/<stem>_seg_colors.png with exact colours: red = wall -> walls;
# blue = door, cyan = window -> openings; black = background, white = interior -> bg.
# StephansdomBauHistorie is skipped: its mask (660x455) does not match the image (648x424) and
# is visibly misaligned even after resizing.
WAFFLE_COLORS = {(0, 0, 255): 1, (255, 0, 0): 2, (255, 255, 0): 2}  # BGR
# 5-class variant matching the WAFFLE paper's Table 4: 0 bg, 1 walls, 2 doors, 3 windows, 4 interior
WAFFLE_COLORS5 = {(0, 0, 255): 1, (255, 0, 0): 2, (255, 255, 0): 3, (255, 255, 255): 4}
WAFFLE_SKIP = {'StephansdomBauHistorie'}


def waffle_loader(img_path, seg_path, colors=WAFFLE_COLORS):
    def load():
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if rgba.ndim == 2:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        if rgba.shape[2] == 4:
            a = rgba[..., 3:].astype(np.float32) / 255.
            image = np.round(rgba[..., :3] * a + 255. * (1. - a)).astype(np.uint8)
        else:
            image = rgba
        seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)
        assert image.shape[:2] == seg.shape[:2], img_path
        mask = np.zeros(seg.shape[:2], np.uint8)
        for bgr, cls in colors.items():
            mask[np.all(seg == bgr, axis=-1)] = cls
        return image, mask
    return load


def build_waffle(root, out_dir, five_class=False):
    samples = []
    for img in sorted(glob.glob(os.path.join(root, 'pngs', '*.png'))):
        stem = os.path.splitext(os.path.basename(img))[0]
        if stem in WAFFLE_SKIP:
            continue
        seg = os.path.join(root, 'segmented_descrete_pngs', f'{stem}_seg_colors.png')
        assert os.path.exists(seg), seg
        samples.append((stem, waffle_loader(img, seg, WAFFLE_COLORS5 if five_class else WAFFLE_COLORS)))
    if five_class:
        write_records(samples, os.path.join(out_dir, 'waffle5_all.tfrecords'), n_classes=5)
    else:
        write_records(samples, os.path.join(out_dir, 'waffle_all.tfrecords'))
    print(f'Skipped (mask/image misaligned): {sorted(WAFFLE_SKIP)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['r3d', 'cvc', 'waffle', 'waffle5'], required=True)
    args = ap.parse_args()
    if args.dataset == 'r3d':
        build_r3d('data/r3d', 'data/tfrecords/r3d')
    elif args.dataset == 'cvc':
        build_cvc('data/cvc', 'data/tfrecords/cvc')
    else:
        build_waffle('data/waffle/benchmark', 'data/tfrecords/waffle', five_class=args.dataset == 'waffle5')
