"""
Tiled (sliding-window) inference for plans that are too large to resize to the model's usual ~812 px input,
and the MLSTRUCT-FP scoring helpers (bg / wall at native ground-truth resolution).

predict_tiled(): square tiles (multiple of 64), `overlap` fraction between neighbours, softmax maps averaged
with a separable Hann-like weight (never exactly 0, so borders of the plan are still covered). Images smaller
than a tile are padded with white paper (255), as `predict.pad_to` does. The 6 CubiCasa outputs are reduced
per tile with an aggregation matrix `agg` (6 x K), so memory is H x W x K floats.
"""
import numpy as np
import tensorflow as tf

# CubiCasa output order: bg, walls, railings, doors, windows, stairs_all
AGG_WALL = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]], np.float32)  # walls vs everything else
# Tiles per forward pass. CAB2 B4 (UNet3+ decoder, 32-512 filters at full-scale skips) crashes with
# CUDA_ERROR_ILLEGAL_ADDRESS at 8 x 768^2 and needs ~2x the memory of the V2M models, so it runs one tile at a time.
TILE_BATCH = {('cab2', 'EfficientNetB4'): 1}


def tile_batch(model_name, backbone, default=8):
    return TILE_BATCH.get((model_name, backbone), default)


def _window(n, floor=1e-2):
    w = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    return np.maximum(w, floor).astype(np.float32)


def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride)) + [length - tile]
    return sorted(set(s))


def _tile_fn(model, agg, tile, batch):
    """tf.function with a fixed (batch, tile, tile, 3) signature, cached on the model: one trace, and cuDNN
    autotunes a single shape (the last, partial batch is padded instead of changing the shape)."""
    key = (agg.tobytes(), agg.shape, tile, batch)
    cache = model.__dict__.setdefault('_tiled_fns', {})
    if key not in cache:
        agg_t = tf.constant(agg)

        @tf.function(input_signature=[tf.TensorSpec((batch, tile, tile, 3), tf.float32)])
        def fn(x):
            y = model(x, training=False)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return tf.einsum('bhwc,ck->bhwk', tf.cast(y[..., :agg.shape[0]], tf.float32), agg_t)
        cache[key] = fn
    return cache[key]


def predict_tiled(model, img, agg=AGG_WALL, tile=768, overlap=0.25, batch=8):
    """img: H x W x 3 (uint8/float, 0-255, BGR as in training). Returns H x W x K float32 probabilities."""
    assert tile % 64 == 0
    h0, w0 = img.shape[:2]
    H, W = max(h0, tile), max(w0, tile)
    x = np.full((H, W, 3), 255, np.float32)
    x[:h0, :w0] = img
    stride = max(64, int(round(tile * (1 - overlap))))
    win = np.outer(_window(tile), _window(tile))[..., None]
    k = agg.shape[1]
    acc = np.zeros((H, W, k), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    coords = [(r, c) for r in _starts(H, tile, stride) for c in _starts(W, tile, stride)]
    fn = _tile_fn(model, agg, tile, batch)
    xb = np.empty((batch, tile, tile, 3), np.float32)
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        for j, (r, c) in enumerate(cs):
            xb[j] = x[r:r + tile, c:c + tile]
        xb[len(cs):] = 255.
        y = fn(tf.constant(xb)).numpy()
        for (r, c), p in zip(cs, y):
            acc[r:r + tile, c:c + tile] += p * win
            wsum[r:r + tile, c:c + tile] += win
    return (acc / wsum)[:h0, :w0]


def wall_cm(prob_wall, gt):
    """2x2 confusion matrix (rows = GT bg/wall). prob_wall is resized (bilinear) to gt's shape if needed."""
    if prob_wall.shape != gt.shape:
        prob_wall = _resize_prob(prob_wall, gt.shape)
    p, t = prob_wall > 0.5, gt > 0
    tp = np.count_nonzero(p & t)
    n_p, n_t = np.count_nonzero(p), np.count_nonzero(t)
    fp, fn = n_p - tp, n_t - tp
    return np.array([[t.size - tp - fp - fn, fp], [fn, tp]], np.int64)


def _resize_prob(p, shape):
    import cv2
    return cv2.resize(p.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def wall_metrics(cm):
    """Metrics in both conventions: v1 Table 9 (wall Rec/F1/IoU) and Paek Table 3 (bg+wall mIoU, per-class acc,
    overall acc). Per-class accuracy in Paek = recall."""
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    rec = tp / np.maximum(tp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {'wall_rec': rec[1], 'wall_prec': prec[1], 'wall_f1': f1[1], 'wall_iou': iou[1], 'bg_iou': iou[0],
            'miou': iou.mean(), 'bg_acc': rec[0], 'overall_acc': tp.sum() / cm.sum()}
