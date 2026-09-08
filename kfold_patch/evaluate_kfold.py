"""
K-Fold test set evaluation script.

Evaluates all fold models for each model type and reports per-fold
metrics + aggregated confusion matrix across all folds.

The script discovers fold models by scanning the models/ directory for
directories matching the model name prefix and finding the latest one
that contains a valid saved_model.pb for each fold index.

Usage:
    CUDA_VISIBLE_DEVICES=3 python kfold_patch/evaluate_kfold.py

Or specify specific model prefixes:
    CUDA_VISIBLE_DEVICES=3 python kfold_patch/evaluate_kfold.py \\
        --models cab1 cab2 cubicasa5k zeng

The results are saved to the results/ directory.
"""
import ctypes
import glob
import logging
import os
import site
import sys
import time
import argparse

# Add project root to sys.path so local packages (segmentation_models, datasets,
# training, utils) are importable regardless of where the script is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from tabulate import tabulate


# Preload NVIDIA GPU libraries before TensorFlow import
def _preload_gpu_libs():
    search_paths = ["/usr/lib/x86_64-linux-gnu"]
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, "lib"))
    try:
        search_paths.extend(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))
    except Exception:
        pass
    lib_names = [
        "libcuda.so.1", "libcudart.so.11.0", "libcudnn.so.8",
        "libcublas.so.11", "libcublasLt.so.11", "libcufft.so.10",
        "libcurand.so.10", "libcusolver.so.11", "libcusparse.so.11",
    ]
    for lib_name in lib_names:
        for path in search_paths:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_preload_gpu_libs()

import tensorflow as tf
import segmentation_models as sm
from datasets import floorplans
from training.metrics import cm_metrics
import utils

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Map of model type prefix -> (dataset_file, data_root, normalize, n_up_sample_block, classes)
MODEL_CONFIGS = {
    'cab1': {
        'dataset_file': 'cubicasa5k',
        'data_root': 'data/tfrecords/cubicasa5k',
        'normalize': False,
        'n_up_sample_block': 6,
        'classes': ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all'],
    },
    'cab2': {
        'dataset_file': 'cubicasa5k',
        'data_root': 'data/tfrecords/cubicasa5k',
        'normalize': False,
        'n_up_sample_block': 6,
        'classes': ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all'],
    },
    'cubicasa5k': {
        'dataset_file': 'cubicasa5k',
        'data_root': 'data/tfrecords/cubicasa5k',
        'normalize': True,
        'n_up_sample_block': 6,
        'classes': ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all'],
    },
    'zeng': {
        'dataset_file': 'cubicasa5k',
        'data_root': 'data/tfrecords/cubicasa5k',
        'normalize': True,
        'n_up_sample_block': 6,
        'classes': ['bg', 'walls', 'railings', 'doors', 'windows', 'stairs_all'],
    },
}


def extract_timestamp(dir_name):
    base = os.path.basename(dir_name)
    parts = base.split('_')
    if parts:
        ts = parts[-1]
        if '-' in ts and len(ts) == 15 and ts.replace('-', '').isdigit():
            return ts
    try:
        return str(os.path.getmtime(dir_name))
    except Exception:
        return dir_name


def extract_backbone_name(dir_name):
    known = ['EfficientNetB4', 'EfficientNetV2S', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB0', 'VGG16', 'vgg16']
    base = os.path.basename(dir_name)
    for b in known:
        if b.lower() in base.lower():
            return b
    parts = base.split('_')
    return parts[3] if len(parts) > 3 else 'Unknown'


def find_fold_model_dirs(model_prefix, k_fold=10, backbone=None):
    """
    Scans models/ to find the latest valid saved_model.pb for each fold index.
    Optionally filters by backbone name substring.
    Returns a dict: {fold_index: path_to_saved_model_dir}
    """
    # Find all parent dirs matching this model prefix
    all_run_dirs = glob.glob(os.path.join(MODELS_DIR, f'{model_prefix}_*'))
    if backbone:
        all_run_dirs = [d for d in all_run_dirs if backbone.lower() in os.path.basename(d).lower()]

    # Sort by embedded timestamp/mtime (newest last)
    all_run_dirs = sorted(all_run_dirs, key=lambda d: extract_timestamp(d))

    fold_dirs = {}
    for run_dir in all_run_dirs:
        if not os.path.isdir(run_dir):
            continue
        for entry in os.listdir(run_dir):
            if not entry.isdigit():
                continue
            fold_idx = int(entry)
            fold_model_path = os.path.join(run_dir, entry)
            saved_model = os.path.join(fold_model_path, 'saved_model.pb')
            if os.path.exists(saved_model):
                # Always prefer the latest (sorted order, so later overwrites earlier)
                fold_dirs[fold_idx] = fold_model_path

    return fold_dirs


def evaluate_fold(fold_dir, model_cfg, fold_idx=0, split='test'):
    """Evaluate a single fold model on the test or val set. Returns the confusion matrix."""
    sm.set_framework('tf.keras')
    classes = model_cfg['classes']
    num_classes = len(classes)

    print(f"    Loading: {fold_dir}")
    model = tf.keras.models.load_model(fold_dir, compile=False)

    if split == 'val':
        val_path = os.path.join(model_cfg['data_root'], f"{model_cfg['dataset_file']}_fold_{fold_idx}")
        dataset = floorplans.load_dataset(
            val_path,
            normalize=model_cfg['normalize'],
            classes=classes,
            n_upsample=model_cfg['n_up_sample_block']
        )
    else:
        dataset = floorplans.load_test_data(
            classes,
            model_cfg['dataset_file'],
            normalize=model_cfg['normalize'],
            base_dir=model_cfg['data_root'],
            n_upsample=model_cfg['n_up_sample_block']
        )

    # Manual prediction loop — model.evaluate() cannot correctly handle the
    # non-scalar CM metric (Keras reduces it to a scalar internally).
    cm_accum = np.zeros((num_classes, num_classes), dtype=np.int64)
    for img, mask in dataset.batch(1):
        y_pred = model(img, training=False)
        # Handle tuple outputs (e.g. models that return (seg, heatmap))
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        y_true_idx = tf.math.argmax(mask[0, :, :, :num_classes], axis=-1).numpy().flatten()
        y_pred_idx = tf.math.argmax(y_pred[0, :, :, :num_classes], axis=-1).numpy().flatten()
        cm_batch = np.bincount(
            y_true_idx * num_classes + y_pred_idx,
            minlength=num_classes * num_classes
        ).reshape((num_classes, num_classes))
        cm_accum += cm_batch

    tf.keras.backend.clear_session()
    return cm_accum


def print_and_save_results(cm_agg, fold_accs, model_prefix, model_cfg, metadata=None, split='test'):
    classes = model_cfg['classes']
    columns = cm_metrics(cm_agg)

    table = []
    table.append(['Accuracy', np.diag(cm_agg).sum() / cm_agg.sum()])
    cm_nobg = np.copy(cm_agg)
    cm_nobg[0] = 0
    table.append(['Accuracy no bg', np.diag(cm_nobg).sum() / cm_nobg.sum()])
    for i, c in enumerate(classes):
        table.append(
            [c] + [column[i] for column in columns[:-1]] + [
                np.array([metrics[i] for metrics in columns[-1]]) / max(
                    sum([metrics[i] for metrics in columns[-1]]), 1e-9)])
    table.append(['Mean'] + [np.nanmean(column) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1], axis=1)) / np.sum(columns[-1], axis=1).sum()])
    table.append(['Mean no bg'] + [np.nanmean(column[1:]) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1][1:], axis=1)) / np.sum(columns[-1][1:], axis=1).sum()])
    headers = ['Class', 'Class Acc', 'Recall', 'Precision', 'F1', 'IoU', 'fwRecall', 'fwIoU', 'TP,FP,TN,FN']

    backbone_str = metadata.get('backbone', '') if metadata else ''
    exp_name_str = metadata.get('exp_name', '') if metadata else ''

    print(f"\n  Aggregated {split} set results ({model_prefix} {backbone_str}, {len(fold_accs)} folds):")
    print(f"  Per-fold accuracy: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"  Mean {split} accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(tabulate(table, headers=headers, floatfmt=".4f"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"_{backbone_str}" if backbone_str else ""
    results_path = os.path.join(RESULTS_DIR, f'{split}_kfold_{model_prefix}{suffix}_{timestr}.txt')
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_prefix}\n")
        if backbone_str:
            f.write(f"Backbone: {backbone_str}\n")
        if exp_name_str:
            f.write(f"Experiment Name: {exp_name_str}\n")
        f.write(f"Dataset: {model_cfg['dataset_file']}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Folds evaluated: {len(fold_accs)}\n")
        f.write(f"Per-fold accuracy: {[f'{a:.4f}' for a in fold_accs]}\n")
        f.write(f"Mean {split} accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}\n")
        if metadata and 'fold_dirs' in metadata:
            f.write("\nFold Checkpoint Sources:\n")
            for f_idx, f_path in sorted(metadata['fold_dirs'].items()):
                f.write(f"  Fold {f_idx}: {f_path}\n")
        f.write("\n")
        f.write(tabulate(table, headers=headers, floatfmt=".4f"))
    print(f"  Results saved to: {results_path}")
    return results_path


def _evaluate_model_split(model_prefix, k_fold=10, backbone=None, split='test'):
    print(f"\n{'='*60}")
    print(f"Model: {model_prefix} (Backbone: {backbone or 'auto-detect latest'}) | Split: {split}")
    print('='*60)

    if model_prefix not in MODEL_CONFIGS:
        print(f"  ERROR: Unknown model prefix '{model_prefix}'. Add it to MODEL_CONFIGS.")
        return

    model_cfg = MODEL_CONFIGS[model_prefix]
    fold_dirs = find_fold_model_dirs(model_prefix, k_fold, backbone=backbone)

    if not fold_dirs:
        bb_msg = f" with backbone '{backbone}'" if backbone else ""
        print(f"  ERROR: No saved fold models found for prefix '{model_prefix}'{bb_msg}")
        return

    # Extract metadata from discovered fold directories
    sample_fold_dir = fold_dirs[sorted(fold_dirs.keys())[0]]
    parent_dir_name = os.path.basename(os.path.dirname(sample_fold_dir))
    detected_backbone = extract_backbone_name(parent_dir_name)
    parts = parent_dir_name.split('_')
    exp_name = parts[1] if len(parts) > 1 else 'Unknown'

    metadata = {
        'backbone': detected_backbone,
        'exp_name': exp_name,
        'fold_dirs': fold_dirs,
    }

    print(f"  Found {len(fold_dirs)} fold models for {model_prefix} ({detected_backbone}): folds {sorted(fold_dirs.keys())}")
    if len(fold_dirs) < k_fold:
        print(f"  WARNING: Expected {k_fold} folds but only found {len(fold_dirs)}!")

    cms = []
    fold_accs = []
    for fold_idx in sorted(fold_dirs.keys()):
        print(f"\n  Fold {fold_idx} ({split}):")
        tic = time.time()
        cm = evaluate_fold(fold_dirs[fold_idx], model_cfg, fold_idx=fold_idx, split=split)
        toc = time.time()
        acc = np.diag(cm).sum() / cm.sum()
        fold_accs.append(acc)
        cms.append(cm)
        print(f"    Accuracy: {acc:.4f} (evaluated in {(toc-tic)/60:.1f} min)")

    cm_agg = np.sum(cms, axis=0)
    print_and_save_results(cm_agg, fold_accs, model_prefix, model_cfg, metadata=metadata, split=split)


def evaluate_model(model_prefix, k_fold=10, backbone=None, split='test'):
    splits = ['val', 'test'] if split == 'both' else [split]
    for s in splits:
        _evaluate_model_split(model_prefix, k_fold=k_fold, backbone=backbone, split=s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-Fold evaluation')
    parser.add_argument('--models', nargs='+', default=list(MODEL_CONFIGS.keys()),
                        help=f'Model prefixes to evaluate. Default: all ({list(MODEL_CONFIGS.keys())})')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Filter models by backbone name (e.g. EfficientNetB4, EfficientNetV2S, EfficientNetB2, VGG16)')
    parser.add_argument('--k_fold', type=int, default=10, help='Number of folds (default: 10)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'both'],
                        help='Dataset split to evaluate: test, val (out-of-fold), or both (default: test)')
    args = parser.parse_args()

    for model_prefix in args.models:
        evaluate_model(model_prefix, args.k_fold, backbone=args.backbone, split=args.split)

    print("\n============== Evaluation Complete ==============")
