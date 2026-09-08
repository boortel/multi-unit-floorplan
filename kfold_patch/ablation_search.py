"""
Architecture ablation search for CAB1 / CAB2.

Runs a predefined set of architectural variants each for a single fold (fold 0),
keeping the cosine LR schedule active. Results are printed as a comparison table
and saved to results/ablation_<model>_<timestamp>.txt.

Usage (run each model on its own GPU in separate terminals):
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/ablation_search.py --model cab1
    CUDA_VISIBLE_DEVICES=1 python kfold_patch/ablation_search.py --model cab2

Optional flags:
    --variants baseline B3 no_hhdc  # run only specific variants
    --fold 0                         # which k-fold split to use (default: 0)
    --epochs 100                     # epochs per variant (default: 100)
"""
import argparse
import ctypes
import gc
import glob
import logging
import os
import site
import sys
import time

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── GPU preloader (must run before TF import) ────────────────────────────────
def _preload_gpu_libs():
    search_paths = ["/usr/lib/x86_64-linux-gnu"]
    if (cp := os.environ.get("CONDA_PREFIX")):
        search_paths.append(os.path.join(cp, "lib"))
    try:
        search_paths.extend(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))
    except Exception:
        pass
    for lib in ["libcuda.so.1", "libcudart.so.11.0", "libcudnn.so.8",
                "libcublas.so.11", "libcublasLt.so.11", "libcufft.so.10",
                "libcurand.so.10", "libcusolver.so.11", "libcusparse.so.11"]:
        for p in search_paths:
            full = os.path.join(p, lib)
            if os.path.exists(full):
                try:
                    ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break

_preload_gpu_libs()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
logging.disable(logging.WARNING)

import segmentation_models as sm
sm.set_framework("tf.keras")

from utils import Config, mkdir_or_exist
from kfold_patch.train_config_kfold import train  # noqa: E402

# ── Base configs (mirrors the eval_cab*.py files) ────────────────────────────
_COMMON = dict(
    local_rank=0,
    output_activation="Softmax",
    batch_norm=True,
    loss_functions=["asym_unified_focal_loss", "heatmap_regression_loss",
                    "adaptive_affinity_loss", "AutomaticWeightedLoss"],
    optimizer=dict(type="Adam", learning_rate=1e-4, lossScale=False),
    metrics=["CategoricalAccuracy"],
    checkpoint_weights_only=False,
    run_eagerly=False,
    verbose=2,
    dataset_name="CubiCasa5k",
    dataset_file="cubicasa5k",
    data_root="data/tfrecords/cubicasa5k",
    classes=["walls", "railings", "doors", "windows", "stairs_all"],
    heatmap_inds=[3, 4],
    train_buffer_size=400,
    data_reduction=None,
    baseline=False,
    deep_supervision=False,
    normalize=False,
    backbone_weights="imagenet",
    up_rates=None,
    # LR scheduling
    lr_scheduler="cosine-decay-warmup",
    lr_min=1e-6,
    warmup_epochs=5,
)

BASE_CAB1 = dict(
    **_COMMON,
    model_type="cab1",
    backbone="EfficientNetB2",
    filters=[32, 64, 128, 256, 512],
    aaf=[2, 4],
    hhdc=5,
    cam=3,
    batch_size=2,
)

BASE_CAB2 = dict(
    **_COMMON,
    model_type="cab2",
    backbone="EfficientNetB2",
    filters=[32, 64, 128, 256, 512],
    aaf=[2, 4],
    hhdc=5,
    cam=3,
    batch_size=2,
)

# ── Architecture variants ────────────────────────────────────────────────────
# Each dict overrides the base config. 'name' and 'desc' are metadata only.
VARIANTS = [
    # ── Baselines ──────────────────────────────────────────────────────────
    dict(name="baseline",     desc="Current best (B2, HHDC=5, CAM=3, AAF=[2,4])"),

    # ── Backbone ──────────────────────────────────────────────────────────
    dict(name="B0_small",     desc="EfficientNetB0, smaller filters",
         backbone="EfficientNetB0", filters=[16, 32, 64, 128, 256]),
    dict(name="B3",           desc="EfficientNetB3",
         backbone="EfficientNetB3"),
    dict(name="B4",           desc="EfficientNetB4",
         backbone="EfficientNetB4"),
    dict(name="B5",           desc="EfficientNetB5",
         backbone="EfficientNetB5"),
    dict(name="V2B3",         desc="EfficientNetV2B3",
         backbone="EfficientNetV2B3"),
    dict(name="V2S",          desc="EfficientNetV2S",
         backbone="EfficientNetV2S"),
    dict(name="V2M",          desc="EfficientNetV2M",
         backbone="EfficientNetV2M"),

    # ── Filter size ────────────────────────────────────────────────────────
    dict(name="large_filters", desc="Larger feature maps [64,128,256,512,1024]",
         filters=[64, 128, 256, 512, 1024]),

    # ── HHDC ablation ─────────────────────────────────────────────────────
    dict(name="no_hhdc",      desc="Remove HHDC module",    hhdc=False),
    dict(name="hhdc_3",       desc="HHDC kernel=3",          hhdc=3),
    dict(name="hhdc_7",       desc="HHDC kernel=7",          hhdc=7),

    # ── CAM ablation ──────────────────────────────────────────────────────
    dict(name="no_cam",       desc="Remove CAM module",      cam=False),
    dict(name="cam_1",        desc="CAM scale=1",            cam=1),
    dict(name="cam_5",        desc="CAM scale=5",            cam=5),

    # ── AAF ablation ──────────────────────────────────────────────────────
    dict(name="no_aaf",       desc="Remove AAF module",      aaf=[]),
    dict(name="aaf_248",      desc="AAF dilations=[2,4,8]",  aaf=[2, 4, 8]),
    dict(name="aaf_48",       desc="AAF dilations=[4,8]",    aaf=[4, 8]),
]

VARIANT_NAMES = [v["name"] for v in VARIANTS]


def build_config(base: dict, overrides: dict, fold: int, epochs: int,
                 work_dir: str, save_models: bool = False,
                 batch_size: int = None, patience: int = None,
                 data_reduction: int = None) -> Config:
    """Merge base + overrides into a Config object ready for train()."""
    cfg_dict = dict(**base)
    for k, v in overrides.items():
        if k not in ("name", "desc"):
            cfg_dict[k] = v

    # n_up_sample_block must match number of filters (+1 for VGG-style models)
    cfg_dict["n_up_sample_block"] = len(cfg_dict["filters"]) + 1

    cfg_dict["epochs"] = epochs
    cfg_dict["kFold"] = 10       # full 10-fold split (for consistent train/val)
    cfg_dict["fold"] = fold      # but only this one fold is evaluated
    cfg_dict["work_dir"] = os.path.abspath(work_dir)
    cfg_dict["hist"] = []

    cfg_dict["checkpoint_callback"] = save_models
    cfg_dict["tensorboard_callback"] = save_models

    if batch_size is not None:
        cfg_dict["batch_size"] = batch_size
    if patience is not None:
        cfg_dict["early_stopping_patience"] = patience
    if data_reduction is not None:
        cfg_dict["data_reduction"] = data_reduction

    return Config(cfg_dict=cfg_dict)


def run_ablation(model: str, selected: list[str], fold: int, epochs: int,
                 save_models: bool = False, batch_size: int = None,
                 patience: int = 20, data_reduction: int = None,
                 backbone: str = None):
    base = dict(BASE_CAB1 if model == "cab1" else BASE_CAB2)
    if backbone:
        base["backbone"] = backbone
    results = []

    print(f"\n{'='*60}")
    print(f"Ablation: {model.upper()}  |  fold={fold}  |  epochs={epochs}")
    print(f"Variants to run: {selected}")
    print(f"Save models: {save_models} | Batch size: {batch_size or base.get('batch_size', 2)} | EarlyStop patience: {patience}")
    print("=" * 60)

    for variant in VARIANTS:
        vname = variant["name"]
        if vname not in selected:
            continue

        print(f"\n{'─'*60}")
        print(f"Variant: {vname}  —  {variant['desc']}")
        print("─" * 60)

        exp_name = f"{model}_ablation_{vname}"
        cfg = build_config(base, variant, fold, epochs, work_dir=".",
                           save_models=save_models, batch_size=batch_size,
                           patience=patience, data_reduction=data_reduction)
        cfg.exp_name = exp_name

        tic = time.time()
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()

            history = train(cfg)
            elapsed = (time.time() - tic) / 60

            hist_data = history.history if hasattr(history, "history") else history
            best_loss = min(hist_data.get("val_loss", [float("inf")]))
            best_acc  = max(hist_data.get("val_categorical_accuracy", [0.0]))
            n_epochs  = len(hist_data.get("val_loss", []))

            results.append(dict(variant=vname, desc=variant["desc"],
                                val_loss=best_loss, val_acc=best_acc,
                                epochs=n_epochs, elapsed_min=elapsed))
            print(f"✓ {vname}: val_loss={best_loss:.4f}  val_acc={best_acc:.4f}"
                  f"  ({n_epochs} epochs, {elapsed:.0f} min)")
        except Exception as e:
            elapsed = (time.time() - tic) / 60
            results.append(dict(variant=vname, desc=variant["desc"],
                                val_loss=None, val_acc=None,
                                epochs=0, elapsed_min=elapsed, error=str(e)))
            print(f"✗ {vname} FAILED: {e}")
        finally:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()

    # ── Print results table ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"ABLATION RESULTS — {model.upper()} (fold {fold})")
    print("=" * 70)
    header = f"{'Variant':<16} {'Val Loss':<12} {'Val Acc':<12} {'Epochs':<8} {'Min':<8}  Description"
    print(header)
    print("-" * 70)
    for r in results:
        if r.get("val_loss") is not None:
            row = (f"{r['variant']:<16} {r['val_loss']:<12.4f} "
                   f"{r['val_acc']:<12.4f} {r['epochs']:<8} "
                   f"{r['elapsed_min']:<8.0f}  {r['desc']}")
        else:
            row = f"{r['variant']:<16} {'ERROR':<12} {'':<12} {'':<8} {r['elapsed_min']:<8.0f}  {r.get('error','')}"
        print(row)

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"results/ablation_{model}_fold{fold}_{timestr}.txt"
    with open(out_path, "w") as f:
        f.write(header + "\n" + "-" * 70 + "\n")
        for r in results:
            if r.get("val_loss") is not None:
                f.write(f"{r['variant']:<16} {r['val_loss']:<12.4f} "
                        f"{r['val_acc']:<12.4f} {r['epochs']:<8} "
                        f"{r['elapsed_min']:<8.0f}  {r['desc']}\n")
            else:
                f.write(f"{r['variant']:<16} ERROR  {r.get('error','')}\n")
    print(f"\nResults saved → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture ablation search")
    parser.add_argument("--model",    choices=["cab1", "cab2"], required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        help=f"Subset of variants to run. All: {VARIANT_NAMES}")
    parser.add_argument("--fold",   type=int, default=0,
                        help="K-fold split index to use (default: 0)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs per variant (default: 100)")
    parser.add_argument("--save_models", action="store_true", default=False,
                        help="Save model checkpoints and tensorboard logs to models/ (default: False)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: 2)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in epochs (default: 20)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override base backbone (default: EfficientNetB2, e.g. EfficientNetV2S)")
    parser.add_argument("--data_reduction", type=int, default=None,
                        help="Image downscale factor for fast exploratory runs, e.g. 2 (default: None)")
    args = parser.parse_args()

    unknown = set(args.variants) - set(VARIANT_NAMES)
    if unknown:
        print(f"Unknown variant(s): {unknown}")
        print(f"Available: {VARIANT_NAMES}")
        sys.exit(1)

    run_ablation(args.model, args.variants, args.fold, args.epochs,
                 save_models=args.save_models, batch_size=args.batch_size,
                 patience=args.patience, data_reduction=args.data_reduction,
                 backbone=args.backbone)
