"""
Optuna hyperparameter optimisation for CAB1 / CAB2.

Searches over backbone, filters, HHDC, CAM, AAF, and LR-schedule params
using a single fold (fold 0) as the validation set.  Trials are pruned
early using MedianPruner so bad configs are stopped after ~20-30 epochs.
The study is persisted in an SQLite file so it survives container restarts.

Usage (run each model on its own GPU):
    CUDA_VISIBLE_DEVICES=0 python kfold_patch/optuna_hpo.py --model cab1 --trials 50
    CUDA_VISIBLE_DEVICES=1 python kfold_patch/optuna_hpo.py --model cab2 --trials 50

The study can be inspected / resumed at any time:
    python kfold_patch/optuna_hpo.py --model cab1 --trials 50  # just re-run; it resumes
    optuna-dashboard sqlite:///optuna_cab1.db                  # web UI (pip install optuna-dashboard)
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

import optuna
import segmentation_models as sm
sm.set_framework("tf.keras")
import tensorflow as tf
from utils import Config, mkdir_or_exist
from kfold_patch.train_config_kfold import train  # noqa: E402

# ── Optuna pruning callback ──────────────────────────────────────────────────
class OptunaPruneCallback(tf.keras.callbacks.Callback):
    """Reports val_loss to Optuna each epoch and raises TrialPruned when requested."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")


# ── Base config (shared between cab1 and cab2) ───────────────────────────────
_COMMON_DICT = dict(
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
    kFold=10,   # use standard 10-fold split for consistent train/val sets
    fold=0,     # always evaluate on fold 0
    hist=[],
)


def suggest_params(trial: optuna.Trial, model: str, fixed_backbone: str = None) -> dict:
    """Define the hyperparameter search space and sample from it."""

    if fixed_backbone:
        backbone = fixed_backbone
    else:
        backbone = trial.suggest_categorical(
            "backbone", ["EfficientNetB0", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5", "EfficientNetV2B3", "EfficientNetV2S", "EfficientNetV2M"])

    filter_preset = trial.suggest_categorical(
        "filter_preset", ["small", "base", "large"])
    filters = {
        "small": [16, 32, 64, 128, 256],
        "base":  [32, 64, 128, 256, 512],
        "large": [64, 128, 256, 512, 1024],
    }[filter_preset]

    hhdc = trial.suggest_categorical("hhdc", [False, 3, 5, 7])
    cam  = trial.suggest_categorical("cam",  [False, 1, 3, 5])

    aaf_preset = trial.suggest_categorical("aaf_preset", ["none", "[2,4]", "[4,8]", "[2,4,8]"])
    aaf = {
        "none":    [],
        "[2,4]":   [2, 4],
        "[4,8]":   [4, 8],
        "[2,4,8]": [2, 4, 8],
    }[aaf_preset]

    lr_scheduler = trial.suggest_categorical(
        "lr_scheduler", ["cosine-decay-warmup", "cosine-decay", "reduce-lr-on-plateau"])
    lr_min        = trial.suggest_float("lr_min", 1e-7, 1e-5, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 10)

    return dict(
        model_type=model,
        backbone=backbone,
        filters=filters,
        n_up_sample_block=len(filters) + 1,
        aaf=aaf,
        hhdc=hhdc,
        cam=cam,
        batch_size=2,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        warmup_epochs=warmup_epochs,
    )


# ── Objective ────────────────────────────────────────────────────────────────
def make_objective(model: str, epochs: int, work_dir: str, fixed_backbone: str = None):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model, fixed_backbone=fixed_backbone)

        # Build config
        cfg_dict = dict(**_COMMON_DICT, **params)
        cfg_dict["exp_name"] = f"{model}_hpo_trial{trial.number}"
        cfg_dict["epochs"]   = epochs
        cfg_dict["work_dir"] = os.path.abspath(work_dir)
        cfg = Config(cfg_dict=cfg_dict)

        prune_cb = OptunaPruneCallback(trial, monitor="val_loss")

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
            history = train(cfg, extra_callbacks=[prune_cb])
            hist_data = history.history if hasattr(history, "history") else history
            best_val_loss = min(hist_data.get("val_loss", [float("inf")]))
            return best_val_loss
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")
        finally:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for CAB1/CAB2")
    parser.add_argument("--model",   choices=["cab1", "cab2"], required=True)
    parser.add_argument("--trials",  type=int, default=50,
                        help="Total number of Optuna trials (default: 50)")
    parser.add_argument("--epochs",  type=int, default=100,
                        help="Max epochs per trial (default: 100)")
    parser.add_argument("--fold",    type=int, default=0,
                        help="K-fold split to use (default: 0)")
    parser.add_argument("--backbone", default=None,
                        help="Fix backbone (e.g. EfficientNetV2S) instead of searching over backbones")
    parser.add_argument("--db",      default=None,
                        help="SQLite DB path (default: optuna_<model>.db)")
    parser.add_argument("--study",   default=None,
                        help="Study name (default: hpo_<model>)")
    args = parser.parse_args()

    db_path   = args.db    or f"optuna_{args.model}.db"
    study_name = args.study or f"hpo_{args.model}"
    storage   = f"sqlite:///{db_path}"

    _COMMON_DICT["fold"] = args.fold

    # MedianPruner: prune if trial is worse than median of completed trials
    # Start pruning after 5 completed trials, check from epoch 15 onwards
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=15, interval_steps=5)

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,   # resume if study already exists in DB
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, args.trials - n_completed)

    print(f"\nStudy: {study_name}")
    print(f"DB:    {db_path}")
    print(f"Model: {args.model}  |  fold={args.fold}  |  epochs={args.epochs}")
    if args.backbone:
        print(f"Fixed Backbone: {args.backbone}")
    print(f"Completed trials: {n_completed}  |  Remaining: {n_remaining}")
    if n_completed > 0:
        print(f"Best so far: val_loss={study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    if n_remaining == 0:
        print("All requested trials already completed.")
    else:
        objective = make_objective(args.model, args.epochs, work_dir=".", fixed_backbone=args.backbone)
        study.optimize(objective, n_trials=n_remaining,
                       catch=(Exception,))

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"HPO COMPLETE — {args.model.upper()}")
    print("=" * 60)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    print(f"\nTop 10 trials (val_loss):")
    print(f"{'Rank':<6} {'Trial':<8} {'val_loss':<12}  Params")
    print("-" * 60)
    for rank, t in enumerate(completed[:10], 1):
        print(f"{rank:<6} {t.number:<8} {t.value:<12.4f}  {t.params}")

    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  val_loss = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    # Save summary
    os.makedirs("results", exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = f"results/hpo_{args.model}_{timestr}.txt"
    with open(out, "w") as f:
        f.write(f"HPO results: {args.model}\n")
        f.write(f"Best val_loss: {study.best_value:.4f}\n")
        f.write(f"Best params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k} = {v}\n")
        f.write("\nAll completed trials:\n")
        for t in completed:
            f.write(f"  trial={t.number}  val_loss={t.value:.4f}  {t.params}\n")
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
