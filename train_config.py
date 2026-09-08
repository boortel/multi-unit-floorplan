import ctypes
import glob
import logging
import os
import site
import time


# Preload NVIDIA GPU libraries BEFORE importing TensorFlow.
# Setting LD_LIBRARY_PATH in Python doesn't affect dlopen() in the current process
# (the dynamic linker only reads it at startup), so we must explicitly load each
# shared library into the global symbol table via ctypes.
def _preload_gpu_libs():
    search_paths = ["/usr/lib/x86_64-linux-gnu"]
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, "lib"))
    try:
        search_paths.extend(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))
    except Exception:
        pass

    # Libraries TensorFlow needs for GPU support
    lib_names = [
        "libcuda.so.1",
        "libcudart.so.11.0",
        "libcudnn.so.8",
        "libcublas.so.11",
        "libcublasLt.so.11",
        "libcufft.so.10",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.11",
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
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer

# Limit visible physical GPUs to at most 2 devices if not explicitly set by environment
try:
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 2 and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        tf.config.set_visible_devices(gpus[:2], 'GPU')
        print(f"Restricting visible GPUs to at most 2 devices: {[g.name for g in gpus[:2]]}")
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"GPU device setup note: {e}")

# Enable Mixed Precision policy (FP16 compute with FP32 weights/scaling for Tensor Cores)
mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision global policy: {mixed_precision.global_policy().name}")

# Enable XLA JIT Compilation for operator fusion
tf.config.optimizer.set_jit(True)

import segmentation_models as sm
from datasets import floorplans
from segmentation_models.models import zeng, r2v
from segmentation_models.models.cab1.cab1 import cab1
from segmentation_models.models.cab2.cab2 import cab2
from segmentation_models.models.ours_multi.ours_multi import ours_multi
from segmentation_models.models.unet3plus.model_unet_2d import unet_2d
from segmentation_models.models.unet3plus.model_unet_3plus_2d import unet_3plus_2d
from training import Trainer, loss_functions
from training.schedulers import SchedulerType, get as get_scheduler
from training.AutomaticWeightedLoss import AutomaticWeightedLoss, AutomaticWeightedLossCallback

from tqdm.keras import TqdmCallback
from utils import Config, mkdir_or_exist, get_args_dict

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def main():
    sm.set_framework('tf.keras')

    # Get default options
    help_cfg = Config.fromfile('configs/base/train_ops_help.py')
    # Parse args
    parser = help_cfg.auto_argparser('Train a model options')

    # Process configs
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args_dict = get_args_dict(args)

    configs = args_dict.pop('config')
    if args_dict.get('work_dir', None) is None:
        train_folder = args_dict.get('exp_name', None)
        if train_folder is None:
            args_dict['work_dir'] = os.path.abspath('./')
        else:
            # create sub folder train_folder
            args_dict['work_dir'] = os.path.abspath(
                os.path.join('./', os.path.splitext(os.path.basename(train_folder))[0]))
    # create work_dir
    mkdir_or_exist(args_dict['work_dir'])
    # Run for each config
    for config in configs:
        cfg = Config.fromfile(config)
        cfg.merge_from_dict(args_dict)

        # Print warning for options which are not specified
        for key in help_cfg.to_dict().keys():
            if key not in cfg.to_dict().keys():
                print(f'Warning: option {key} is not in config.')

        tic = time.time()
        k_fold = cfg.get('kFold', cfg.get('k_fold', 0))
        target_fold = cfg.get('target_fold', None)
        cfg.hist = []
        if k_fold > 0:
            import numpy as np
            histories = []
            folds_to_run = [target_fold] if (target_fold is not None and isinstance(target_fold, int) and target_fold >= 0) else list(range(k_fold))
            for f in folds_to_run:
                tic_fold = time.time()
                print(f"Starting fold {f}/{k_fold}")
                cfg.fold = f
                hist = train(cfg)
                if hist is not None:
                    histories.append(hist.history)
                    cfg.hist.append(hist.history)
                cfgs = Config(cfg_dict=dict(train_cfg=cfg.to_dict()))
                cfgs.dump(os.path.join(cfg.log_dir_fold, 'training_cfg.py'))
                toc_fold = time.time()
            
            if histories:
                print("\n--- K-Fold Cross Validation Results ---")
                metrics_summary = {}
                for key in histories[0].keys():
                    if 'val_' in key:
                        if 'loss' in key:
                            best_vals = [min(h[key]) for h in histories]
                        else:
                            best_vals = [max(h[key]) for h in histories]
                        mean_val = np.mean(best_vals)
                        std_val = np.std(best_vals)
                        print(f"{key}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
                        metrics_summary[key] = {'mean': mean_val, 'std': std_val}
                
                with open(os.path.join(cfg.log_dir, 'kfold_summary.txt'), 'w') as f:
                    for k, v in metrics_summary.items():
                        f.write(f"{k}: Mean = {v['mean']:.4f}, Std = {v['std']:.4f}\n")
        else:
            train(cfg)
        toc = time.time()

        cfg.training_time = toc - tic

        cfg = Config(cfg_dict=dict(train_cfg=cfg.to_dict()))
        cfg.dump(os.path.join(cfg.train_cfg.log_dir, 'training_cfg.py'))

        print('total training time = {} minutes'.format((toc - tic) / 60))
        print()
        print('Waiting 1 min')
        time.sleep(60)
        print('Resuming')
        print()
    print('Finished')


def train(config, extra_callbacks=None):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    exp_name = config.get('exp_name', '')
    # Parameters
    model_type = config.get('model_type', None)
    backbone = config.get('backbone', None)
    resume_from = config.get('resume_from', None)
    deep_supervision = config.get('deep_supervision', False)
    filters = config.get('filters', [])
    up_rates = config.get('up_rates', [])
    n_up_sample_block = config.get('n_up_sample_block', None)
    output_activation = config.get('output_activation', None)
    backbone_weights = config.get('backbone_weights', None)
    hhdc = config.get('hhdc', None)
    cam = config.get('cam', None)
    aaf = config.get('aaf', [])
    aaf_count = len(aaf)
    w_edge = None
    w_not_edge = None
    baseline = config.get('baseline', False)
    batch_norm = config.get('batch_norm', False)

    classes = ['bg'] + config.get('classes', [])
    heatmap_inds = config.get('heatmap_inds', [])
    dataset = config.get('dataset_file', None)
    normalize = config.get('normalize', False)
    data_root = config.get('data_root', None)
    data_reduction = config.get('data_reduction', None)

    epochs = config.get('epochs', 1)
    batch_size = config.get('batch_size', 1)
    train_buffer_size = config.get('train_buffer_size', 1)
    checkpoint_weights_only = config.get('checkpoint_weights_only', False)
    loss_function = config.get('loss_functions', [])
    optimizer = config.get('optimizer', None)
    cfg_metrics = config.get('metrics', [])
    run_eagerly = config.get('run_eagerly', False)
    verbose = config.get('verbose', 0)

    if not dataset:
        ValueError('Dataset must be specify!')
    if not os.path.exists(data_root):
        ValueError(f'Data dir: {data_root} does not exist!')
    if not optimizer:
        ValueError('Optimizer must be specify!')
    if model_type not in ['zeng', 'cubicasa5k', 'unet', 'unetpp', 'unet3p', 'ours_multi', 'cab1', 'cab2']:
        ValueError(f'Model: {model_type} is not implemented!')

    mkdir_or_exist(os.path.join(config.work_dir, 'models'))
    config.log_dir = os.path.join(config.work_dir, 'models', '_'.join(
        [model_type, exp_name, str(backbone or ''), ','.join(map(str, filters)), dataset,
         time.strftime("%Y%m%d-%H%M%S")]))
         
    if hasattr(config, 'fold'):
        config.log_dir_fold = config.log_dir + f"/{config.fold}"
        mkdir_or_exist(config.log_dir_fold)
    else:
        config.log_dir_fold = config.log_dir

    with strategy.scope():
        metrics = []
        for m in cfg_metrics:
            if m == 'CategoricalAccuracy':
                metrics.append(tf.metrics.CategoricalAccuracy())
            else:
                ValueError(f'Metric: {m} is not Implemented!')

        if optimizer.get('type', None) == 'Adam':
            lr = optimizer.get('learning_rate', 1e-4)
            base_opt = tf.keras.optimizers.Adam(learning_rate=lr)
            if mixed_precision.global_policy().name == 'mixed_float16':
                optimizer = mixed_precision.LossScaleOptimizer(base_opt)
            elif optimizer.get('lossScale', None):
                optimizer = LossScaleOptimizer(base_opt, loss_scale='dynamic')
            else:
                optimizer = base_opt
        else:
            ValueError(f'Not implemented optimizer: {optimizer}')

        if not loss_function:
            ValueError('Loss function must be specify!')
        if len(loss_function) == 1:
            lf = loss_function[0]
            if lf == 'balanced_entropy':
                loss_function = loss_functions.balanced_entropy(len(classes))
            elif lf == 'categorical_crossentropy':
                loss_function = loss_functions.categorical_crossentropy(len(classes))
            if lf == 'asym_unified_focal_loss':
                loss_function = loss_functions.asym_unified_focal_loss(len(classes))
            else:
                ValueError(f'Not implemented loss function: {lf}')
        else:
            loss_funcs, names, inds, dec = [], [], [], []
            for lf in loss_function:
                if lf == 'asym_unified_focal_loss':
                    loss_funcs.append(loss_functions.asym_unified_focal_loss(len(classes)))
                    names.append(lf)
                    inds.append(0)
                    dec.append(False)
                if lf == 'heatmap_regression_loss_nomean':
                    loss_funcs.append(
                        loss_functions.heatmap_regression_loss_nomean(len(classes), heatmap_inds))
                    names.append(lf)
                    inds.append(len(inds))
                    dec.append(False)
                if lf == 'heatmap_regression_loss':
                    loss_funcs.append(
                        loss_functions.heatmap_regression_loss(len(classes), heatmap_inds))
                    names.append(lf)
                    inds.append(len(inds))
                    dec.append(False)
                if lf == 'adaptive_affinity_loss':
                    if aaf_count > 0:
                        init_w = tf.constant_initializer(1 / aaf_count)
                        w_edge = tf.Variable(
                            name='edge_w',
                            initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                            dtype=tf.float32,
                            trainable=True)
                        w_not_edge = tf.Variable(
                            name='nonedge_w',
                            initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                            dtype=tf.float32,
                            trainable=True)
                        aaf_ind = len(inds)
                        for a, s in enumerate(aaf):
                            loss_funcs.append(
                                loss_functions.adaptive_affinity_loss(size=s, k=a, num_classes=len(classes),
                                                                      w_edge=w_edge, w_not_edge=w_not_edge))
                            names.append('AAF({0}x{0})'.format(2 * s + 1))
                            inds.append(aaf_ind)
                            dec.append(True)
            loss_function = AutomaticWeightedLoss(loss_funcs, names, inds, dec, epochs, config.log_dir_fold)

        if resume_from:
            print('Resuming training')
            custom_objects = {'loss_function': loss_function}
            unet_model = tf.keras.models.load_model(resume_from, custom_objects=custom_objects)
        else:
            if model_type == 'zeng':
                unet_model = zeng.deepfloorplanModel(classes)
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'cubicasa5k':
                unet_model = r2v.hg_furukawa_original(len(classes))
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unet':
                unet_model = unet_2d((None, None, 3), n_labels=len(classes), backbone=backbone,
                                     filter_num=filters, output_activation=output_activation, batch_norm=batch_norm,
                                     weights=backbone_weights, aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                # Apply loss scaling for optimizer
                unet_model.compile(loss=loss_function.combined_loss(), optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unetpp':
                unet_model = sm.Xnet(backbone_name=backbone, classes=len(classes), decoder_filters=filters,
                                     activation=output_activation, encoder_weights=backbone_weights,
                                     upsample_rates=up_rates)
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unet3p':
                unet_model = unet_3plus_2d((None, None, 3), n_labels=len(classes), backbone=backbone,
                                           filter_num_down=filters, output_activation=output_activation,
                                           batch_norm=batch_norm, weights=backbone_weights, aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge

                # Apply loss scaling for optimizer
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'ours_multi':
                unet_model = ours_multi((None, None, 3), n_labels=len(classes), backbone=backbone,
                                        filter_num_down=filters,
                                        output_activation=output_activation,
                                        batch_norm=batch_norm,
                                        weights=backbone_weights,
                                        aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'cab1':
                unet_model = cab1((None, None, 3), n_labels=len(classes), backbone=backbone,
                                  filter_num_down=filters,
                                  output_activation=output_activation,
                                  batch_norm=batch_norm,
                                  deep_supervision=deep_supervision,
                                  weights=backbone_weights,
                                  aaf=(aaf_count > 0),
                                  use_hhdc=hhdc,
                                  use_cam=cam)

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'cab2':
                if baseline:
                    unet_model = cab2((None, None, 3), n_labels=len(classes), backbone=backbone,
                                      filter_num_down=filters,
                                      output_activation=output_activation,
                                      batch_norm=batch_norm,
                                      deep_supervision=deep_supervision,
                                      weights=backbone_weights,
                                      aaf=(aaf_count > 0),
                                      use_hhdc=hhdc,
                                      use_cam=cam)
                    unet_model.compile(loss=loss_function,
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       run_eagerly=run_eagerly)
                else:
                    unet_model = cab2((None, None, 3), n_labels=len(classes), backbone=backbone,
                                      filter_num_down=filters,
                                      output_activation=output_activation,
                                      batch_norm=batch_norm,
                                      deep_supervision=deep_supervision,
                                      weights=backbone_weights,
                                      aaf=(aaf_count > 0),
                                      use_hhdc=hhdc,
                                      use_cam=cam)

                    unet_model.automatic_loss = loss_function
                    unet_model.loss_sigmas = loss_function.sigmas
                    if aaf_count > 0:
                        unet_model.w_edge = w_edge
                        unet_model.w_not_edge = w_not_edge
                    unet_model.compile(loss=loss_function.combined_loss(),
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       run_eagerly=run_eagerly)

        print('Used config: ', config)
        train_dataset, validation_dataset, test_dataset = floorplans.load_train_data(classes, dataset,
                                                                                     normalize=normalize,
                                                                                     buffer_size=train_buffer_size,
                                                                                     base_dir=data_root,
                                                                                     n_upsample=n_up_sample_block,
                                                                                     reduction_ratio=data_reduction,
                                                                                     fold=getattr(config, 'fold', None),
                                                                                     k_fold=config.get('kFold', config.get('k_fold', None)))
        patience = config.get('early_stopping_patience', config.get('patience', 30))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
        progbar = TqdmCallback(verbose=2)
        callbacks = [early_stop, progbar]

        # --- LR Scheduling ---
        # Read from config; default to cosine-decay-warmup for all models.
        # Supported values: 'cosine-decay-warmup', 'cosine-decay', 'reduce-lr-on-plateau', None
        lr_scheduler_type = config.get('lr_scheduler', 'cosine-decay-warmup')
        lr_min = config.get('lr_min', 1e-6)
        warmup_epochs = config.get('warmup_epochs', 5)
        learning_rate = optimizer.lr.numpy() if hasattr(optimizer, 'lr') else config.get('optimizer', {}).get('learning_rate', 1e-4)

        if lr_scheduler_type in ('cosine-decay-warmup', 'cosine-decay'):
            # Compute total steps from the actual training dataset size
            train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
            if train_dataset_size < 0:
                k_fold_val = config.get('kFold', config.get('k_fold', 0))
                if k_fold_val > 0:
                    train_dataset_size = 4140  # 9 training folds x 460 samples for CubiCasa5k
                else:
                    train_dataset_size = train_buffer_size
            scheduler_enum = (SchedulerType.COSINE_DECAY_WITH_WARMUP
                              if lr_scheduler_type == 'cosine-decay-warmup'
                              else SchedulerType.COSINE_DECAY)
            cosine_cb = get_scheduler(
                scheduler_enum,
                train_dataset_size=train_dataset_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                min_lr=lr_min,
                warmup_epochs=warmup_epochs,
            )
            callbacks.append(cosine_cb)
            print(f'LR scheduler: {lr_scheduler_type} (lr={learning_rate:.2e} -> {lr_min:.2e}, '
                  f'warmup_epochs={warmup_epochs if lr_scheduler_type == "cosine-decay-warmup" else 0})')

        if lr_scheduler_type == 'reduce-lr-on-plateau' or config.get('use_reduce_lr_on_plateau', False):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=lr_min, verbose=verbose)
            callbacks.append(reduce_lr)
            print(f'LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10, min_lr={lr_min:.2e})')

        if hasattr(unet_model, 'automatic_loss'):
            callbacks.append(AutomaticWeightedLossCallback(aaf_count > 0))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        checkpoint_callback = config.get('checkpoint_callback', True)
        tensorboard_callback = config.get('tensorboard_callback', True)
        trainer = Trainer(checkpoint_callback=checkpoint_callback, checkpoint_weights_only=checkpoint_weights_only,
                          learning_rate_scheduler=None, tensorboard_callback=tensorboard_callback,
                          tensorboard_images_callback=False, callbacks=callbacks,
                          log_dir_path=config.log_dir_fold)
        history = trainer.fit(unet_model,
                              train_dataset,
                              validation_dataset,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose)

    del unet_model
    tf.keras.backend.clear_session()
    print('========== DONE ==========')
    return history


if __name__ == "__main__":
    main()
