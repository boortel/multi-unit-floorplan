_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab1'
exp_name = 'cab1_eval_cubicasa_v2s'
backbone = 'EfficientNetV2S'
filters = [32, 64, 128, 256, 512]
n_up_sample_block = len(filters) + 1  # Fix for up_sample_block bug
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4]
hhdc = 7
cam = 5

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

# Training
batch_size = 4          # 4 = 2 GPUs x 2 samples/GPU (or 4 on single GPU)
epochs = 100

# LR scheduling: 'cosine-decay-warmup', 'cosine-decay', 'reduce-lr-on-plateau', or None
lr_scheduler = 'cosine-decay-warmup'
lr_min = 1e-6           # final LR at end of cosine decay
warmup_epochs = 5       # linear warmup before cosine decay begins

## K-Fold specs
kFold = 10
target_fold = 4         # Retrain only Fold 4

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
