_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab2'
exp_name = 'cab2_eval_cubicasa_b4'
backbone = 'EfficientNetB4'
filters = [32, 64, 128, 256, 512]
n_up_sample_block = len(filters) + 1
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4]
hhdc = False             # Disabled based on ablation search (1.5462 val loss, best setting)
cam = 3                  # Kept at 3 based on ablation search (1.5481 val loss, clear optimum)

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

# Training
batch_size = 4          # 4 = 2 samples/GPU x 2 GPUs (or 4 on a single GPU)
epochs = 100

# LR scheduling: 'cosine-decay-warmup', 'cosine-decay', 'reduce-lr-on-plateau', or None
lr_scheduler = 'cosine-decay-warmup'
lr_min = 1e-6           # final LR at end of cosine decay
warmup_epochs = 5       # linear warmup before cosine decay begins

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
