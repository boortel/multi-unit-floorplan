_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab1'
exp_name = 'cab1_hpo_trial21_v2m'
backbone = 'EfficientNetV2M'
filters = [16, 32, 64, 128, 256]
n_up_sample_block = len(filters) + 1
output_activation = 'Softmax'
batch_norm = True
aaf = [4, 8]
hhdc = False
cam = 5

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

# Training
batch_size = 4
epochs = 100

lr_scheduler = 'reduce-lr-on-plateau'
lr_min = 1.83e-07
warmup_epochs = 2

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
