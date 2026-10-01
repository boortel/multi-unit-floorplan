_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab1'
exp_name = 'cab1_hpo_trial52_b4'
backbone = 'EfficientNetB4'
filters = [16, 32, 64, 128, 256]
n_up_sample_block = len(filters) + 1
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4, 8]
hhdc = False
cam = 3

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

# Training
batch_size = 4
epochs = 100

lr_scheduler = 'cosine-decay-warmup'
lr_min = 7.08e-07
warmup_epochs = 4

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
