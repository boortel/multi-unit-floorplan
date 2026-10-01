_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab2'
exp_name = 'cab2_hpo_trial5_b0'
backbone = 'EfficientNetB0'
filters = [32, 64, 128, 256, 512]
n_up_sample_block = len(filters) + 1
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4, 8]
hhdc = 7
cam = False

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

# Training
batch_size = 4
epochs = 100

lr_scheduler = 'reduce-lr-on-plateau'
lr_min = 5.34e-07
warmup_epochs = 10

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
