_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cab1'
exp_name = 'cab1_eval_cubicasa'
backbone = 'EfficientNetB2'
filters = [32, 64, 128, 256, 512]
n_up_sample_block = len(filters) + 1  # Fix for up_sample_block bug
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4]
hhdc = 5
cam = 3

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

batch_size = 2
epochs = 100

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
