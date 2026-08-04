_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

model_type = 'cubicasa5k'
exp_name = 'cubicasa5k_eval_cubicasa'
backbone = 'vgg16'
normalize = True
batch_norm = True
n_up_sample_block = 6  # Force padding to multiple of 64 for r2v skip connections


batch_size = 2
epochs = 100

## K-Fold specs
kFold = 10

## Dataset overrides
data_root = 'data/tfrecords/cubicasa5k'
