# Necessary fields for training
_base_ = ['../configs/base/default_runtime.py', '../configs/base/default_model.py', '../configs/datasets/cubicasa5k.py']

## Train specs
exp_name = 'CubiCasa5k_Original'

## Model specs
model_type = 'unetpp'
backbone = 'efficientnetb5'
backbone_weights = '/home/simonbilik/Programming/multi-unit-floorplan/b5_notop.h5'
output_activation = 'softmax'
filters = [512, 256, 128, 64, 32]
up_rates = (2, 2, 2, 2, 2)
n_up_sample_block = len(up_rates)+1
batch_norm = True

## K-Fold specs
kFold = 10 # Set this to the number of folds you generated

## Dataset overrides
# Point this to wherever you generated the un-augmented TFRecords!
data_root = 'data/tfrecords/cubicasa5k'

## Runtime specs
batch_size = 1
epochs = 100 # Change this to your desired number of epochs
