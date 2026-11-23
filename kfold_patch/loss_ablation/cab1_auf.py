# Loss ablation L1: asymmetric unified focal loss only (+ AWL, single term). No heatmap, no AAF.
_base_ = ['../eval_cab1_hpo_v2m.py']
exp_name = 'lossabl_cab1_auf_v2m'
loss_functions = ['asym_unified_focal_loss', 'AutomaticWeightedLoss']
aaf = []
