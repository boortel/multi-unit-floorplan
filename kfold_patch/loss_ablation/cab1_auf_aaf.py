# Loss ablation L2: focal + AAF (+ AWL). No heatmap regression. AAF sizes as in the champion.
_base_ = ['../eval_cab1_hpo_v2m.py']
exp_name = 'lossabl_cab1_auf_aaf_v2m'
loss_functions = ['asym_unified_focal_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']
