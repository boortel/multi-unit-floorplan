# 2D Floor Plan to 3D model
Python framework of several existing methods and datasets for image segmentation.
This repository origin from work Gijs de Jong: [repository](https://github.com/TheOnlyError/2d3d)

Thesis: **[Multi-Unit Floor Plan Recognition and Reconstruction](https://repository.tudelft.nl/islandora/object/uuid%3A158f6745-0b43-4796-b21d-6388a35f5a2d?collection=education)**
<br>
[Gijs de Jong](https://github.com/TheOnlyError)
<br>

Contents:
* [Creation of virtual environment](#creation-of-virtual-environment)
* [Data preparation](#data-preparation)
* [Training](#training)
* [Testing](#testing)
* [Evaluation](#evaluation)
* [Experiments & Evaluation Results](#experiments--evaluation-results)
* [Repository layout](#repository-layout)

<details>
<summary>Supported models and datasets</summary>

Models:
- [CubiCasa5k](https://arxiv.org/pdf/1904.01920.pdf)
- [DFPR](https://arxiv.org/pdf/1908.11025.pdf)
- [Unet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- [Unet++](https://arxiv.org/pdf/1912.05074.pdf)
- [Unet3+](https://www.semanticscholar.org/paper/UNet-3%2B%3A-A-Full-Scale-Connected-UNet-for-Medical-Huang-Lin/0b444f74dd9cc06c2833dd15f9258ef5e169e6ea)
- Our models CAB1, CAB2

Dataset:
- [R3D](https://www.cs.toronto.edu/~fidler/projects/rent3D.html)
- [CubiCasa5k](https://zenodo.org/records/2613548)
- [MURF](resources)
- [CVC-FP](http://dag.cvc.uab.es/resources/floorplans/)
- [WAFFLE](https://tau-vailab.github.io/WAFFLE/)
- [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLSTRUCT-FP)


</details>

## Creation of virtual environment
To create and setup venv install `python3.8` and `virtualenv`:

For Linux users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/bin/activate  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt  # Or you could try 'specific_requirements.txt' to get identical environment
```
For Windows users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/Scripts/activate.bat  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt  # Or you could try 'specific_requirements.txt' to get identical environment
```
## Data preparation
For data preparation follow example of dataset creation:

[Dataset example](data_create_example.py)

Framework use `tfrecord` for training and have some tools for its creation: 
> (Here is example for **R3D**)
> - [split](datasets/split_dataset.py) - generate data description from folder
> - [create_mask](datasets/create_data_mask.py) - generate binary mask for each class
> - [create_dataset](datasets/floorplans.py) - generate tfrecords
> - [Config](utils/config.py) - generate Config file used for training
> - (optional) [create_overlay](datasets/create_data_mask.py) - visualize mask on original data
> - (optional) [augmentation](datasets/augment.py) - create more data with rotation
> - (optional) [create_heatmap_from_paths](datasets/create_heatmap.py) - generate heatmaps for training

## Training
For training, you will need data in `tfrecord` file and Config with model parameters. For generation one could follow
[train_example](configs/train_example.py).

When you have prepared data and config, you can run training by this command:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active
```bash
python train_config.py <config> 
```
You can add more configs to run, just by adding them and add some parameters from CLI to update your config. 
But be careful, if you add more configs, CLI parameters overwrite parameters in all passed configs. All parameters could be seen by using `-h`.

## Testing
```bash
python predict.py
```
## Evaluation
```bash
python evaluate.py
```
Or 
```bash
python evaluate_config.py <config>
```

## Experiments & Evaluation Results
Comprehensive benchmarks, 10-fold cross-validation evaluations, and ablation studies on the **CubiCasa5k** dataset are documented in:
* **[Experiment Registry Manifest](results/EXPERIMENT_REGISTRY.md):** Complete registry of all training runs, cross-validation logs, and test evaluations.
* **[Experiment Analysis & 10-Fold Cross-Validation Evaluation](results/experiment_analysis_and_kfold_evaluation.md):** Analysis comparing CAB1 & CAB2 (EfficientNetV2M HPO, EfficientNetB4, EfficientNetV2S) against CubiCasa5k and Zeng baselines on the 400 test images.
* **[Hyperparameter Search & Recommendations](results/hyperparameter_recommendations.md):** Architecture ablation study (backbones B0–B4, HHDC, CAM, AAF), Optuna HPO studies and the recommended configs.
* **[Test Set Evaluation Guide & Runbook](results/TEST_EVALUATION_GUIDE.md):** Execution runbook and instructions for running 10-fold test evaluations on GPU.
* **[Training Accelerations & Optimizations](docs/TRAINING_OPTIMIZATIONS.md):** Mixed precision (`mixed_float16`), XLA JIT, and vectorized loss optimizations providing a ~5×–7× training speedup.
* **[Codebase Audit](docs/CODEBASE_AUDIT.md):** Bugs found and fixed in training, evaluation, data pipeline and models (Sept 2026).
* **[Overview across all datasets](results/OVERVIEW_all_datasets.md):** our models on CubiCasa5k, R3D, CVC-FP and WAFFLE in one comparable 3-class format (zero-shot and fine-tuned with the backbone frozen or unfrozen), the best models against SOTA, and a short summary (Nov 11, 2026; unfrozen fine-tuning added Nov 13).
* **[Comparison with the paper & SOTA](results/COMPARISON_paper_sota.md):** CAB1/CAB2 (B4, V2M) against arXiv:2408.01526 and published methods on CubiCasa5k, R3D, CVC-FP and WAFFLE, both zero-shot and fine-tuned (Nov 2026). Raw results are in [`results/cross_dataset/`](results/cross_dataset/SUMMARY.md), [`results/finetune/`](results/finetune/SUMMARY.md) (backbone frozen) and [`results/finetune_unfrozen/`](results/finetune_unfrozen/SUMMARY.md) (backbone unfrozen).

> **Key Finding (Oct 14, 2026):** The HPO-tuned **EfficientNetV2M** models are the best on the 400-image test set. **CAB1 V2M reaches 96.37% accuracy and 80.16% foreground (no-BG) accuracy. CAB2 V2M reaches 61.99% macro IoU.** CubiCasa5k scores 95.61% / 61.65% / 51.89%. Door, stair and railing IoU roughly doubled to tripled compared with the earlier B4 models. Note that these models were also the first trained after the audit fixes. See [Section 0 of the Evaluation Report](results/experiment_analysis_and_kfold_evaluation.md#0-hpo-champion-models--efficientnetv2m-october-2026).

> **Cross-dataset (Nov 10, 2026):** Trained only on CubiCasa5k, the best models reach walls IoU of 82.1 on R3D and 66.5 on CVC-FP zero-shot. That is the strongest zero-shot CVC-FP result we found. After fine-tuning only the decoder, walls IoU is **91.2–93.4 on R3D (paper: 90)** and 79–81 on CVC-FP (paper: 88–91). On the WAFFLE benchmark (historic and public buildings, Nov 11) mean IoU goes from 38–39 zero-shot to 55–57 fine-tuned (walls IoU 74–76). Zero-shot walls IoU (54.5–58.1) beats the WAFFLE paper's CubiCasa5K model (48.8). See [the comparison](results/COMPARISON_paper_sota.md).

> **Backbone unfrozen (Nov 13, 2026):** Fine-tuning the encoder as well as the decoder improves every model on R3D (mean IoU +1.0 to +3.1; best **CAB2 V2M 83.2**, walls IoU 91.7–93.7, openings recall 84–87 vs DeepFloorplan's 83) and on CVC-FP (mean IoU +1.9 to +6.0; best **CAB1 V2M 79.8**, walls IoU 82.6, openings IoU 77.0 vs 67.4 frozen). CVC-FP walls stay below the paper's in-domain 88–91. WAFFLE barely changes (mean IoU 56–59). See [the overview](results/OVERVIEW_all_datasets.md).

## Repository layout
```
configs/           Model and dataset configs (mmcv-style)
datasets/          Dataset parsing, masks, heatmaps, tfrecord creation
segmentation_models/, training/, utils/   Models, losses, trainer, helpers
kfold_patch/       10-fold CV variants, k-fold eval configs, Optuna HPO, evaluator
scripts/           Shell runners for k-fold training & evaluation (run from repo root)
results/           Evaluation outputs and result write-ups (see EXPERIMENT_REGISTRY.md)
docs/              Codebase audit and training-optimization notes
optuna_*.db        Active Optuna studies (default path used by kfold_patch/optuna_hpo.py)
models/, logs/, data/   Checkpoints, run logs, datasets (not tracked)
```

## TODO:
- [x] Add Config environment - Done from [mmcv](https://github.com/open-mmlab/mmcv/tree/v1.7.1)
- [ ] Add [InternImage](https://github.com/OpenGVLab/InternImage/tree/master) backbone
- [ ] Add [FloorPlanCAD](https://floorplancad.github.io/) dataset, [model](https://arxiv.org/pdf/2105.07147v2.pdf)
- [x] Add [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLSTRUCT-FP)
- [x] Add [CVC-FP](http://dag.cvc.uab.es/resources/floorplans/)

# License
[Apache License v2.0](LICENSE)
