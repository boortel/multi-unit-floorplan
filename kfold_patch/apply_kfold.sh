#!/bin/bash

# Navigate to the workspace directory
cd /workspaces/multi-unit-floorplan || cd /home/simonbilik/Programming/multi-unit-floorplan

echo "Copying modified files to apply K-Fold CV logic..."
cp kfold_patch/split_dataset_kfold.py datasets/split_dataset.py
cp kfold_patch/floorplans_kfold.py datasets/floorplans.py
cp kfold_patch/data_create_example_kfold.py data_create_example.py
cp kfold_patch/train_config_kfold.py train_config.py
cp kfold_patch/dataset_statistics_kfold.py datasets/dataset_statistics.py
cp kfold_patch/post_create_command_kfold.sh .devcontainer/post_create_command.sh

echo "Successfully updated:"
echo "- datasets/split_dataset.py"
echo "- datasets/floorplans.py"
echo "- datasets/dataset_statistics.py"
echo "- data_create_example.py"
echo "- train_config.py"

