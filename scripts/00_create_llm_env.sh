#!/bin/bash

# set up conda
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# create llm environment
mamba create -y -n llm-env python=3.11
conda activate llm-env

# prepare for downloads
# add module for pytorch (cuda)
module load cuda

# add packages
which pip #should result in: ~/.conda/envs/llm-env/bin/pip

~/.conda/envs/llm-env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
~/.conda/envs/llm-env/bin/pip install --no-build-isolation axolotl

# deactivate
conda deactivate
