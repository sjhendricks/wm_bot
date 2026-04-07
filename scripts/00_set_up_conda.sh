#!/bin/bash

# set up conda
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# create environment
mamba create -y -n wmbot-env
conda activate wmbot-env

# deactivate
conda deactivate
