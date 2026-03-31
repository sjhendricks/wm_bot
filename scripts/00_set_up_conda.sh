#!/bin/bash

# set up conda
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh”

# create environment
mamba create -y -n wmbot-env python=3.12
conda activate wmbot-env

# install packages
conda install -c conda-forge beautifulsoup4
pip install "trafilatura[all]"

conda deactivate wmbot-env

