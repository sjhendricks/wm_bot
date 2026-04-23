#!/bin/bash

# activate conda environment
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# if need to create env, run this
conda env create -f wmbot-env.yaml

# activate environment
conda activate wmbot-env

# now, install extra packages via pip
which pip #should result in this: ~/.conda/envs/wmbot-env/bin/pip

~/.conda/envs/wmbot-env/bin/pip install "trafilatura[all]"

conda deactivate
