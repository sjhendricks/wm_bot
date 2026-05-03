#!/bin/bash
set -ueo pipefail

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# create  1st scraping env
conda env create -f wmbot-env_full.yaml

conda activate wmbot-env
pip install -r requirements-wmbot.txt
conda deactivate

# create 2nd axolotl env
module load cuda
conda env create -f llm-env_new.yaml

# create 3rd bot env
conda env create -f wmbot-stable.yaml
