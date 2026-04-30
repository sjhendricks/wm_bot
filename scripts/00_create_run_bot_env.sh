#!/bin/bash

# load in needed modules
module load miniforge3
module load cuda

# source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# create environment
conda create -n wmbot-stable python=3.11 -y
conda activate wmbot-stable

# ensure versions are correct
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft sentence-transformers faiss-cpu rank_bm25

conda deactivate
