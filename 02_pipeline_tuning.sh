#!/bin/bash
#SBATCH --job-name=fine-tuning
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6 # how many cores (bora max is 20)
#SBATCH --time=08:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=64G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=hrsweazey@wm.edu # who to email
#SBATCH -o logs/fine_tuning/fine-tuning_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e logs/fine_tuning/fine-tuning_%j.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-env

export HF_TOKEN="YOUR HUGGING FACE TOKEN"

python ./scripts/07_fine_tuning.py
conda deactivate
conda activate wmbot-env

python ./scripts/08_conversation_format.py
conda deactivate