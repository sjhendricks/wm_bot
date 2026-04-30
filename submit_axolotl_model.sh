#!/bin/bash
#SBATCH --job-name=axolotl
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=8 # how many cores (bora max is 20)
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=64G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=YOUR EMAIL # who to email
#SBATCH -o logs/axolotl_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e logs/axolotl_%j.err #STDERR to file (%j is jobID)

# 1. Read the first argument, default to 'mistral' if no argument is provided
MODEL_NAME=${1:-mistral}

# 2. Validate the input and assign the corresponding YAML config
case "$MODEL_NAME" in
    llama)
        CONFIG_FILE="bot_test_resources/llama/axolotl_llama.yaml"
        ;;
    gemma)
        CONFIG_FILE="bot_test_resources/gemma/axolotl_gemma.yaml"
        ;;
    mistral)
        CONFIG_FILE="bot_test_resources/mistral/axolotl_mistral.yaml"
        ;;
    *)
        # Catch invalid arguments and stop the job safely
        echo "============================================================"
        echo "ERROR: Invalid model name ('$MODEL_NAME')!"
        echo "Usage: sbatch axolotl_train.sh [llama|gemma|mistral]"
        echo "Note: If no argument is provided, it defaults to 'mistral'."
        echo "============================================================"
        exit 1
        ;;
esac

echo "============================================================"
echo "Starting Axolotl Training Task"
echo "Target Model: $MODEL_NAME"
echo "Config File:  $CONFIG_FILE"
echo "============================================================"

# Load modules and environment
module load miniforge3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-env

# Export Hugging Face Token
export HF_TOKEN="YOUR HUGGING FACE TOKEN"

# 3. Execute Accelerate with the dynamic config file variable
accelerate launch -m axolotl.cli.train "$CONFIG_FILE"

# Clean up
conda deactivate
echo "Job Completed."