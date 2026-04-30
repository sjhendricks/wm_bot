#!/bin/bash
#SBATCH --job-name=scraping
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=3-00:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=128G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=hrsweazey@wm.edu # who to email
#SBATCH -o logs/scraping/scraping_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e logs/scraping/scraping_%j.err #STDERR to file (%j is jobID)

export start=$1
export domain=$2
export folder=$3

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wmbot-env

python /sciclone/scr10/gzdata440/wm_bot/scripts/01_scraper.py
conda deactivate
