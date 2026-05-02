#!/bin/bash
#SBATCH --job-name=scraping
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=3-00:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=128G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=YOUR EMAIL@wm.edu # who to email
#SBATCH -o logs/scraping_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e logs/scraping_%j.err #STDERR to file (%j is jobID)

# export start=$1
# export domain=$2
# export folder=$3

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wmbot-env


INPUT_FILE="./metadata/metadata.csv"


# Check if the file exists before starting
if [ ! -f "$INPUT_FILE" ]; then
   echo "Error: File not found."
   exit 1
fi
# Just to ensure the file system is ready (optional)
while IFS=',' read -r var1 var2 var3
do
   # Check if var1 (the URL) is actually present
   if [ -n "$var1" ]; then
       echo "Processing: $var1"
       python ./scripts/01_scraping.py "$var1" "$var2" "./data/raw/${var3}"
   fi
done < "$INPUT_FILE"
conda deactivate
