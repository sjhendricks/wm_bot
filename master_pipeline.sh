#!/bin/bash
#SBATCH --job-name=cleaning_chunking
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=10 # how many cores (bora max is 20)
#SBATCH --time=08:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=64G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=hrsweazey@wm.edu # who to email
#SBATCH -o logs/cleaning_chunking_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e logs/cleaning_chunking_%j.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wmbot-env

echo "🔹 Step 0: Formatting Scraped Files"
# Point to the parent directory where all your raw folders live
for dir_path in /sciclone/scr10/gzdata440/wm_bot/data/raw/*; do
    
    # Check if it's actually a directory (skips random files)
    if [ -d "$dir_path" ]; then
        
        # 1. Extract just the folder name (e.g., 'catalog' or 'flathat')
        export data=$(basename "$dir_path")
        
        echo "Processing folder: $data"

        python /sciclone/scr10/gzdata440/wm_bot/scripts/02_format_files.py
        
    fi
done

echo "🔹 Step 1: Cleaning"
python /sciclone/scr10/gzdata440/wm_bot/scripts/clean2.py

echo "🔹 Step 2: Chunking"
python /sciclone/scr10/gzdata440/wm_bot/scripts/chunk_data2.py

echo "🔹 Step 3: Embedding + FAISS"
python /sciclone/scr10/gzdata440/wm_bot/scripts/embed_faiss.py

echo "🔹 Building BM25..."
python /sciclone/scr10/gzdata440/wm_bot/scripts/build_bm25.py

echo "✅ Pipeline Complete!"

conda deactivate