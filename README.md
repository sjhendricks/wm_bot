# wm_bot

## Project Goal
The goal of this project is to collect and organize information from William & Mary websites so we can build a system that answers questions about William & Mary. 

We scrape relevant webpages, extract clean text, and store the results in a structured format that can later be used for model training or retrieval. 

---

## Project Structure
- `scripts/` -> code for setup and scraping
- `data_raw/` -> raw scraped outputs (JSONL)
- `data_clean/` -> cleaned/processed data 
- `metadata/` -> seed URLs and tracking files
- `logs/` -> run logs and error logs 
- `README.md` -> project overview and instructions

---

## Scripts 
- `00_set_up_conda.sh` -> sets up the conda environment
- `scraper.py` -> main scraper for collecting W&M webpages
- `wm_spider.py` -> alternative/test version of scraper
- `02_format_files.py` -> formats raw text files for downstream use 
- `submit.sh` -> submits jobs to HPC using SLURM

---

## Pipeline Overview 
1. Define seed URLs in `metadata/seed_urls.txt`
2. Run scraper to collect webpage data
3. Save raw output to `data_raw/`
4. Format and clean outputs into `data_clean/`
5. Prepare dataset for use in a W&M question-answering system

---

## Inputs
- Seed URLs from `metadata/seed_urls.txt`

---

## Outputs
- Raw text files in `data_raw/`
- Cleaned/processed files in `data_clean/`

---

## How to Run 

1. Navigate to project directory:
   cd wm_bot

2. Set up environment:
   bash scripts/00_set_up_conda.sh

3. Activate environment:
   conda activate wmbot-env

4. Run scraper:
   python scripts/wm_spider.py

---

## Scraper Overview
The scraper:
- starts from selected W&M seed URLs
- uses `trafilatura` to extract readable webpage content
- follows links within wm.edu to find more pages
- skips non-text files (PDFs, images, etc.)
- saves results in JSONL format 

---

## Environment
Required tools:
- Python 3.12 
- trafilatura
- beautifulsoup4

Environment is created using:
- conda / mamba (`00_set_up_conda.sh`)

---

## Current Status 
- Repo structure created 
- Seed URLs added 
- Metadata system set up 
- Scraper integrated and ready for testing

---

## Notes
- Scraper is currently limited by `max_pages` for testing
- Designed to scale on HPC (SciClone) 
- Logs can be saved for debugging and tracking runs 

