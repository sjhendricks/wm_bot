# wm_bot

> A retrieval-augmented generation (RAG) chatbot for answering questions about William & Mary, powered by a 3-stage hybrid retrieval pipeline (FAISS + BM25 + CrossEncoder reranking) and a QLoRA fine-tuned LLM (Llama 3, Gemma, or Mistral) served via a Gradio UI.

**GitHub Repository:** https://github.com/sjhendricks/wm_bot

--- 

## Project Goal
The goal of this project is to collect and organize information from William & Mary websites to build a system that answers questions about William & Mary.

We scrape relevant webpages, extract clean text, and store results in a structured format used for retrieval-augmented generation (RAG). A 3-stage hybrid retrieval system (FAISS → BM25 → CrossEncoder reranking) fetches the most relevant passages, which are passed as context to a QLoRA fine-tuned LLM to generate grounded answers via a Gradio interface.

The pipeline supports fine-tuning and inference with three interchangeable models: **Llama 3**, **Gemma**, and **Mistral**.

---

## Project Structure

```
wm_bot/
├── scripts/                    # All pipeline scripts
├── envs/
│   ├── wmbot-env_full.yaml     # Full scraping environment spec
│   ├── wmbot-stable.yaml       # Stable scraping environment spec
│   └── llm-env_new.yaml        # LLM/retrieval environment spec
├── data/
│   ├── raw/                    # Raw scraped files (organized by site)
│   ├── clean/                  # Cleaned and formatted text
│   ├── rag/                    # chunks.json, faiss.index, bm25.pkl
│   └── fine_tuning/            # Training data + qlora-out/ adapters
├── metadata/                   # Seed URLs and tracking files
├── logs/                       # Run logs and error logs
└── README.md
```
---

## Scripts

### Core Pipeline

| Script | Description | 
|--------|-------------|
| `01_scraper.py` | scrapes data from different sites (listed in metadata) |
| `02_format_files.py` | uses trifilatura to clean html extras from sites |
| `03_clean.py` | cleans the data |
| `04_chunk_data.py` | chunks the data |
| `05_embed_faiss.py` | FAISS embeddings|
| `06_build_bm25.py` | builds BM25 for retrieval |
| `07_fine_tuning.py` | fine-tunes |
| `08_conversation_format.py` | formats jsonl for input to axolotl |
| `09_app.py` | full RAG chatbot: retrieval + LLM generation + Gradio UI | 

### Environment Setup

| Script | Description | 
|--------|-------------|
| `00_create_scraping_env.sh` | creates environment for scraping and preparing data |
| `00_create_llm_env.sh` | creates environment for training the llm using axolotl |
| `00_create_run_bot_env.sh` | creates environment for running the bot using pytorch | 

### HPC (SLURM) Submit Scripts 

| Script | Description | 
|--------|-------------|
| `submit_scraper.sh` | submits scraping job |
| `submit_cleaner.sh` | submits formatting job |
| `submit_reformat.sh` | submits conversation formatting job |
| `submit_axolotl_model.sh` | submits fine-tuning with selectable model (see Model Selection) |
| `submit_retrieve.sh` | submits retrieval job |

### Supporting Scripts

| Script | Description |
|--------|-------------|
| `cleaning_pipeline_pt1.sh` | allows filename changes before cleaning |
| `cleaning_pipeline_pt2.sh` | runs scripts 03–06 as a pipeline |
| `master_pipeline.sh` | end-to-end pipeline runner |
| `wm_bot_rag.py` | RAG pipeline module |

---

## Pipeline Overview 

```mermaid
flowchart TD

    %% ===== DATA PIPELINE =====
    A["Scrape<br/>01_scraper.py"] --> 
    B["Format<br/>02_format_files.py"] --> 
    C["Clean<br/>03_clean.py"] --> 
    D["Chunk<br/>04_chunk.py"]

    %% ===== RETRIEVAL INDEX =====
    D --> E["Embed + Index<br/>FAISS / BM25<br/>05_embed_faiss.py + 06_build_bm25.py"]

    %% ===== TRAINING =====
    D --> L["Create Training Data"]
    L --> M["Fine-tuning Preparations<br/>07_fine_tuning.py"]
    M --> N["Axolotl Fine-Tuning<br/>QLoRA / DPO<br/> placeholder"]

    %% ===== INFERENCE =====
    G[User Query] --> H[Retriever]
    E --> H

    H --> I[Build Prompt]
    G --> I

    I --> J["LLM<br/>(Base or Fine-tuned)"]
    J --> K[Response]

    N --> J
```

---

## Retrieval Architecture

The retrieval system uses 3 stages to maximize answer quality:

1. **FAISS semantic search** — encodes the query with `all-MiniLM-L6-v2` and retrieves the top-K semantically similar chunks
2. **BM25 keyword search** — runs a parallel keyword search to catch exact-match results that semantic search may miss
3. **CrossEncoder reranking** — combines and deduplicates both result sets, reranks with `ms-marco-MiniLM-L-6-v2`, and passes the top 3 passages as LLM context

---

## Model
 
The pipeline supports three interchangeable models for fine-tuning and inference:

```bash
sbatch scripts/submit_axolotl_model.sh llama    # Meta-Llama-3-8B-Instruct
sbatch scripts/submit_axolotl_model.sh gemma    # Gemma
sbatch scripts/submit_axolotl_model.sh mistral  # Mistral (default)
```

| Component | Value |
|-----------|-------|
| Supported LLMs | Llama 3 8B Instruct, Gemma, Mistral |
| Default model | Mistral |
| Fine-tuning method | QLoRA via Axolotl |
| Adapter path | `data/fine_tuning/qlora-out/` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| UI | Gradio |

System prompt: *"You are a helpful William and Mary Advisor. Be concise and professional."*

---

## Data Sources

Scraped content is organized by site under `data/raw/`:

| Source | Directory |
|--------|-----------|
| W&M main site | `wm_edu/` |
| Course catalog | `catalog/` |
| W&M news | `news/` |
| W&M library | `wm_library/` |
| Law school | `law/` |
| Flat Hat (student newspaper) | `flathat/` |
| Dining | `dininghub/` |
| Recreation | `rec/` |
| VIMS | `vims/` |
| Visit Williamsburg | `visit_wmburg/` |
| Mason School of Business | `mason/` |
| CDSP | `cdsp/` |
| Colonial Williamsburg | `cw/` |
| ScholarWorks | `scholarworks/` |

---

### Local

```bash
cd wm_bot

# Create environments
bash scripts/00_create_scraping_env.sh
bash scripts/00_create_llm_env.sh
bash scripts/00_create_llm_env.sh

# Steps 1–4: Scrape + process data
conda activate wmbot-env
python scripts/01_scraper.py
python scripts/02_format_files.py
python scripts/03_clean.py
python scripts/04_chunk.py
conda deactivate

# Steps 5–6: Build retrieval indexes
conda activate llm-env
python scripts/05_embed_faiss.py
python scripts/06_build_bm25.py

# Steps 7–8: Prepare fine-tuning data
python scripts/07_fine_tuning.py
python scripts/08_conversation_format.py

# Step 9: Launch chatbot
conda activate wmbot-stable
srun -p batch --gres=gpu:1 --mem=64G -t 01:00:00 --pty bash
python bot_test_resources/[MODEL NAME]_bot.py
```

### HPC (SLURM)

```bash
# Scraping (with arguments url, domain, folder name)
sbatch scripts/submit_scraper.sh {URL} {DOMAIN} {FOLDER NAME}
# Cleaning/formatting (run stages together)
bash master_pipeline.sh  # runs scripts 02–06

# Fine-tuning
sbatch scripts/submit_axolotl_model.sh mistral

# Launch chatbot interactively, replace [MODEL NAME] with model option of choice (gemma, llama, mistral)
conda activate llm-env
srun -p batch --gres=gpu:1 --mem=64G -t 01:00:00 --pty bash
python bot_test_resources/[MODEL NAME]_bot.py
```

---

## Environments

### `wmbot-env` - Scraping & Processing
Used for steps 1–4.
```bash
bash scripts/00_create_scraping_env.sh
# uses envs/wmbot-env_full.yaml 
```
Key packages: `trafilatura`, `beautifulsoup4`, `requests`, `pandas`, `numpy`

### `llm-env` - Fine-tuning via Axolotl
Used for steps 5–8.
```bash
bash scripts/00_create_llm_env.sh
# uses envs/llm-env_new.yaml
```
Key packages: `axolotl`, `torch`, `transformers`, `peft`, `trl`, `accelerate`

### `run-bot-env` - Inference & UI
Used for step 9.
```bash
bash scripts/00_create_run_bot_env.sh
```
Key packages: `faiss`, `sentence-transformers`, `gradio`, `rank-bm25`, `peft`, `torch`

---

## Demo & Results

### Example Chatbot Outputs

| User Query | Bot Response |
|------------|--------------|
| *"What majors are in arts & sciences?"* | *"At William & Mary, the Arts & Sciences majors include but are not limited to: Anthropology, Biology, Chemistry, Classical Studies, Computer Science, Economics, English, Environmental Science, Geology, History, Mathematics, Physics, Psychology, Sociology, and Theatre, Speech, and Dance. However, this is not an exhaustive list. I would recommend visiting the W&M website or contacting the appropriate department or faculty member for more information on majors in Arts & Sciences."* |
| *"What resources can I use at the Swem Library?"* | *"The Swem Library offers a variety of resources to help you with your academic journey. You can access databases, e-books, journals, and newspapers, as well as physical books, DVDs, and more. Additionally, the library has a number of study spaces, including individual study rooms, group study rooms, and quiet study areas. You can also visit the library's website to see what events are happening, such as workshops and lectures. For more information, you can visit the library's website or speak with a librarian."* |
| *"What degrees can I pursue at the Business School at W&M?"* | *"The Business School, also known as the Raymond A. Mason School of Business, offers undergraduate degrees in Business Administration. Specifically, you can pursue a Business Administration degree with majors in Accounting, Business Analytics, Business Management, Economics, Entrepreneurship, International Business, Marketing, and Real Estate. Additionally, the Business School offers a joint degree program with the School of Law, which allows students to earn both a Juris Doctor (J.D.) and a Master of Business Administration (M.B.A.) in approximately six years. For more information, I recommend visiting the Business School's website or reaching out to an academic advisor."* |

### Evaluation 

We tested the chatbot on...

### User Interface 

Below is the gradio interface that can be run through the .ipynb

---

## Team Contributions

| Member | Contributions |
|--------|---------------|
| [Name 1] | Web scraping pipeline, seed URL curation, data organization across 14 site categories, project restructuring, multi-model selection feature, README documentation |
| [Name 2] | Retrieval system - FAISS, BM25, CrossEncoder reranking |
| [Name 3] | Fine-tuning - QLoRA/Axolotl training, conversation formatting |
| [Name 4] | Gradio UI, app integration, testing |

---

## Key Dependencies & Citations

| Tool | Purpose | Reference |
|------|---------|-----------|
| [trafilatura](https://github.com/adbar/trafilatura) | Web text extraction | @inproceedings{barbaresi-2021-trafilatura,<br> title = {{Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction}}, <br>  author = "Barbaresi, Adrien", <br>  booktitle = "Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations", <br>  pages = "122--131",<br>  publisher = "Association for Computational Linguistics",<br>  url = "https://aclanthology.org/2021.acl-demo.15",<br> year = 2021, <br>} |
| [FAISS](https://github.com/facebookresearch/faiss) | Semantic vector search | @article{johnson2019billion,<br> title={Billion-scale similarity search with {GPUs}},<br> author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},<br> journal={IEEE Transactions on Big Data},<br> volume={7},<br> number={3},<br> pages={535--547}, <br> year={2019}, <br>  publisher={IEEE} <br> } |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Keyword retrieval | @software{rank_bm25,<br> author = {Dorian Brown},<br> title = {{Rank-BM25: A Collection of BM25 Algorithms in Python}},<br> year = 2020,<br> publisher = {Zenodo},<br> doi = {10.5281/zenodo.4520057},<br> url = {https://doi.org/10.5281/zenodo.4520057} <br> } |
| [sentence-transformers](https://www.sbert.net/) | Text embeddings | @inproceedings{reimers-2019-sentence-bert,<br> title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",<br> author = "Reimers, Nils and Gurevych, Iryna",<br> booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",<br> month = "11",<br> year = "2019",<br> publisher = "Association for Computational Linguistics",<br> url = "https://arxiv.org/abs/1908.10084",<br> } <br> |
| [CrossEncoder](https://www.sbert.net/) | Retrieval reranking |  
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
} 
|
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Fine-tuning framework |
 @software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
} 
|
| [Pytorch](https://docs.pytorch.org/docs/2.5/index.html) | GPU capabilities for training | cff-version: 1.2.0
message: If you use this software, please cite it as below.
title: PyTorch
authors:
  - family-names: PyTorch Team
url: https://pytorch.org
preferred-citation:
  type: conference-paper
  title: "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"
  authors:
    - family-names: Ansel
      given-names: Jason
    - family-names: Yang
      given-names: Edward
    - family-names: He
      given-names: Horace
    - family-names: Gimelshein
      given-names: Natalia
    - family-names: Jain
      given-names: Animesh
    - family-names: Voznesensky
      given-names: Michael
    - family-names: Bao
      given-names: Bin
    - family-names: Bell
      given-names: Peter
    - family-names: Berard
      given-names: David
    - family-names: Burovski
      given-names: Evgeni
    - family-names: Chauhan
      given-names: Geeta
    - family-names: Chourdia
      given-names: Anjali
    - family-names: Constable
      given-names: Will
    - family-names: Desmaison
      given-names: Alban
    - family-names: DeVito
      given-names: Zachary
    - family-names: Ellison
      given-names: Elias
    - family-names: Feng
      given-names: Will
    - family-names: Gong
      given-names: Jiong
    - family-names: Gschwind
      given-names: Michael
    - family-names: Hirsh
      given-names: Brian
    - family-names: Huang
      given-names: Sherlock
    - family-names: Kalambarkar
      given-names: Kshiteej
    - family-names: Kirsch
      given-names: Laurent
    - family-names: Lazos
      given-names: Michael
    - family-names: Lezcano
      given-names: Mario
    - family-names: Liang
      given-names: Yanbo
    - family-names: Liang
      given-names: Jason
    - family-names: Lu
      given-names: Yinghai
    - family-names: Luk
      given-names: CK
    - family-names: Maher
      given-names: Bert
    - family-names: Pan
      given-names: Yunjie
    - family-names: Puhrsch
      given-names: Christian
    - family-names: Reso
      given-names: Matthias
    - family-names: Saroufim
      given-names: Mark
    - family-names: Siraichi
      given-names: Marcos Yukio
    - family-names: Suk
      given-names: Helen
    - family-names: Suo
      given-names: Michael
    - family-names: Tillet
      given-names: Phil
    - family-names: Wang
      given-names: Eikan
    - family-names: Wang
      given-names: Xiaodong
    - family-names: Wen
      given-names: William
    - family-names: Zhang
      given-names: Shunting
    - family-names: Zhao
      given-names: Xu
    - family-names: Zhou
      given-names: Keren
    - family-names: Zou
      given-names: Richard
    - family-names: Mathews
      given-names: Ajit
    - family-names: Chanan
      given-names: Gregory
    - family-names: Wu
      given-names: Peng
    - family-names: Chintala
      given-names: Soumith
  collection-title: "29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)"
  collection-type: proceedings
  month: 4
  year: 2024
  publisher:
    name: ACM
  doi: "10.1145/3620665.3640366"
  url: "https://docs.pytorch.org/assets/pytorch2-2.pdf"
|
| [PEFT](https://github.com/huggingface/peft) | QLoRA adapter loading | 
@Misc{peft,
  title =        {{PEFT}: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan and Marian Tietz},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
} 
|
| [Beautifulsoup4](https://pypi.org/project/beautifulsoup4/) | Data Scraping |
@article{richardson2007beautiful,
  title={Beautiful soup documentation},
  author={Richardson, Leonard},
  journal={April},
  year={2007}
}
|
| [Gradio](https://www.gradio.app/) | Chatbot UI | 
cff-version: 1.2.0
message: Please cite this project using these metadata.
title: "Gradio: Hassle-free sharing and testing of ML models in the wild"
abstract: >-
  Accessibility is a major challenge of machine learning (ML).
  Typical ML models are built by specialists and require
  specialized hardware/software as well as ML experience to
  validate. This makes it challenging for non-technical
  collaborators and endpoint users (e.g. physicians) to easily
  provide feedback on model development and to gain trust in
  ML. The accessibility challenge also makes collaboration
  more difficult and limits the ML researcher's exposure to
  realistic data and scenarios that occur in the wild. To
  improve accessibility and facilitate collaboration, we
  developed an open-source Python package, Gradio, which
  allows researchers to rapidly generate a visual interface
  for their ML models. Gradio makes accessing any ML model as
  easy as sharing a URL. Our development of Gradio is informed
  by interviews with a number of machine learning researchers
  who participate in interdisciplinary collaborations. Their
  feedback identified that Gradio should support a variety of
  interfaces and frameworks, allow for easy sharing of the
  interface, allow for input manipulation and interactive
  inference by the domain expert, as well as allow embedding
  the interface in iPython notebooks. We developed these
  features and carried out a case study to understand Gradio's
  usefulness and usability in the setting of a machine
  learning collaboration between a researcher and a
  cardiologist.
authors:
  - family-names: Abid
    given-names: Abubakar
  - family-names: Abdalla
    given-names: Ali
  - family-names: Abid
    given-names: Ali
  - family-names: Khan
    given-names: Dawood
  - family-names: Alfozan
    given-names: Abdulrahman
  - family-names: Zou
    given-names: James
doi: 10.48550/arXiv.1906.02569
date-released: 2019-06-06
url: https://arxiv.org/abs/1906.02569
|

---

## Known Limitations

- Scraping is scoped to selected W&M sites; not a full site crawl
- Benchmarking is currently manual
