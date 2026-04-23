import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

INPUT_FILE = "/sciclone/scr10/gzdata440/wm_bot/data/chunks.json"
INDEX_FILE = "/sciclone/scr10/gzdata440/wm_bot/data/faiss.index"
CHUNKS_FILE = "/sciclone/scr10/gzdata440/wm_bot/data/chunks.pkl"

# Load chunks
with open(INPUT_FILE, "r") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ✅ chunks are already strings
texts = chunks

# Load model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Embed
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Save
faiss.write_index(index, INDEX_FILE)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index built successfully.")
