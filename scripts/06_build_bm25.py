import json
import pickle
from rank_bm25 import BM25Okapi

INPUT_FILE = "./data/chunks.json"
OUTPUT_FILE = "./data/bm25.pkl"

with open(INPUT_FILE, "r") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

tokenized = [t.split() for t in texts]

bm25 = BM25Okapi(tokenized)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(bm25, f)

print("✅ BM25 index saved")
