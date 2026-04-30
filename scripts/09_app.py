import json
import pickle

import faiss
import gradio as gr
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =====================================================
# AUTH
# =====================================================

token = "YOUR HUGGING FACETOKEN"

# =====================================================
# PATHS
# =====================================================

DATA_DIR = "/sciclone/scr10/gzdata440/wm_bot/data"

FAISS_PATH = f"{DATA_DIR}/rag/faiss.index"
BM25_PATH = f"{DATA_DIR}/rag/bm25.pkl"
CHUNKS_PATH = f"{DATA_DIR}/rag/chunks.json"

ADAPTER_PATH = f"{DATA_DIR}/fine_tuning/qlora-out"

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =====================================================
# LOAD DATA
# =====================================================

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)

print("Loading BM25...")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("Loading chunks...")
with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)


# =====================================================
# MODELS
# =====================================================

print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# =====================================================
# LLM
# =====================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)

print("Loading QLoRA adapters...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True
)

model.eval()


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve(query, top_k=5):

    # ---------------- FAISS ----------------
    query_embedding = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    faiss_results = []

    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            faiss_results.append({
                "text": chunks[idx],
                "score": float(distances[0][i])
            })

    # ---------------- BM25 ----------------
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25 = np.argsort(bm25_scores)[::-1][:top_k]

    bm25_results = []

    for idx in top_bm25:
        if 0 <= idx < len(chunks):
            bm25_results.append({
                "text": chunks[idx],
                "score": float(bm25_scores[idx])
            })

    # ---------------- MERGE ----------------
    combined = faiss_results + bm25_results

    # FIX: dedupe safely using string
    unique = {}
    for r in combined:
        text = r["text"]
        if isinstance(text, dict):
            text = text.get("text", str(text))
        unique[text] = {"text": text, "score": r["score"]}

    combined = list(unique.values())

    # ---------------- RERANK ----------------
    pairs = [(query, r["text"]) for r in combined]

    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        combined[i]["rerank_score"] = float(score)

    combined = sorted(
        combined,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return combined[:3]


# =====================================================
# GENERATION
# =====================================================

def generate_answer(query):

    retrieved = retrieve(query)

    context = "\n\n".join(r["text"] for r in retrieved)

    prompt = f"""You are an expert academic advisor for William & Mary.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# =====================================================
# GRADIO
# =====================================================

def chat(message, history=None):
    return generate_answer(message)


iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about William & Mary..."),
    outputs="text",
    title="William & Mary AI Assistant",
    description="RAG + FAISS + BM25 + QLoRA"
)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )