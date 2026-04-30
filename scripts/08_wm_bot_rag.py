import faiss
import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
MODEL_PATH = "/sciclone/scr10/gzdata440/wm_bot/output/final_model"  # your axolotl output
CHUNKS_FILE = "/sciclone/scr10/gzdata440/wm_bot/data/chunks.json"
FAISS_INDEX  = "/sciclone/scr10/gzdata440/wm_bot/data/faiss.index"
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"  # or whichever embedder you used

class WMRetriever:
    def __init__(self):
        with open(CHUNKS_FILE) as f:
            self.chunks = json.load(f)
        
        # FAISS
        self.index = faiss.read_index(FAISS_INDEX)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        
        # BM25
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=3):
        # Dense retrieval (FAISS)
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        _, faiss_ids = self.index.search(np.array(q_vec, dtype="float32"), top_k)
        dense_hits = set(faiss_ids[0].tolist())

        # Sparse retrieval (BM25)
        scores = self.bm25.get_scores(query.lower().split())
        bm25_ids = np.argsort(scores)[::-1][:top_k].tolist()

        # Merge (simple union, deduplicated)
        combined_ids = list(dict.fromkeys(bm25_ids + list(dense_hits)))[:top_k]
        return [self.chunks[i]["text"] for i in combined_ids if i < len(self.chunks)]


class WMChatbot:
    def __init__(self):
        self.retriever = WMRetriever()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )

    def chat(self, user_query, history):
        context_chunks = self.retriever.retrieve(user_query)
        context = "\n\n".join(context_chunks)

        system_msg = (
            "You are a William & Mary Academic Advisor. Be concise, professional, and accurate. "
            "Answer ONLY using the provided context. If the answer isn't there, say you don't know "
            "and direct the student to the relevant W&M office. Do not use filler phrases."
        )
        user_msg = f"CONTEXT:\n{context}\n\nSTUDENT QUESTION: {user_query}"

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()