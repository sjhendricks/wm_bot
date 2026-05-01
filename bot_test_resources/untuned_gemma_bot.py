import faiss
import json
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DATA_DIR = "/sciclone/scr10/gzdata440/wm_bot/data/rag"
BASE_MODEL_ID = "google/gemma-2-9b-it"

# ==========================================
# 1. LOAD RETRIEVAL ASSETS
# ==========================================
print("--- Loading Retrieval Assets ---")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

index = faiss.read_index(f"{DATA_DIR}/faiss.index")

with open(f"{DATA_DIR}/chunks.json", "r") as f:
    text_chunks = json.load(f)

with open(f"{DATA_DIR}/bm25.pkl", "rb") as f:
    bm25_obj = pickle.load(f)

# ==========================================
# 2. LOAD UNTUNED LLM
# ==========================================
def load_untuned_llm():
    print("--- Loading Untuned Base Brain ---")
    tk = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    return base_model, tk

llm, tokenizer = load_untuned_llm()

# ==========================================
# 3. TRUE HYBRID RETRIEVAL LOGIC (FIXED)
# ==========================================
def hybrid_retrieve(query, embed_model, cross_encoder, index, text_chunks, bm25, top_k=5):
    # 🔹 Step 1: BM25 (Get Top 20 based on keywords)
    # Fixed: lowercase query for accurate BM25 matching
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:20]

    # 🔹 Step 2: FAISS (Get Top 20 based on semantic meaning)
    query_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, faiss_top = index.search(query_emb, 20) 
    faiss_top = faiss_top[0]

    # 🔹 Merge the two lists of indices and remove duplicates
    candidates = sorted(list(set(bm25_top).union(set(faiss_top))))

    # 🔥 Step 3: Cross-encoder reranking
    # Fixed: Safe dictionary extraction checking 'text' and 'content'
    pairs = [[query, text_chunks[i].get('text', text_chunks[i].get('content', ''))] for i in candidates]
    ce_scores = cross_encoder.predict(pairs)

    # Sort the merged candidates by CE score
    final_idx_positions = np.argsort(ce_scores)[::-1][:top_k]

    results = []
    for pos in final_idx_positions:
        i = candidates[pos]
        results.append({
            "score": float(ce_scores[pos]),
            # Fixed: Safe extraction for final output
            "text": text_chunks[i].get('text', text_chunks[i].get('content', '')),
            "meta": text_chunks[i]  
        })

    return results

# ==========================================
# 4. CHAT LOOP
# ==========================================
print("\n" + "="*50)
print("W&M ADVISOR BOT ACTIVE (Type 'exit' to stop)")
print("="*50 + "\n")

# Standardized System Prompt
SYSTEM_PROMPT = (
    "You are a professional William & Mary Academic Advisor. Your sole purpose is to assist "
    "students with W&M-related inquiries. If a student asks a question that is unrelated to "
    "William & Mary, you must politely decline to answer and offer to help them with their "
    "academic journey instead. Do not provide general knowledge or instructions outside of this scope."
)

while True:
    user_input = input("\nStudent: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    # Run the True Hybrid Retrieval
    retrieved_context_list = hybrid_retrieve(
        user_input, embed_model, cross_encoder, index, text_chunks, bm25_obj
    )
    
    # Combine retrieved text chunks into a single string
    retrieved_context_str = "\n\n".join([item['text'] for item in retrieved_context_list])

    # Standardized User/RAG Instruction (updated with the referral logic)
    user_instruction = f"""Using the following W&M context, answer the student's question. If the context does not contain the answer, politely state that you do not have that information in your current records and suggest the appropriate W&M department, office, or faculty member the student should contact for help. If the question is entirely unrelated to W&M, politely decline to answer.

    Context:
    {retrieved_context_str}

    Student Question:
    {user_input}"""

    # --- GEMMA 2 SPECIFIC FIX ---
    # We combine the system and user messages into one user message to avoid the TemplateError
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_instruction}"}
    ]

    # Apply formatting
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    
    outputs = llm.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id 
    )

    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\nAdvisor: {response.strip()}")
    print("-" * 30)
    print(f"Source Context Used: {str(retrieved_context_list)[:150]}...")