import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import pickle
import numpy as np
import os
import sys

# --- HPC SAFETY & PATHS ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_PATH = "/sciclone/scr10/gzdata440/wm_bot"
DATA_DIR = f"{BASE_PATH}/data/rag"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- STANDARDIZED PROMPTS ---
SYSTEM_PROMPT = (
    "You are a professional William & Mary Academic Advisor. Your sole purpose is to assist "
    "students with W&M-related inquiries. If a student asks a question that is unrelated to "
    "William & Mary, you must politely decline to answer and offer to help them with their "
    "academic journey instead. Do not provide general knowledge or instructions outside of this scope."
)

def load_retrieval_system():
    print("--- Loading Retrieval Assets ---", flush=True)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    index = faiss.read_index(f"{DATA_DIR}/faiss.index")
    with open(f"{DATA_DIR}/chunks.json", "r") as f:
        chunks = json.load(f)
    with open(f"{DATA_DIR}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
        
    return embed_model, cross_encoder, index, chunks, bm25

def load_untuned_llm():
    print("--- Loading Untuned Base Brain ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Llama 3 often needs the pad_token explicitly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"":"cuda:0"} 
    )
    
    # Returning the raw base model without PEFT weights
    return base_model, tokenizer

def hybrid_retrieve(query, embed_model, cross_encoder, index, text_chunks, bm25, top_k=5):
    # 🔹 Step 1: BM25 (Get Top 20 based on keywords)
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
    pairs = [[query, text_chunks[i].get('text', text_chunks[i].get('content', ''))] for i in candidates]
    ce_scores = cross_encoder.predict(pairs)

    # Sort the merged candidates by CE score
    final_idx_positions = np.argsort(ce_scores)[::-1][:top_k]

    results = []
    for pos in final_idx_positions:
        i = candidates[pos]
        results.append({
            "score": float(ce_scores[pos]),
            "text": text_chunks[i].get('text', text_chunks[i].get('content', '')),
            "meta": text_chunks[i]
        })

    return results

def ask_advisor(model, tokenizer, question, context_list):
    # Safe extraction logic
    extracted_texts = []
    for item in context_list:
        if isinstance(item, dict):
            text = item.get('text') or item.get('content') or item.get('chunk')
            if text:
                extracted_texts.append(text)
            elif 'meta' in item and isinstance(item['meta'], dict):
                text = item['meta'].get('text') or item['meta'].get('content')
                extracted_texts.append(text if text else str(item))
            else:
                extracted_texts.append(str(item))
        else:
            extracted_texts.append(str(item))

    retrieved_context_str = "\n\n".join(extracted_texts)
    
    # Standardized User/RAG Instruction
    user_instruction = f"""Using the following W&M context, answer the student's question. If the question is entirely unrelated to W&M, politely decline to answer. If the context does not contain the answer, politely state that you do not have that information in your current records and suggest the appropriate W&M department, office, or faculty member the student should contact for help.

    Context:
    {retrieved_context_str}

    Student Question:
    {question}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the assistant's part
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # Load everything
    e_model, ce_model, idx, text_chunks, bm25_obj = load_retrieval_system()
    llm, tk = load_untuned_llm()
    
    print("\n" + "="*50)
    print("W&M ADVISOR BOT ACTIVE (Type 'exit' to stop)")
    print("="*50)
    
    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # 1. Retrieve the fact
        retrieved_context = hybrid_retrieve(user_input, e_model, ce_model, idx, text_chunks, bm25_obj)
        
        # 2. Generate the advisor's response
        answer = ask_advisor(llm, tk, user_input, retrieved_context)
        
        print(f"\nAdvisor: {answer}")
        print("-" * 30)
        print(f"Source Context Used: {str(retrieved_context)[:150]}...")