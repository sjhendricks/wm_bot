import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import random
import sys

# --- HPC SAFETY SETTINGS ---
# 1. Prevent tokenizer deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. Force Offline Mode (Skip all internet checks)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# Make sure these are ABSOLUTE paths to avoid confusion
BASE_PATH = "/sciclone/scr10/gzdata440/wm_bot"
INPUT_FILE = os.path.join(BASE_PATH, "data/chunks.json")
OUTPUT_FILE = os.path.join(BASE_PATH, "data/fine_tuning/fine_tuning_data.jsonl")

TOTAL_EXAMPLES = 3000
POS_COUNT = 2550  
NEG_COUNT = 300   
OUT_COUNT = 150   

def get_llama3_prompt(system_content, user_content):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Use flush=True on all prints to see them in .out file IMMEDIATELY
    print("--- STEP 1: Initializing Model on GPU ---", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # We specify device 0 to avoid device_map auto-hangs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"":"cuda:0"} 
    )

    print(f"--- STEP 2: Loading Data from {INPUT_FILE} ---", flush=True)
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)
    
    random.shuffle(all_chunks)
    pos_chunks = all_chunks[:POS_COUNT]
    neg_chunks = all_chunks[POS_COUNT : POS_COUNT + NEG_COUNT]
    distractor_pool = all_chunks[POS_COUNT + NEG_COUNT:] 
    
    print("--- STEP 3: Starting Generation Loop ---", flush=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Factual Loop
        for i, chunk in enumerate(pos_chunks):
            title = chunk.get('page_title', 'W&M Catalog')
            section = chunk.get('section', 'Overview')
            text = chunk.get('text', '')

            # Neutralized System Message
            sys_msg = (
                "You are a William & Mary Academic Advisor. You are professional, grounded, and concise. "
                "Your goal is to provide accurate responses based strictly on the provided reference material. "
                "Do not use introductory AI filler (e.g., 'I'm happy to help', 'Sure thing'). "
                "If the reference doesn't contain the answer, say so and suggest a relevant W&M office."
            )

            # Neutralized User Message (No "Catalog" references)
            user_msg = (
                f"SOURCE MATERIAL [{title} - {section}]:\n{text}\n\n"
                "TASK:\n"
                "1. Write a natural, conversational student question about a specific detail in the SOURCE MATERIAL.\n"
                "2. Write a professional, concise Advisor response that answers the question accurately.\n\n"
                "FORMAT:\n"
                "STUDENT: [Question]\n"
                "ADVISOR: [Response]"
            )
            response = generate(model, tokenizer, sys_msg, user_msg)
            
            human_query = f"Regarding {title}, what can you tell me about {section}?"
            save_example(f_out, human_query, response)
            
            if i % 10 == 0: # Log more frequently (every 10) for debugging
                print(f"  [+] Progress: {i}/{TOTAL_EXAMPLES} | Current: {title}", flush=True)

        # Distractor Loop
        print("--- PHASE 2: Negative Examples ---", flush=True)
        # ... (rest of logic remains the same) ...

def generate(model, tokenizer, sys, user):
    prompt = get_llama3_prompt(sys, user)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("assistant")[-1].strip()

def save_example(file_obj, human_val, assistant_val):
    entry = {"conversations": [{"from": "human", "value": human_val}, {"from": "gpt", "value": assistant_val}]}
    file_obj.write(json.dumps(entry) + "\n")
    file_obj.flush() # Forces the write to disk immediately

if __name__ == "__main__":
    main()