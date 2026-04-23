import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import random
import sys
import re

# --- HPC SAFETY SETTINGS ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_PATH = "/sciclone/scr10/gzdata440/wm_bot"
INPUT_FILE = os.path.join(BASE_PATH, "data/chunks.json")
# We use a separate file so you can 'cat' them together later
OUTPUT_FILE = os.path.join(BASE_PATH, "data/fine_tuning/special_cases.jsonl")

NEG_COUNT = 300   
OUT_COUNT = 150   

def get_llama3_prompt(system_content, user_content):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def clean_metadata(title, section):
    """Shortens titles and removes (Part X) artifacts."""
    section = re.sub(r'\s\(Part\s\d+\)', '', section)
    parts = re.split(r' - | \| | : ', title)
    # Take top two breadcrumbs for a natural sounding query
    short_title = f"{parts[0]} ({parts[1]})" if len(parts) > 1 else parts[0]
    return short_title, section

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("--- STEP 1: Initializing Model (Offline Mode) ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"":"cuda:0"} 
    )

    print(f"--- STEP 2: Loading Data for Special Cases ---", flush=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    random.shuffle(all_chunks)
    
    # We use the shuffled list to pick topics and distractors
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        # --- PHASE 1: NEGATIVE EXAMPLES (I DON'T KNOW) ---
        print(f"--- PHASE 1: Generating {NEG_COUNT} Negative Pairs ---", flush=True)
        offices = ["the Registrar", "the Dean of Students", "the Cohen Career Center", "Financial Aid", "the IT help desk"]

        for i in range(NEG_COUNT):
            target = all_chunks[i]
            distractor = all_chunks[-(i+1)] 
            
            title, section = clean_metadata(target.get('page_title', ''), target.get('section', ''))
            wrong_context = distractor.get('text', '')
            pivot = "Swem Library" if "Swem" in title else random.choice(offices)

            sys_msg = (
                "You are a W&M Advisor. Be professional, direct, and concise. "
                "DO NOT use introductory filler or repeat long titles. "
                "If the records don't match the topic, admit it and pivot to the right office."
            )
            
            user_msg = (
                f"TOPIC: {title} - {section}\n"
                f"RECORDS AT HAND: {wrong_context}\n\n"
                f"TASK: Generate a 1-sentence student question about the TOPIC. "
                f"Then, generate a 2-sentence Advisor response stating you don't have that info and suggesting {pivot}.\n"
                "FORMAT:\nSTUDENT: [Question]\nADVISOR: [Response]"
            )
            
            raw_output = generate(model, tokenizer, sys_msg, user_msg)
            try:
                parts = raw_output.split("ADVISOR:")
                student_q = parts[0].replace("STUDENT:", "").strip()
                advisor_a = parts[1].strip()
                save_example(f_out, student_q, advisor_a)
            except:
                continue

            if i % 25 == 0:
                print(f"  [Negative] Progress: {i}/{NEG_COUNT}", flush=True)

        # --- PHASE 2: OUT-OF-BOUNDS (REFUSALS) ---
        print(f"--- PHASE 2: Generating {OUT_COUNT} Refusal Pairs ---", flush=True)
        topics = ["UVA admissions", "cooking recipes", "coding a game", "weather in Blacksburg", "Super Bowl winners"]
        
        for i in range(OUT_COUNT):
            topic = random.choice(topics)
            sys_msg = "You are a W&M Advisor. Politely refuse non-W&M academic questions in 1 sentence."
            user_msg = (
                f"Generate a natural student question about {topic}.\n"
                "FORMAT:\nSTUDENT: [Question]\nADVISOR: [Response]"
            )
            
            raw_output = generate(model, tokenizer, sys_msg, user_msg)
            try:
                parts = raw_output.split("ADVISOR:")
                save_example(f_out, parts[0].replace("STUDENT:", "").strip(), parts[1].strip())
            except:
                continue

            if i % 50 == 0:
                print(f"  [Refusal] Progress: {i}/{OUT_COUNT}", flush=True)

    print(f"--- Success! Special cases saved to {OUTPUT_FILE} ---", flush=True)

def generate(model, tokenizer, sys, user):
    prompt = get_llama3_prompt(sys, user)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

def save_example(file_obj, human_val, assistant_val):
    entry = {"conversations": [{"from": "human", "value": human_val}, {"from": "gpt", "value": assistant_val}]}
    file_obj.write(json.dumps(entry) + "\n")
    file_obj.flush()

if __name__ == "__main__":
    main()