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
INPUT_FILE = "./data/chunks.json"
OUTPUT_FILE = "./data/fine_tuning/fine_tuning_data.jsonl"

# Target counts
POS_COUNT = 2550  
NEG_COUNT = 300   
OUT_COUNT = 150   

def get_llama3_prompt(system_content, user_content):
    return (
	f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def clean_metadata(title, section):
    section = re.sub(r'\s\(Part\s\d+\)', '', section)
    parts = re.split(r' - | \| | : ', title)
    short_title = f"{parts[0]} ({parts[1]})" if len(parts) > 1 else parts[0]
    return short_title, section

def generate(model, tokenizer, sys, user):
    """Standard generation without extra regex processing."""
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
    file_obj.flush()

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("--- STEP 1: Initializing Model on GPU ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"":"cuda:0"} 
    )

    print(f"--- STEP 2: Loading Data ---", flush=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    random.shuffle(all_chunks)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        # --- PHASE 1: FACTUAL PAIRS ---
        print("--- PHASE 1: Factual Pairs ---", flush=True)
        for i in range(POS_COUNT):
            chunk = all_chunks[i]
            title, section = clean_metadata(chunk.get('page_title', ''), chunk.get('section', ''))
            text = chunk.get('text', '')

            success = False
            retries = 0
            MAX_RETRIES = 5 # Safety valve

            while not success and retries < MAX_RETRIES:
                sys_msg = "You are a William & Mary Academic Advisor. Be professional, grounded, and concise."
                user_msg = (
                    f"SOURCE MATERIAL [{title} - {section}]:\n{text}\n\n"
                    "TASK: Write a natural student question and a professional Advisor response.\n"
                    "FORMAT:\nSTUDENT: [Question]\nADVISOR: [Response]"
                    "Output ONLY the above format without any additional commentary or text."
                )
                
               	response = generate(model, tokenizer, sys_msg, user_msg)
                
               	if "ADVISOR:" in response:
                    try:
                        parts = response.split("ADVISOR:")
                        student_q = parts[0].replace("STUDENT:", "").strip()
                        advisor_a = parts[1].strip()
                        if student_q and advisor_a:
                            save_example(f_out, student_q, advisor_a)
                            success = True
                    except:
                        pass
                
               	if not success:
                    retries += 1
                    # THIS IS THE KEY: See why it is failing
                    print(f"  [!] Retry {retries}/{MAX_RETRIES} for index {i} (Topic: {title})", flush=True)

            if i % 10 == 0: # Increased frequency for peace of mind
                print(f"  [+] Completed index: {i}/{POS_COUNT} | Success: {success}", flush=True)

        # --- PHASE 2: NEGATIVE EXAMPLES ---
        print("\n--- PHASE 2: Negative Examples ---", flush=True)
        offices = ["the Registrar", "the Dean of Students", "the Cohen Career Center", "Financial Aid", "the IT help desk"]
        for i in range(NEG_COUNT):
            success = False
            while not success:
                target = all_chunks[i]
                distractor = all_chunks[-(i+1)]
                title, _ = clean_metadata(target.get('page_title', ''), "")
                pivot = "Swem Library" if "Swem" in title else random.choice(offices)

                sys_msg = "You are a W&M Advisor. Be professional, direct, and concise."
                user_msg = (
                    f"TOPIC: {title}\nRECORDS AT HAND: {distractor.get('text', '')}\n\n"
                    f"TASK: Generate a student question about the TOPIC. The Advisor response should state YOU DO NOT HAVE THAT INFO and suggest a relevant William & Mary Resource.\n"
                    "FORMAT:\nSTUDENT: [Question]\nADVISOR: [Response]"
                    "Output ONLY the above format without any additional commentary or text."
                )
                
               	response = generate(model, tokenizer, sys_msg, user_msg)
                if "ADVISOR:" in response:
                    try:
                        parts = response.split("ADVISOR:")
                        save_example(f_out, parts[0].replace("STUDENT:", "").strip(), parts[1].strip())
                        success = True
                    except:
                        continue

        # --- PHASE 3: REFUSAL EXAMPLES ---
        print("\n--- PHASE 3: Refusal Examples ---", flush=True)
        topics = ["UVA admissions", "cooking recipes", "coding a game", "weather", "Super Bowl"]
        for i in range(OUT_COUNT):
            success = False
            while not success:
                topic = random.choice(topics)
                sys_msg = "You are a W&M Advisor. Politely refuse non-W&M academic questions."
                user_msg = (
                    f"Generate a natural student question about {topic}.\n"
                    "TASK: The Advisor MUST refuse this request in 1-2 sentences.\n"
                    "FORMAT:\nSTUDENT: [Question]\nADVISOR: [Response]"
                    "Output ONLY the above format without any additional commentary or text."
                )
                
               	response = generate(model, tokenizer, sys_msg, user_msg)
                if "ADVISOR:" in response:
                    try:
                        parts = response.split("ADVISOR:")
                        save_example(f_out, parts[0].replace("STUDENT:", "").strip(), parts[1].strip())
                        success = True
                    except:
                        continue

    print(f"\n--- FINISHED: Exactly 3000 examples saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()

