import trafilatura
import json
import os
import re

def process_txt_to_openai_jsonl(input_folder, output_file="wm_refined_training.jsonl"):
    """
    Takes a directory of .txt files, cleans them into Markdown using Trafilatura,
    and formats them for OpenAI-style training.
    """
    
    # --- CONFIGURATION ---
    SYSTEM_PROMPT = (
        "You are an expert academic advisor for the College of William & Mary. "
        "Use the provided university documentation to answer questions accurately "
        "using Markdown formatting for clarity."
    )
    
    print(f"--- W&M Data Refinery Engaged ---")
    print(f"Reading from: {input_folder}")
    
    # Get list of all .txt files in the directory
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("No .txt files found in the directory!")
        return

    processed_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in txt_files:
            file_path = os.path.join(input_folder, filename)
            
            try:
                # 1. READ THE FILE
                with open(file_path, "r", encoding="utf-8") as in_f:
                    raw_content = in_f.read()

                # 2. EXTRACT CLEAN MARKDOWN
                # Even if it's already text, Trafilatura's extract handles 
                # converting HTML remnants into clean Markdown.
                clean_markdown = trafilatura.extract(
                    raw_content,
                    output_format='markdown',
                    include_tables=True,
                    include_formatting=True
                )

                # Skip files that are empty or just "noise" (less than 150 chars)
                if not clean_markdown or len(clean_markdown) < 150:
                    continue

                # 3. GENERATE A CLEAN TITLE FOR THE 'USER' PROMPT
                # We'll use the filename (removing .txt and replacing hyphens/underscores)
                display_name = filename.replace('.txt', '').replace('-', ' ').replace('_', ' ').title()

                # 4. CONSTRUCT OPENAI MESSAGES FORMAT
                openai_row = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"I need details regarding: {display_name}. What does the documentation say?"},
                        {"role": "assistant", "content": clean_markdown.strip()}
                    ]
                }

                # 5. SAVE & FLUSH (HPC Monitor Readiness)
                out_f.write(json.dumps(openai_row) + "\n")
                out_f.flush() # Ensures you can monitor progress on Jumbotron
                
                processed_count += 1
                print(f"[{processed_count}] Processed: {filename}")

            except Exception as e:
                print(f"Error refining {filename}: {e}")

    print(f"\n--- Refinery Complete! ---")
    print(f"Total entries created: {processed_count}")
    print(f"File ready for Axolotl: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Change 'my_scraped_pages' to the folder name where your BeautifulSoup script 
    # saves its .txt files.
    INPUT_DIR = "scraped_data_folder" 
    
    process_txt_to_openai_jsonl(INPUT_DIR)
