import trafilatura
import json
import os

def process_txt_to_openai_jsonl(input_folder, output_file="data/clean/catalog_training.jsonl"):
    SYSTEM_PROMPT = (
        "You are an expert academic advisor for the College of William & Mary. "
        "Use the provided university documentation to answer questions accurately."
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    abs_input_path = os.path.abspath(input_folder)
    
    print(f"--- W&M Data Refinery: Precision Mode Engaged ---")
    txt_files = [f for f in os.listdir(abs_input_path) if f.endswith('.txt')]
    
    processed_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in txt_files:
            file_path = os.path.join(abs_input_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    lines = in_f.readlines()

                display_name = "William & Mary Information"
                content_start_index = 0
                for i, line in enumerate(lines):
                    clean_line = line.strip()
                    if clean_line and "Source URL" not in clean_line and "=" not in clean_line:
                        display_name = clean_line.replace("Page Title:", "").split('|')[0].strip()
                        
                        if not display_name or len(display_name) < 2 or display_name in [".", "A"]:
                            display_name = filename.replace('www_wm_edu_', '').replace('.txt', '').replace('_', ' ').title()
                        
                        # FIXED: Changed from i + 1 to i + 2 to skip the '====' separator line
                        content_start_index = i + 2 
                        break

                actual_body_raw = "".join(lines[content_start_index:]).strip()

                clean_markdown = trafilatura.extract(
                    actual_body_raw,
                    output_format='markdown',
                    include_tables=True,
                    include_formatting=True,
                    favor_precision=True,
                    no_fallback=True
                )

                # --- NECESSARY CHANGE: STRICTOR DECISION LOGIC ---
                if clean_markdown and len(clean_markdown) > 100:
                    content_to_use = clean_markdown.strip()
                else:
                    # FIXED: Instead of dumping raw HTML when Trafilatura fails, 
                    # we skip the file to keep the training data high-quality.
                    # Technical UI pages like 'Quicktabs' are not helpful for training.
                    print(f"  [!] Skipping {filename}: Could not extract a clean body.")
                    continue
                
                if len(content_to_use) < 100:
                    continue

                openai_row = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"I need details regarding: {display_name}."},
                        {"role": "assistant", "content": content_to_use}
                    ]
                }

                out_f.write(json.dumps(openai_row) + "\n")
                out_f.flush()
                processed_count += 1
                print(f"  [+] Refined: {display_name}")

            except Exception as e:
                print(f"  [X] Error refining {filename}: {e}")

    print(f"\nRefinery Complete! Created {processed_count} precision entries.")

if __name__ == "__main__":
    INPUT_DIR = "data/raw/catalog"
    process_txt_to_openai_jsonl(INPUT_DIR)