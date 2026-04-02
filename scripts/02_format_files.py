import trafilatura
import json
import os

def process_txt_to_openai_jsonl(input_folder, output_file="data_clean/wm_refined_training.jsonl"):
    """
    Refines raw HTML files into OpenAI-formatted JSONL.
    Optimized for files with 'Source URL' and 'Page Title' headers.
    """
    SYSTEM_PROMPT = (
        "You are an expert academic advisor for the College of William & Mary. "
        "Use the provided university documentation to answer questions accurately."
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    abs_input_path = os.path.abspath(input_folder)
    
    print(f"--- W&M Data Refinery: HTML Mode Engaged ---")
    txt_files = [f for f in os.listdir(abs_input_path) if f.endswith('.txt')]
    
    processed_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in txt_files:
            file_path = os.path.join(abs_input_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    lines = in_f.readlines()

                # --- 1. EXTRACT TITLE FROM HEADER ---
                # Your scraper now explicitly saves "Page Title: X" on the second line
                display_name = "William & Mary Information"
                html_start_index = 0
                
                for i, line in enumerate(lines):
                    if line.startswith("Page Title:"):
                        display_name = line.replace("Page Title:", "").strip()
                    if "=====" in line:
                        html_start_index = i + 1
                        break

                # --- 2. SEPARATE HTML FROM METADATA ---
                # We only want to feed the raw HTML to Trafilatura
                raw_html = "".join(lines[html_start_index:]).strip()

                if not raw_html:
                    continue

                # --- 3. PRECISION EXTRACTION ---
                # Because we are providing raw HTML, Trafilatura will use its 
                # structural algorithms to strip navbars and footers automatically.
                clean_markdown = trafilatura.extract(
                    raw_html,
                    output_format='markdown',
                    include_tables=True,
                    include_formatting=True,
                    favor_precision=True # Prioritizes body text over menus
                )

                # --- 4. DECISION & CLEANUP ---
                # If Trafilatura successfully finds the body, use it.
                if clean_markdown and len(clean_markdown) > 150:
                    content_to_use = clean_markdown.strip()
                else:
                    # Fallback: If it's a short page, Trafilatura might be too strict.
                    # We skip these to avoid training the model on "Search" menus.
                    print(f"  [!] Skipping {filename}: No clear body content found.")
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

    print(f"\nRefinery Complete! Created {processed_count} high-quality entries.")

if __name__ == "__main__":
    # Ensure this points to your raw data directory
    INPUT_DIR = "data_raw/test_data" 
    process_txt_to_openai_jsonl(INPUT_DIR)