import trafilatura
import json
import os
data = os.environ.get('data', 'default')

def process_txt_to_catalog_json(input_folder, output_file=f"data/clean/{data}_cleaned.json"):
    """
    Refines raw TXT files into a clean JSON list with 'text' and 'page_title'.
    This format is optimized for both RAG embeddings and Synthetic Fine-Tuning.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    abs_input_path = os.path.abspath(input_folder)
    
    print(f"--- W&M Data Refinery: Metadata Extraction Mode ---")
    txt_files = [f for f in os.listdir(abs_input_path) if f.endswith('.txt')]
    
    catalog_data = [] # We'll collect all objects in a list for a standard .json file
    processed_count = 0

    for filename in txt_files:
        file_path = os.path.join(abs_input_path, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as in_f:
                lines = in_f.readlines()

            # --- 1. TITLE EXTRACTION LOGIC ---
            display_name = ""
            content_start_index = 0
            
            for i, line in enumerate(lines):
                clean_line = line.strip()
                # Look for the Page Title header usually added by scrapers
                if clean_line.startswith("Page Title:"):
                    display_name = clean_line.replace("Page Title:", "").split('|')[0].strip()
                    content_start_index = i + 2 # Skip the title line and the separator
                    break
            
            # Fallback to filename if title line is missing or garbage
            if not display_name or len(display_name) < 3:
                display_name = filename.replace('www_wm_edu_', '').replace('.txt', '').replace('_', ' ').title()

            # --- 2. BODY EXTRACTION LOGIC ---
            actual_body_raw = "".join(lines[content_start_index:]).strip()

            clean_markdown = trafilatura.extract(
                actual_body_raw,
                output_format='markdown',
                include_tables=True,
                include_formatting=True,
                favor_precision=True,
                no_fallback=True
            )

            # Quality Filter
            if not clean_markdown or len(clean_markdown) < 150:
                continue

            # --- 3. THE NEW SIMPLIFIED STRUCTURE ---
            # This provides the 'page_title' metadata you need for the next script
            refined_entry = {
                "page_title": display_name,
                "text": clean_markdown.strip()
            }

            catalog_data.append(refined_entry)
            processed_count += 1
            print(f"  [+] Refined: {display_name}")

        except Exception as e:
            print(f"  [X] Error refining {filename}: {e}")

    # Save as a single JSON list (easier for your next script to load)
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(catalog_data, out_f, indent=2, ensure_ascii=False)

    print(f"\nRefinery Complete! Created {processed_count} metadata-tagged entries.")

if __name__ == "__main__":
    # Update these paths for your Sciclone setup
    INPUT_DIR = f"data/raw/{data}"
    OUTPUT_JSON = f"data/clean/{data}_cleaned.json"
    process_txt_to_catalog_json(INPUT_DIR, OUTPUT_JSON)