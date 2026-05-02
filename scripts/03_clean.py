import json
import os
import re
from glob import glob

# --- CONFIGURATION ---
INPUT_FOLDER = "./data/clean/"
OUTPUT_FILE = "./data/cleaned.json"

def clean_text(text):
    """
    Standardizes text by removing HTML artifacts and collapsing whitespace.
    """
    # 1. Remove leftover HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 2. Normalize whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

def is_high_quality(text, page_title):
    """
    Detects 'Pipe Forests' and low-content extractions.
    Returns False if the text appears to be a broken table or junk.
    """
    if not text.strip(): 
        return False
    
    # 1. Count actual alphabet letters
    letters = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text)
    
    if total_chars == 0:
        return False

    # 2. Letter Density Check
    # If less than 25% of the characters are letters, it's likely a broken table/junk
    density = letters / total_chars
    
    # 3. Pipe Overload Check
    # Broken tables often have a massive number of | relative to actual content
    pipes = text.count('|')
    
    if density < 0.25:
        print(f"  [!] Dropping low-density entry: {page_title} ({density:.2f} density)")
        return False
    
    if pipes > (letters * 1.5):
        print(f"  [!] Dropping 'Pipe Forest': {page_title} ({pipes} pipes vs {letters} letters)")
        return False
        
    return True

def main():
    all_data = []
    seen_texts = set()

    # Search for the .json files produced by your refinery
    files = glob(os.path.join(INPUT_FOLDER, "*.json"))

    print(f"🔹 Found {len(files)} catalog files to process")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"  [>] Processing: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)

                for obj in data_list:
                    raw_text = obj.get("text", "")
                    page_title = obj.get("page_title", "William & Mary Information")

                    # Apply basic cleaning
                    text = clean_text(raw_text)

                    # --- FILTERS ---
                    
                    # 1. Quality/Table Filter (The fix for the pipes)
                    if not is_high_quality(text, page_title):
                        continue

                    # 2. Size Filter: Skip fragments (increased threshold slightly)
                    if len(text) < 150:
                        continue
                    
                    # 3. Junk Keyword Filter
                    junk_keywords = ["Resource Not Found", "404 Error", "Page not found", "Forbidden"]
                    if any(kw in text for kw in junk_keywords):
                        continue

                    # 4. Deduplication
                    if text in seen_texts:
                        continue

                    seen_texts.add(text)

                    # --- STRUCTURE ---
                    all_data.append({
                        "page_title": page_title,
                        "text": text,
                        "source": filename 
                    })

        except Exception as e:
            print(f"  [X] Failed to parse {filename}: {e}")

    # Save merged cleaned file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Clean-up Complete!")
    print(f"   Unique documents retained: {len(all_data)}")
    print(f"   Documents discarded: {len(seen_texts) - len(all_data) if len(seen_texts) > len(all_data) else 'N/A'}")
    print(f"   Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()