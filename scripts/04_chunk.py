import json
import re
import os

# --- CONFIGURATION ---
INPUT_FILE = "data/cleaned.json"
OUTPUT_FILE = "data/rag/chunks.json"

# RAG Best Practices:
# 2000 chars is roughly 400-500 tokens, perfect for Llama-3-8B.
MAX_CHUNK_CHARS = 2000 
CHUNK_OVERLAP = 200

def split_by_chars(text, max_chars=2000, overlap=200):
    """
    Sub-chunks a long section into smaller overlapping blocks.
    Ensures that if a section is 5000 chars, it's broken into ~3 parts.
    """
    sub_chunks = []
    if not text:
        return sub_chunks
        
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        sub_chunks.append(chunk)
        
        # Move start point forward, but stay back by the overlap amount
        start += (max_chars - overlap)
        
        # Safety break to avoid infinite loop on tiny text/huge overlap
        if start >= len(text) or max_chars <= overlap:
            break
            
    return sub_chunks

def split_sections(text):
    """
    Splits a document into sections based on Markdown headers (H1-H3).
    Returns a list of tuples: (header_name, section_content)
    """
    # Regex keeps the header as a separate element in the split list
    sections = re.split(r"(^#{1,3}\s.*)", text, flags=re.MULTILINE)

    results = []
    current_header = "Overview" 

    for part in sections:
        if not part.strip():
            continue
            
        if part.startswith("#"):
            current_header = part.strip("# ").strip()
        else:
            content = part.strip()
            # Initial filter: ignore tiny fragments
            if len(content) > 50:
                results.append((current_header, content))

    return results

def main():
    print(f"--- W&M Recursive Chunker: Header + Character Mode ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []

    for doc in data:
        page_title = doc.get("page_title", "Unknown Page")
        raw_text = doc.get("text", "")
        source_file = doc.get("source", "unknown")

        # Step 1: Split by logical headers
        logical_sections = split_sections(raw_text)

        for section_name, content in logical_sections:
            # Step 2: Check if the section is too long for the LLM context
            if len(content) <= MAX_CHUNK_CHARS:
                # Section is fine as is
                all_chunks.append({
                    "page_title": page_title,
                    "section": section_name,
                    "text": content,
                    "metadata": {
                        "source": source_file,
                        "chunk_type": "semantic"
                    }
                })
            else:
                # Step 3: Recursive sub-chunking for massive sections
                sub_blocks = split_by_chars(content, MAX_CHUNK_CHARS, CHUNK_OVERLAP)
                for i, block in enumerate(sub_blocks):
                    all_chunks.append({
                        "page_title": page_title,
                        "section": f"{section_name} (Part {i+1})",
                        "text": block,
                        "metadata": {
                            "source": source_file,
                            "chunk_type": "recursive_split"
                        }
                    })

    # Save as a standard JSON list
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed {len(data)} documents into {len(all_chunks)} recursive chunks.")
    print(f"   Max Chunk Size: {MAX_CHUNK_CHARS} chars | Overlap: {CHUNK_OVERLAP} chars")

if __name__ == "__main__":
    main()