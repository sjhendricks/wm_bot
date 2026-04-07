import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import sys

def run_wm_llm_scraper(start_url, output_file="wm_catalog_markdown.jsonl", max_pages=50):
    """
    Scrapes W&M content, converts it to Markdown for better LLM reasoning,
    and saves it in a format ready for Axolotl fine-tuning.
    """
    print(f"--- Hardened W&M Scraper Started ---")
    print(f"Targeting: {start_url}")
    print(f"Output File: {output_file}")
    
    queue = [start_url]
    seen = {start_url}
    processed_count = 0

    # Ensure we use the latest 2026 extraction standards for Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        while queue and processed_count < max_pages:
            url = queue.pop(0)
            
            try:
                # 1. Fetch the raw HTML
                raw_html = trafilatura.fetch_url(url)
                if not raw_html:
                    continue

                # 2. Extract clean Markdown (Best for tables and hierarchy)
                # include_tables=True is vital for the W&M Course Catalog
                markdown_content = trafilatura.extract(
                    raw_html, 
                    output_format='markdown',
                    include_tables=True,
                    include_formatting=True,
                    include_comments=False
                )
                
                if markdown_content:
                    # Parse for metadata (title)
                    result_json = trafilatura.extract(raw_html, output_format='json')
                    page_title = json.loads(result_json).get('title', 'William & Mary Information')

                    # 3. Format for Axolotl (Alpaca/Instruction Style)
                    entry = {
                        "instruction": f"Explain the academic requirements or details regarding: {page_title}",
                        "input": f"Source: {url}",
                        "output": markdown_content.strip()
                    }
                    
                    # Write and Force-Flush (Real-time monitoring)
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    
                    processed_count += 1
                    print(f"[{processed_count}/{max_pages}] Successfully processed: {url}")

                # 4. Discover new W&M links using BeautifulSoup
                soup = BeautifulSoup(raw_html, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    full_link = urljoin(url, link).split('#')[0] # Clean anchors

                    # Filter: Must be W&M, unvisited, and not a binary file
                    if "wm.edu" in full_link and full_link not in seen:
                        if not any(ext in full_link.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx']):
                            seen.add(full_link)
                            queue.append(full_link)

            except Exception as e:
                print(f"Error processing {url}: {e}")

            # Politeness delay for university servers
            time.sleep(1.5)

    print(f"\n--- Scrape Complete! Total pages saved: {processed_count} ---")

if __name__ == "__main__":
    # Starting seed: The W&M Academic Regulations Catalog
    # This URL is the most dense 'knowledge silo' for the university.
    seed_url = "https://catalog.wm.edu/content.php?catoid=34&navoid=5177"
    
    # Increase max_pages when moving to the HPC (e.g., 5000)
    run_wm_llm_scraper(seed_url, max_pages=15)
