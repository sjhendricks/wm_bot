import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time

def wm_bulletproof_crawler(start_url, output_file="wm_hpc_ready.jsonl", max_pages=15):
    print(f"--- Hardened Crawler Initialized (Python 3.12 compatible) ---")
    
    queue = [start_url]
    seen = {start_url}
    processed_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        while queue and processed_count < max_pages:
            url = queue.pop(0)
            print(f"[{processed_count + 1}] Crawling: {url}")

            try:
                # 1. Fetch raw HTML
                raw_html = trafilatura.fetch_url(url)
                if not raw_html:
                    continue

                # 2. Extract clean text for the LLM
                # We use trafilatura here because its 'cleaner' is top-tier
                clean_content = trafilatura.extract(raw_html, output_format='json', include_tables=True)
                
                if clean_content:
                    page_data = json.loads(clean_content)
                    entry = {
                        "instruction": f"Provide detailed information about {page_data.get('title')}",
                        "input": f"Source: {url}",
                        "output": page_data.get('text', '').strip()
                    }
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    processed_count += 1

                # 3. Find NEW links using BeautifulSoup (The Bulletproof Way)
                soup = BeautifulSoup(raw_html, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    # Convert relative links (/about) to absolute (https://wm.edu/about)
                    full_link = urljoin(url, link)
                    
                    # Clean the link (remove #anchors)
                    full_link = full_link.split('#')[0]

                    # Logic: Must be W&M, not visited yet, and not a file
                    if "wm.edu" in full_link and full_link not in seen:
                        if not any(ext in full_link.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx']):
                            seen.add(full_link)
                            queue.append(full_link)

            except Exception as e:
                print(f"Error on {url}: {e}")

            time.sleep(1) # Be a good neighbor

    print(f"\n--- Success! {processed_count} pages captured in {output_file} ---")

if __name__ == "__main__":
    # Test on the Catalog—this will now capture those .php links easily
    seed_url = "https://catalog.wm.edu/content.php?catoid=34&navoid=5177"
    wm_bulletproof_crawler(seed_url, max_pages=10)