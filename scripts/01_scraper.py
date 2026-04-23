import requests
from bs4 import BeautifulSoup
import time
import os
import hashlib
import re
from urllib.parse import urljoin, urlparse

start = os.environ.get('start', 'default')
domain = os.environ.get('domain', 'default')
folder = os.environ.get('folder', 'default')

# --- NECESSARY CHANGE: Update seed URL to the Catalog ---
START_URL = start
DOMAIN = domain
OUTPUT_DIR = f"data/raw/{folder}"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def crawl_site(start_url, max_pages=10000):
    visited_urls = set()
    queue = [start_url]
    pages_scraped = 0

    headers = {
        "User-Agent": "Mozilla/5.0 (HPC Research Crawler; Contact: your_email@wm.edu)"
    }

    while queue and pages_scraped < max_pages:
        current_url = queue.pop(0)

        if current_url in visited_urls:
            continue

        try:
            print(f"[{pages_scraped + 1}] Scraping: {current_url} | Queue size: {len(queue)}")
            response = requests.get(current_url, headers=headers, timeout=10)
            visited_urls.add(current_url)
            
            if "text/html" in response.headers.get("Content-Type", ""):
                soup = BeautifulSoup(response.content, "html.parser")
                html_content = str(soup)
                
                # --- NECESSARY CHANGE: Robust Title Extraction ---
                # This prevents the 'NoneType' split error if a page has no title[cite: 48].
                title_tag = soup.title
                if title_tag and title_tag.string:
                    page_title = title_tag.string.split('|')[0].strip()
                else:
                    page_title = "W&M Catalog Information"
                
                # --- FILENAME LOGIC ---
                readable_part = current_url.replace("https://", "").replace("http://", "")
                readable_part = re.sub(r'[^a-zA-Z0-9]', '_', readable_part).strip('_')
                if not readable_part: readable_part = "catalog_home"
                
                url_hash = hashlib.md5(current_url.encode('utf-8')).hexdigest()
                safe_name = f"{readable_part[:100]}_{url_hash}"
                
                file_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: {current_url}\n")
                    f.write(f"Page Title: {page_title}\n")
                    f.write("="*50 + "\n\n")
                    f.write(html_content)

                pages_scraped += 1
                
                # --- LINK PARSER ---
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link['href'])
                    # Normalizing fragment anchors (#) is critical for catalog navigation
                    full_url = full_url.split('#')[0].rstrip('/')
                    
                    parsed_url = urlparse(full_url)
                    if DOMAIN in parsed_url.netloc and full_url not in visited_urls:
                        queue.append(full_url)

            # Standard delay for HPC politeness
            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"Failed to scrape {current_url}: {e}")

    print(f"\nFinished. Scraped {pages_scraped} pages and visited {len(visited_urls)} URLs.")

if __name__ == "__main__":
    crawl_site(START_URL, max_pages=10000)
