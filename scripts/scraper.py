import requests
from bs4 import BeautifulSoup
import time
import os
import hashlib
import re
from urllib.parse import urljoin, urlparse

START_URL = "https://www.wm.edu/"
DOMAIN = "wm.edu"
OUTPUT_DIR = "wm_website_data"

# Create a folder to hold our text files
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_text(text):
    # Remove excessive blank lines and whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def crawl_site(start_url, max_pages=5000): # Increased default limit
    visited_urls = set()
    queue = [start_url]
    pages_scraped = 0

    headers = {
        "User-Agent": "Mozilla/5.0 (HPC Research Crawler; Contact: your_email@domain.edu)"
    }

    while queue and pages_scraped < max_pages:
        # We already clean the URL before it enters the queue now, 
        # but popping it is the same.
        current_url = queue.pop(0)

        if current_url in visited_urls:
            continue

        try:
            print(f"[{pages_scraped + 1}] Scraping: {current_url} | Queue size: {len(queue)}")
            response = requests.get(current_url, headers=headers, timeout=10)
            visited_urls.add(current_url)
            
            if "text/html" in response.headers.get("Content-Type", ""):
                soup = BeautifulSoup(response.content, "html.parser")
                
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()
                
                
                raw_text = soup.get_text(separator=' ')
                clean_content = clean_text(raw_text)
                
		# 1. Clean the URL for readability
                readable_part = current_url.replace("https://", "").replace("http://", "")
                readable_part = re.sub(r'[^a-zA-Z0-9]', '_', readable_part).strip("_")
                if not readable_part: readable_part = "home"
                
                # 2. Create a unique 32-character hash from the full URL
                url_hash = hashlib.md5(current_url.encode('utf-8')).hexdigest()
                
                # 3. Combine a truncated readable part (max 100 chars) with the hash
                safe_name = f"{readable_part[:100]}_{url_hash}"
                
                file_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: {current_url}\n")
                    f.write("="*50 + "\n\n")
                    f.write(clean_content)

                pages_scraped += 1
                
                # --- THE MODIFIED LINK PARSER ---
                for link in soup.find_all("a", href=True):
                    # Join the relative link to the base domain
                    full_url = urljoin(current_url, link['href'])
                    
                    # NORMALIZE: Strip anchors (#) and trailing slashes (/)
                    # This prevents treating /about and /about/ as different pages
                    full_url = full_url.split('#')[0].rstrip('/')
                    
                    parsed_url = urlparse(full_url)

                    # Check domain and ensure we haven't seen it yet
                    if DOMAIN in parsed_url.netloc and full_url not in visited_urls and full_url not in queue:
                        queue.append(full_url)

            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"Failed to scrape {current_url}: {e}")

    print(f"\nFinished. Scraped {pages_scraped} pages and visited {len(visited_urls)} unique URLs.")

if __name__ == "__main__":
    # 10,000 pages will take about 5.5 hours to run with a 2-second delay
    crawl_site(START_URL, max_pages=10000)
