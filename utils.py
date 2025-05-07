
import re
import requests
from bs4 import BeautifulSoup

def clean_text(text):
    """Clean input text by removing special characters and extra whitespace."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def scrape_shl_product(url):
    """
    Extract the main product title and description from an SHL product URL.
    Assumes a consistent HTML structure.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('h1').text.strip() if soup.find('h1') else "No title found"
        paragraphs = soup.find_all('p')
        content = " ".join([p.text.strip() for p in paragraphs])

        return {
            "title": title,
            "description": content
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
