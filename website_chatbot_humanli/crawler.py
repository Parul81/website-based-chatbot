"""
Website Crawler Module
Extracts clean text content from a given URL.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional


def crawl_website(url: str) -> Optional[dict]:
    """
    Crawl a website and extract clean text content.
    
    Args:
        url: The URL to crawl
        
    Returns:
        Dictionary with 'text' and 'title' keys, or None if error
    """
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "Untitled Page"
        
        # Extract main content
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            # Get text and clean it
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'text': text,
            'title': title_text,
            'url': url
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None

