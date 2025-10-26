"""
Web Text Extraction Service
Extracts important text content from web pages for ad context analysis.
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import json
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebTextExtractor:
    """Extracts and processes important text content from web pages."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def extract_page_content(self, url: str) -> Dict[str, any]:
        """
        Extract important text content from a web page.
        
        Args:
            url: The URL of the web page to extract content from
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract different types of content
            content = {
                'url': url,
                'title': self._extract_title(soup),
                'headings': self._extract_headings(soup),
                'paragraphs': self._extract_paragraphs(soup),
                'meta_description': self._extract_meta_description(soup),
                'keywords': self._extract_keywords(soup),
                'main_content': self._extract_main_content(soup),
                'domain': urlparse(url).netloc,
                'page_type': self._classify_page_type(soup, url)
            }
            
            # Generate summary text for ad context
            content['summary_text'] = self._generate_summary(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'summary_text': f"Content from {urlparse(url).netloc}"
            }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings (h1-h6)."""
        headings = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = tag.get_text().strip()
            if text:
                headings.append(text)
        return headings
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraph text."""
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 20:  # Only include substantial paragraphs
                paragraphs.append(text)
        return paragraphs
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '') if meta_desc else ""
    
    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords from meta tags and content."""
        keywords = []
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            keywords.extend([k.strip() for k in meta_keywords.get('content', '').split(',')])
        
        # Extract keywords from headings and paragraphs
        all_text = ' '.join(self._extract_headings(soup) + self._extract_paragraphs(soup))
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words as keywords
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords.extend([word for word, freq in frequent_words if freq > 2])
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content area."""
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '#content'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content.get_text().strip()
        
        # Fallback to body content
        body = soup.find('body')
        return body.get_text().strip() if body else ""
    
    def _classify_page_type(self, soup: BeautifulSoup, url: str) -> str:
        """Classify the type of page (news, blog, e-commerce, etc.)."""
        url_lower = url.lower()
        title = self._extract_title(soup).lower()
        
        # Simple classification based on URL patterns and content
        if any(word in url_lower for word in ['news', 'article', 'post']):
            return 'news'
        elif any(word in url_lower for word in ['shop', 'store', 'buy', 'product']):
            return 'ecommerce'
        elif any(word in url_lower for word in ['blog', 'blogger']):
            return 'blog'
        elif any(word in title for word in ['news', 'breaking', 'update']):
            return 'news'
        else:
            return 'general'
    
    def _generate_summary(self, content: Dict) -> str:
        """Generate a summary text for ad context."""
        parts = []
        
        if content.get('title'):
            parts.append(f"Page: {content['title']}")
        
        if content.get('meta_description'):
            parts.append(f"Description: {content['meta_description']}")
        
        if content.get('headings'):
            # Take first few headings
            headings_text = ' '.join(content['headings'][:3])
            parts.append(f"Topics: {headings_text}")
        
        if content.get('keywords'):
            keywords_text = ', '.join(content['keywords'][:5])
            parts.append(f"Keywords: {keywords_text}")
        
        return ' | '.join(parts)

# Example usage and testing
if __name__ == "__main__":
    extractor = WebTextExtractor()
    
    # Test with a sample URL
    test_url = "https://example.com"
    result = extractor.extract_page_content(test_url)
    
    print("Extracted Content:")
    print(json.dumps(result, indent=2))

