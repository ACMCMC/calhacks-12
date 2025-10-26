"""
Simple Gemini API Integration for Ad Customization
Uses requests library instead of Python SDK for better reliability
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdCustomizationRequest(BaseModel):
    """Request model for ad customization."""
    ad_context: Dict[str, Any]
    web_context: Dict[str, Any]
    customization_preferences: Optional[Dict[str, Any]] = None

class AdCustomizationResponse(BaseModel):
    """Response model for customized ad."""
    customized_ad_text: str
    original_ad_id: str
    customization_applied: Dict[str, Any]
    confidence_score: float
    generation_metadata: Dict[str, Any]

class GeminiAdCustomizer:
    """Service for customizing ads."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the customizer."""
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        
        if not self.api_key:
            logger.warning("No GOOGLE_AI_API_KEY found - using template-based customization only")
        else:
            logger.info(f"✅ Gemini API key configured (first 20 chars: {self.api_key[:20]}...)")
    
    def customize_ad(self, request: AdCustomizationRequest) -> AdCustomizationResponse:
        """Customize an ad based on web context."""
        try:
            ad_id = request.ad_context.get('ad_id', 'unknown')
            original_description = request.ad_context.get('description', '')
            web_keywords = request.web_context.get('keywords', [])
            page_type = request.web_context.get('page_type', 'general')
            page_title = request.web_context.get('title', 'this page')
            
            logger.info(f"Customizing ad {ad_id} for {page_type} page: {page_title}")
            
            # Try Gemini first if API key exists
            if self.api_key:
                try:
                    customized_text = self._customize_with_gemini_http(
                        original_description,
                        page_type,
                        web_keywords,
                        page_title
                    )
                    logger.info(f"✅ Gemini customization successful")
                    return AdCustomizationResponse(
                        customized_ad_text=customized_text,
                        original_ad_id=ad_id,
                        customization_applied={
                            'method': 'gemini_http',
                            'web_context_integrated': True,
                            'keywords_incorporated': len(web_keywords) > 0
                        },
                        confidence_score=0.85,
                        generation_metadata={'model': 'gemini-2.0-flash-001', 'method': 'http'}
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Gemini failed ({type(e).__name__}: {str(e)[:100]}), falling back to template")
            
            # Fallback to template-based
            customized_text = self._customize_with_template(
                original_description,
                page_type,
                web_keywords,
                page_title
            )
            logger.info(f"Using template-based customization")
            
            return AdCustomizationResponse(
                customized_ad_text=customized_text,
                original_ad_id=ad_id,
                customization_applied={
                    'method': 'template',
                    'web_context_integrated': True,
                    'keywords_incorporated': len(web_keywords) > 0
                },
                confidence_score=0.65,
                generation_metadata={'model': 'template_based'}
            )
            
        except Exception as e:
            logger.error(f"Error in customize_ad: {e}", exc_info=True)
            raise
    
    def _customize_with_gemini_http(self, ad_text: str, page_type: str, keywords: List[str], page_title: str) -> str:
        """Use Gemini API via direct HTTP request."""
        keywords_str = ', '.join(keywords[:5]) if keywords else 'your needs'
        
        prompt = f"""You are an expert copywriter. Create a short, compelling ad (2-3 sentences) that:
1. Mentions the core value: {ad_text}
2. Relates to this page: {page_title}
3. Incorporates these keywords naturally: {keywords_str}
4. Matches the tone for a {page_type} page

Return ONLY the ad text, nothing else."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        logger.info(f"Calling Gemini API via HTTP (gemini-2.0-flash-001)...")
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            
            logger.info(f"Gemini response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text[:200]
                logger.error(f"Gemini API error: {response.status_code} - {error_text}")
                raise Exception(f"Gemini API returned {response.status_code}")
            
            data = response.json()
            
            # Extract text from response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                content = candidate.get('content', {})
                parts = content.get('parts', [])
                if parts and len(parts) > 0:
                    text = parts[0].get('text', '').strip()
                    if text:
                        logger.info(f"Got response: {text[:80]}...")
                        return text
            
            raise Exception("No valid text in Gemini response")
            
        except requests.Timeout:
            logger.error("❌ Gemini API timeout after 15 seconds")
            raise
        except requests.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error parsing Gemini response: {e}")
            raise
    
    def _customize_with_template(self, ad_text: str, page_type: str, keywords: List[str], page_title: str) -> str:
        """Simple template-based customization when Gemini isn't available."""
        
        # Build a customized version using templates
        if keywords:
            keyword = keywords[0]
            base = f"{ad_text} Perfect for {keyword}."
        else:
            base = ad_text
        
        # Adapt tone based on page type
        if page_type == 'news':
            return f"Breaking: {base} Don't miss out."
        elif page_type == 'education':
            return f"Learn more: {base} Enhance your knowledge."
        elif page_type == 'wellness':
            return f"Health tip: {base} Improve your wellbeing."
        elif page_type == 'ecommerce':
            return f"Shop now: {base} Limited time offer!"
        else:
            return f"{base} Discover today!"

# Initialize globally
gemini_customizer = GeminiAdCustomizer()