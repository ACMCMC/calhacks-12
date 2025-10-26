"""
Gemini API Integration for Ad Customization
Uses Google's Gemini API to generate customized text-based ads.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # Only look in current directory

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
    """Service for customizing ads using Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini ad customizer.
        
        Args:
            api_key: Google AI API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            logger.warning("No Google AI API key provided. Set GOOGLE_AI_API_KEY environment variable.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Gemini API: {e}")
                self.model = None
    
    def customize_ad(self, request: AdCustomizationRequest) -> AdCustomizationResponse:
        """
        Generate a customized ad based on web context and ad features.
        
        Args:
            request: AdCustomizationRequest containing ad and web context
            
        Returns:
            AdCustomizationResponse with customized ad text
        """
        try:
            if not self.model:
                return self._fallback_customization(request)
            
            # Prepare the prompt
            prompt = self._build_customization_prompt(request)
            
            # Generate customized ad
            response = self.model.generate_content(prompt)
            customized_text = response.text.strip()
            
            # Extract metadata
            generation_metadata = {
                'model_used': 'gemini-pro',
                'prompt_length': len(prompt),
                'response_length': len(customized_text),
                'safety_ratings': getattr(response, 'safety_ratings', [])
            }
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(request, customized_text)
            
            # Determine customization applied
            customization_applied = self._analyze_customization_applied(request, customized_text)
            
            return AdCustomizationResponse(
                customized_ad_text=customized_text,
                original_ad_id=request.ad_context.get('ad_id', 'unknown'),
                customization_applied=customization_applied,
                confidence_score=confidence_score,
                generation_metadata=generation_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in ad customization: {e}")
            return self._fallback_customization(request)
    
    def _build_customization_prompt(self, request: AdCustomizationRequest) -> str:
        """Build the prompt for Gemini API."""
        ad_context = request.ad_context
        web_context = request.web_context
        
        # Extract key information
        ad_description = ad_context.get('description', '')
        ad_features = ad_context.get('ad_features', {})
        web_summary = web_context.get('summary_text', '')
        page_type = web_context.get('page_type', 'general')
        web_keywords = web_context.get('keywords', [])
        
        prompt = f"""
You are an expert advertising copywriter tasked with creating a customized text-based advertisement.

ORIGINAL AD CONTEXT:
- Description: {ad_description}
- Category: {ad_features.get('category', 'general')}
- Product Type: {ad_features.get('product_type', 'product')}
- Target Audience: {ad_features.get('target_audience', 'general')}
- Price Range: {ad_features.get('price_range', 'standard')}

WEB PAGE CONTEXT:
- Page Type: {page_type}
- Content Summary: {web_summary}
- Key Topics: {', '.join(web_keywords[:5]) if web_keywords else 'general'}

CUSTOMIZATION REQUIREMENTS:
1. Adapt the ad language to match the web page's tone and context
2. Incorporate relevant keywords from the web content naturally
3. Make the ad feel native to the page content
4. Maintain the core product value proposition
5. Keep the ad concise but compelling (2-3 sentences max)
6. Use appropriate call-to-action based on page context

GENERATE A CUSTOMIZED AD TEXT that:
- Feels natural and relevant to the current web page
- Maintains the original ad's core message
- Incorporates web page context seamlessly
- Is engaging and likely to convert

Format your response as just the customized ad text, nothing else.
"""
        
        return prompt.strip()
    
    def _calculate_confidence_score(self, request: AdCustomizationRequest, customized_text: str) -> float:
        """Calculate confidence score for the customization."""
        score = 0.5  # Base score
        
        # Check if customization incorporates web context
        web_keywords = request.web_context.get('keywords', [])
        if web_keywords:
            keyword_matches = sum(1 for keyword in web_keywords[:5] 
                                if keyword.lower() in customized_text.lower())
            score += (keyword_matches / len(web_keywords[:5])) * 0.3
        
        # Check if original ad message is preserved
        original_description = request.ad_context.get('description', '')
        if original_description:
            # Simple check for key product terms
            original_words = set(original_description.lower().split())
            customized_words = set(customized_text.lower().split())
            overlap = len(original_words.intersection(customized_words))
            if original_words:
                score += (overlap / len(original_words)) * 0.2
        
        return min(score, 1.0)
    
    def _analyze_customization_applied(self, request: AdCustomizationRequest, customized_text: str) -> Dict[str, Any]:
        """Analyze what customizations were applied."""
        customizations = {
            'web_context_integration': False,
            'keyword_incorporation': False,
            'tone_adaptation': False,
            'length_optimization': False
        }
        
        # Check web context integration
        web_keywords = request.web_context.get('keywords', [])
        if web_keywords:
            keyword_matches = sum(1 for keyword in web_keywords[:5] 
                                if keyword.lower() in customized_text.lower())
            customizations['web_context_integration'] = keyword_matches > 0
            customizations['keyword_incorporation'] = keyword_matches > 0
        
        # Check tone adaptation (simple heuristic)
        page_type = request.web_context.get('page_type', 'general')
        if page_type == 'news' and any(word in customized_text.lower() 
                                      for word in ['breaking', 'latest', 'update']):
            customizations['tone_adaptation'] = True
        elif page_type == 'ecommerce' and any(word in customized_text.lower() 
                                             for word in ['buy', 'shop', 'deal']):
            customizations['tone_adaptation'] = True
        
        # Check length optimization
        original_length = len(request.ad_context.get('description', ''))
        customized_length = len(customized_text)
        customizations['length_optimization'] = customized_length < original_length * 1.5
        
        return customizations
    
    def _fallback_customization(self, request: AdCustomizationRequest) -> AdCustomizationResponse:
        """Fallback customization when Gemini API is not available."""
        ad_context = request.ad_context
        web_context = request.web_context
        
        # Simple template-based customization
        original_description = ad_context.get('description', '')
        web_keywords = web_context.get('keywords', [])
        page_type = web_context.get('page_type', 'general')
        
        # Add web context keywords if available
        if web_keywords:
            keyword_text = f" {web_keywords[0]}"
            customized_text = f"{original_description}{keyword_text}. Perfect for your needs!"
        else:
            customized_text = f"{original_description}. Discover more today!"
        
        # Adjust for page type
        if page_type == 'news':
            customized_text = f"Breaking: {customized_text}"
        elif page_type == 'ecommerce':
            customized_text = f"Shop Now: {customized_text}"
        
        return AdCustomizationResponse(
            customized_ad_text=customized_text,
            original_ad_id=ad_context.get('ad_id', 'unknown'),
            customization_applied={
                'web_context_integration': bool(web_keywords),
                'keyword_incorporation': bool(web_keywords),
                'tone_adaptation': True,
                'length_optimization': True
            },
            confidence_score=0.6,  # Lower confidence for fallback
            generation_metadata={
                'model_used': 'fallback_template',
                'fallback_reason': 'Gemini API not available'
            }
        )

# Initialize the customizer
gemini_customizer = GeminiAdCustomizer()

# Example usage and testing
if __name__ == "__main__":
    # Test the customizer
    test_request = AdCustomizationRequest(
        ad_context={
            'ad_id': 'test_ad_001',
            'description': 'Revolutionary smartphone with AI-powered camera',
            'ad_features': {
                'category': 'technology',
                'product_type': 'smartphone',
                'target_audience': 'tech_enthusiasts'
            }
        },
        web_context={
            'summary_text': 'Latest tech news and smartphone reviews',
            'page_type': 'news',
            'keywords': ['smartphone', 'tech', 'review', 'camera', 'ai']
        }
    )
    
    result = gemini_customizer.customize_ad(test_request)
    print("Customized Ad:")
    print(result.customized_ad_text)
    print(f"\nConfidence Score: {result.confidence_score}")
    print(f"Customizations Applied: {result.customization_applied}")

