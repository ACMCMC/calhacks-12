"""
Main Web Ad Service API
Orchestrates the complete web ad customization pipeline.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # Only look in current directory

# Import our services
from web_text_extractor import WebTextExtractor
from best_ad_service import BestAdService
from gemini_customizer import GeminiAdCustomizer
from ad_injection import AdInjectionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
text_extractor = WebTextExtractor()
best_ad_service = BestAdService()
gemini_customizer = GeminiAdCustomizer()
injection_service = AdInjectionService()

# FastAPI app
app = FastAPI(
    title="PrivAds Web Ad Service",
    description="Complete web ad customization pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class WebAdRequest(BaseModel):
    """Complete web ad customization request."""
    url: str
    user_embedding: Optional[List[float]] = None
    user_id: Optional[str] = None
    injection_method: str = "web_component"
    position: str = "bottom"
    customization_preferences: Optional[Dict[str, Any]] = None

class WebAdResponse(BaseModel):
    """Complete web ad customization response."""
    success: bool
    ad_id: str
    customized_ad_text: str
    injection_code: str
    web_context: Dict[str, Any]
    ad_context: Dict[str, Any]
    customization_metadata: Dict[str, Any]
    error_message: Optional[str] = None

class QuickAdRequest(BaseModel):
    """Quick ad request for testing."""
    url: str
    ad_text: str

class QuickAdResponse(BaseModel):
    """Quick ad response."""
    injection_code: str
    success: bool

@app.post("/web_ad/complete", response_model=WebAdResponse)
async def get_complete_web_ad(request: WebAdRequest):
    """
    Complete web ad customization pipeline:
    1. Extract web content
    2. Find best ad
    3. Customize with Gemini
    4. Generate injection code
    """
    try:
        logger.info(f"Processing complete web ad request for URL: {request.url}")
        
        # Step 1: Extract web content
        web_content = text_extractor.extract_page_content(request.url)
        logger.info(f"Extracted web content: {web_content.get('page_type', 'unknown')} page")
        
        # Step 2: Find best ad
        from best_ad_service import WebContentRequest as BestAdReq
        best_ad_req = BestAdReq(
            url=request.url,
            user_embedding=request.user_embedding,
            user_id=request.user_id
        )
        
        best_ad_response = best_ad_service.get_best_ad(best_ad_req)
        logger.info(f"Found best ad: {best_ad_response.ad_id}")
        
        # Step 3: Customize ad with Gemini
        from gemini_customizer import AdCustomizationRequest as CustomizationReq
        customization_req = CustomizationReq(
            ad_context={
                'ad_id': best_ad_response.ad_id,
                'description': best_ad_response.description,
                'ad_features': best_ad_response.ad_features,
                'content_signals': best_ad_response.content_signals
            },
            web_context=web_content,
            customization_preferences=request.customization_preferences
        )
        
        customized_response = gemini_customizer.customize_ad(customization_req)
        logger.info(f"Customized ad with confidence: {customized_response.confidence_score}")
        
        # Step 4: Generate injection code
        from ad_injection import AdInjectionRequest as InjectionReq
        injection_req = InjectionReq(
            customized_ad={
                'customized_ad_text': customized_response.customized_ad_text,
                'original_ad_id': customized_response.original_ad_id,
                'confidence_score': customized_response.confidence_score
            },
            target_url=request.url,
            injection_method=request.injection_method,
            position=request.position
        )
        
        injection_response = injection_service.inject_ad(injection_req)
        logger.info(f"Generated injection code: {injection_response.success}")
        
        return WebAdResponse(
            success=True,
            ad_id=best_ad_response.ad_id,
            customized_ad_text=customized_response.customized_ad_text,
            injection_code=injection_response.injection_code,
            web_context=web_content,
            ad_context={
                'ad_id': best_ad_response.ad_id,
                'description': best_ad_response.description,
                'similarity_score': best_ad_response.similarity_score,
                'ad_features': best_ad_response.ad_features
            },
            customization_metadata={
                'confidence_score': customized_response.confidence_score,
                'customization_applied': customized_response.customization_applied,
                'generation_metadata': customized_response.generation_metadata
            }
        )
        
    except Exception as e:
        logger.error(f"Error in complete web ad pipeline: {e}")
        return WebAdResponse(
            success=False,
            ad_id="",
            customized_ad_text="",
            injection_code="",
            web_context={},
            ad_context={},
            customization_metadata={},
            error_message=str(e)
        )

@app.post("/web_ad/quick", response_model=QuickAdResponse)
async def get_quick_ad(request: QuickAdRequest):
    """
    Quick ad injection for testing - bypasses ML pipeline.
    """
    try:
        # Generate simple injection code
        from ad_injection import AdInjectionRequest
        injection_req = AdInjectionRequest(
            customized_ad={
                'customized_ad_text': request.ad_text,
                'original_ad_id': 'quick_test',
                'confidence_score': 1.0
            },
            target_url=request.url,
            injection_method="web_component"
        )
        
        injection_response = injection_service.inject_ad(injection_req)
        
        return QuickAdResponse(
            injection_code=injection_response.injection_code,
            success=injection_response.success
        )
        
    except Exception as e:
        logger.error(f"Error in quick ad: {e}")
        return QuickAdResponse(
            injection_code="",
            success=False
        )

@app.get("/web_ad/extract/{url:path}")
async def extract_web_content(url: str):
    """
    Extract web content from a URL for testing.
    """
    try:
        content = text_extractor.extract_page_content(url)
        return {
            "success": True,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/web_ad/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "text_extractor": "active",
            "best_ad_service": "active" if best_ad_service.encoder else "fallback",
            "gemini_customizer": "active" if gemini_customizer.model else "fallback",
            "injection_service": "active"
        }
    }

@app.get("/web_ad/test")
async def test_endpoint():
    """Test endpoint with sample data."""
    sample_url = "https://example.com"
    
    try:
        # Test web content extraction
        web_content = text_extractor.extract_page_content(sample_url)
        
        # Test quick ad injection
        test_request = QuickAdRequest(
            url=sample_url,
            ad_text="Test advertisement for demonstration purposes"
        )
        
        quick_response = await get_quick_ad(test_request)
        
        return {
            "success": True,
            "web_content_extracted": bool(web_content.get('summary_text')),
            "quick_ad_generated": quick_response.success,
            "sample_injection_code": quick_response.injection_code[:200] + "..." if len(quick_response.injection_code) > 200 else quick_response.injection_code
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

