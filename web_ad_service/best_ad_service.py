"""
Enhanced Best Ad API Service
Provides intelligent ad matching based on web content and user context.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import numpy as np
from pathlib import Path
import torch
from scipy.spatial.distance import cosine
import logging
from web_text_extractor import WebTextExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebContentRequest(BaseModel):
    """Request model for web content analysis."""
    url: str
    user_embedding: Optional[List[float]] = None
    user_id: Optional[str] = None

class AdContext(BaseModel):
    """Ad context information."""
    ad_id: str
    description: str
    source_url: str
    similarity_score: float
    projected_embedding: List[float]
    ad_features: Dict[str, Any]
    content_signals: Dict[str, Any]

class BestAdResponse(BaseModel):
    """Response model for best ad recommendation."""
    ad_id: str
    description: str
    source_url: str
    similarity_score: float
    projected_embedding: List[float]
    ad_features: Dict[str, Any]
    content_signals: Dict[str, Any]
    web_context: Dict[str, Any]
    recommendation_reason: str

class AdCustomizationRequest(BaseModel):
    """Request for ad customization."""
    ad_context: AdContext
    web_context: Dict[str, Any]
    customization_preferences: Optional[Dict[str, Any]] = None

class AdCustomizationResponse(BaseModel):
    """Response for customized ad."""
    customized_ad_text: str
    original_ad_id: str
    customization_applied: Dict[str, Any]
    confidence_score: float

class BestAdService:
    """Service for finding the best ad based on web content and user context."""
    
    def __init__(self):
        self.text_extractor = WebTextExtractor()
        self.ad_metadata = self._load_ad_metadata()
        self.encoder = None
        self.projector = None
        self._load_models()
    
    def _load_ad_metadata(self) -> Dict:
        """Load ad metadata from the scraped data."""
        try:
            metadata_path = Path("../ad_creatives/scraped_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Ad metadata not found, using mock data")
                return self._create_mock_ad_data()
        except Exception as e:
            logger.error(f"Error loading ad metadata: {e}")
            return self._create_mock_ad_data()
    
    def _create_mock_ad_data(self) -> Dict:
        """Create mock ad data for testing."""
        return {
            "ads": [
                {
                    "id": "mock_ad_001",
                    "description": "Revolutionary smartphone with AI-powered camera and all-day battery life",
                    "source_url": "https://example.com/ads/mock_ad_001.jpg",
                    "features": {
                        "category": "technology",
                        "product_type": "smartphone",
                        "price_range": "premium",
                        "target_audience": "tech_enthusiasts"
                    }
                },
                {
                    "id": "mock_ad_002", 
                    "description": "Sustainable fashion collection made from recycled materials",
                    "source_url": "https://example.com/ads/mock_ad_002.jpg",
                    "features": {
                        "category": "fashion",
                        "product_type": "clothing",
                        "price_range": "mid_range",
                        "target_audience": "eco_conscious"
                    }
                },
                {
                    "id": "mock_ad_003",
                    "description": "Professional development course for data science and machine learning",
                    "source_url": "https://example.com/ads/mock_ad_003.jpg",
                    "features": {
                        "category": "education",
                        "product_type": "online_course",
                        "price_range": "affordable",
                        "target_audience": "professionals"
                    }
                }
            ]
        }
    
    def _load_models(self):
        """Load the ad encoder and projector models."""
        try:
            # Import the encoder and projector classes
            import sys
            sys.path.append('../pipeline/training')
            
            from ad_encoder import AdEncoder
            from projector import Projector
            
            # Initialize models
            self.encoder = AdEncoder(device="cpu")
            self.projector = Projector(d_ad=self.encoder.d_ad, d_user=128)
            
            # Load projector weights
            projector_path = Path("../models/projector.pt")
            if projector_path.exists():
                self.projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
                self.projector.eval()
                logger.info("Models loaded successfully")
            else:
                logger.warning("Projector weights not found, using random initialization")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.encoder = None
            self.projector = None
    
    def get_best_ad(self, request: WebContentRequest) -> BestAdResponse:
        """
        Find the best ad based on web content and user context.
        
        Args:
            request: WebContentRequest containing URL and user info
            
        Returns:
            BestAdResponse with the best matching ad
        """
        try:
            # Extract web content
            web_content = self.text_extractor.extract_page_content(request.url)
            
            # Get user embedding (use provided or generate default)
            if request.user_embedding:
                user_embedding = np.array(request.user_embedding)
            else:
                # Use global mean for cold start
                user_embedding = self._get_default_user_embedding()
            
            # Find best matching ad
            best_ad, similarity_score, projected_embedding = self._find_best_ad(
                web_content, user_embedding
            )
            
            # Generate content signals
            content_signals = self._generate_content_signals(web_content)
            
            # Generate recommendation reason
            reason = self._generate_recommendation_reason(
                best_ad, web_content, similarity_score
            )
            
            return BestAdResponse(
                ad_id=best_ad['id'],
                description=best_ad['description'],
                source_url=best_ad['source_url'],
                similarity_score=float(similarity_score),
                projected_embedding=projected_embedding.tolist(),
                ad_features=best_ad.get('features', {}),
                content_signals=content_signals,
                web_context=web_content,
                recommendation_reason=reason
            )
            
        except Exception as e:
            logger.error(f"Error in get_best_ad: {e}")
            raise HTTPException(status_code=500, detail=f"Ad matching error: {str(e)}")
    
    def _find_best_ad(self, web_content: Dict, user_embedding: np.ndarray) -> tuple:
        """Find the best matching ad based on content and user embedding."""
        best_ad = None
        best_similarity = -1
        best_projected = None
        
        for ad in self.ad_metadata['ads']:
            if self.encoder and self.projector:
                # Use ML models for sophisticated matching
                similarity, projected = self._ml_based_matching(
                    ad, web_content, user_embedding
                )
            else:
                # Fallback to simple keyword matching
                similarity = self._keyword_based_matching(ad, web_content)
                projected = user_embedding  # Use user embedding as fallback
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_ad = ad
                best_projected = projected
        
        return best_ad, best_similarity, best_projected
    
    def _ml_based_matching(self, ad: Dict, web_content: Dict, user_embedding: np.ndarray) -> tuple:
        """Use ML models for ad matching."""
        try:
            # Encode ad text
            ad_embedding = self.encoder.encode(text=ad['description'])
            
            # Project to user space
            ad_tensor = torch.tensor(ad_embedding, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                projected_ad = self.projector(ad_tensor).squeeze(0)
            
            # Calculate similarity
            similarity = 1 - cosine(projected_ad.numpy(), user_embedding)
            
            # Boost similarity based on content relevance
            content_boost = self._calculate_content_relevance(ad, web_content)
            similarity = similarity * (1 + content_boost * 0.2)  # 20% boost max
            
            return similarity, projected_ad
            
        except Exception as e:
            logger.error(f"ML matching error: {e}")
            return self._keyword_based_matching(ad, web_content), user_embedding
    
    def _keyword_based_matching(self, ad: Dict, web_content: Dict) -> float:
        """Fallback keyword-based matching."""
        ad_text = ad['description'].lower()
        web_text = web_content.get('summary_text', '').lower()
        
        # Simple keyword overlap scoring
        ad_words = set(ad_text.split())
        web_words = set(web_text.split())
        
        if not web_words:
            return 0.5  # Default score
        
        overlap = len(ad_words.intersection(web_words))
        total_words = len(web_words)
        
        return overlap / total_words if total_words > 0 else 0.5
    
    def _calculate_content_relevance(self, ad: Dict, web_content: Dict) -> float:
        """Calculate how relevant the ad is to the web content."""
        relevance_score = 0.0
        
        # Check page type alignment
        page_type = web_content.get('page_type', 'general')
        ad_features = ad.get('features', {})
        
        if page_type == 'news' and ad_features.get('category') == 'technology':
            relevance_score += 0.3
        elif page_type == 'ecommerce' and ad_features.get('category') == 'fashion':
            relevance_score += 0.3
        elif page_type == 'blog' and ad_features.get('category') == 'education':
            relevance_score += 0.3
        
        # Check keyword alignment
        web_keywords = web_content.get('keywords', [])
        ad_text = ad['description'].lower()
        
        keyword_matches = sum(1 for keyword in web_keywords if keyword in ad_text)
        if web_keywords:
            relevance_score += (keyword_matches / len(web_keywords)) * 0.4
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _generate_content_signals(self, web_content: Dict) -> Dict[str, Any]:
        """Generate signals about the web content for ad targeting."""
        return {
            'page_type': web_content.get('page_type', 'general'),
            'domain': web_content.get('domain', ''),
            'top_keywords': web_content.get('keywords', [])[:5],
            'content_length': len(web_content.get('main_content', '')),
            'has_commerce_signals': any(word in web_content.get('summary_text', '').lower() 
                                      for word in ['buy', 'shop', 'price', 'sale', 'deal']),
            'has_tech_signals': any(word in web_content.get('summary_text', '').lower() 
                                  for word in ['tech', 'software', 'app', 'digital', 'ai']),
            'has_news_signals': any(word in web_content.get('summary_text', '').lower() 
                                  for word in ['news', 'breaking', 'update', 'report'])
        }
    
    def _generate_recommendation_reason(self, ad: Dict, web_content: Dict, similarity_score: float) -> str:
        """Generate a human-readable reason for the ad recommendation."""
        reasons = []
        
        if similarity_score > 0.8:
            reasons.append("High relevance match")
        elif similarity_score > 0.6:
            reasons.append("Good relevance match")
        
        page_type = web_content.get('page_type', 'general')
        ad_features = ad.get('features', {})
        
        if page_type == 'news' and ad_features.get('category') == 'technology':
            reasons.append("Tech content matches tech product")
        elif page_type == 'ecommerce' and ad_features.get('category') == 'fashion':
            reasons.append("Shopping context matches fashion product")
        
        if not reasons:
            reasons.append("General content match")
        
        return " | ".join(reasons)
    
    def _get_default_user_embedding(self) -> np.ndarray:
        """Get default user embedding for cold start."""
        try:
            global_mean_path = Path("../models/global_mean.npy")
            if global_mean_path.exists():
                return np.load(global_mean_path)
            else:
                # Return random embedding as fallback
                return np.random.randn(128).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading default user embedding: {e}")
            return np.random.randn(128).astype(np.float32)

# Initialize the service
best_ad_service = BestAdService()

# FastAPI app for the service
app = FastAPI(title="Best Ad Service", version="1.0.0")

@app.post("/get_best_ad", response_model=BestAdResponse)
async def get_best_ad_endpoint(request: WebContentRequest):
    """Get the best ad for a given web page and user context."""
    return best_ad_service.get_best_ad(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "best_ad_service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

