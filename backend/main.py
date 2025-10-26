"""
Project Aura Backend API
FastAPI application providing ad serving and search endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
import os
from pathlib import Path
import joblib
import json

# Import our core modules
from privads_core import PrivAdsCore
from elastic_search import search_ads_elastic

app = FastAPI(
    title="Project Aura API",
    description="AI-Native Growth Platform Backend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
core = PrivAdsCore()

# Initialize Chroma client
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print("✓ ChromaDB initialized")
except Exception as e:
    print(f"⚠ ChromaDB initialization failed: {e}")
    chroma_client = None

# Load ML model for click prediction
try:
    model_path = Path("../models/click_predictor.pkl")
    scaler_path = Path("../models/feature_scaler.pkl")
    features_path = Path("../models/feature_names.json")
    
    click_model = joblib.load(model_path)
    feature_scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
        
    print(f"Loaded click prediction model with {len(feature_names)} features: {feature_names}")
except Exception as e:
    print(f"Warning: Could not load click prediction model: {e}")
    click_model = None
    feature_scaler = None
    feature_names = []

class AdRequest(BaseModel):
    user_id: str
    p_receptive: float
    site_context: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str

class AdResponse(BaseModel):
    decision: str
    ad_id: Optional[str] = None
    creative_url: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None

class SearchResult(BaseModel):
    ad_id: str
    thumbnail_url: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]

class ClickPredictionRequest(BaseModel):
    features: Dict[str, float]

class ClickPredictionResponse(BaseModel):
    probability: float
    features_used: List[str]

class ClickPredictionRequest(BaseModel):
    features: Dict[str, float]

class ClickPredictionResponse(BaseModel):
    probability: float
    features_used: List[str]

@app.post("/get_ad", response_model=AdResponse)
async def get_ad(request: AdRequest):
    """
    Core ad serving endpoint that combines user preference, receptiveness, and context.
    """
    try:
        # Check receptiveness threshold
        if request.p_receptive < 0.7:
            return AdResponse(
                decision="BLOCK",
                reason="User not receptive"
            )

        # Get user embedding (or global mean for cold start)
        user_embedding = core.get_user_embedding(request.user_id)

        # Apply any contextual modulation (placeholder for now)
        final_embedding = user_embedding  # Could modulate based on site_context

        # Query Chroma with context filtering
        results = core.query_ads_chroma(
            query_embedding=final_embedding,
            site_context=request.site_context,
            n_results=10
        )

        if not results:
            return AdResponse(
                decision="BLOCK",
                reason="No relevant ads found for context"
            )

        # Select best ad (already ranked by Chroma)
        best_ad = results[0]
        ad_id = best_ad['id']
        score = best_ad['distance']  # Cosine similarity

        # Get creative URL (placeholder - would come from metadata)
        creative_url = f"https://example.com/ads/{ad_id}.jpg"

        return AdResponse(
            decision="SERVE",
            ad_id=ad_id,
            creative_url=creative_url,
            score=score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ad serving error: {str(e)}")

@app.post("/search_ads", response_model=SearchResponse)
async def search_ads(request: SearchRequest):
    """
    Natural language search over ad metadata using Elastic Agent Builder.
    """
    try:
        results = search_ads_elastic(request.query)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                ad_id=result['ad_id'],
                thumbnail_url=result.get('thumbnail_url', f"https://example.com/thumbnails/{result['ad_id']}.jpg"),
                metadata=result.get('metadata', {})
            ))

        return SearchResponse(results=formatted_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/model/interaction_predictor.onnx")
async def get_onnx_model():
    """Serve the ONNX model file with correct MIME type"""
    model_path = Path("../models/interaction_predictor.onnx")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    def file_generator():
        with open(model_path, "rb") as f:
            yield from f

    return StreamingResponse(
        file_generator(),
        media_type="application/wasm",
        headers={"Content-Disposition": "attachment; filename=interaction_predictor.onnx"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)