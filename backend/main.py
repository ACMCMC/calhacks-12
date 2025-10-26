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
import sys

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

class BestAdRequest(BaseModel):
    web_text: str
    user_embedding: List[float]

class BestAdResponse(BaseModel):
    ad_id: str
    description: str
    source_url: str
    similarity_score: float
    projected_embedding: List[float]

@app.post("/get_best_ad", response_model=BestAdResponse)
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

@app.post("/get_best_ad", response_model=BestAdResponse)
async def get_best_ad(request: BestAdRequest):
    """
    Find the best ad by projecting ad embeddings into user space and matching with user embedding.
    """
    try:
        import torch
        from pathlib import Path
        import json
        from scipy.spatial.distance import cosine

        # Load ad metadata
        ad_metadata_path = Path("../ad_creatives/scraped_metadata.json")
        if not ad_metadata_path.exists():
            raise HTTPException(status_code=404, detail="Ad metadata not found")

        with open(ad_metadata_path, 'r') as f:
            ad_data = json.load(f)

        # Load ad encoder and projector
        ad_encoder_path = Path("../pipeline/training/ad_encoder.py")
        projector_path = Path("../pipeline/training/projector.py")

        if not ad_encoder_path.exists() or not projector_path.exists():
            raise HTTPException(status_code=404, detail="Model files not found")

        # Import modules dynamically
        import importlib.util

        # Load ad encoder
        spec = importlib.util.spec_from_file_location("ad_encoder", ad_encoder_path)
        ad_encoder_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ad_encoder_module)
        AdEncoder = ad_encoder_module.AdEncoder

        # Load projector
        spec = importlib.util.spec_from_file_location("projector", projector_path)
        projector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projector_module)
        Projector = projector_module.Projector

        # Initialize encoder and projector
        encoder = AdEncoder(device="cpu")  # Use CPU for API calls
        projector = Projector(d_ad=encoder.d_ad, d_user=128)

        # Load projector weights
        projector_path = Path("../models/projector.pt")
        if not projector_path.exists():
            raise HTTPException(status_code=404, detail="Projector model not found")

        projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
        projector.eval()

        # Convert user embedding to tensor
        user_embedding = torch.tensor(request.user_embedding, dtype=torch.float32)

        best_ad = None
        best_similarity = -1
        best_projected = None

        # Evaluate each ad
        for ad in ad_data['ads']:
            # Encode ad text into embedding
            ad_embedding = encoder.encode(text=ad['description'])

            # Convert to tensor and project into user space
            ad_tensor = torch.tensor(ad_embedding, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                projected_ad = projector(ad_tensor).squeeze(0)

            # Calculate similarity with user embedding
            similarity = 1 - cosine(projected_ad.numpy(), user_embedding.numpy())

            if similarity > best_similarity:
                best_similarity = similarity
                best_ad = ad
                best_projected = projected_ad.tolist()

        if not best_ad:
            raise HTTPException(status_code=404, detail="No ads found")

        return BestAdResponse(
            ad_id=best_ad['id'],
            description=best_ad['description'],
            source_url=best_ad['source_url'],
            similarity_score=float(best_similarity),
            projected_embedding=best_projected
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ad matching error: {str(e)}")

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