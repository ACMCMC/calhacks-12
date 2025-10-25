"""
Project Aura Backend API
FastAPI application providing ad serving and search endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
import os
from pathlib import Path

# Import our core modules
from privads_core import PrivAdsCore
from elastic_search import search_ads_elastic

app = FastAPI(
    title="Project Aura API",
    description="AI-Native Growth Platform Backend",
    version="1.0.0"
)

# Initialize core components
core = PrivAdsCore()

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Project Aura Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)