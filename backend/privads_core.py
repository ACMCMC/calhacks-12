"""
PrivAds Core Logic for Project Aura
Handles user embeddings, projector, and Chroma database interactions.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
import sys
import os

# Add project root to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.training.projector import Projector

class PrivAdsCore:
    """Core PrivAds functionality for ad serving."""

    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)

        # Load trained models
        self._load_models()

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("ads")

    def _load_models(self):
        """Load user embeddings, global mean, and projector."""
        try:
            self.user_embeddings = np.load(self.models_dir / "user_embeddings.npy")
            self.global_mean = np.load(self.models_dir / "global_mean.npy")

            # Load projector
            from pipeline.training.projector import Projector
            self.projector = Projector(d_ad=1024, d_user=128)  # Jina CLIP v2 is 1024D
            self.projector.load_state_dict(torch.load(self.models_dir / "projector.pt"))
            self.projector.eval()

            print("✓ Loaded PrivAds models")
        except FileNotFoundError as e:
            print(f"⚠ Model files not found: {e}")
            print("Run training pipeline first: python pipeline/training/train_models.py")
            # Initialize with defaults for development
            self.user_embeddings = np.random.randn(1000, 128)
            self.global_mean = np.mean(self.user_embeddings, axis=0)

    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """
        Get user embedding by ID, or return global mean for cold start.
        In production, this would map string IDs to indices.
        """
        try:
            # Simple mapping: assume user_id is numeric for now
            user_idx = int(user_id) if user_id.isdigit() else hash(user_id) % len(self.user_embeddings)
            if 0 <= user_idx < len(self.user_embeddings):
                return self.user_embeddings[user_idx]
            else:
                return self.global_mean
        except (ValueError, IndexError):
            return self.global_mean

    def query_ads_chroma(
        self,
        query_embedding: np.ndarray,
        site_context: Dict[str, Any],
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query Chroma database for relevant ads using embedding similarity and metadata filtering.
        """
        # Build metadata filter from site_context
        where_clause = self._build_chroma_filter(site_context)

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause
        )

        # Format results
        formatted_results = []
        if results['ids']:
            for i, ad_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': ad_id,
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return formatted_results

    def _build_chroma_filter(self, site_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Chroma where clause from site_context.
        """
        filters = []

        # Add keyword filtering
        if 'keywords' in site_context and site_context['keywords']:
            keywords = site_context['keywords']
            # Chroma supports $in for list matching
            filters.append({"tags": {"$in": keywords}})

        # Add type filtering
        if 'type' in site_context:
            filters.append({"type": site_context['type']})

        # Add alignment filtering
        if 'alignment' in site_context:
            filters.append({"alignment": site_context['alignment']})

        # Combine filters with $and if multiple
        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            return {"$and": filters}
        else:
            return None

    def project_ad_embedding(self, ad_embedding: np.ndarray) -> np.ndarray:
        """
        Project raw ad embedding (1024D) to user space (128D).
        """
        with torch.no_grad():
            tensor_emb = torch.tensor(ad_embedding, dtype=torch.float32).unsqueeze(0)
            projected = self.projector(tensor_emb)
            return projected.squeeze(0).numpy()

    def add_ad_to_chroma(self, ad_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Add an ad to the Chroma collection.
        """
        self.collection.add(
            ids=[ad_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def search_similar_ads(self, ad_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find ads similar to a given ad embedding.
        """
        results = self.collection.query(
            query_embeddings=[ad_embedding.tolist()],
            n_results=n_results
        )

        formatted_results = []
        if results['ids']:
            for i, ad_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': ad_id,
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return formatted_results