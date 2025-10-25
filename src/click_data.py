"""Modular click data sources: synthetic and real."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


ClickRecord = Tuple[int, int, bool, int]  # (user_id, ad_id, clicked, position)


class ClickDataSource(ABC):
    """Abstract interface for click data."""
    
    @abstractmethod
    def get_clicks(self, n_samples: int) -> List[ClickRecord]:
        """Return list of (user_id, ad_id, clicked, position)."""
        pass


class SyntheticClickGenerator(ClickDataSource):
    """Generate synthetic clicks from user archetypes."""
    
    def __init__(
        self,
        n_users: int = 1000,
        n_ads: int = 500,
        d_user: int = 128,
        ad_embeddings: dict = None,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.n_users = n_users
        self.n_ads = n_ads
        self.d_user = d_user
        
        # Create user archetypes (clusters)
        n_archetypes = 10
        archetypes = np.random.randn(n_archetypes, d_user)
        archetypes = archetypes / np.linalg.norm(archetypes, axis=1, keepdims=True)
        
        # Assign users to archetypes with noise
        self.user_preferences = {}
        for uid in range(n_users):
            arch_idx = np.random.choice(n_archetypes)
            noise = np.random.randn(d_user) * 0.2
            u = archetypes[arch_idx] + noise
            self.user_preferences[uid] = u / np.linalg.norm(u)
        
        # Store ad embeddings
        if ad_embeddings is None:
            # Random ad embeddings for initialization
            self.ad_embeddings = {
                aid: np.random.randn(d_user) for aid in range(n_ads)
            }
            for aid in self.ad_embeddings:
                self.ad_embeddings[aid] /= np.linalg.norm(self.ad_embeddings[aid])
        else:
            self.ad_embeddings = ad_embeddings
        
        print(f"✓ Synthetic generator: {n_users} users, {n_ads} ads, {n_archetypes} archetypes")
    
    def get_clicks(self, n_samples: int) -> List[ClickRecord]:
        """Generate synthetic click data."""
        clicks = []
        
        for _ in range(n_samples):
            uid = np.random.randint(self.n_users)
            aid = np.random.randint(self.n_ads)
            position = np.random.randint(1, 6)  # positions 1-5
            
            # Get user preference and ad embedding
            u = self.user_preferences[uid]
            a = self.ad_embeddings.get(aid, np.random.randn(self.d_user))
            if aid not in self.ad_embeddings:
                a = a / np.linalg.norm(a)
            
            # Click probability: sigmoid(similarity * scale - position_penalty)
            sim = np.dot(u, a[:len(u)])  # handle dimension mismatch
            logit = 5 * sim - 0.3 * position
            p_click = 1 / (1 + np.exp(-logit))
            clicked = np.random.rand() < p_click
            
            clicks.append((uid, aid, clicked, position))
        
        return clicks


class RealClickData(ClickDataSource):
    """Load real click data (for future use)."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        print(f"✓ Real data loader: {data_path}")
    
    def get_clicks(self, n_samples: int) -> List[ClickRecord]:
        """Load from CSV/database."""
        import pandas as pd
        df = pd.read_csv(self.data_path)
        if len(df) > n_samples:
            df = df.sample(n_samples)
        
        return list(df[['user_id', 'ad_id', 'clicked', 'position']].itertuples(
            index=False, name=None
        ))


if __name__ == "__main__":
    # Test synthetic generator
    gen = SyntheticClickGenerator(n_users=100, n_ads=50)
    clicks = gen.get_clicks(1000)
    
    n_clicked = sum(1 for _, _, clicked, _ in clicks if clicked)
    print(f"Generated {len(clicks)} impressions, {n_clicked} clicks (CTR: {n_clicked/len(clicks):.2%})")
    
    # Show sample
    for i, (uid, aid, clicked, pos) in enumerate(clicks[:5]):
        print(f"  {i+1}. user={uid}, ad={aid}, clicked={clicked}, pos={pos}")
