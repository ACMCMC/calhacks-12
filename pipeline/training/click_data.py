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
        self.n_archetypes = 10
        n_archetypes = self.n_archetypes
        archetypes = np.random.randn(n_archetypes, d_user)
        archetypes = archetypes / np.linalg.norm(archetypes, axis=1, keepdims=True)
        self.clean_archetypes = archetypes.copy()  # Store clean archetypes

        # Assign users to archetypes with less noise (more distinct)
        self.user_preferences = {}
        self.user_archetype = {}
        for uid in range(n_users):
            arch_idx = np.random.choice(n_archetypes)
            noise = np.random.randn(d_user) * 0.05  # less noise
            u = archetypes[arch_idx] + noise
            self.user_preferences[uid] = u / np.linalg.norm(u)
            self.user_archetype[uid] = arch_idx

        # Assign each ad to an archetype and make its embedding close to that archetype
        self.ad_archetype = {}
        if ad_embeddings is None:
            self.ad_embeddings = {}
            for aid in range(n_ads):
                arch_idx = np.random.choice(n_archetypes)
                self.ad_archetype[aid] = arch_idx
                noise = np.random.randn(d_user) * 0.05  # ad noise
                a = archetypes[arch_idx] + noise
                self.ad_embeddings[aid] = a / np.linalg.norm(a)
        else:
            self.ad_embeddings = ad_embeddings
            for aid in range(n_ads):
                self.ad_archetype[aid] = np.random.choice(n_archetypes)
        print(f"✓ Synthetic generator: {n_users} users, {n_ads} ads, {n_archetypes} archetypes (ad embeddings archetype-aligned)")
    
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
            
            # Click probability: strong archetype preference + embedding similarity
            u_arch = self.user_archetype[uid]
            a_arch = self.ad_archetype.get(aid, -1)
            
            # Base probability depends on archetype match
            if a_arch == u_arch:
                base_prob = 0.4  # High click probability for matching archetype
            else:
                base_prob = 0.05  # Low click probability for non-matching
            
            # Add embedding similarity bonus (scaled down)
            sim = np.dot(u, a[:len(u)])  # handle dimension mismatch
            similarity_bonus = 0.1 * sim  # Much smaller bonus
            
            # Position penalty
            position_penalty = 0.05 * position
            
            logit = np.log(base_prob / (1 - base_prob)) + similarity_bonus - position_penalty
            p_click = 1 / (1 + np.exp(-logit))
            clicked = np.random.rand() < p_click
            
            clicks.append((uid, aid, clicked, position))
        
        # Confusion matrix: user archetype vs ad archetype for clicked ads
        cm = np.zeros((self.n_archetypes, self.n_archetypes), dtype=int)
        for uid, aid, clicked, _ in clicks:
            if clicked:
                u_arch = self.user_archetype[uid]
                a_arch = self.ad_archetype.get(aid, -1)
                if a_arch >= 0:
                    cm[u_arch, a_arch] += 1
        print("User-Archetype vs Ad-Archetype Confusion Matrix (clicked only):")
        print(cm)
        print("Diagonal (matching):", np.diag(cm))
        print("Off-diagonal (non-matching):", cm.sum() - np.diag(cm).sum())

        # Save confusion matrix as heatmap and CSV
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Use absolute path relative to script location
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        eval_dir = os.path.join(script_dir, "backend", "evaluation_results")
        os.makedirs(eval_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("User-Archetype vs Ad-Archetype Confusion Matrix (clicked only)")
        plt.xlabel("Ad Archetype")
        plt.ylabel("User Archetype")
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "archetype_confusion.png"))
        plt.close()
        np.savetxt(os.path.join(eval_dir, "archetype_confusion.csv"), cm, delimiter=",", fmt="%d")
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
