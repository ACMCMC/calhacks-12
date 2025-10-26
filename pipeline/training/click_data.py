
import pandas as pd
import numpy as np
from typing import List, Tuple

ClickRecord = Tuple[int, int, bool, int]

class ClickDataSource:
    def get_clicks(self):
        raise NotImplementedError("Subclasses must implement this method")

class RealClickDataQuantile(ClickDataSource):
    """
    Loads real click data from a CSV with click_probability, binarizing clicks at a given quantile threshold.
    Expects columns: persona_id, ad_id, click_probability
    """
    def __init__(self, data_path: str, quantile: float = 0.85):
        df = pd.read_csv(data_path)
        threshold = df["click_probability"].quantile(quantile)
        df["clicked"] = (df["click_probability"] > threshold).astype(int)
        self.clicks = [
            (int(row["persona_id"]), int(row["ad_id"]), int(row["clicked"]), 1)
            for _, row in df.iterrows()
        ]

    def get_clicks(self) -> List[ClickRecord]:
        """Return all binarized real clicks."""
        return self.clicks


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
