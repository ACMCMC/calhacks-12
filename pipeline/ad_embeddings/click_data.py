import pandas as pd
import numpy as np

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
            if row["clicked"] == 1
        ]

    def get_clicks(self):
        return self.clicks