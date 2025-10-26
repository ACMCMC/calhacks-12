"""
Quick feature correlation analysis for ad click prediction.
Extracts features for each ad, computes correlation with click label, and prints ranked list.
"""

import pandas as pd
import numpy as np
from feature_extractor import extract_ad_features
from tqdm import tqdm


# Load click data (expects columns: user_id, ad_id, clicked, ...)
df = pd.read_csv("../persona_ad_clicks.csv")

# Downsample to 100 unique ads for speed
ad_ids = df["ad_id"].unique()[:100]
df = df[df["ad_id"].isin(ad_ids)]

# For each ad_id, extract features ONCE (assume image path is ad_creatives/hf_ad_{ad_id:06d}.jpg)
ad_features = {}
for ad_id in tqdm(ad_ids, desc="Extracting ad features"):
    img_path = f"../ad_creatives/hf_ad_{int(ad_id):06d}.jpg"
    try:
        feats = extract_ad_features(image_path=img_path)
    except Exception as e:
        feats = {}
    ad_features[ad_id] = feats


# Merge features into main df
def flatten_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            continue  # skip complex types for now
        out[k] = v
    return out


feature_df = pd.DataFrame(
    {ad_id: flatten_dict(feats) for ad_id, feats in ad_features.items()}
).T
feature_df["ad_id"] = feature_df.index

df = df.merge(feature_df, on="ad_id", how="left")

# Compute correlation matrix (features + click_probability)
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = [c for c in feature_df.columns if c not in ["ad_id"]]
numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(feature_df[c])]

df_corr = df[["click_probability"] + numeric_cols].copy()
df_corr = df_corr.apply(pd.to_numeric, errors="coerce")
corr = df_corr.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix (including click_probability)")
plt.tight_layout()
plt.savefig("feature_correlation_matrix.png")
print("Saved correlation matrix plot as feature_correlation_matrix.png")
