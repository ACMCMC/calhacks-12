import json
import numpy as np
from pathlib import Path
from ad_processing.encoder import AdEncoder

# Load ad metadata
with open("ad_creatives/scraped_metadata.json", "r") as f:
    ads = json.load(f)["ads"]

encoder = AdEncoder()
ad_embeddings = {}

for ad in ads:
    ad_id = ad["id"]
    image_path = ad["local_path"]
    text = ad.get("description", None)
    emb = encoder.encode(text=text, image_path=image_path)
    ad_embeddings[ad_id] = emb

# Save as .npz
np.savez("data/ad_embeddings_raw.npz", **ad_embeddings)
print(f"Saved {len(ad_embeddings)} ad embeddings to data/ad_embeddings_raw.npz")
