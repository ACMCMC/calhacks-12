
import json
import torch
from pathlib import Path
from ad_processing.encoder import AdEncoder
from tqdm import tqdm

# Load ad metadata
with open("ad_creatives/scraped_metadata.json", "r") as f:
    ads = json.load(f)["ads"]

encoder = AdEncoder()
ad_embeddings = {}


for ad in tqdm(ads, desc="Encoding ads"):
    ad_id = ad["id"]
    image_path = ad["local_path"]
    text = ad.get("description", None)
    emb = encoder.encode(text=text, image_path=image_path)
    ad_embeddings[str(ad_id)] = emb  # ensure string keys

# Save as .pt (torch tensor dict)
torch.save(ad_embeddings, "data/ad_embeddings_raw.pt")
print(f"Saved {len(ad_embeddings)} ad embeddings to data/ad_embeddings_raw.pt")
