import json
import torch
from pathlib import Path
from pipeline.training.ad_encoder import AdEncoder
from tqdm import tqdm

# Load ad metadata
with open("ad_creatives/scraped_metadata.json", "r") as f:
    ads = json.load(f)["ads"]

encoder = AdEncoder()
ad_embeddings = {}

BATCH_SIZE = 32
for i in tqdm(range(0, len(ads), BATCH_SIZE), desc="Encoding ads (batched)"):
    batch = ads[i:i+BATCH_SIZE]
    texts = [ad.get("description", None) for ad in batch]
    image_paths = [ad["local_path"] for ad in batch]
    embs = [encoder.encode(text=text, image=image_path) for text, image_path in zip(texts, image_paths)]
    for ad, emb in zip(batch, embs):
        ad_id = ad["id"]
        ad_embeddings[str(ad_id)] = emb  # ensure string keys

# Save as .pt (torch tensor dict)
torch.save(ad_embeddings, "data/ad_embeddings_raw.pt")
print(f"Saved {len(ad_embeddings)} ad embeddings to data/ad_embeddings_raw.pt")
