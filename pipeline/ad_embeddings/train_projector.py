import numpy as np
import torch
from pipeline.training.projector import Projector

# Load ad embeddings and user embeddings
ad_embeddings = np.load("data/ad_embeddings_raw.npz")
user_graph = np.load("data/user_graph.npz", allow_pickle=True)
user_ids = user_graph["user_ids"]
edges = user_graph["edges"]

# Dummy: Random user embeddings for demo
user_emb = np.random.randn(len(user_ids), 128)
user_emb = user_emb / np.linalg.norm(user_emb, axis=1, keepdims=True)

# Prepare ad embedding matrix (order by ad_id)
ad_id_list = list(ad_embeddings.keys())
ad_matrix = np.stack([ad_embeddings[aid] for aid in ad_id_list])
ad_id_to_idx = {aid: i for i, aid in enumerate(ad_id_list)}

# Projector
projector = Projector(d_ad=2048, d_user=128)
projector.train()
opt = torch.optim.AdamW(projector.parameters(), lr=1e-3)

# Simple test train loop (InfoNCE-style, not full pipeline)
B = 16
epochs = 3
for epoch in range(epochs):
    losses = []
    for i in range(0, len(edges), B):
        batch = list(edges)[i:i+B]
        if len(batch) < 2:
            continue
        u1_idx = [b[0] for b in batch]
        u2_idx = [b[1] for b in batch]
        # Pick random ad for each user
        a_idx = np.random.randint(0, ad_matrix.shape[0], size=len(batch))
        z_ad = torch.tensor(ad_matrix[a_idx], dtype=torch.float32)
        u1 = torch.tensor(user_emb[u1_idx], dtype=torch.float32)
        u2 = torch.tensor(user_emb[u2_idx], dtype=torch.float32)
        # Project ads
        p_ad = projector(z_ad)
        # InfoNCE: align p_ad with u1, u2
        logits = torch.matmul(p_ad, u1.T) / 0.07
        labels = torch.arange(len(batch))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}: mean loss = {np.mean(losses):.4f}")
print("Test train complete.")
