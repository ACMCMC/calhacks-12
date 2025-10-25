"""Train projector to map ad embeddings to user space."""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from projector import Projector


def train_projector(
    clicks,
    user_embeddings: np.ndarray,
    ad_embeddings_raw: dict,
    d_ad: int = 768,
    d_user: int = 128,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.07,
    centroid_weight: float = 0.1,
    device: str = "cuda"
):
    """
    Train projector with InfoNCE + centroid auxiliary loss.
    
    Args:
        clicks: List of (user_id, ad_id, clicked, position)
        user_embeddings: (n_users, d_user) trained user vectors
        ad_embeddings_raw: dict {ad_id: z_ad vector (d_ad,)}
        
    Returns:
        projector: trained Projector module
    """
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Training projector on {device}...")
    
    # Filter to only clicks
    click_pairs = [(uid, aid) for uid, aid, clicked, pos in clicks if clicked]
    print(f"Training on {len(click_pairs)} clicks")
    
    if len(click_pairs) == 0:
        print("⚠ No clicks found!")
        return Projector(d_ad, d_user)
    
    # Compute ad centroids (mean of user embeddings who clicked each ad)
    ad_to_users = defaultdict(list)
    for uid, aid in click_pairs:
        ad_to_users[aid].append(user_embeddings[uid])
    
    ad_centroids = {}
    for aid, users in ad_to_users.items():
        centroid = np.mean(users, axis=0)
        ad_centroids[aid] = centroid / np.linalg.norm(centroid)
    
    print(f"Computed centroids for {len(ad_centroids)} ads")
    
    # Model
    projector = Projector(d_ad, d_user).to(device)
    opt = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(epochs):
        np.random.shuffle(click_pairs)
        total_infonce = 0
        total_centroid = 0
        n_batches = 0
        
        for i in range(0, len(click_pairs), batch_size):
            batch = click_pairs[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            # Get data
            u_ids = [uid for uid, _ in batch]
            a_ids = [aid for _, aid in batch]
            
            # Skip if any ad missing
            if not all(aid in ad_embeddings_raw for aid in a_ids):
                continue
            
            u_emb = torch.tensor(
                np.stack([user_embeddings[uid] for uid in u_ids]),
                dtype=torch.float32,
                device=device
            )
            z_ad = torch.tensor(
                np.stack([ad_embeddings_raw[aid] for aid in a_ids]),
                dtype=torch.float32,
                device=device
            )
            
            # Project ads to user space
            p_ad = projector(z_ad)
            
            # InfoNCE: ad should be close to its user
            logits = torch.matmul(p_ad, u_emb.T) / temperature
            labels = torch.arange(len(batch), device=device)
            loss_infonce = nn.CrossEntropyLoss()(logits, labels)
            
            # Centroid loss: p_ad should match mean of clickers
            centroids = torch.tensor(
                np.stack([ad_centroids[aid] for aid in a_ids]),
                dtype=torch.float32,
                device=device
            )
            loss_centroid = ((p_ad - centroids) ** 2).mean()
            
            # Combined loss
            loss = loss_infonce + centroid_weight * loss_centroid
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_infonce += loss_infonce.item()
            total_centroid += loss_centroid.item()
            n_batches += 1
        
        if n_batches > 0 and epoch % 5 == 0:
            avg_infonce = total_infonce / n_batches
            avg_centroid = total_centroid / n_batches
            print(f"Epoch {epoch:3d}/{epochs}: InfoNCE={avg_infonce:.4f}, Centroid={avg_centroid:.4f}")
    
    print(f"✓ Projector trained")
    return projector


if __name__ == "__main__":
    from click_data import SyntheticClickGenerator
    from train_user_embeddings import train_user_embeddings
    
    # Generate data
    gen = SyntheticClickGenerator(n_users=100, n_ads=50, d_user=128)
    clicks = gen.get_clicks(5000)
    
    # Train user embeddings
    user_emb, _ = train_user_embeddings(clicks, n_users=100, d_user=128, epochs=20, device="cpu")
    
    # Create fake ad embeddings (in real system, these come from AdEncoder)
    ad_emb_raw = {aid: np.random.randn(768) for aid in range(50)}
    for aid in ad_emb_raw:
        ad_emb_raw[aid] /= np.linalg.norm(ad_emb_raw[aid])
    
    # Train projector
    proj = train_projector(
        clicks=clicks,
        user_embeddings=user_emb,
        ad_embeddings_raw=ad_emb_raw,
        d_ad=768,
        d_user=128,
        epochs=20,
        device="cpu"
    )
    
    # Test
    z = torch.randn(1, 768)
    p = proj(z)
    print(f"Test projection: {z.shape} -> {p.shape}, norm={p.norm().item():.4f}")
