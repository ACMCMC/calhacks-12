"""Train user embeddings from co-click graph using InfoNCE."""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def build_co_click_graph(clicks, min_shared: int = 1):
    """Build user-user edges from shared ad clicks."""
    ad_to_users = defaultdict(set)
    
    # Build ad->users mapping
    for uid, aid, clicked, pos in clicks:
        if clicked:
            ad_to_users[aid].add(uid)
    
    # Create edges between users who clicked same ads
    edges = []
    edge_weights = defaultdict(int)
    
    for aid, users in ad_to_users.items():
        users = list(users)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u1, u2 = sorted([users[i], users[j]])
                edge_weights[(u1, u2)] += 1
    
    # Filter by min_shared and create edge list
    edges = [(u1, u2) for (u1, u2), w in edge_weights.items() if w >= min_shared]
    
    print(f"✓ Co-click graph: {len(edges)} edges from {len(ad_to_users)} ads")
    return edges


class UserEmbeddings(nn.Module):
    """Learnable user embedding table."""
    
    def __init__(self, n_users: int, d_user: int = 128):
        super().__init__()
        self.embeddings = nn.Embedding(n_users, d_user)
        nn.init.normal_(self.embeddings.weight, std=0.02)
    
    def forward(self, user_ids):
        emb = self.embeddings(user_ids)
        return emb / emb.norm(dim=-1, keepdim=True)


def train_user_embeddings(
    clicks,
    n_users: int,
    d_user: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    device: str = "cuda",
    synthetic_generator=None  # For supervised archetype loss
):
    """
    Train user embeddings with InfoNCE on co-click graph.
    
    Returns:
        user_embeddings: np.array (n_users, d_user)
        global_mean: np.array (d_user,)
        embedding_changes: list of average embedding changes per epoch
    """
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Training user embeddings on {device}...")
    
    # Build graph
    edges = build_co_click_graph(clicks)
    
    if len(edges) == 0:
        print("⚠ No co-click edges found! Using random initialization.")
        user_embeddings = np.random.randn(n_users, d_user)
        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        global_mean = user_embeddings.mean(axis=0)
        global_mean = global_mean / np.linalg.norm(global_mean)
        return user_embeddings, global_mean
    
    # Model
    model = UserEmbeddings(n_users, d_user).to(device)
    
    # Initialize embeddings close to archetype centroids if generator provided
    if synthetic_generator is not None:
        print("Initializing user embeddings near clean archetype centroids...")
        with torch.no_grad():
            for uid in range(n_users):
                arch_idx = synthetic_generator.user_archetype[uid]
                # Use the clean archetype vector as initialization, with small noise
                arch_vector = synthetic_generator.clean_archetypes[arch_idx]
                noise = np.random.randn(d_user) * 0.01  # Very small noise
                init_emb = arch_vector + noise
                init_emb = init_emb / np.linalg.norm(init_emb)
                model.embeddings.weight.data[uid] = torch.tensor(init_emb, dtype=torch.float32)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Track embedding changes over epochs
    embedding_changes = []
    prev_embeddings = None
    
    # Training loop
    for epoch in range(epochs):
        np.random.shuffle(edges)
        total_loss = 0
        n_batches = 0
        
    # Training loop
    for epoch in range(epochs):
        np.random.shuffle(edges)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            u_ids = torch.tensor([e[0] for e in batch], device=device)
            v_ids = torch.tensor([e[1] for e in batch], device=device)
            
            u_emb = model(u_ids)
            v_emb = model(v_ids)
            
            # InfoNCE: positive pairs (u, v); negatives are other users in batch
            logits = torch.matmul(u_emb, v_emb.T) / temperature
            labels = torch.arange(len(batch), device=device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if epoch % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        # Track embedding change
        model.eval()
        with torch.no_grad():
            current_embeddings = model.embeddings.weight.cpu().numpy()
            if prev_embeddings is not None:
                change = np.mean(np.linalg.norm(current_embeddings - prev_embeddings, axis=1))
                embedding_changes.append(change)
            prev_embeddings = current_embeddings.copy()
        model.train()
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(n_users, device=device)
        user_embeddings = model(all_ids).cpu().numpy()
        global_mean = user_embeddings.mean(axis=0)
        global_mean = global_mean / np.linalg.norm(global_mean)
    
    print(f"✓ User embeddings trained: shape={user_embeddings.shape}")
    return user_embeddings, global_mean, embedding_changes


if __name__ == "__main__":
    from click_data import SyntheticClickGenerator
    
    # Generate synthetic clicks
    gen = SyntheticClickGenerator(n_users=100, n_ads=50)
    clicks = gen.get_clicks(5000)
    
    # Train
    user_emb, global_mean, changes = train_user_embeddings(
        clicks=clicks,
        n_users=100,
        d_user=128,
        epochs=30,
        device="cpu"
    )
    
    print(f"Embedding changes over epochs: {changes}")
    
    print(f"Global mean shape: {global_mean.shape}")
    print(f"Global mean norm: {np.linalg.norm(global_mean):.4f}")
