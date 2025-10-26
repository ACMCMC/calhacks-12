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
    import torch
    from pipeline.training.click_data import RealClickDataQuantile
    from pipeline.training.projector import Projector
    import numpy as np
    from tqdm import tqdm

    # Load real click data
    clicks = RealClickDataQuantile("persona_ad_clicks.csv").get_clicks()
    user_ids = sorted(set(c[0] for c in clicks))
    ad_ids = sorted(set(c[1] for c in clicks))
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    ad_id_to_idx = {aid: i for i, aid in enumerate(ad_ids)}

    # Load ad embeddings (torch tensors)
    ad_emb_dict = torch.load("data/ad_embeddings_raw.pt", weights_only=False)

    # Initialize user embeddings as a leaf Parameter
    d_user = 128
    user_emb = torch.nn.Parameter(torch.randn(len(user_ids), d_user))
    user_emb.data = user_emb.data / user_emb.data.norm(dim=1, keepdim=True)

    # Initialize projector
    projector = Projector(d_ad=2048, d_user=128)
    projector.train()
    opt = torch.optim.Adam([user_emb] + list(projector.parameters()), lr=1e-2)

    # Build co-click graph: users who clicked the same ad
    from collections import defaultdict
    ad_to_users = defaultdict(list)
    for u, a, clicked, _ in clicks:
        if clicked:
            ad_to_users[a].append(u)

    # Prepare positive/negative ad-user pairs and user-user pairs
    pos_ad_pairs = []
    neg_ad_pairs = []
    user_user_pos = []
    user_user_neg = []
    for u, a, clicked, _ in clicks:
        u_idx = user_id_to_idx[u]
        ad_id = int(a)
        ad_key = f"hf_ad_{ad_id:06d}"
        if ad_key not in ad_emb_dict:
            print(f"Warning: ad embedding for ad_id {a} (key {ad_key}) missing, skipping.")
            continue
        a_emb = ad_emb_dict[ad_key]
        # Convert to torch tensor if needed
        if isinstance(a_emb, np.ndarray):
            a_emb = torch.tensor(a_emb, dtype=torch.float32, device=user_emb.device)
        if clicked:
            pos_ad_pairs.append((u_idx, a_emb))
        else:
            neg_ad_pairs.append((u_idx, a_emb))

    # User-user positive: users who clicked same ad
    for users in ad_to_users.values():
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                user_user_pos.append((user_id_to_idx[users[i]], user_id_to_idx[users[j]]))

    # User-user negative: sample random pairs who never clicked same ad
    import random
    for _ in range(len(user_user_pos)):
        u1, u2 = random.sample(user_ids, 2)
        if all(u1 not in ad_to_users[a] or u2 not in ad_to_users[a] for a in ad_ids):
            user_user_neg.append((user_id_to_idx[u1], user_id_to_idx[u2]))

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        np.random.shuffle(pos_ad_pairs)
        np.random.shuffle(neg_ad_pairs)
        np.random.shuffle(user_user_pos)
        np.random.shuffle(user_user_neg)
        losses = []
        with tqdm(total=min(len(pos_ad_pairs), len(neg_ad_pairs)), desc=f"Epoch {epoch+1}") as t:
            for i in range(min(len(pos_ad_pairs), len(neg_ad_pairs))):
                # Ad-user InfoNCE
                u_idx, a_emb = pos_ad_pairs[i]
                neg_u_idx, neg_a_emb = neg_ad_pairs[i]
                u = user_emb[u_idx]
                p_a = projector(a_emb.unsqueeze(0)).squeeze(0)
                pos_sim = torch.dot(u, p_a)
                neg_sim = torch.dot(user_emb[neg_u_idx], projector(neg_a_emb.unsqueeze(0)).squeeze(0))
                ad_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
                # User-user InfoNCE
                if i < min(len(user_user_pos), len(user_user_neg)):
                    uu1, uu2 = user_user_pos[i]
                    nu1, nu2 = user_user_neg[i]
                    uu_pos_sim = torch.dot(user_emb[uu1], user_emb[uu2])
                    uu_neg_sim = torch.dot(user_emb[nu1], user_emb[nu2])
                    uu_loss = -torch.log(torch.exp(uu_pos_sim) / (torch.exp(uu_pos_sim) + torch.exp(uu_neg_sim)))
                else:
                    uu_loss = 0.0
                loss = ad_loss + uu_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                user_emb.data = user_emb.data / user_emb.data.norm(dim=1, keepdim=True)
                losses.append(loss.item())
                t.set_postfix(loss=loss.item(), avg_loss=np.mean(losses))
                t.update(1)
                # Evaluation metrics every 100 steps
                if (i+1) % 100 == 0:
                    with torch.no_grad():
                        from sklearn.metrics import roc_auc_score
                        cos_sims = []
                        recall_at_10 = []
                        aucs = []
                        # Prepare all ad embeddings projected to user space
                        ad_keys = list(ad_emb_dict.keys())
                        ad_embs = []
                        for k in ad_keys:
                            emb = ad_emb_dict[k]
                            if isinstance(emb, np.ndarray):
                                emb = torch.tensor(emb, dtype=torch.float32, device=user_emb.device)
                            ad_embs.append(emb)
                        ad_embs = torch.stack([projector(ae.unsqueeze(0)).squeeze(0) for ae in ad_embs])  # (n_ads, d_user)
                        ad_embs = ad_embs / ad_embs.norm(dim=1, keepdim=True)
                        # Sample 50 positive pairs for recall/auc eval
                        for u_idx_eval, a_emb_eval in pos_ad_pairs[:50]:
                            u_eval = user_emb[u_idx_eval]
                            p_a_eval = projector(a_emb_eval.unsqueeze(0)).squeeze(0)
                            sim = torch.nn.functional.cosine_similarity(u_eval, p_a_eval, dim=0).item()
                            cos_sims.append(sim)
                            # Recall@10: does the true ad appear in top 10 for this user?
                            sims = torch.matmul(ad_embs, u_eval)
                            topk = torch.topk(sims, 10).indices.cpu().numpy()
                            true_ad_idx = ad_keys.index(f"hf_ad_{int(ad_ids[u_idx_eval]):06d}") if f"hf_ad_{int(ad_ids[u_idx_eval]):06d}" in ad_keys else None
                            if true_ad_idx is not None and true_ad_idx in topk:
                                recall_at_10.append(1)
                            else:
                                recall_at_10.append(0)
                            # AUC: true ad is positive, all others negative
                            if true_ad_idx is not None:
                                labels = np.zeros(len(ad_keys), dtype=int)
                                labels[true_ad_idx] = 1
                                try:
                                    auc = roc_auc_score(labels, sims.cpu().numpy())
                                    aucs.append(auc)
                                except Exception:
                                    pass
                        mean_cos_sim = np.mean(cos_sims)
                        std_cos_sim = np.std(cos_sims)
                        recall10 = np.mean(recall_at_10) if recall_at_10 else float('nan')
                        auc_val = np.mean(aucs) if aucs else float('nan')
                        norms = user_emb.data.norm(dim=1).cpu().numpy()
                        print(f"[Eval step {i+1}] mean_cos_sim={mean_cos_sim:.3f}, std_cos_sim={std_cos_sim:.3f}, recall@10={recall10:.3f}, auc={auc_val:.3f}, emb_norm_mean={norms.mean():.3f}, emb_norm_std={norms.std():.3f}")
        print(f"Epoch {epoch+1}: mean loss = {np.mean(losses):.4f}")

    # Save user embeddings
    torch.save({"user_ids": user_ids, "user_emb": user_emb.detach()}, "data/user_embeddings.pt")
    print("User embedding + projector joint training complete and saved.")



