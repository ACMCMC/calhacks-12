import numpy as np
import torch
from .click_data import RealClickDataQuantile
from tqdm import tqdm
from pipeline.training.projector import Projector

# Load clicks (persona_id, ad_id, clicked, position)
clicks = RealClickDataQuantile("persona_ad_clicks.csv").get_clicks()

# Build user and ad id sets
user_ids = sorted(set(c[0] for c in clicks))
ad_ids = sorted(set(c[1] for c in clicks))
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
ad_id_to_idx = {aid: i for i, aid in enumerate(ad_ids)}

# Load ad embeddings (torch tensors)
ad_emb_dict = torch.load("data/ad_embeddings_raw.pt")

# Initialize user embeddings randomly
d_user = 128
user_emb = torch.randn(len(user_ids), d_user, requires_grad=True)
user_emb = user_emb / user_emb.norm(dim=1, keepdim=True)

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
    a_emb = ad_emb_dict[str(a)]
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
user_set = set(user_ids)
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
            # Total loss (equal weight)
            loss = ad_loss + uu_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Re-normalize
            user_emb.data = user_emb.data / user_emb.data.norm(dim=1, keepdim=True)
            losses.append(loss.item())
            t.set_postfix(loss=loss.item(), avg_loss=np.mean(losses))
    print(f"Epoch {epoch+1}: mean loss = {np.mean(losses):.4f}")

# Save user embeddings
torch.save({"user_ids": user_ids, "user_emb": user_emb.detach()}, "data/user_embeddings.pt")
print("User embedding + projector joint training complete and saved.")
