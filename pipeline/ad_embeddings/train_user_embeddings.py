import numpy as np
from .click_data import RealClickDataQuantile

# Load clicks (persona_id, ad_id, clicked, position)
clicks = RealClickDataQuantile("persona_ad_clicks.csv").get_clicks()

# Build user and ad id sets
user_ids = sorted(set(c[0] for c in clicks))
ad_ids = sorted(set(c[1] for c in clicks))
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
ad_id_to_idx = {aid: i for i, aid in enumerate(ad_ids)}

# Build co-click graph (user-user edges if they clicked same ad)
from collections import defaultdict
user_to_ads = defaultdict(set)
for u, a, clicked, _ in clicks:
    if clicked:
        user_to_ads[u].add(a)

edges = set()
for a in ad_ids:
    users = [u for u in user_ids if a in user_to_ads[u]]
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            edges.add((user_id_to_idx[users[i]], user_id_to_idx[users[j]]))

# Dummy: Save user_ids and edges for downstream training
np.savez("data/user_graph.npz", user_ids=user_ids, edges=list(edges))
print(f"Saved user graph with {len(user_ids)} users and {len(edges)} edges to data/user_graph.npz")
