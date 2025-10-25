"""Quick test of all components without full training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

print("=" * 60)
print("Testing PrivAds Components")
print("=" * 60)

# Test 1: Projector
print("\n1. Testing Projector...")
from projector import Projector
import torch
proj = Projector(d_ad=768, d_user=128)
z = torch.randn(4, 768)
p = proj(z)
print(f"   ✓ Input: {z.shape} → Output: {p.shape}")
print(f"   ✓ Output norms: {p.norm(dim=-1).tolist()}")

# Test 2: Synthetic Click Data
print("\n2. Testing Synthetic Click Generator...")
from click_data import SyntheticClickGenerator
gen = SyntheticClickGenerator(n_users=100, n_ads=50, d_user=128)
clicks = gen.get_clicks(1000)
n_clicked = sum(1 for _, _, clicked, _ in clicks if clicked)
print(f"   ✓ Generated {len(clicks)} impressions")
print(f"   ✓ {n_clicked} clicks (CTR: {n_clicked/len(clicks):.2%})")
print(f"   ✓ Sample: user={clicks[0][0]}, ad={clicks[0][1]}, clicked={clicks[0][2]}, pos={clicks[0][3]}")

# Test 3: User Embeddings (small scale)
print("\n3. Testing User Embedding Training (5 epochs)...")
from train_user_embeddings import train_user_embeddings
user_emb, global_mean, _ = train_user_embeddings(
    clicks=clicks,
    n_users=100,
    d_user=128,
    epochs=5,
    batch_size=32,
    device="cpu"
)
print(f"   ✓ User embeddings shape: {user_emb.shape}")
print(f"   ✓ Global mean shape: {global_mean.shape}")
print(f"   ✓ Global mean norm: {np.linalg.norm(global_mean):.4f}")

# Test 5: Evaluation (quick test)
print("\n5. Testing Evaluation (quick)...")
from evaluate import compute_precision_at_k, plot_user_embeddings_tsne, plot_recommendation_heatmap

# Create small test data
test_user_emb = np.random.randn(50, 128)
test_user_emb = test_user_emb / np.linalg.norm(test_user_emb, axis=1, keepdims=True)

test_ad_emb = {aid: np.random.randn(128) for aid in range(30)}
for aid in test_ad_emb:
    test_ad_emb[aid] = test_ad_emb[aid] / np.linalg.norm(test_ad_emb[aid])

# Create synthetic test clicks
test_clicks = []
for _ in range(200):
    uid = np.random.randint(50)
    aid = np.random.randint(30)
    clicked = np.random.random() < 0.15  # 15% CTR
    pos = np.random.randint(1, 6)
    test_clicks.append((uid, aid, clicked, pos))

precision = compute_precision_at_k(test_user_emb, test_ad_emb, test_clicks, k=10)
print(f"   ✓ Precision@10: {precision:.3f}")

print("   ✓ t-SNE and heatmap plotting functions available")
