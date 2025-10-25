"""Quick test of all components without full training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
user_emb, global_mean = train_user_embeddings(
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

# Test 4: Projector Training (small scale)
print("\n4. Testing Projector Training (5 epochs)...")
import numpy as np
ad_emb_raw = {aid: np.random.randn(768) for aid in range(50)}
for aid in ad_emb_raw:
    ad_emb_raw[aid] /= np.linalg.norm(ad_emb_raw[aid])

from train_projector import train_projector
proj = train_projector(
    clicks=clicks,
    user_embeddings=user_emb,
    ad_embeddings_raw=ad_emb_raw,
    d_ad=768,
    d_user=128,
    epochs=5,
    batch_size=32,
    device="cpu"
)
print(f"   ✓ Projector trained")

# Test projection
z_test = torch.randn(1, 768)
p_test = proj(z_test)
print(f"   ✓ Test projection: {z_test.shape} → {p_test.shape}, norm={p_test.norm().item():.4f}")

print("\n" + "=" * 60)
print("✅ All components working!")
print("=" * 60)
print("\nReady for full pipeline. Run:")
print("  python main.py")
print("=" * 60)
