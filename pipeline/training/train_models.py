"""
Training Pipeline Orchestration for PrivAds
Runs the complete PrivAds training pipeline.
"""

import sys
import os
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.training.click_data import SyntheticClickGenerator
from pipeline.training.train_user_embeddings import train_user_embeddings
from pipeline.training.train_projector import train_projector
from pipeline.training.ad_encoder import AdEncoder
import numpy as np
import torch

def main():
    print("=" * 60)
    print("PrivAds: PrivAds Training Pipeline")
    print("=" * 60)

    # Config
    N_USERS = 3000
    N_ADS = 1500
    N_CLICKS = 60000
    D_USER = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths - relative to project root
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    models_dir = script_dir / "backend" / "models"
    data_dir = script_dir / "backend" / "data"
    models_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n📊 Config: {N_USERS} users, {N_ADS} ads, {N_CLICKS} interactions")
    print(f"🖥️  Device: {DEVICE}")

    # Step 1: Load Ad Encoder (Jina CLIP v2 - Frozen)
    print("\n" + "=" * 60)
    print("Step 1: Load Ad Encoder (Jina CLIP v2 - Frozen)")
    print("=" * 60)
    encoder = AdEncoder(device=DEVICE)
    d_ad = encoder.d_ad

    # Step 2: Generate synthetic ad embeddings
    print("\n" + "=" * 60)
    print("Step 2: Generate Ad Embeddings")
    print("=" * 60)
    print("Encoding sample ads...")

    sample_ads = [
        "Sustainable children's clothing, eco-friendly",
        "Luxury watch, premium quality, expensive",
        "Affordable kids toys, colorful, fun",
        "Organic food delivery, healthy meals",
        "Gaming laptop, high performance"
    ]

    ad_embeddings_raw = {}
    for i, text in enumerate(sample_ads):
        z_ad = encoder.encode(text=text)
        ad_embeddings_raw[i] = z_ad
        print(f"  Ad {i}: '{text[:40]}...' -> {z_ad.shape}")

    # Fill remaining ads with random (simulate)
    print(f"Filling {N_ADS - len(sample_ads)} more ads with random embeddings (simulated)...")
    for aid in range(len(sample_ads), N_ADS):
        ad_embeddings_raw[aid] = np.random.randn(d_ad)
        ad_embeddings_raw[aid] /= np.linalg.norm(ad_embeddings_raw[aid])

    np.savez(data_dir / "ad_embeddings_raw.npz", **{str(k): v for k, v in ad_embeddings_raw.items()})
    print(f"✓ Saved {len(ad_embeddings_raw)} ad embeddings to {data_dir / 'ad_embeddings_raw.npz'}")

    # Step 3: Generate synthetic clicks
    print("\n" + "=" * 60)
    print("Step 3: Generate Synthetic Click Data")
    print("=" * 60)
    click_gen = SyntheticClickGenerator(
        n_users=N_USERS,
        n_ads=N_ADS,
        d_user=D_USER,
        ad_embeddings=ad_embeddings_raw  # Use actual embeddings now
    )
    clicks = click_gen.get_clicks(n_samples=N_CLICKS)

    # Split into train/test (80/20)
    np.random.shuffle(clicks)
    split_idx = int(0.8 * len(clicks))
    train_clicks = clicks[:split_idx]
    test_clicks = clicks[split_idx:]

    n_train_clicked = sum(1 for _, _, clicked, _ in train_clicks if clicked)
    n_test_clicked = sum(1 for _, _, clicked, _ in test_clicks if clicked)
    print(f"✓ Train: {len(train_clicks)} impressions, {n_train_clicked} clicks (CTR: {n_train_clicked/len(train_clicks):.2%})")
    print(f"✓ Test:  {len(test_clicks)} impressions, {n_test_clicked} clicks (CTR: {n_test_clicked/len(test_clicks):.2%})")

    # Step 4: Train user embeddings
    print("\n" + "=" * 60)
    print("Step 4: Train User Embeddings (InfoNCE on co-clicks)")
    print("=" * 60)
    user_embeddings, global_mean, embedding_changes = train_user_embeddings(
        clicks=train_clicks,
        n_users=N_USERS,
        d_user=D_USER,
        epochs=120,
        batch_size=512,
        device=DEVICE,
        synthetic_generator=click_gen
    )

    np.save(models_dir / "user_embeddings.npy", user_embeddings)
    np.save(models_dir / "global_mean.npy", global_mean)
    print(f"✓ Saved user embeddings to {models_dir / 'user_embeddings.npy'}")
    print(f"✓ Saved global_mean to {models_dir / 'global_mean.npy'}")

    # Step 5: Train projector
    print("\n" + "=" * 60)
    print("Step 5: Train Projector (ad space → user space)")
    print("=" * 60)
    projector = train_projector(
        clicks=train_clicks,
        user_embeddings=user_embeddings,
        ad_embeddings_raw=ad_embeddings_raw,
        d_ad=d_ad,
        d_user=D_USER,
        epochs=80,
        batch_size=256,
        device=DEVICE
    )

    torch.save(projector.state_dict(), models_dir / "projector.pt")
    print(f"✓ Saved projector to {models_dir / 'projector.pt'}")

    # Step 6: Precompute projected ad embeddings
    print("\n" + "=" * 60)
    print("Step 6: Precompute Projected Ad Embeddings")
    print("=" * 60)
    projector.eval()
    with torch.no_grad():
        p_ads = {}
        for aid, z_ad in ad_embeddings_raw.items():
            z = torch.tensor(z_ad, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            p = projector(z).cpu().numpy()[0]
            p_ads[aid] = p

    np.savez(data_dir / "ad_projected.npz", **{str(k): v for k, v in p_ads.items()})
    print(f"✓ Saved {len(p_ads)} projected ad embeddings to {data_dir / 'ad_projected.npz'}")

    print("\n" + "=" * 60)
    print("✅ Training Pipeline Complete!")
    print("=" * 60)
    print(f"📁 Models saved to: {models_dir.absolute()}")
    print(f"📁 Data saved to: {data_dir.absolute()}")
    print("\nNext steps:")
    print("  - Run ad processing pipeline: python pipeline/run_ad_pipeline.py")
    print("  - Load data into databases: python pipeline/load_databases.py")
    print("  - Start backend API: cd backend && python main.py")
    print("=" * 60)

if __name__ == "__main__":
    main()