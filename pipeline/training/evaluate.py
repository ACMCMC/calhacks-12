from sklearn.metrics import silhouette_score
"""Evaluation and visualization for PrivAds."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_models_and_data(models_dir: Path = Path("models"), data_dir: Path = Path("data")):
    """Load trained models and precomputed embeddings."""
    user_embeddings = np.load(models_dir / "user_embeddings.npy")
    global_mean = np.load(models_dir / "global_mean.npy")

    # Load ad embeddings
    ad_embeddings_raw = dict(np.load(data_dir / "ad_embeddings_raw.npz"))
    ad_embeddings_projected = dict(np.load(data_dir / "ad_projected.npz"))

    # Dynamically determine d_ad from ad_embeddings_raw
    first_ad = next(iter(ad_embeddings_raw.values()))
    d_ad = first_ad.shape[0]

    from projector import Projector
    projector = Projector(d_ad=d_ad)
    projector.load_state_dict(torch.load(models_dir / "projector.pt"))
    projector.eval()

    return {
        'user_embeddings': user_embeddings,
        'global_mean': global_mean,
        'projector': projector,
        'ad_embeddings_raw': ad_embeddings_raw,
        'ad_embeddings_projected': ad_embeddings_projected
    }


def compute_precision_at_k(
    user_embeddings: np.ndarray,
    ad_embeddings_projected: Dict[int, np.ndarray],
    test_clicks: List[Tuple[int, int, bool, int]],
    k: int = 10
) -> float:
    """
    Compute Precision@K for test clicks.

    For each user who has test clicks, check if their top-K recommended
    ads include the ads they actually clicked on.
    """
    # Build test click matrix: user_id -> set of clicked ad_ids
    test_clicks_by_user = defaultdict(set)
    for uid, aid, clicked, pos in test_clicks:
        if clicked:
            test_clicks_by_user[uid].add(aid)

    # Only evaluate users who have test clicks
    users_with_clicks = list(test_clicks_by_user.keys())
    if len(users_with_clicks) == 0:
        print("No test clicks found!")
        return 0.0

    # Get all ad_ids
    all_ad_ids = list(ad_embeddings_projected.keys())

    precisions = []
    for uid in users_with_clicks:
        # Get user's embedding
        u_emb = user_embeddings[uid]

        # Compute similarities to all ads
        similarities = {}
        for aid in all_ad_ids:
            p_ad = ad_embeddings_projected[aid]
            sim = np.dot(u_emb, p_ad)  # Cosine similarity (normalized vectors)
            similarities[aid] = sim

        # Get top-K recommendations
        top_k_ads = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_ad_ids = {aid for aid, _ in top_k_ads}

        # Check how many test clicks are in top-K
        actual_clicks = test_clicks_by_user[uid]
        hits = len(top_k_ad_ids & actual_clicks)
        precision = hits / k
        precisions.append(precision)

    return np.mean(precisions)


def plot_user_embeddings_tsne(
    user_embeddings: np.ndarray,
    synthetic_generator=None,
    save_path: Path = None
):
    """Create t-SNE visualization of user embeddings colored by archetype."""
    # If we have synthetic generator, we can color by true archetypes
    if synthetic_generator is not None:
        # Get archetype assignments
        archetype_labels = []
        for uid in range(len(user_embeddings)):
            # Find closest archetype
            u_pref = synthetic_generator.user_preferences[uid]
            archetypes = np.array(list(synthetic_generator.user_preferences.values())[:10])  # First 10 are archetypes
            similarities = np.dot(archetypes, u_pref)
            archetype_id = np.argmax(similarities)
            archetype_labels.append(archetype_id)
    else:
        # No ground truth, just show all as one color
        archetype_labels = [0] * len(user_embeddings)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(user_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=archetype_labels,
        cmap='tab10',
        alpha=0.7,
        s=50
    )

    plt.title('User Embeddings (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Archetype' if synthetic_generator else 'User Group')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()


def plot_recommendation_heatmap(
    user_embeddings: np.ndarray,
    ad_embeddings_projected: Dict[int, np.ndarray],
    n_users: int = 20,
    n_ads: int = 50,
    save_path: Path = None
):
    """Create heatmap showing user-ad similarity matrix."""
    # Sample users and ads
    user_ids = np.random.choice(len(user_embeddings), size=min(n_users, len(user_embeddings)), replace=False)
    ad_ids = np.random.choice(list(ad_embeddings_projected.keys()), size=min(n_ads, len(ad_embeddings_projected)), replace=False)

    # Compute similarity matrix
    similarity_matrix = np.zeros((len(user_ids), len(ad_ids)))
    for i, uid in enumerate(user_ids):
        u_emb = user_embeddings[uid]
        for j, aid in enumerate(ad_ids):
            p_ad = ad_embeddings_projected[aid]
            similarity_matrix[i, j] = np.dot(u_emb, p_ad)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        similarity_matrix,
        cmap='RdYlBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=[f'Ad {aid}' for aid in ad_ids],
        yticklabels=[f'User {uid}' for uid in user_ids],
        cbar_kws={'label': 'Cosine Similarity'}
    )

    plt.title('User-Ad Recommendation Similarities', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved recommendation heatmap to {save_path}")
    else:
        plt.show()


def evaluate_model(
    models_dir: Path = Path("../../backend/models"),
    data_dir: Path = Path("../../backend/data"),
    test_clicks: List[Tuple[int, int, bool, int]] = None,
    synthetic_generator=None,
    output_dir: Path = Path("../../backend/evaluation_results"),
    embedding_changes: List[float] = None
):
    """
    Run full evaluation suite.

    Args:
        test_clicks: Test click data for precision evaluation
        synthetic_generator: For ground truth archetype coloring
    """
    output_dir.mkdir(exist_ok=True)

    print("Loading models and data...")
    data = load_models_and_data(models_dir, data_dir)

    # Precision@10
    if test_clicks:
        print("Computing Precision@10...")
        precision_10 = compute_precision_at_k(
            data['user_embeddings'],
            data['ad_embeddings_projected'],
            test_clicks,
            k=10
        )
        print(".4f")

        # Save metric
        with open(output_dir / "precision_10.txt", "w") as f:
            f.write(".4f")
    else:
        print("No test clicks provided, skipping Precision@10")


    # Silhouette score (if synthetic_generator is provided)
    if synthetic_generator is not None:
        print("Computing user embedding silhouette score...")
        # Get true archetype labels for each user
        user_labels = [synthetic_generator.user_archetype[uid] for uid in range(len(data['user_embeddings']))]
        sil_score = silhouette_score(data['user_embeddings'], user_labels)
        print(f"User embedding silhouette score: {sil_score:.4f}")
        
        # Also compute within/between cluster distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(data['user_embeddings'])
        
        within_cluster = []
        between_cluster = []
        for i in range(len(user_labels)):
            for j in range(i+1, len(user_labels)):
                if user_labels[i] == user_labels[j]:
                    within_cluster.append(distances[i,j])
                else:
                    between_cluster.append(distances[i,j])
        
        avg_within = np.mean(within_cluster) if within_cluster else 0
        avg_between = np.mean(between_cluster) if between_cluster else 0
        
        print(f"Average within-archetype distance: {avg_within:.4f}")
        print(f"Average between-archetype distance: {avg_between:.4f}")
        print(f"Distance ratio (within/between): {avg_within/avg_between:.4f}" if avg_between > 0 else "Distance ratio: inf")
        
        with open(output_dir / "user_silhouette.txt", "w") as f:
            f.write(f"{sil_score:.4f}\n")
            f.write(f"within_dist: {avg_within:.4f}\n")
            f.write(f"between_dist: {avg_between:.4f}\n")
            f.write(f"ratio: {avg_within/avg_between:.4f}\n" if avg_between > 0 else "ratio: inf\n")

    # Embedding changes (if provided)
    if embedding_changes is not None:
        print("Saving embedding changes over training...")
        with open(output_dir / "embedding_changes.txt", "w") as f:
            f.write("Epoch,Change\n")
            for i, change in enumerate(embedding_changes):
                f.write(f"{i+1},{change:.6f}\n")
        print(f"Embedding changes saved to {output_dir / 'embedding_changes.txt'}")

    # t-SNE plot
    print("Creating user embedding t-SNE plot...")
    plot_user_embeddings_tsne(
        data['user_embeddings'],
        synthetic_generator,
        save_path=output_dir / "user_embeddings_tsne.png"
    )

    # Recommendation heatmap
    print("Creating recommendation heatmap...")
    plot_recommendation_heatmap(
        data['user_embeddings'],
        data['ad_embeddings_projected'],
        save_path=output_dir / "recommendation_heatmap.png"
    )

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with synthetic data split
    from click_data import SyntheticClickGenerator

    print("Running evaluation with synthetic data...")

    # Recreate the same synthetic generator for ground truth
    gen = SyntheticClickGenerator(n_users=1000, n_ads=500, d_user=128, seed=42)

    # Generate data and split for evaluation
    all_clicks = gen.get_clicks(20000)  # More data for better evaluation

    # Split into train/test (80/20)
    np.random.shuffle(all_clicks)
    split_idx = int(0.8 * len(all_clicks))
    train_clicks = all_clicks[:split_idx]
    test_clicks = all_clicks[split_idx:]

    print(f"Train clicks: {len(train_clicks)}, Test clicks: {len(test_clicks)}")

    # For now, assume models are already trained
    # In practice, you'd train on train_clicks and evaluate on test_clicks
    evaluate_model(
        test_clicks=test_clicks,
        synthetic_generator=gen
    )