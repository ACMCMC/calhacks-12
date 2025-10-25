"""
Visualization utilities for PrivAds embeddings and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from typing import List, Tuple
import pandas as pd

sns.set_style("whitegrid")


def visualize_embeddings_2d(
    embeddings: np.ndarray,
    labels: List[str] = None,
    method: str = 'tsne',
    title: str = 'Embedding Visualization',
    save_path: str = None
):
    """
    Visualize embeddings in 2D using PCA or t-SNE.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Optional labels for coloring points
        method: 'pca' or 'tsne'
        title: Plot title
        save_path: Path to save figure
    """
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_.sum()
        subtitle = f"PCA (explained variance: {explained_var:.2%})"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        subtitle = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label,
                alpha=0.6,
                s=50
            )
        plt.legend()
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    plt.title(f"{title}\n{subtitle}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation losses over epochs.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', marker='s')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Click loss
    axes[0, 1].plot(epochs, history['train_click_loss'], label='Train Click Loss', marker='o')
    axes[0, 1].plot(epochs, history['val_click_loss'], label='Val Click Loss', marker='s')
    axes[0, 1].set_title('Click Prediction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Triplet loss
    axes[1, 0].plot(epochs, history['train_triplet_loss'], label='Triplet Loss', marker='o', color='green')
    axes[1, 0].set_title('User Contrastive Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Loss components comparison
    axes[1, 1].plot(epochs, history['train_click_loss'], label='Click Loss', marker='o')
    axes[1, 1].plot(epochs, history['train_triplet_loss'], label='Triplet Loss', marker='s')
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_similarity_distribution(
    similarities_clicked: np.ndarray,
    similarities_not_clicked: np.ndarray,
    save_path: str = None
):
    """
    Plot distribution of similarity scores for clicked vs non-clicked pairs.
    
    Args:
        similarities_clicked: Similarity scores for clicked ads
        similarities_not_clicked: Similarity scores for non-clicked ads
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(similarities_clicked, bins=50, alpha=0.6, label='Clicked', color='green')
    plt.hist(similarities_not_clicked, bins=50, alpha=0.6, label='Not Clicked', color='red')
    
    # Add vertical lines for means
    plt.axvline(similarities_clicked.mean(), color='green', linestyle='--', 
                label=f'Clicked Mean: {similarities_clicked.mean():.3f}')
    plt.axvline(similarities_not_clicked.mean(), color='red', linestyle='--',
                label=f'Not Clicked Mean: {similarities_not_clicked.mean():.3f}')
    
    separation = similarities_clicked.mean() - similarities_not_clicked.mean()
    plt.title(f'User-Ad Similarity Distribution\nSeparation: {separation:.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_ranking_metrics(metrics: dict, save_path: str = None):
    """
    Plot ranking metrics (Precision@K, Recall@K, NDCG@K).
    
    Args:
        metrics: Dictionary with metric names as keys
        save_path: Path to save figure
    """
    # Extract k values and metrics
    precision_metrics = {k: v for k, v in metrics.items() if 'precision' in k}
    recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k}
    ndcg_metrics = {k: v for k, v in metrics.items() if 'ndcg' in k}
    
    k_values = [int(k.split('@')[1]) for k in precision_metrics.keys()]
    
    precision_values = list(precision_metrics.values())
    recall_values = list(recall_metrics.values())
    ndcg_values = list(ndcg_metrics.values())
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision@K
    axes[0].plot(k_values, precision_values, marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Precision@K', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_values)
    
    # Recall@K
    axes[1].plot(k_values, recall_values, marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_title('Recall@K', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('K', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_values)
    
    # NDCG@K
    axes[2].plot(k_values, ndcg_values, marker='^', linewidth=2, markersize=8, color='orange')
    axes[2].set_title('NDCG@K', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('K', fontsize=12)
    axes[2].set_ylabel('NDCG', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(k_values)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_user_ad_heatmap(
    user_embeddings: np.ndarray,
    ad_embeddings: np.ndarray,
    user_labels: List[str] = None,
    ad_labels: List[str] = None,
    max_items: int = 20,
    save_path: str = None
):
    """
    Plot heatmap of user-ad similarities.
    
    Args:
        user_embeddings: User embeddings (n_users, dim)
        ad_embeddings: Ad embeddings (n_ads, dim)
        user_labels: Labels for users
        ad_labels: Labels for ads
        max_items: Maximum number of users/ads to show
        save_path: Path to save figure
    """
    # Sample if too many
    if len(user_embeddings) > max_items:
        indices = np.random.choice(len(user_embeddings), max_items, replace=False)
        user_embeddings = user_embeddings[indices]
        if user_labels:
            user_labels = [user_labels[i] for i in indices]
    
    if len(ad_embeddings) > max_items:
        indices = np.random.choice(len(ad_embeddings), max_items, replace=False)
        ad_embeddings = ad_embeddings[indices]
        if ad_labels:
            ad_labels = [ad_labels[i] for i in indices]
    
    # Compute similarity matrix
    similarities = np.dot(user_embeddings, ad_embeddings.T)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        similarities,
        cmap='RdYlGn',
        center=0,
        xticklabels=ad_labels if ad_labels else range(len(ad_embeddings)),
        yticklabels=user_labels if user_labels else range(len(user_embeddings)),
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title('User-Ad Similarity Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Ads', fontsize=12)
    plt.ylabel('Users', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_dashboard(
    history: dict,
    metrics: dict,
    similarities_clicked: np.ndarray,
    similarities_not_clicked: np.ndarray,
    save_path: str = None
):
    """
    Create a comprehensive dashboard with multiple plots.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Validation', marker='s')
    ax1.set_title('Training Progress', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    metric_text = "Key Metrics\n" + "="*20 + "\n"
    for k, v in list(metrics.items())[:5]:
        metric_text += f"{k}: {v:.4f}\n"
    
    ax2.text(0.1, 0.5, metric_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    # Similarity distribution
    ax3 = fig.add_subplot(gs[1, :])
    ax3.hist(similarities_clicked, bins=40, alpha=0.6, label='Clicked', color='green')
    ax3.hist(similarities_not_clicked, bins=40, alpha=0.6, label='Not Clicked', color='red')
    ax3.axvline(similarities_clicked.mean(), color='green', linestyle='--')
    ax3.axvline(similarities_not_clicked.mean(), color='red', linestyle='--')
    ax3.set_title('Similarity Distribution', fontweight='bold')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ranking metrics
    precision_metrics = {k: v for k, v in metrics.items() if 'precision' in k}
    recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k}
    
    if precision_metrics and recall_metrics:
        k_values = [int(k.split('@')[1]) for k in precision_metrics.keys()]
        precision_values = list(precision_metrics.values())
        recall_values = list(recall_metrics.values())
        
        ax4 = fig.add_subplot(gs[2, :])
        x = np.arange(len(k_values))
        width = 0.35
        
        ax4.bar(x - width/2, precision_values, width, label='Precision', alpha=0.8)
        ax4.bar(x + width/2, recall_values, width, label='Recall', alpha=0.8)
        
        ax4.set_title('Ranking Performance', fontweight='bold')
        ax4.set_xlabel('K')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(k_values)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('PrivAds Model Dashboard', fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Visualization utilities for PrivAds")
    print("\nAvailable functions:")
    print("  - visualize_embeddings_2d()")
    print("  - plot_training_history()")
    print("  - plot_similarity_distribution()")
    print("  - plot_ranking_metrics()")
    print("  - plot_user_ad_heatmap()")
    print("  - create_dashboard()")
