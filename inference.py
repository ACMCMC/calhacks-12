"""
Inference and Evaluation Script

Load a trained model and run inference on test data.
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import argparse
from pathlib import Path

from model import PrivAdsModel, predict_top_k_ads
from triplet_mining import ClickDataProcessor


class Evaluator:
    """
    Evaluate model performance on various metrics.
    """
    
    def __init__(self, model: PrivAdsModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def compute_ranking_metrics(
        self, 
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute ranking metrics: Precision@K, Recall@K, NDCG@K, MRR
        """
        # Group by user
        user_groups = test_df.groupby('user_id')
        
        metrics = {f'precision@{k}': [] for k in k_values}
        metrics.update({f'recall@{k}': [] for k in k_values})
        metrics.update({f'ndcg@{k}': [] for k in k_values})
        metrics['mrr'] = []
        
        for user_id, group in tqdm(user_groups, desc='Evaluating'):
            # Get ground truth clicks
            clicked_ads = set(group[group['clicked']]['ad_id'].values)
            
            if len(clicked_ads) == 0:
                continue  # Skip users with no clicks
            
            # Get all ads for this user
            candidate_ads = [
                (row['ad_id'], row['ad_text']) 
                for _, row in group.iterrows()
            ]
            
            # Predict rankings
            rankings = predict_top_k_ads(
                self.model, 
                user_id=user_id,
                candidate_ads=candidate_ads,
                k=max(k_values),
                device=self.device
            )
            
            # Extract ranked ad IDs
            ranked_ad_ids = [ad_id for ad_id, _ in rankings]
            
            # Compute metrics
            for k in k_values:
                top_k = set(ranked_ad_ids[:k])
                
                # Precision@K
                precision = len(top_k & clicked_ads) / k
                metrics[f'precision@{k}'].append(precision)
                
                # Recall@K
                recall = len(top_k & clicked_ads) / len(clicked_ads)
                metrics[f'recall@{k}'].append(recall)
                
                # NDCG@K
                ndcg = self._compute_ndcg(ranked_ad_ids[:k], clicked_ads)
                metrics[f'ndcg@{k}'].append(ndcg)
            
            # MRR
            mrr = self._compute_mrr(ranked_ad_ids, clicked_ads)
            metrics['mrr'].append(mrr)
        
        # Average metrics
        return {key: np.mean(values) for key, values in metrics.items()}
    
    @staticmethod
    def _compute_ndcg(ranked_list: List[int], relevant_items: set) -> float:
        """Compute NDCG (Normalized Discounted Cumulative Gain)."""
        dcg = sum(
            (1.0 if item in relevant_items else 0.0) / np.log2(i + 2)
            for i, item in enumerate(ranked_list)
        )
        
        # Ideal DCG (all relevant items at top)
        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(relevant_items), len(ranked_list)))
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def _compute_mrr(ranked_list: List[int], relevant_items: set) -> float:
        """Compute MRR (Mean Reciprocal Rank)."""
        for i, item in enumerate(ranked_list):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    @torch.no_grad()
    def compute_embedding_statistics(self, num_users: int = 1000) -> Dict[str, float]:
        """
        Analyze user embedding distributions.
        """
        user_ids = torch.arange(min(num_users, self.model.user_embeddings.embeddings.num_embeddings))
        user_ids = user_ids.to(self.device)
        
        embeddings = self.model.encode_users(user_ids)
        embeddings = embeddings.cpu().numpy()
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarity)
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diagonal_sims = similarities[mask]
        
        return {
            'mean_similarity': off_diagonal_sims.mean(),
            'std_similarity': off_diagonal_sims.std(),
            'min_similarity': off_diagonal_sims.min(),
            'max_similarity': off_diagonal_sims.max(),
            'embedding_norm_mean': np.linalg.norm(embeddings, axis=1).mean(),
            'embedding_norm_std': np.linalg.norm(embeddings, axis=1).std()
        }


def load_model(checkpoint_path: str, num_users: int, user_dim: int = 256) -> PrivAdsModel:
    """Load trained model from checkpoint."""
    model = PrivAdsModel(num_users=num_users, user_dim=user_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='PrivAds Inference & Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--num-users', type=int, required=True, help='Number of users')
    parser.add_argument('--user-dim', type=int, default=256, help='User embedding dimension')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20])
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PrivAds: Inference & Evaluation")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.num_users, args.user_dim)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(args.test_data)
    print(f"✓ Loaded {len(test_df):,} test interactions")
    print(f"  Unique users: {test_df['user_id'].nunique():,}")
    print(f"  Unique ads: {test_df['ad_id'].nunique():,}")
    print(f"  CTR: {test_df['clicked'].mean():.2%}")
    
    # Create evaluator
    evaluator = Evaluator(model, device=args.device)
    
    # Compute ranking metrics
    print("\nComputing ranking metrics...")
    ranking_metrics = evaluator.compute_ranking_metrics(test_df, k_values=args.k_values)
    
    print("\n" + "=" * 60)
    print("RANKING METRICS")
    print("=" * 60)
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compute embedding statistics
    print("\nComputing embedding statistics...")
    emb_stats = evaluator.compute_embedding_statistics(num_users=min(1000, args.num_users))
    
    print("\n" + "=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)
    for stat, value in emb_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    sample_user_id = test_df['user_id'].iloc[0]
    sample_user_data = test_df[test_df['user_id'] == sample_user_id]
    
    candidate_ads = [
        (row['ad_id'], row['ad_text'])
        for _, row in sample_user_data.head(10).iterrows()
    ]
    
    top_ads = predict_top_k_ads(
        model, 
        user_id=sample_user_id,
        candidate_ads=candidate_ads,
        k=5,
        device=args.device
    )
    
    print(f"\nTop 5 ads for user {sample_user_id}:")
    for i, (ad_id, score) in enumerate(top_ads, 1):
        ad_text = test_df[test_df['ad_id'] == ad_id]['ad_text'].iloc[0]
        clicked = test_df[
            (test_df['user_id'] == sample_user_id) & 
            (test_df['ad_id'] == ad_id)
        ]['clicked'].iloc[0]
        
        click_indicator = "✓" if clicked else "✗"
        print(f"  {i}. Ad {ad_id} ({click_indicator}): {score:.4f}")
        print(f"     \"{ad_text}\"")


if __name__ == '__main__':
    main()
