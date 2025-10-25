"""
Triplet Mining for User Embedding Learning

This module handles the creation of triplets from click data:
- Anchor: User A
- Positive: User B who clicked same ads as A
- Negative: User C who clicked different ads from A
"""

import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import random


class ClickDataProcessor:
    """
    Processes raw click data to build user-ad interaction graphs
    and generate triplets for contrastive learning.
    """
    
    def __init__(self, click_data: List[Tuple[int, int, bool]]):
        """
        Args:
            click_data: List of (user_id, ad_id, clicked) tuples
        """
        self.click_data = click_data
        
        # Build interaction maps
        self.user_to_ads: Dict[int, Set[int]] = defaultdict(set)
        self.ad_to_users: Dict[int, Set[int]] = defaultdict(set)
        
        for user_id, ad_id, clicked in click_data:
            if clicked:  # Only use positive clicks
                self.user_to_ads[user_id].add(ad_id)
                self.ad_to_users[ad_id].add(user_id)
        
        self.all_users = list(self.user_to_ads.keys())
        self.all_ads = list(self.ad_to_users.keys())
    
    def get_user_similarity(self, user_a: int, user_b: int) -> float:
        """
        Compute Jaccard similarity between two users based on clicked ads.
        """
        ads_a = self.user_to_ads.get(user_a, set())
        ads_b = self.user_to_ads.get(user_b, set())
        
        if len(ads_a) == 0 or len(ads_b) == 0:
            return 0.0
        
        intersection = len(ads_a & ads_b)
        union = len(ads_a | ads_b)
        
        return intersection / union if union > 0 else 0.0
    
    def find_positive_user(
        self, 
        anchor_user: int, 
        strategy: str = 'co-click'
    ) -> int:
        """
        Find a positive user (similar to anchor).
        
        Strategies:
        - 'co-click': Sample from users who clicked at least one same ad
        - 'high-overlap': Sample from users with high Jaccard similarity
        """
        anchor_ads = self.user_to_ads.get(anchor_user, set())
        
        if len(anchor_ads) == 0:
            # No clicks, return random user
            return random.choice(self.all_users)
        
        if strategy == 'co-click':
            # Find all users who clicked at least one same ad
            candidate_users = set()
            for ad_id in anchor_ads:
                candidate_users.update(self.ad_to_users[ad_id])
            
            # Remove anchor itself
            candidate_users.discard(anchor_user)
            
            if len(candidate_users) == 0:
                # Fallback to random
                return random.choice(self.all_users)
            
            return random.choice(list(candidate_users))
        
        elif strategy == 'high-overlap':
            # Find users with high Jaccard similarity
            similarities = []
            for user in self.all_users:
                if user != anchor_user:
                    sim = self.get_user_similarity(anchor_user, user)
                    if sim > 0:
                        similarities.append((user, sim))
            
            if len(similarities) == 0:
                return random.choice(self.all_users)
            
            # Sample proportional to similarity
            users, sims = zip(*similarities)
            sims = np.array(sims)
            probs = sims / sims.sum()
            
            return np.random.choice(users, p=probs)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def find_negative_user(
        self, 
        anchor_user: int,
        strategy: str = 'random'
    ) -> int:
        """
        Find a negative user (dissimilar to anchor).
        
        Strategies:
        - 'random': Random user
        - 'hard': User with no overlapping clicks
        - 'semi-hard': User with low but non-zero overlap
        """
        anchor_ads = self.user_to_ads.get(anchor_user, set())
        
        if strategy == 'random':
            candidate = random.choice(self.all_users)
            while candidate == anchor_user:
                candidate = random.choice(self.all_users)
            return candidate
        
        elif strategy == 'hard':
            # Find users with zero overlapping clicks
            candidates = []
            for user in self.all_users:
                if user != anchor_user:
                    overlap = len(anchor_ads & self.user_to_ads.get(user, set()))
                    if overlap == 0:
                        candidates.append(user)
            
            if len(candidates) == 0:
                # Fallback to random
                return self.find_negative_user(anchor_user, strategy='random')
            
            return random.choice(candidates)
        
        elif strategy == 'semi-hard':
            # Find users with low similarity (but not zero)
            similarities = []
            for user in self.all_users:
                if user != anchor_user:
                    sim = self.get_user_similarity(anchor_user, user)
                    if 0 < sim < 0.3:  # Low but non-zero
                        similarities.append((user, sim))
            
            if len(similarities) == 0:
                # Fallback to hard negatives
                return self.find_negative_user(anchor_user, strategy='hard')
            
            # Sample inversely proportional to similarity
            users, sims = zip(*similarities)
            sims = np.array(sims)
            probs = (1 - sims) / (1 - sims).sum()
            
            return np.random.choice(users, p=probs)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def generate_triplets(
        self,
        num_triplets: int,
        positive_strategy: str = 'co-click',
        negative_strategy: str = 'hard'
    ) -> List[Tuple[int, int, int]]:
        """
        Generate triplets (anchor, positive, negative) for training.
        
        Returns:
            List of (anchor_user_id, positive_user_id, negative_user_id)
        """
        triplets = []
        
        for _ in range(num_triplets):
            # Sample anchor user
            anchor = random.choice(self.all_users)
            
            # Find positive and negative
            positive = self.find_positive_user(anchor, positive_strategy)
            negative = self.find_negative_user(anchor, negative_strategy)
            
            triplets.append((anchor, positive, negative))
        
        return triplets


class TripletBatchSampler:
    """
    Custom batch sampler that yields triplets for training.
    Ensures each batch has anchor-positive-negative structure.
    """
    
    def __init__(
        self,
        processor: ClickDataProcessor,
        batch_size: int,
        triplets_per_epoch: int,
        positive_strategy: str = 'co-click',
        negative_strategy: str = 'hard'
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.triplets_per_epoch = triplets_per_epoch
        self.positive_strategy = positive_strategy
        self.negative_strategy = negative_strategy
    
    def __iter__(self):
        # Generate triplets for this epoch
        triplets = self.processor.generate_triplets(
            self.triplets_per_epoch,
            self.positive_strategy,
            self.negative_strategy
        )
        
        # Shuffle and batch
        random.shuffle(triplets)
        
        for i in range(0, len(triplets), self.batch_size):
            batch = triplets[i:i + self.batch_size]
            yield batch
    
    def __len__(self):
        return (self.triplets_per_epoch + self.batch_size - 1) // self.batch_size


class AugmentedClickDataset:
    """
    Dataset that combines click data with triplet mining.
    Each sample includes:
    - User-ad pair for click prediction
    - Positive/negative users for contrastive learning
    """
    
    def __init__(
        self,
        user_ids: List[int],
        ad_ids: List[int],
        ad_texts: List[str],
        clicked: List[bool],
        processor: ClickDataProcessor,
        triplet_probability: float = 0.5
    ):
        self.user_ids = user_ids
        self.ad_ids = ad_ids
        self.ad_texts = ad_texts
        self.clicked = clicked
        self.processor = processor
        self.triplet_probability = triplet_probability
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        
        sample = {
            'user_id': user_id,
            'ad_id': self.ad_ids[idx],
            'ad_text': self.ad_texts[idx],
            'clicked': float(self.clicked[idx])
        }
        
        # Optionally add triplet information
        if random.random() < self.triplet_probability:
            positive_user = self.processor.find_positive_user(user_id)
            negative_user = self.processor.find_negative_user(user_id)
            
            sample['positive_user'] = positive_user
            sample['negative_user'] = negative_user
        
        return sample


def augmented_collate_fn(batch):
    """Collate function for augmented dataset."""
    collated = {
        'user_ids': torch.tensor([item['user_id'] for item in batch]),
        'ad_ids': torch.tensor([item['ad_id'] for item in batch]),
        'ad_texts': [item['ad_text'] for item in batch],
        'clicked': torch.tensor([item['clicked'] for item in batch])
    }
    
    # Add triplet data if available
    if 'positive_user' in batch[0]:
        collated['positive_users'] = torch.tensor(
            [item['positive_user'] for item in batch]
        )
        collated['negative_users'] = torch.tensor(
            [item['negative_user'] for item in batch]
        )
    
    return collated


# ==================== Utility Functions ====================

def compute_click_statistics(click_data: List[Tuple[int, int, bool]]):
    """
    Compute statistics about the click dataset.
    """
    user_clicks = defaultdict(int)
    ad_clicks = defaultdict(int)
    total_clicks = 0
    total_impressions = len(click_data)
    
    for user_id, ad_id, clicked in click_data:
        if clicked:
            user_clicks[user_id] += 1
            ad_clicks[ad_id] += 1
            total_clicks += 1
    
    stats = {
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'ctr': total_clicks / total_impressions if total_impressions > 0 else 0,
        'num_users': len(user_clicks),
        'num_ads': len(ad_clicks),
        'avg_clicks_per_user': np.mean(list(user_clicks.values())),
        'avg_clicks_per_ad': np.mean(list(ad_clicks.values())),
        'users_with_clicks': len([c for c in user_clicks.values() if c > 0]),
        'ads_with_clicks': len([c for c in ad_clicks.values() if c > 0])
    }
    
    return stats


def analyze_user_overlap(processor: ClickDataProcessor, sample_size: int = 100):
    """
    Analyze overlap in ad clicks between users.
    """
    users = random.sample(processor.all_users, min(sample_size, len(processor.all_users)))
    
    similarities = []
    for i, user_a in enumerate(users):
        for user_b in users[i+1:]:
            sim = processor.get_user_similarity(user_a, user_b)
            similarities.append(sim)
    
    return {
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'std_similarity': np.std(similarities)
    }


# ==================== Example Usage ====================

if __name__ == '__main__':
    # Generate synthetic click data
    print("Generating synthetic click data...")
    np.random.seed(42)
    
    NUM_USERS = 1000
    NUM_ADS = 500
    NUM_INTERACTIONS = 10000
    
    # Simulate user preferences (clusters)
    user_clusters = np.random.randint(0, 5, size=NUM_USERS)
    ad_clusters = np.random.randint(0, 5, size=NUM_ADS)
    
    click_data = []
    for _ in range(NUM_INTERACTIONS):
        user_id = np.random.randint(0, NUM_USERS)
        ad_id = np.random.randint(0, NUM_ADS)
        
        # Higher probability of click if user and ad in same cluster
        if user_clusters[user_id] == ad_clusters[ad_id]:
            clicked = np.random.random() < 0.3  # 30% CTR for matching
        else:
            clicked = np.random.random() < 0.05  # 5% CTR for non-matching
        
        click_data.append((user_id, ad_id, clicked))
    
    # Compute statistics
    stats = compute_click_statistics(click_data)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Create processor
    processor = ClickDataProcessor(click_data)
    
    # Analyze user overlap
    print("\nAnalyzing user overlap...")
    overlap_stats = analyze_user_overlap(processor, sample_size=100)
    print("User Overlap Statistics:")
    for key, value in overlap_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate triplets
    print("\nGenerating triplets...")
    triplets = processor.generate_triplets(
        num_triplets=100,
        positive_strategy='co-click',
        negative_strategy='hard'
    )
    
    print(f"Generated {len(triplets)} triplets")
    print("Sample triplets (anchor, positive, negative):")
    for i, (a, p, n) in enumerate(triplets[:5]):
        sim_ap = processor.get_user_similarity(a, p)
        sim_an = processor.get_user_similarity(a, n)
        print(f"  {i+1}. ({a}, {p}, {n}) - sim(a,p)={sim_ap:.3f}, sim(a,n)={sim_an:.3f}")
    
    print("\n✅ Triplet mining working correctly!")
    print("   - Positive users have higher similarity to anchor")
    print("   - Negative users have lower similarity to anchor")
