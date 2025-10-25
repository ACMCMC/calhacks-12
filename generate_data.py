"""
Data Generation Utilities

Generate synthetic click data for testing and development.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import random


class SyntheticDataGenerator:
    """
    Generate synthetic user-ad click data with realistic patterns.
    """
    
    def __init__(
        self,
        num_users: int = 10000,
        num_ads: int = 1000,
        num_categories: int = 10,
        seed: int = 42
    ):
        self.num_users = num_users
        self.num_ads = num_ads
        self.num_categories = num_categories
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate user and ad characteristics
        self.user_interests = self._generate_user_interests()
        self.ad_categories, self.ad_features = self._generate_ad_features()
    
    def _generate_user_interests(self) -> np.ndarray:
        """
        Generate user interest vectors.
        Each user has preferences over categories (soft clustering).
        """
        # Some users have focused interests, others are diverse
        concentration = np.random.choice([1.0, 3.0, 10.0], size=self.num_users)
        
        interests = np.random.dirichlet(
            alpha=np.ones(self.num_categories) * concentration[:, None],
            size=1
        )[0]
        
        return interests
    
    def _generate_ad_features(self) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Generate ad category assignments and textual features.
        """
        # Each ad belongs to one primary category
        categories = np.random.randint(0, self.num_categories, size=self.num_ads)
        
        # Feature vocabulary
        price_features = ['affordable', 'mid-range', 'expensive', 'premium', 'budget']
        audience_features = ['for children', 'for adults', 'for seniors', 'for everyone', 'for professionals']
        style_features = ['casual', 'formal', 'sporty', 'elegant', 'modern', 'vintage']
        quality_features = ['sustainable', 'eco-friendly', 'durable', 'handmade', 'luxury']
        category_names = ['electronics', 'clothing', 'food', 'travel', 'entertainment', 
                         'education', 'health', 'automotive', 'home', 'sports']
        
        # Generate feature text for each ad
        ad_features = []
        for ad_id in range(self.num_ads):
            category = categories[ad_id]
            
            # Select random features
            features = [
                category_names[category % len(category_names)],
                random.choice(price_features),
                random.choice(audience_features),
                random.choice(style_features)
            ]
            
            # Sometimes add quality features
            if random.random() < 0.3:
                features.append(random.choice(quality_features))
            
            ad_features.append(features)
        
        return categories, ad_features
    
    def generate_interactions(
        self, 
        num_interactions: int = 50000,
        ctr_match: float = 0.25,
        ctr_mismatch: float = 0.03
    ) -> pd.DataFrame:
        """
        Generate user-ad interactions with clicks.
        
        Args:
            num_interactions: Number of impressions to generate
            ctr_match: Click-through rate when user interest matches ad category
            ctr_mismatch: CTR when they don't match
        
        Returns:
            DataFrame with columns: user_id, ad_id, ad_text, clicked
        """
        data = []
        
        for _ in range(num_interactions):
            # Sample user and ad
            user_id = np.random.randint(0, self.num_users)
            ad_id = np.random.randint(0, self.num_ads)
            
            # Get ad category
            ad_category = self.ad_categories[ad_id]
            
            # Compute click probability based on user interest
            user_interest_in_category = self.user_interests[user_id, ad_category]
            
            # Higher interest → higher CTR
            if user_interest_in_category > 0.2:
                base_ctr = ctr_match
            else:
                base_ctr = ctr_mismatch
            
            # Add noise
            click_prob = base_ctr * (1 + user_interest_in_category)
            clicked = np.random.random() < click_prob
            
            # Convert features to text
            ad_text = ', '.join(self.ad_features[ad_id])
            
            data.append({
                'user_id': user_id,
                'ad_id': ad_id,
                'ad_text': ad_text,
                'clicked': clicked
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_train_test_split(
        self,
        num_interactions: int = 50000,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate train and test datasets.
        """
        df = self.generate_interactions(num_interactions)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        return train_df, test_df


def generate_cold_start_test_data(
    train_df: pd.DataFrame,
    num_new_users: int = 100,
    num_new_ads: int = 50
) -> pd.DataFrame:
    """
    Generate cold-start test data with new users and ads.
    """
    max_user_id = train_df['user_id'].max()
    max_ad_id = train_df['ad_id'].max()
    
    # Sample from existing patterns but with new IDs
    sample_interactions = train_df.sample(n=num_new_users * 5, replace=True)
    
    new_data = []
    for i, row in sample_interactions.iterrows():
        # Replace with new user/ad IDs
        new_user_id = max_user_id + 1 + (i % num_new_users)
        new_ad_id = max_ad_id + 1 + (i % num_new_ads)
        
        new_data.append({
            'user_id': new_user_id,
            'ad_id': new_ad_id,
            'ad_text': row['ad_text'],
            'clicked': row['clicked']
        })
    
    return pd.DataFrame(new_data)


if __name__ == '__main__':
    print("Generating synthetic PrivAds dataset...")
    
    # Generate data
    generator = SyntheticDataGenerator(
        num_users=5000,
        num_ads=1000,
        num_categories=8,
        seed=42
    )
    
    train_df, test_df = generator.generate_train_test_split(
        num_interactions=50000,
        test_size=0.2
    )
    
    print(f"\nGenerated {len(train_df):,} training interactions")
    print(f"Generated {len(test_df):,} test interactions")
    
    # Compute statistics
    print("\nTraining Data Statistics:")
    print(f"  Unique users: {train_df['user_id'].nunique():,}")
    print(f"  Unique ads: {train_df['ad_id'].nunique():,}")
    print(f"  Overall CTR: {train_df['clicked'].mean():.2%}")
    print(f"  Clicks: {train_df['clicked'].sum():,}")
    
    print("\nSample interactions:")
    print(train_df.head(10).to_string(index=False))
    
    # Save to CSV
    output_dir = './data'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train_clicks.csv', index=False)
    test_df.to_csv(f'{output_dir}/test_clicks.csv', index=False)
    
    print(f"\n✅ Data saved to {output_dir}/")
    print("   - train_clicks.csv")
    print("   - test_clicks.csv")
    
    # Generate cold start data
    cold_start_df = generate_cold_start_test_data(train_df, num_new_users=100)
    cold_start_df.to_csv(f'{output_dir}/cold_start_test.csv', index=False)
    print("   - cold_start_test.csv")
