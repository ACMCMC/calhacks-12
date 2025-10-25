"""
PrivAds: Privacy-First Advertising Pipeline
Multi-Modal Embedding System

This module implements the core components:
1. Ad Encoder: Text → Embedding
2. User Embeddings: Learned from clicks
3. Projector: Ad space → User space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import numpy as np


# ==================== Component 1: Ad Encoder ====================

class AdEncoder(nn.Module):
    """
    Encodes ad text features into dense embeddings using a pre-trained LM.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', freeze: bool = True):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        if freeze:
            # Freeze the encoder weights for stability
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, ad_texts: List[str]) -> torch.Tensor:
        """
        Args:
            ad_texts: List of ad descriptions, e.g., ["for children, expensive"]
        
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = self.encoder.encode(
            ad_texts, 
            convert_to_tensor=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings


# ==================== Component 2: User Embeddings ====================

class UserEmbeddings(nn.Module):
    """
    Learnable embedding table for users based on their click behavior.
    """
    def __init__(self, num_users: int, embedding_dim: int = 256):
        super().__init__()
        self.embeddings = nn.Embedding(num_users, embedding_dim)
        
        # Initialize with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
    
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
        
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        embs = self.embeddings(user_ids)
        # L2 normalize for cosine similarity
        return F.normalize(embs, p=2, dim=1)


# ==================== Component 3: Projector ====================

class AdToUserProjector(nn.Module):
    """
    Projects ad embeddings into user embedding space.
    """
    def __init__(self, ad_dim: int, user_dim: int, hidden_dim: int = None):
        super().__init__()
        
        if hidden_dim is None:
            # Simple linear projection
            self.projector = nn.Linear(ad_dim, user_dim)
        else:
            # MLP projection with non-linearity
            self.projector = nn.Sequential(
                nn.Linear(ad_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, user_dim)
            )
    
    def forward(self, ad_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ad_embeddings: Tensor of shape (batch_size, ad_dim)
        
        Returns:
            projected: Tensor of shape (batch_size, user_dim)
        """
        projected = self.projector(ad_embeddings)
        # L2 normalize to match user embedding space
        return F.normalize(projected, p=2, dim=1)


# ==================== Full Model ====================

class PrivAdsModel(nn.Module):
    """
    Complete pipeline: Ad text → Ad embedding → Projected embedding
    Computes similarity with user embeddings.
    """
    def __init__(
        self, 
        num_users: int,
        user_dim: int = 256,
        ad_model_name: str = 'all-MiniLM-L6-v2',
        projector_hidden_dim: int = None,
        freeze_ad_encoder: bool = True
    ):
        super().__init__()
        
        # Initialize components
        self.ad_encoder = AdEncoder(ad_model_name, freeze=freeze_ad_encoder)
        self.user_embeddings = UserEmbeddings(num_users, user_dim)
        self.projector = AdToUserProjector(
            ad_dim=self.ad_encoder.embedding_dim,
            user_dim=user_dim,
            hidden_dim=projector_hidden_dim
        )
    
    def encode_ads(self, ad_texts: List[str]) -> torch.Tensor:
        """Encode ads and project to user space."""
        ad_embs = self.ad_encoder(ad_texts)
        return self.projector(ad_embs)
    
    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings."""
        return self.user_embeddings(user_ids)
    
    def compute_similarity(
        self, 
        user_ids: torch.Tensor, 
        ad_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute similarity between users and ads.
        
        Returns:
            similarities: Tensor of shape (batch_size,), range [-1, 1]
        """
        ad_projected = self.encode_ads(ad_texts)
        user_embs = self.encode_users(user_ids)
        
        # Cosine similarity (since embeddings are normalized)
        similarities = (user_embs * ad_projected).sum(dim=1)
        return similarities


# ==================== Dataset ====================

class ClickDataset(Dataset):
    """
    Dataset for user-ad click interactions.
    Data format: (user_id, ad_id, ad_text, clicked)
    """
    def __init__(
        self, 
        user_ids: List[int],
        ad_ids: List[int],
        ad_texts: List[str],
        clicked: List[bool]
    ):
        assert len(user_ids) == len(ad_ids) == len(ad_texts) == len(clicked)
        
        self.user_ids = user_ids
        self.ad_ids = ad_ids
        self.ad_texts = ad_texts
        self.clicked = clicked
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'ad_id': self.ad_ids[idx],
            'ad_text': self.ad_texts[idx],
            'clicked': float(self.clicked[idx])
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        'user_ids': torch.tensor([item['user_id'] for item in batch]),
        'ad_ids': torch.tensor([item['ad_id'] for item in batch]),
        'ad_texts': [item['ad_text'] for item in batch],
        'clicked': torch.tensor([item['clicked'] for item in batch])
    }


# ==================== Loss Functions ====================

class ClickPredictionLoss(nn.Module):
    """
    Binary cross-entropy loss for click prediction.
    Aligns ad and user embeddings based on clicks.
    """
    def forward(self, similarities: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            similarities: Cosine similarity scores, range [-1, 1]
            labels: Binary labels (1 = clicked, 0 = not clicked)
        """
        # Convert similarity to probability
        logits = similarities * 10  # Scale for stability
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for user embeddings based on co-click behavior.
    Users who click the same ads should be close; different ads → far.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ):
        """
        Args:
            anchor, positive, negative: User embeddings, shape (batch_size, dim)
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines click prediction loss and user contrastive loss.
    """
    def __init__(self, click_weight: float = 1.0, triplet_weight: float = 0.5):
        super().__init__()
        self.click_loss = ClickPredictionLoss()
        self.triplet_loss = TripletLoss()
        self.click_weight = click_weight
        self.triplet_weight = triplet_weight
    
    def forward(
        self,
        similarities: torch.Tensor,
        click_labels: torch.Tensor,
        user_embeddings: torch.Tensor,
        positive_users: torch.Tensor = None,
        negative_users: torch.Tensor = None
    ):
        # Click prediction loss
        loss_click = self.click_loss(similarities, click_labels)
        
        # User triplet loss (if triplets provided)
        loss_triplet = 0
        if positive_users is not None and negative_users is not None:
            loss_triplet = self.triplet_loss(
                user_embeddings, 
                positive_users, 
                negative_users
            )
        
        total_loss = (
            self.click_weight * loss_click + 
            self.triplet_weight * loss_triplet
        )
        
        return total_loss, loss_click, loss_triplet


# ==================== Training Loop ====================

def train_epoch(
    model: PrivAdsModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: str = 'cpu'
):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    total_click_loss = 0
    total_triplet_loss = 0
    
    for batch in dataloader:
        user_ids = batch['user_ids'].to(device)
        ad_texts = batch['ad_texts']
        clicked = batch['clicked'].to(device)
        
        # Forward pass
        similarities = model.compute_similarity(user_ids, ad_texts)
        user_embs = model.encode_users(user_ids)
        
        # TODO: Sample positive/negative users for triplet loss
        # For now, just use click loss
        loss, click_loss, triplet_loss = criterion(
            similarities, clicked, user_embs
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_click_loss += click_loss.item()
        if isinstance(triplet_loss, torch.Tensor):
            total_triplet_loss += triplet_loss.item()
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'click_loss': total_click_loss / n_batches,
        'triplet_loss': total_triplet_loss / n_batches
    }


# ==================== Inference ====================

def predict_top_k_ads(
    model: PrivAdsModel,
    user_id: int,
    candidate_ads: List[Tuple[int, str]],  # List of (ad_id, ad_text)
    k: int = 10,
    device: str = 'cpu'
) -> List[Tuple[int, float]]:
    """
    Predict top-K ads for a given user.
    
    Returns:
        List of (ad_id, similarity_score) sorted by score descending
    """
    model.eval()
    
    with torch.no_grad():
        user_ids = torch.tensor([user_id] * len(candidate_ads)).to(device)
        ad_texts = [ad_text for _, ad_text in candidate_ads]
        
        similarities = model.compute_similarity(user_ids, ad_texts)
        similarities = similarities.cpu().numpy()
        
        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        
        results = [
            (candidate_ads[idx][0], similarities[idx])
            for idx in ranked_indices[:k]
        ]
    
    return results


# ==================== Example Usage ====================

if __name__ == '__main__':
    # Hyperparameters
    NUM_USERS = 10000
    USER_DIM = 256
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    
    # Sample data
    user_ids = [0, 1, 2, 0, 1]
    ad_ids = [100, 100, 101, 102, 103]
    ad_texts = [
        "for children, affordable, educational",
        "for children, affordable, educational",
        "luxury, adult, premium",
        "outdoor, camping, sustainable",
        "tech, gadgets, innovative"
    ]
    clicked = [True, True, False, True, False]
    
    # Create dataset
    dataset = ClickDataset(user_ids, ad_ids, ad_texts, clicked)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = PrivAdsModel(
        num_users=NUM_USERS,
        user_dim=USER_DIM,
        projector_hidden_dim=192
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CombinedLoss(click_weight=1.0, triplet_weight=0.5)
    
    # Train
    print("Training started...")
    for epoch in range(5):
        metrics = train_epoch(model, dataloader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Click Loss={metrics['click_loss']:.4f}")
    
    # Inference
    print("\nPredicting top ads for user 0...")
    candidate_ads = [
        (200, "for children, toys, fun"),
        (201, "luxury, expensive, premium"),
        (202, "outdoor, camping, adventure")
    ]
    
    top_ads = predict_top_k_ads(model, user_id=0, candidate_ads=candidate_ads, k=3)
    for ad_id, score in top_ads:
        print(f"  Ad {ad_id}: similarity={score:.4f}")
