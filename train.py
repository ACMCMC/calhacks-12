"""
Complete Training Script for PrivAds Model

This script orchestrates the full training pipeline:
1. Load and preprocess click data
2. Initialize model components
3. Train with combined loss (click prediction + user contrastive)
4. Evaluate and save model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from model import PrivAdsModel, CombinedLoss
from triplet_mining import (
    ClickDataProcessor, 
    AugmentedClickDataset,
    augmented_collate_fn,
    compute_click_statistics
)


class Trainer:
    """
    Trainer class for PrivAds model.
    """
    
    def __init__(
        self,
        model: PrivAdsModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: CombinedLoss,
        device: str = 'cpu',
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_click_loss': [],
            'train_triplet_loss': [],
            'val_loss': [],
            'val_click_loss': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_click_loss = 0
        total_triplet_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move to device
            user_ids = batch['user_ids'].to(self.device)
            ad_texts = batch['ad_texts']
            clicked = batch['clicked'].to(self.device)
            
            # Forward pass
            similarities = self.model.compute_similarity(user_ids, ad_texts)
            user_embs = self.model.encode_users(user_ids)
            
            # Get positive/negative users if available
            positive_users = None
            negative_users = None
            if 'positive_users' in batch:
                positive_ids = batch['positive_users'].to(self.device)
                negative_ids = batch['negative_users'].to(self.device)
                
                positive_users = self.model.encode_users(positive_ids)
                negative_users = self.model.encode_users(negative_ids)
            
            # Compute loss
            loss, click_loss, triplet_loss = self.criterion(
                similarities, clicked, user_embs,
                positive_users, negative_users
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_click_loss += click_loss.item()
            if isinstance(triplet_loss, torch.Tensor):
                total_triplet_loss += triplet_loss.item()
            
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'click': f'{click_loss.item():.4f}'
            })
        
        return {
            'loss': total_loss / n_batches,
            'click_loss': total_click_loss / n_batches,
            'triplet_loss': total_triplet_loss / n_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_click_loss = 0
        n_batches = 0
        
        # Metrics for evaluation
        all_similarities = []
        all_labels = []
        
        for batch in tqdm(self.val_dataloader, desc='Validation'):
            user_ids = batch['user_ids'].to(self.device)
            ad_texts = batch['ad_texts']
            clicked = batch['clicked'].to(self.device)
            
            # Forward pass
            similarities = self.model.compute_similarity(user_ids, ad_texts)
            
            # Click loss only (no triplet loss in validation)
            click_loss = self.criterion.click_loss(similarities, clicked)
            
            total_loss += click_loss.item()
            total_click_loss += click_loss.item()
            n_batches += 1
            
            # Store for metrics
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(clicked.cpu().numpy())
        
        # Compute additional metrics
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # Binary classification metrics
        predictions = (all_similarities > 0).astype(int)
        accuracy = (predictions == all_labels).mean()
        
        # AUC-like metric
        positive_sims = all_similarities[all_labels == 1]
        negative_sims = all_similarities[all_labels == 0]
        
        return {
            'loss': total_loss / n_batches,
            'click_loss': total_click_loss / n_batches,
            'accuracy': accuracy,
            'avg_positive_sim': positive_sims.mean() if len(positive_sims) > 0 else 0,
            'avg_negative_sim': negative_sims.mean() if len(negative_sims) > 0 else 0,
            'separation': positive_sims.mean() - negative_sims.mean() if len(positive_sims) > 0 and len(negative_sims) > 0 else 0
        }
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Click: {train_metrics['click_loss']:.4f}, "
                  f"Triplet: {train_metrics['triplet_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(Accuracy: {val_metrics['accuracy']:.4f})")
            print(f"  Similarity Separation: {val_metrics['separation']:.4f} "
                  f"(Pos: {val_metrics['avg_positive_sim']:.4f}, "
                  f"Neg: {val_metrics['avg_negative_sim']:.4f})")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_click_loss'].append(train_metrics['click_loss'])
            self.history['train_triplet_loss'].append(train_metrics['triplet_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_click_loss'].append(val_metrics['click_loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved!")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save training history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.save_dir}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def load_data(data_path: str) -> Tuple[List, List, List, List]:
    """
    Load click data from file.
    Expected format: CSV with columns [user_id, ad_id, ad_text, clicked]
    """
    import pandas as pd
    
    df = pd.read_csv(data_path)
    
    user_ids = df['user_id'].tolist()
    ad_ids = df['ad_id'].tolist()
    ad_texts = df['ad_text'].tolist()
    clicked = df['clicked'].astype(bool).tolist()
    
    return user_ids, ad_ids, ad_texts, clicked


def create_dataloaders(
    user_ids: List[int],
    ad_ids: List[int],
    ad_texts: List[str],
    clicked: List[bool],
    train_split: float = 0.8,
    batch_size: int = 64,
    use_triplets: bool = True
) -> Tuple[DataLoader, DataLoader, ClickDataProcessor]:
    """Create train and validation dataloaders."""
    
    # Split data
    n_train = int(len(user_ids) * train_split)
    
    train_user_ids = user_ids[:n_train]
    train_ad_ids = ad_ids[:n_train]
    train_ad_texts = ad_texts[:n_train]
    train_clicked = clicked[:n_train]
    
    val_user_ids = user_ids[n_train:]
    val_ad_ids = ad_ids[n_train:]
    val_ad_texts = ad_texts[n_train:]
    val_clicked = clicked[n_train:]
    
    # Create processor for triplet mining
    click_data = list(zip(train_user_ids, train_ad_ids, train_clicked))
    processor = ClickDataProcessor(click_data)
    
    # Compute statistics
    stats = compute_click_statistics(click_data)
    print("\nTraining Data Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Create datasets
    if use_triplets:
        train_dataset = AugmentedClickDataset(
            train_user_ids, train_ad_ids, train_ad_texts, train_clicked,
            processor, triplet_probability=0.5
        )
    else:
        from model import ClickDataset
        train_dataset = ClickDataset(
            train_user_ids, train_ad_ids, train_ad_texts, train_clicked
        )
    
    from model import ClickDataset
    val_dataset = ClickDataset(
        val_user_ids, val_ad_ids, val_ad_texts, val_clicked
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=augmented_collate_fn if use_triplets else None
    )
    
    from model import collate_fn
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader, processor


def main():
    parser = argparse.ArgumentParser(description='Train PrivAds Model')
    parser.add_argument('--data', type=str, required=True, help='Path to click data CSV')
    parser.add_argument('--num-users', type=int, required=True, help='Total number of users')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--user-dim', type=int, default=256, help='User embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--use-triplets', action='store_true', help='Use triplet loss')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PrivAds Model Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    user_ids, ad_ids, ad_texts, clicked = load_data(args.data)
    print(f"Loaded {len(user_ids):,} interactions")
    
    # Create dataloaders
    train_dataloader, val_dataloader, processor = create_dataloaders(
        user_ids, ad_ids, ad_texts, clicked,
        batch_size=args.batch_size,
        use_triplets=args.use_triplets
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = PrivAdsModel(
        num_users=args.num_users,
        user_dim=args.user_dim,
        projector_hidden_dim=192,
        freeze_ad_encoder=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss function
    criterion = CombinedLoss(
        click_weight=1.0,
        triplet_weight=0.5 if args.use_triplets else 0.0
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
