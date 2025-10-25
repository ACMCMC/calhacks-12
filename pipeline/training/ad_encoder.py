"""Ad encoder using Jina CLIP v2 for unified text+image embeddings."""

from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
from typing import Optional, Union
import numpy as np


class AdEncoder:
    """Encode ad text and optional image into single unified embedding."""
    
    def __init__(self, model_name: str = "jinaai/jina-clip-v2", device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}...")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy = self.processor(text=["test"], return_tensors="pt").to(self.device)
            self.d_ad = self.model.get_text_features(**dummy).shape[-1]
        
        print(f"✓ Loaded {model_name}, embedding dim: {self.d_ad}")
    
    @torch.no_grad()
    def encode(
        self, 
        text: str, 
        image: Optional[Union[Image.Image, str]] = None
    ) -> np.ndarray:
        """
        Encode ad into single unified embedding.
        
        Args:
            text: Ad description text
            image: PIL Image or path to image file (optional)
        
        Returns:
            z_ad: Normalized embedding vector (d_ad,)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if image is not None:
            # Multimodal: text + image
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get unified embedding - use image features for multimodal
            embedding = self.model.get_image_features(**inputs)
        else:
            # Text only
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            embedding = self.model.get_text_features(**inputs)
        
        # L2 normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        # Convert to float32 for numpy compatibility
        return embedding.cpu().float().numpy()[0]
    
    def encode_batch(
        self,
        texts: list[str],
        images: Optional[list[Optional[Union[Image.Image, str]]]] = None
    ) -> np.ndarray:
        """
        Encode batch of ads.
        
        Args:
            texts: List of ad descriptions
            images: List of images (None for text-only ads)
        
        Returns:
            embeddings: (batch_size, d_ad)
        """
        if images is None:
            images = [None] * len(texts)
        
        embeddings = []
        for text, image in zip(texts, images):
            emb = self.encode(text=text, image=image)
            embeddings.append(emb)
        
        return np.stack(embeddings)


if __name__ == "__main__":
    # Quick test
    encoder = AdEncoder()
    
    # Test text-only
    z = encoder.encode("Sustainable fashion for children, eco-friendly materials")
    print(f"Text-only embedding shape: {z.shape}")
    print(f"Norm: {np.linalg.norm(z):.4f}")
    
    # Test batch
    texts = [
        "Luxury watch, expensive, premium",
        "Kids toys, affordable, colorful"
    ]
    embeddings = encoder.encode_batch(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
