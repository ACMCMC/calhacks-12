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
        print(f"Loading jinaai/jina-clip-v2 on {self.device}...")
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2",
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "jinaai/jina-clip-v2",
            trust_remote_code=True
        )
        self.model.eval()
        self.d_ad = 1024
        print(f"✓ Loaded jinaai/jina-clip-v2, embedding dim: {self.d_ad}")
    
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
        inputs = {}
        if text:
            inputs.update(self.processor(text=[text], return_tensors="pt"))
        img = None
        if image is not None:
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
            else:
                img = image
        if img is not None:
            inputs.update(self.processor(images=[img], return_tensors="pt"))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        image_emb = outputs["image_embeds"][0].to(torch.float32).cpu() if "image_embeds" in outputs else None
        text_emb = outputs["text_embeds"][0].to(torch.float32).cpu() if "text_embeds" in outputs else None
        if image_emb is not None and text_emb is not None:
            embedding = torch.cat([image_emb, text_emb])
        elif image_emb is not None:
            embedding = image_emb
        elif text_emb is not None:
            embedding = text_emb
        else:
            raise ValueError("No image or text embedding produced by model.")
        embedding = embedding / embedding.norm()
        return embedding.numpy()
    
    def encode_batch(
        self,
        texts: list,
        images: Optional[list] = None
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
