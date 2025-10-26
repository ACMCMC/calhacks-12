"""
Jina CLIP v2 Encoder for PrivAds
Handles encoding of ad creatives (text + optional images).
"""

import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from typing import Optional, Union
from pathlib import Path
import PIL.Image

class AdEncoder:
    """Jina CLIP v2 wrapper for encoding ad creatives."""

    def __init__(self, device: str = "cuda"):
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

        # Jina CLIP v2 outputs 1024D embeddings
        self.d_ad = 1024
        print(f"✓ Loaded jinaai/jina-clip-v2, embedding dim: {self.d_ad}")

    @torch.no_grad()
    def encode(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        image: Optional[PIL.Image.Image] = None
    ) -> np.ndarray:
        """
        Encode ad creative to embedding vector.

        Args:
            text: Ad text content
            image_path: Path to image file
            image: PIL Image object

        Returns:
            L2-normalized embedding vector
        """
        inputs = {}

        # Process text
        if text:
            inputs.update(self.processor(text=[text], return_tensors="pt"))

        # Process image
        if image_path:
            image = PIL.Image.open(image_path).convert("RGB")
        if image:
            inputs.update(self.processor(images=[image], return_tensors="pt"))

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        # Extract embeddings
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
        # L2 normalize
        embedding = embedding / embedding.norm()
        return embedding

    def encode_batch(self, texts: list, image_paths: Optional[list] = None) -> np.ndarray:
        """
        Encode multiple ad creatives in batch.

        Args:
            texts: List of ad texts
            image_paths: Optional list of image paths

        Returns:
            Array of normalized embeddings (n_ads, d_ad)
        """
        embeddings = []

        for i, text in enumerate(texts):
            image_path = image_paths[i] if image_paths and i < len(image_paths) else None
            emb = self.encode(text=text, image_path=image_path)
            embeddings.append(emb)

        return np.array(embeddings)

    def __call__(self, text=None, image_path=None, image=None):
        """Convenience method for encoding."""
        return self.encode(text=text, image_path=image_path, image=image)