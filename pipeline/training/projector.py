"""Projector: maps ad embedding space to user embedding space."""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """MLP projector from ad space (d_ad) to user space (d_user)."""
    def __init__(self, d_ad: int, d_user: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_ad, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_user),
            nn.LayerNorm(d_user)
        )

    def forward(self, z_ad: torch.Tensor) -> torch.Tensor:
        p = self.net(z_ad)
        return p / p.norm(dim=-1, keepdim=True)


if __name__ == "__main__":
    proj = Projector(d_ad=768, d_user=128)
    z = torch.randn(4, 768)
    p = proj(z)
    print(f"Input: {z.shape}, Output: {p.shape}")
    print(f"Output norms: {p.norm(dim=-1)}")
