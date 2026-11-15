"""Add continuous role embeddings to all spatial memory values."""

import torch
from torch import nn


class SoftRegionValueAdapter(nn.Module):
    def __init__(self, heads=1, head_dim=256, max_relative=0.1):
        super().__init__()
        self.role_embeddings = nn.Parameter(torch.zeros(heads, 3, head_dim))
        self.max_relative = float(max_relative)

    def forward(self, native, query, key, value, memberships, pointer_tokens,
                mode="regional_value", *, sdpa=None):
        if mode != "regional_value":
            raise ValueError("Expected regional_value mode")
        if value.ndim != 4 or key.shape != value.shape or query.ndim != 4:
            raise ValueError("Expected [batch, heads, tokens, head_dim] attention tensors")
        batch, heads, tokens, channels = value.shape
        spatial = tokens - pointer_tokens
        if not 0 < spatial <= tokens or memberships.shape != (batch, spatial, 3):
            raise ValueError("Soft memberships must align with spatial keys only")
        if memberships.device != value.device or not torch.isfinite(memberships).all():
            raise ValueError("Soft memberships must be finite and on the attention device")
        if (memberships < 0).any() or (memberships > 1).any():
            raise ValueError("Soft memberships must lie in [0, 1]")
        if not torch.allclose(memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5):
            raise ValueError("Soft role memberships must sum to one")
        if self.role_embeddings.shape != (heads, 3, channels):
            raise ValueError("Adapter dimensions do not match attention values")
        if sdpa is None:
            raise ValueError("A scaled dot-product attention function is required")

        spatial_values = value[..., :spatial, :]
        rms = spatial_values.float().square().mean(dim=-2, keepdim=True).sqrt().to(value.dtype)
        centered = self.role_embeddings - self.role_embeddings.mean(dim=1, keepdim=True)
        role_vectors = centered.tanh()
        role_vectors = role_vectors - role_vectors.mean(dim=1, keepdim=True)
        role_vectors = self.max_relative * role_vectors
        delta = torch.einsum("hrd,bnr->bhnd", role_vectors, memberships).to(value.dtype)
        adapted_spatial = spatial_values + delta * rms
        adapted_values = torch.cat((adapted_spatial, value[..., spatial:, :]), dim=-2)
        output = sdpa(query, key, adapted_values, dropout_p=0.0)
        valid = torch.ones((batch, 3), dtype=torch.bool, device=value.device)
        return output, valid


class SoftRegionValueHeads(nn.Module):
    def __init__(self, layers=4, heads=1, head_dim=256):
        super().__init__()
        self.layers = nn.ModuleList(
            SoftRegionValueAdapter(heads, head_dim) for _ in range(layers)
        )
