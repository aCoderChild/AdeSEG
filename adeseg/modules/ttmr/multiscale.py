"""Multiscale *scores* for native memory tokens (no additional stored tokens).

This is an offline/shadow-mode scoring primitive. It does not change the
Lean-SAM2 memory bank, RoPE, token ordering, or temporal propagation.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_multiscale_memory(
    memory_maps: torch.Tensor,
    anchor_tokens: torch.Tensor,
    *,
    scales: Sequence[int] = (2, 4),
    context_weight: float = 0.5,
) -> torch.Tensor:
    """Score [F,C,H,W] memory at original locations using pooled context.

    Return a flat [F*H*W] score vector, ordered by source frame then row-major
    spatial position. The corresponding token vector order is exactly::

        memory_maps.permute(0, 2, 3, 1).reshape(-1, C)

    Each pooled feature is used only for scoring its original-resolution
    locations. Pooling never appends tokens to temporal memory.
    """
    if (
        not isinstance(memory_maps, torch.Tensor)
        or memory_maps.ndim != 4
        or memory_maps.shape[0] < 1
        or memory_maps.shape[1] < 2
        or not torch.is_floating_point(memory_maps)
        or not torch.isfinite(memory_maps).all()
    ):
        raise ValueError("Expected finite floating memory maps [F,C,H,W]")
    frames, channels, height, width = memory_maps.shape
    if (
        not isinstance(anchor_tokens, torch.Tensor)
        or anchor_tokens.ndim != 2
        or anchor_tokens.shape[0] < 1
        or anchor_tokens.shape[1] != channels
        or not torch.is_floating_point(anchor_tokens)
        or not torch.isfinite(anchor_tokens).all()
    ):
        raise ValueError("Expected finite floating anchors [A,C] matching memory channels")
    if not isinstance(context_weight, (float, int)) or not 0 <= context_weight <= 1:
        raise ValueError("context_weight must be in [0,1]")
    if not isinstance(scales, (tuple, list)) or not scales or any(
        type(s) is not int or s < 2 or height % s or width % s for s in scales
    ):
        raise ValueError("scales must be integer pooling factors >=2 dividing H and W")

    features = memory_maps.detach().float()
    anchors = F.normalize(
        anchor_tokens.detach().to(device=features.device, dtype=torch.float32),
        dim=-1, eps=1e-6,
    )
    local = F.normalize(
        features.permute(0, 2, 3, 1).reshape(-1, channels), dim=-1, eps=1e-6
    ) @ anchors.T
    local = local.amax(dim=-1)
    if context_weight == 0:
        return local

    pooled_scores = []
    for scale in scales:
        coarse = F.avg_pool2d(features, kernel_size=scale, stride=scale)
        tokens = coarse.permute(0, 2, 3, 1).reshape(-1, channels)
        cosine = (F.normalize(tokens, dim=-1, eps=1e-6) @ anchors.T).amax(dim=-1)
        score_map = cosine.reshape(frames, 1, height // scale, width // scale)
        pooled_scores.append(
            F.interpolate(
                score_map, size=(height, width), mode="bilinear", align_corners=False
            ).reshape(-1)
        )
    context = torch.stack(pooled_scores).mean(dim=0)
    return ((1.0 - context_weight) * local + context_weight * context).clamp(-1, 1)


__all__ = ["score_multiscale_memory"]
