"""One learned temporal state for heuristic-free MedSAM2 prompting.

The adapter receives only frozen values from the current frame:

* decoder mask logits,
* image-encoder features, and
* the decoder object pointer.

It combines them with one recurrent state tensor.  That state is the only
cross-frame information owned by this module; it is a lossy compression of all
earlier masks, image features, and object pointers.  The module returns a
dense prompt in the normal SAM2 mask-prompt format and never computes a
reliability score or threshold.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ImplicitTemporalState(nn.Module):
    """Fuse frozen per-frame evidence into one causal dense-prompt state."""

    def __init__(
        self,
        hidden_channels: int = 32,
        image_feature_channels: int = 256,
        object_pointer_dim: int = 256,
        maximum_prompt_logit: float = 8.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 8 or hidden_channels % 8:
            raise ValueError("hidden_channels must be at least 8 and divisible by 8.")
        if image_feature_channels < 1 or object_pointer_dim < 1:
            raise ValueError("image_feature_channels and object_pointer_dim must be positive.")
        if maximum_prompt_logit <= 0:
            raise ValueError("maximum_prompt_logit must be positive.")

        self.hidden_channels = hidden_channels
        self.image_feature_channels = image_feature_channels
        self.object_pointer_dim = object_pointer_dim
        self.maximum_prompt_logit = float(maximum_prompt_logit)

        # The mask branch sees the current proposal and the prior prompt
        # reconstructed from the single recurrent state.
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_feature_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        # FiLM lets the frozen object token modulate spatial evidence without
        # inventing an explicit confidence scalar.
        self.pointer_film = nn.Linear(object_pointer_dim, 2 * hidden_channels)

        # A convolutional GRU update retains one state map across the video.
        self.gates = nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.prompt_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward_step(
        self,
        mask_logits: torch.Tensor,
        image_features: torch.Tensor,
        object_pointer: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(dense_prompt_logits, next_state)`` for one frame.

        Args:
            mask_logits: Frozen decoder proposal, shape ``[B, 1, H, W]``.
            image_features: Frozen image feature, shape ``[B, C, h, w]``.
            object_pointer: Frozen decoder token, shape ``[B, D]``.
            state: The single prior state, shape ``[B, hidden_channels, H, W]``.
        """
        self._validate_inputs(mask_logits, image_features, object_pointer, state)
        previous_prompt = (
            torch.zeros_like(mask_logits)
            if state is None
            else self._prompt_from_state(state)
        )
        mask_evidence = self.mask_encoder(
            torch.cat((mask_logits, mask_logits.sigmoid(), previous_prompt), dim=1)
        )
        image_evidence = self.image_encoder(
            F.interpolate(
                image_features,
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        scale, bias = self.pointer_film(object_pointer).chunk(2, dim=1)
        evidence = (mask_evidence + image_evidence) * (1.0 + scale.tanh()[..., None, None])
        evidence = evidence + bias[..., None, None]

        if state is None:
            next_state = evidence
        else:
            reset, update = self.gates(torch.cat((evidence, state), dim=1)).chunk(2, dim=1)
            candidate = self.candidate(torch.cat((evidence, reset.sigmoid() * state), dim=1)).tanh()
            next_state = (1.0 - update.sigmoid()) * state + update.sigmoid() * candidate
        return self._prompt_from_state(next_state), next_state

    def _prompt_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.maximum_prompt_logit * self.prompt_head(state).tanh()

    def _validate_inputs(
        self,
        mask_logits: torch.Tensor,
        image_features: torch.Tensor,
        object_pointer: torch.Tensor,
        state: torch.Tensor | None,
    ) -> None:
        batch_size, _, height, width = mask_logits.shape if mask_logits.ndim == 4 else (0, 0, 0, 0)
        if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
            raise ValueError("mask_logits must have shape [batch, 1, height, width].")
        if image_features.ndim != 4 or image_features.shape[:2] != (batch_size, self.image_feature_channels):
            raise ValueError("image_features has an incompatible shape.")
        if object_pointer.shape != (batch_size, self.object_pointer_dim):
            raise ValueError("object_pointer has an incompatible shape.")
        if state is not None and state.shape != (batch_size, self.hidden_channels, height, width):
            raise ValueError("state has an incompatible shape.")
