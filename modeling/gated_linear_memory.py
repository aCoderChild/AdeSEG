"""Gated linear key-value state used by the LiVOS-style research prototype.

This module implements only the recurrent matching state. Key, value, and gate
projections belong to the surrounding segmentation model and must be trained
with its decoder; this state is intentionally not wired into MedSAM2 yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _normalized_keys(keys: torch.Tensor) -> torch.Tensor:
    """Apply LiVOS's channel-wise softmax key normalization."""
    if keys.ndim != 4:
        raise ValueError("keys must have shape [B, K, H, W].")
    return (keys - keys.amax(dim=1, keepdim=True)).softmax(dim=1)


@dataclass
class GatedLinearKVState:
    """Constant-in-time linear-memory state with LiVOS-style normalization.

    ``state`` has shape ``[B, K, V]`` and ``key_sum`` has shape
    ``[B, K, H, W]``. The latter is required by LiVOS's readout normalizer.
    Both have fixed size for a fixed feature-map resolution, regardless of
    video length.
    """

    state: torch.Tensor
    key_sum: torch.Tensor

    @classmethod
    def initialize(cls, keys: torch.Tensor, values: torch.Tensor) -> "GatedLinearKVState":
        """Initialize the state from one key/value feature map."""
        cls._validate_pair(keys, values)
        keys = _normalized_keys(keys)
        return cls(
            state=torch.einsum("bkhw,bvhw->bkv", keys, values),
            key_sum=keys,
        )

    @staticmethod
    def _validate_pair(keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have shapes [B, C, H, W].")
        if keys.shape[0] != values.shape[0] or keys.shape[-2:] != values.shape[-2:]:
            raise ValueError("keys and values must have matching batch and spatial dimensions.")

    def read(self, query_keys: torch.Tensor) -> torch.Tensor:
        """Read a value map for the current query before writing that frame."""
        query_keys = _normalized_keys(query_keys)
        if query_keys.shape != self.key_sum.shape:
            raise ValueError("query keys must match the stored key-sum shape.")
        if query_keys.shape[0] != self.state.shape[0] or query_keys.shape[1] != self.state.shape[1]:
            raise ValueError("query key batch/key dimensions must match the state.")
        readout = torch.einsum("bkhw,bkv->bvhw", query_keys, self.state)
        normalizer = torch.einsum("bkhw,bkhw->b", query_keys, self.key_sum)
        normalizer = normalizer.clamp_min(torch.finfo(readout.dtype).eps)
        return readout / normalizer[:, None, None, None]

    def write(self, keys: torch.Tensor, values: torch.Tensor, gate: torch.Tensor) -> None:
        """Apply LiVOS's value-channel gate and add this frame's association."""
        self._validate_pair(keys, values)
        keys = _normalized_keys(keys)
        if keys.shape != self.key_sum.shape:
            raise ValueError("new keys must match the stored key-sum shape.")
        if values.shape[:1] != self.state.shape[:1] or values.shape[1] != self.state.shape[2]:
            raise ValueError("value batch/channel dimensions must match the state.")
        if gate.shape != (self.state.shape[0], self.state.shape[2]):
            raise ValueError("gate must have shape [B, V].")
        self.state = self.state * gate[:, None, :].to(self.state)
        self.state = self.state + torch.einsum("bkhw,bvhw->bkv", keys, values)
        self.key_sum = self.key_sum + keys
