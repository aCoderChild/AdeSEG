from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DynamicMemoryState:
    """The memory tensor carried from frame t-1 to frame t."""

    tensor: torch.Tensor | None = None


class PromptMemoryAttention(nn.Module):
    """Read previous memory with feature-similarity attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        memory_strength: float = 0.25,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        del num_heads  # kept for API compatibility
        self.prompt_to_feature = nn.Linear(embed_dim, embed_dim)
        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.memory_strength = float(memory_strength)
        self.temperature = float(temperature)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        frame_features: torch.Tensor,
        previous_memory: torch.Tensor | None,
        sparse_prompts: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = frame_features.shape
        query = frame_features.flatten(2).transpose(1, 2)

        if previous_memory is None or self.memory_strength <= 0:
            return frame_features

        memory = _match_spatial(previous_memory, (h, w))
        memory_tokens = memory.flatten(2).transpose(1, 2)
        read_memory = self._read_memory(query, memory_tokens)
        read_memory_map = read_memory.transpose(1, 2).reshape(b, c, h, w)
        adapter_input = torch.cat(
            [frame_features, read_memory_map, frame_features - read_memory_map],
            dim=1,
        )
        candidate_map = frame_features + self.memory_strength * self.adapter(adapter_input)
        candidate = candidate_map.flatten(2).transpose(1, 2)

        if sparse_prompts.numel() > 0:
            prompt_context = self.prompt_to_feature(sparse_prompts).mean(dim=1, keepdim=True)
            candidate = candidate + 0.01 * prompt_context

        return candidate.transpose(1, 2).reshape(b, c, h, w)

    def _read_memory(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query_norm = F.normalize(query.float(), dim=-1)
        memory_norm = F.normalize(memory.float(), dim=-1)
        affinity = torch.matmul(query_norm, memory_norm.transpose(1, 2))
        affinity = torch.softmax(affinity * self.temperature, dim=-1)
        read = torch.matmul(affinity.to(memory.dtype), memory)
        return read.to(query.dtype)


class ReliabilityGatedMemoryUpdate(nn.Module):
    """M_t = r_t * Z_t + (1 - r_t) * M_{t-1}."""

    def forward(
        self,
        previous_memory: torch.Tensor | None,
        candidate_memory: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if previous_memory is None:
            return candidate_memory

        previous_memory = _match_spatial(previous_memory, candidate_memory.shape[-2:])
        reliability = _as_gate(reliability, candidate_memory)
        return reliability * candidate_memory + (1.0 - reliability) * previous_memory


def _match_spatial(memory: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if memory.shape[-2:] == size:
        return memory
    return F.interpolate(memory, size=size, mode="bilinear", align_corners=False)


def _as_gate(reliability: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while reliability.dim() < like.dim():
        reliability = reliability.unsqueeze(-1)
    return reliability.to(dtype=like.dtype, device=like.device)
