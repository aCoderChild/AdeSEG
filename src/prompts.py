from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class PromptBatch:
    """Prompt inputs accepted by SAM-style prompt encoders."""

    point_coords: torch.Tensor | None = None
    point_labels: torch.Tensor | None = None
    boxes: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    concept_tokens: torch.Tensor | None = None
    concept_text: list[str] | None = None


class ConceptPromptEncoder(nn.Module):
    """
    Tiny conceptual prompt adapter.

    If a MedSAM3 text encoder/tokenizer is available, pass it as ``text_encoder``.
    Otherwise this class uses a small learnable embedding table with deterministic
    hashing so string concepts can still become prompt tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        text_encoder: nn.Module | None = None,
        text_tokenizer=None,
        text_feature_dim: int | None = None,
        vocab_size: int = 4096,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.hash_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.vocab_size = vocab_size

        if text_encoder is None:
            self.proj = nn.Identity()
        else:
            text_feature_dim = text_feature_dim or embed_dim
            self.proj = nn.Linear(text_feature_dim, embed_dim)

    def forward(
        self,
        concept_text: Iterable[str] | None = None,
        concept_tokens: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        if concept_tokens is not None:
            return self.proj(concept_tokens)

        if concept_text is None:
            return None

        concepts = list(concept_text)
        if len(concepts) == 0:
            return None

        if self.text_encoder is not None:
            if self.text_tokenizer is None:
                raise ValueError("A text_tokenizer is required with text_encoder.")
            tokens = self.text_tokenizer(concepts).to(device)
            encoded = self.text_encoder(tokens)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            if encoded.dim() == 2:
                encoded = encoded[:, None, :]
            return self.proj(encoded)

        ids = [_stable_token_id(text, self.vocab_size) for text in concepts]
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.hash_embeddings(token_ids)[None, :, :]


def build_sam_prompt_embeddings(
    prompt_encoder: nn.Module,
    prompts: PromptBatch | None,
    batch_size: int,
    device: torch.device,
    concept_encoder: ConceptPromptEncoder | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse and dense prompt embeddings for a SAM/MedSAM prompt encoder."""

    prompts = prompts or PromptBatch()
    points = None
    if prompts.point_coords is not None and prompts.point_labels is not None:
        points = (prompts.point_coords.to(device), prompts.point_labels.to(device))

    sparse, dense = prompt_encoder(
        points=points,
        boxes=None if prompts.boxes is None else prompts.boxes.to(device),
        masks=None if prompts.mask is None else prompts.mask.to(device),
    )
    if sparse.size(0) == 1 and batch_size > 1:
        sparse = sparse.expand(batch_size, -1, -1)
        dense = dense.expand(batch_size, -1, -1, -1)

    concept_sparse = None
    if concept_encoder is not None:
        concept_sparse = concept_encoder(
            concept_text=prompts.concept_text,
            concept_tokens=prompts.concept_tokens,
            device=device,
        )

    if concept_sparse is not None:
        if concept_sparse.size(0) == 1 and batch_size > 1:
            concept_sparse = concept_sparse.expand(batch_size, -1, -1)
        sparse = torch.cat([sparse, concept_sparse.to(device)], dim=1)

    return sparse, dense


def _stable_token_id(text: str, vocab_size: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % vocab_size
