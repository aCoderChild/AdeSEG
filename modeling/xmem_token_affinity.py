"""XMem-inspired continuous per-token affinity for frozen SAM2 memory reads.

The anisotropic-L2 score is adapted from official XMem code (ECCV 2022, commit
f3b841d50df058910bbf690229ddc15fb1aef7d6). Projection modules here are a local
SAM2-specific adaptation, not imported XMem code: SAM2 supplies 256-channel
queries and 64-channel spatial memories, so the two sides have separate
shape-specific projections into one learned key space. SAM2's pretrained
attention stays intact; the adapter adds a zero-initialized score to spatial
memory logits and never prunes tokens or changes values, pointers, or positions.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_SAM2_ROOT = Path(__file__).resolve().parents[1] / "external" / "sam2"
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))
from sam2.modeling.sam.transformer import RoPEAttention, apply_rotary_enc


class _XMemQueryProjection(nn.Module):
    """SAM2-specific query-side projection with XMem-style key/selection heads."""

    def __init__(self, in_dim: int, key_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        self.selection_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.key_proj(x), torch.sigmoid(self.selection_proj(x))


class _XMemMemoryProjection(nn.Module):
    """SAM2-specific memory-side projection with XMem-style key/shrinkage heads."""

    def __init__(self, in_dim: int, key_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        self.shrinkage_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.key_proj(x), self.shrinkage_proj(x).square() + 1.0


def xmem_anisotropic_l2(
    memory_key: Tensor,
    shrinkage: Tensor,
    query_key: Tensor,
    query_selection: Tensor,
) -> Tensor:
    """Official XMem anisotropic-L2 similarity, returned as [B,Nq,Nm].

    This is algebraically the upstream ``get_similarity`` implementation.
    It uses dense, continuous scores; there is no top-k or hard selection.
    """
    if memory_key.ndim != 3 or query_key.ndim != 3 or query_selection.shape != query_key.shape:
        raise ValueError("Expected flattened [batch, channels, tokens] key/query tensors")
    batch, channels, _ = memory_key.shape
    if query_key.shape[:2] != (batch, channels) or shrinkage.shape != (batch, memory_key.shape[-1], 1):
        raise ValueError("XMem key, shrinkage, and query dimensions do not align")

    mk = memory_key.transpose(1, 2)
    a_sq = mk.square() @ query_selection
    two_ab = 2.0 * (mk @ (query_key * query_selection))
    b_sq = (query_selection * query_key.square()).sum(1, keepdim=True)
    similarity = (-a_sq + two_ab - b_sq) * shrinkage / math.sqrt(channels)
    return similarity.transpose(1, 2)


class XMemTokenAffinity:
    """Build additive per-query/per-memory-token scores from SAM2 features."""

    def __init__(
        self,
        query_projection: _XMemQueryProjection,
        memory_projection: _XMemMemoryProjection,
        key_dim: int,
        granularity: Literal["token", "frame"],
    ) -> None:
        self.query_projection = query_projection
        self.memory_projection = memory_projection
        self.key_dim = key_dim
        self.granularity = granularity

    def project(self, query: Tensor, memory: Tensor, pointer_tokens: int):
        """Project maps once; score query rows in chunks to bound peak memory."""
        if query.ndim != 3 or memory.ndim != 3 or query.shape[0] != memory.shape[0]:
            raise ValueError("Expected batch-first query and memory sequences")
        batch, query_tokens, query_channels = query.shape
        memory_tokens, memory_channels = memory.shape[1:]
        if pointer_tokens < 0 or pointer_tokens > memory_tokens:
            raise ValueError("Invalid object-pointer tail length")
        spatial_count = memory_tokens - pointer_tokens
        side = math.isqrt(query_tokens)
        if side * side != query_tokens or spatial_count % query_tokens:
            raise ValueError("Memory must contain whole frames on the query spatial grid")
        frame_count = spatial_count // query_tokens
        if frame_count < 1:
            raise ValueError("At least one spatial memory frame is required")

        q_map = query.transpose(1, 2).reshape(batch, query_channels, side, side)
        q_map = q_map.to(dtype=self.query_projection.key_proj.weight.dtype)
        q_key, q_selection = self.query_projection(q_map)

        spatial_memory = memory[:, :spatial_count]
        mem_map = spatial_memory.reshape(batch, frame_count, query_tokens, memory_channels)
        mem_map = mem_map.permute(0, 1, 3, 2).reshape(
            batch * frame_count, memory_channels, side, side
        )
        mem_map = mem_map.to(dtype=self.memory_projection.key_proj.weight.dtype)
        m_key, m_shrinkage = self.memory_projection(mem_map)
        m_key = m_key.reshape(batch, frame_count, self.key_dim, query_tokens)
        m_shrinkage = m_shrinkage.reshape(batch, frame_count, 1, query_tokens)

        if self.granularity not in ("token", "frame"):
            raise ValueError(f"Unknown affinity granularity: {self.granularity}")

        m_key = m_key.permute(0, 2, 1, 3).flatten(2)
        m_shrinkage = m_shrinkage.permute(0, 2, 1, 3).flatten(2).transpose(1, 2)
        q_key = q_key.flatten(2)
        q_selection = q_selection.flatten(2)
        return (
            q_key,
            q_selection,
            m_key,
            m_shrinkage,
            pointer_tokens,
            memory_tokens,
            frame_count,
            query_tokens,
            self.granularity,
        )

    @staticmethod
    def score(projected, start: int = 0, stop: int | None = None) -> Tensor:
        """Score a query slice; return [B,Nq_chunk,Nmemory] including pointers."""
        (q_key, q_selection, m_key, m_shrinkage, pointer_tokens, memory_tokens,
         frame_count, tokens_per_frame, granularity) = projected
        stop = q_key.shape[-1] if stop is None else stop
        if not (0 <= start < stop <= q_key.shape[-1]):
            raise ValueError("Invalid query score chunk")
        spatial_score = xmem_anisotropic_l2(
            m_key, m_shrinkage,
            q_key[:, :, start:stop], q_selection[:, :, start:stop],
        )
        if granularity == "frame":
            # Capacity-matched frame control: keep the same learned XMem heads,
            # but remove within-frame score variation. Each query gets one
            # continuous affinity per source frame, broadcast across its spatial
            # keys. Unlike shrinkage-only broadcasting, this truly tests whether
            # token-local differences add value beyond frame-level weighting.
            batch, query_count, spatial_count = spatial_score.shape
            spatial_score = spatial_score.reshape(
                batch, query_count, frame_count, tokens_per_frame
            )
            spatial_score = spatial_score.mean(dim=-1, keepdim=True).expand_as(spatial_score)
            spatial_score = spatial_score.reshape(batch, query_count, spatial_count)
        score = spatial_score[:, None]
        if pointer_tokens:
            # Pointer tokens keep native logits exactly: no added affinity.
            score = F.pad(score, (0, pointer_tokens), value=0.0)
        if score.shape[-1] != memory_tokens:
            raise RuntimeError("Affinity key count differs from the native memory layout")
        return score[:, 0]

    def __call__(self, query: Tensor, memory: Tensor, pointer_tokens: int) -> Tensor:
        """Return the full dense score for diagnostics and small unit fixtures."""
        projected = self.project(query, memory, pointer_tokens)
        return self.score(projected).unsqueeze(1)


def _make_affinity_attention(base_attention: nn.Module, *, key_dim: int, granularity: str) -> nn.Module:
    if not isinstance(base_attention, RoPEAttention):
        raise TypeError("Only SAM2 RoPEAttention memory cross-attention is supported")
    adapter = XMemTokenAffinityRoPEAttention(
        embedding_dim=base_attention.embedding_dim,
        num_heads=base_attention.num_heads,
        downsample_rate=base_attention.embedding_dim // base_attention.internal_dim,
        dropout=base_attention.dropout_p,
        kv_in_dim=base_attention.kv_in_dim,
        key_dim=key_dim,
        granularity=granularity,
    )
    missing, unexpected = adapter.load_state_dict(base_attention.state_dict(), strict=False)
    if unexpected or set(missing) != {"xmem_query_projection.key_proj.weight",
                                     "xmem_query_projection.key_proj.bias",
                                     "xmem_query_projection.selection_proj.weight",
                                     "xmem_query_projection.selection_proj.bias",
                                     "xmem_memory_projection.key_proj.weight",
                                     "xmem_memory_projection.key_proj.bias",
                                     "xmem_memory_projection.shrinkage_proj.weight",
                                     "xmem_memory_projection.shrinkage_proj.bias",
                                     "xmem_strength"}:
        raise RuntimeError(f"SAM2 attention state mismatch: missing={missing}, unexpected={unexpected}")
    adapter.compute_cis = base_attention.compute_cis
    adapter.freqs_cis = base_attention.freqs_cis.detach().clone()
    adapter.rope_k_repeat = base_attention.rope_k_repeat
    adapter.train(base_attention.training)
    first = next(base_attention.parameters())
    return adapter.to(device=first.device, dtype=first.dtype)


class XMemTokenAffinityRoPEAttention(RoPEAttention):
    """SAM2 RoPE attention plus XMem-style dense token affinity.

    The base attention weights are loaded without alteration. At zero strength
    in eval mode, the exact native SDPA call is used. Training keeps the dense
    zero-initialized bias path so ``xmem_strength`` can learn from segmentation
    loss. Spatial positions are still handled by SAM2's unchanged RoPE; the
    adapter receives the unpositioned feature tensors separately.
    """

    uses_xmem_token_affinity = True

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        *,
        key_dim: int = 64,
        granularity: Literal["token", "frame"] = "token",
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=downsample_rate,
            dropout=dropout,
            kv_in_dim=kv_in_dim,
            rope_theta=rope_theta,
            rope_k_repeat=rope_k_repeat,
            feat_sizes=feat_sizes,
        )

        self.xmem_query_projection = _XMemQueryProjection(embedding_dim, key_dim)
        self.xmem_memory_projection = _XMemMemoryProjection(self.kv_in_dim, key_dim)
        self.xmem_strength = nn.Parameter(torch.zeros(()))
        self.xmem_key_dim = key_dim
        self.xmem_granularity = granularity
        self.affinity_gradient_enabled = False

    def _affinity_module(self) -> XMemTokenAffinity:
        return XMemTokenAffinity(
            self.xmem_query_projection,
            self.xmem_memory_projection,
            self.xmem_key_dim,
            self.xmem_granularity,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: int = 0,
        utility_q: Tensor | None = None,
        utility_memory: Tensor | None = None,
    ) -> Tensor:
        if (utility_q is None) != (utility_memory is None):
            raise ValueError("Both unpositioned utility tensors must be supplied together")
        # SAM2's MemoryAttentionLayer passes q=tgt2 (no cross-query PE in the
        # supported configs) and v=memory (unpositioned) to RoPEAttention.
        # These are exactly the clean feature tensors needed by the adapter.
        if utility_q is None:
            utility_q, utility_memory = q, v
        if utility_q is not None and utility_q.shape[:2] != q.shape[:2]:
            raise ValueError("Utility query does not align with the SAM2 query tokens")
        if utility_memory is not None and utility_memory.shape[:2] != k.shape[:2]:
            raise ValueError("Utility memory does not align with the SAM2 memory tokens")
        if num_k_exclude_rope < 0 or num_k_exclude_rope > k.shape[1]:
            raise ValueError("Invalid object-pointer tail length")

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            if not self.rope_k_repeat:
                raise RuntimeError("Native SAM2 attention expected repeated memory RoPE")
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # This candidate trains only added parameters on frozen SAM2 inference
        # features. Keep native evaluation semantics even if a wrapper is put
        # in train() mode; otherwise dropout changes SAM2 independently of the
        # learned affinity.
        dropout_p = 0.0
        zero_strength = float(self.xmem_strength.detach()) == 0.0
        differentiable_identity = (
            zero_strength and self.affinity_gradient_enabled and torch.is_grad_enabled()
        )
        # This is the exact native computation when the added method is neutral.
        # The training gate enables a straight-through identity below so alpha
        # can receive its initial gradient without changing the forward result.
        if zero_strength and not differentiable_identity:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            if utility_q is None or utility_memory is None:
                raise ValueError("Non-neutral affinity requires unpositioned SAM2 features")
            native_out = (
                F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                if differentiable_identity else None
            )
            affinity = self._affinity_module()
            projected = affinity.project(utility_q, utility_memory, num_k_exclude_rope)
            query_ranges = []
            query_chunk = 128
            for start in range(0, q.shape[-2], query_chunk):
                stop = min(start + query_chunk, q.shape[-2])
                query_ranges.append((start, stop))

            # Score the full query grid in one XMem affinity operation. This
            # avoids accumulating shared projection gradients from many
            # independent query-score graphs. The learned scale is likewise
            # applied once; only native SDPA remains query-chunked.
            full_scores = affinity.score(projected)
            full_bias = (full_scores * self.xmem_strength).to(dtype=q.dtype)
            chunks = []
            for start, stop in query_ranges:
                bias = full_bias[:, start:stop].unsqueeze(1)
                attended = F.scaled_dot_product_attention(
                    q[:, :, start:stop], k, v, attn_mask=bias, dropout_p=dropout_p
                )
                if differentiable_identity:
                    # Preserve the native output bit-for-bit at initialization,
                    # while retaining the derivative of the affinity branch.
                    attended = native_out[:, :, start:stop] + (attended - attended.detach())
                chunks.append(attended)
            out = torch.cat(chunks, dim=-2)
        out = self._recombine_heads(out)
        return self.out_proj(out)


def install_xmem_token_affinity(
    model: nn.Module,
    *,
    key_dim: int = 64,
    granularity: Literal["token", "frame"] = "token",
) -> dict[str, object]:
    """Replace only four SAM2 memory cross-attention modules with adapters."""
    if granularity not in ("token", "frame"):
        raise ValueError("granularity must be 'token' or 'frame'")
    memory_attention = getattr(model, "memory_attention", model)
    blocks = getattr(memory_attention, "layers", None)
    if blocks is None or len(blocks) != 4:
        raise ValueError("Expected SAM2's four-layer memory attention")
    entries = []
    for index, block in enumerate(blocks):
        if getattr(block, "pos_enc_at_cross_attn_queries", None) is not False:
            raise ValueError("Adapter requires SAM2's unpositioned cross-attention query")
        old = getattr(block, "cross_attn_image", None)
        new = _make_affinity_attention(old, key_dim=key_dim, granularity=granularity)
        # Verify all pretrained attention tensors are bit-identical after import.
        old_state = old.state_dict()
        new_state = new.state_dict()
        for name, tensor in old_state.items():
            if name not in new_state or not torch.equal(tensor, new_state[name]):
                raise RuntimeError(f"Pretrained SAM2 attention parameter changed at layer {index}: {name}")
        block.cross_attn_image = new
        entries.append({
            "layer": index,
            "native_attention_sha256": _state_digest(old_state),
            "adapter_strength": 0.0,
            "spatial_tokens_changed": False,
        })
    return {
        "layers": entries,
        "granularity": granularity,
        "key_dim": key_dim,
        "trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                     if "xmem_" in n),
        "hard_selection": False,
        "frame_selection": False,
        "pointer_affinity": False,
    }


def freeze_sam2_train_xmem_affinity(model: nn.Module) -> list[nn.Parameter]:
    """Freeze SAM2, keep it in inference mode, and train only the affinity.

    Gradients still flow through the frozen operations to the adapter. Keeping
    the whole model in eval mode prevents SAM2's native dropout layers from
    changing the training signal independently of the new affinity.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    adapter_parameters = [
        parameter for name, parameter in model.named_parameters() if ".xmem_" in name
    ]
    if not adapter_parameters:
        raise ValueError("Install XMem token affinity before configuring training")
    for parameter in adapter_parameters:
        parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, XMemTokenAffinityRoPEAttention):
            module.affinity_gradient_enabled = True
    remaining = [name for name, parameter in model.named_parameters() if parameter.requires_grad
                 and ".xmem_" not in name]
    if remaining:
        raise RuntimeError(f"Unexpected trainable pretrained parameters: {remaining[:5]}")
    return adapter_parameters


def _state_digest(state: dict[str, Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
