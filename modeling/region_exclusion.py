"""Regional key exclusion via native SAM2/PyTorch SDPA, not a new attention.

Reuses the existing observer's key-layout validation and hook lifecycle.
Only mask selection is project integration code. Q/K/V, RoPE positions,
object pointers, image encoder, decoder and memory writer are not replaced.
"""

import hashlib

import torch

from modeling.region_attention import RegionAttentionObserver


def exclusion_keys(labels, units, region, *, seed=None, context=""):
    """Exact region or uniform count-matched samples within each memory frame."""
    labels = labels.cpu()
    if region not in (0, 1, 2) or not ((labels >= -1) & (labels <= 2)).all():
        raise ValueError("Expected one of the three spatial regions")
    selected = torch.zeros_like(labels, dtype=torch.bool)
    counts = []
    for unit in units:
        start, end = unit["start"], unit["end"]
        regional = labels[start:end] == region
        count = int(regional.sum())
        if seed is None:
            selected[start:end] = regional
        elif count:
            token_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|{context}|{region}|{unit['frame_idx']}|{start}".encode()
                ).digest()[:8],
                "little",
            ) % (2**63 - 1)
            generator = torch.Generator().manual_seed(token_seed)
            indices = torch.randperm(end - start, generator=generator)[:count]
            selected[start + indices] = True
        counts.append(count)
        if int(selected[start:end].sum()) != count:
            raise RuntimeError("Random control does not match regional token count")
    if selected[labels < 0].any():
        raise RuntimeError("Object-pointer keys must never be excluded")
    if selected.all():
        raise ValueError("Cannot exclude every key: attention would be undefined")
    return selected, counts


class RegionKeyExclusion(RegionAttentionObserver):
    """Apply the same additive -inf key mask in every memory cross-attention layer."""

    def __init__(self, memory_attention, label_loader):
        super().__init__(memory_attention, label_loader)
        self.enabled = False

    def condition(self, region, frame, sequence, seed=None):
        self.region = region  # None is the exact native no-op path.
        self.frame = frame
        self.sequence = sequence
        self.seed = seed
        self.rows.clear()
        self.enabled = True

    def _context(self, module, args, kwargs):
        if not self.enabled:
            return
        super()._context(module, args, kwargs)
        if any(unit["frame_idx"] >= self.frame for unit in self.units):
            raise ValueError("Intervention accessed current/future spatial memory")
        if self.region is None:
            self.selected = torch.zeros_like(self.labels, dtype=torch.bool)
            self.counts = [0] * len(self.units)
        else:
            self.selected, self.counts = exclusion_keys(
                self.labels,
                self.units,
                self.region,
                seed=self.seed,
                context=f"{self.sequence}|{self.frame}",
            )
        self.bias = None

    def _observe(self, original, layer, q, k, v, attn_mask=None):
        if not self.enabled:
            return original(q, k, v, attn_mask=attn_mask)
        if attn_mask is not None or self.module.training:
            raise ValueError("Do not combine this exclusion with another intervention")
        if k.shape[-2] != len(self.selected):
            raise ValueError("Exclusion/key layout disagreement")
        if self.selected.any() and self.bias is None:
            self.bias = torch.zeros(
                (1, 1, 1, k.shape[-2]), device=q.device, dtype=q.dtype
            )
            self.bias[..., self.selected.to(q.device)] = float("-inf")
        result = original(q, k, v, attn_mask=self.bias)
        self.rows.append(
            {
                "layer": layer,
                "excluded_tokens": int(self.selected.sum()),
                "excluded_per_memory_frame": self.counts,
                "pointer_tokens": int((self.labels < 0).sum()),
                "memory_units": self.units,
                "mask_sha256": hashlib.sha256(
                    self.selected.numpy().tobytes()
                ).hexdigest(),
            }
        )
        return result
