"""Per-key measurement layered onto the existing unchanged-author observer."""

import numpy as np
import torch

from modeling.region_attention import RegionAttentionObserver


class TokenAttentionObserver(RegionAttentionObserver):
    """Save joint query-group/key mass, never substitute a model output."""

    def __init__(self, memory_attention, sink, chunk_size=128):
        super().__init__(memory_attention, None, chunk_size)
        self.sink = sink
        self.native_author = self.author
        self.author = self._collect_chunk

    def _collect_chunk(self, *args, **kwargs):
        result = self.native_author(*args, **kwargs)
        weights = result[1][2]
        count = weights.shape[-2]
        labels = self.query_labels[self.offset : self.offset + count]
        for group in range(3):
            selected = labels == group
            if selected.any():
                self.joint[group] += (
                    weights[:, :, selected, :].float().sum((0, 1, 2)).cpu().double()
                )
        self.offset += count
        return result

    def _observe(self, original, layer, q, k, v, attn_mask=None):
        if not self.enabled:
            return original(q, k, v, attn_mask=attn_mask)
        self.query_labels = torch.as_tensor(self.query_labels).long().flatten().cpu()
        if (
            self.query_labels.numel() != q.shape[-2]
            or not ((self.query_labels >= 0) & (self.query_labels <= 2)).all()
        ):
            raise ValueError("Query labels must partition native query positions")
        self.offset = 0
        self.joint = torch.zeros((3, k.shape[-2]), dtype=torch.float64)
        output = super()._observe(original, layer, q, k, v, attn_mask)
        counts = torch.bincount(self.query_labels, minlength=3)
        joint = (self.joint / (q.shape[0] * q.shape[1] * q.shape[-2])).float().numpy()
        expected = counts.numpy() / q.shape[-2]
        if self.offset != q.shape[-2] or not np.allclose(
            joint.sum(1), expected, atol=1e-4, rtol=0
        ):
            raise RuntimeError("Per-query-group mass/count mismatch")
        if abs(float(joint.sum()) - 1) > 1e-4:
            raise RuntimeError("Saved per-token attention does not sum to one")
        self.sink(layer, joint, counts.numpy(), self.rows[-1])
        return output
