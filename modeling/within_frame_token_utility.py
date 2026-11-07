"""Source-isolated diagnostic using the existing Efficient-SAM2 gather/RoPE path.

The experimental tile plan is project code, not a published token selector.
No attention ranking, regional labels, learned module or memory write is added.
"""
import hashlib
import json

import torch
import torch.nn.functional as F

from modeling.complementary_memory.integration import _tensor_sha256
from modeling.regional_token_selection import NativeRegionalTokenSelection


ARMS = ("native", *(f"tile_{i:02d}" for i in range(16)),
        "whole_source", "union_00_15", "union_03_12")


def removal_plan(arm):
    """Return original row-major cell indices; never compact spatial coordinates."""
    if arm not in ARMS:
        raise ValueError(f"Unknown within-frame intervention: {arm}")
    removed = torch.zeros((64, 64), dtype=torch.bool)
    if arm == "whole_source":
        removed.fill_(True)
    elif arm != "native":
        tiles = ([int(arm[5:])] if arm.startswith("tile_") else
                 [int(i) for i in arm.split("_")[1:]])
        for tile in tiles:
            y, x = divmod(tile, 4)
            removed[y*16:(y+1)*16, x*16:(x+1)*16] = True
    return removed.flatten()


def choose_source(spans, sequence, target, *, seed=20261108):
    """Choose once from actual eligible memories, independent of measured utility."""
    ids = [s["frame"] for s in spans]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected unique native memory frame IDs")
    if spans[0]["t_pos"] != 0 or ids[-1] != target-1 or any(i >= target for i in ids):
        raise ValueError("Expected native prompt/latest and past-only memory layout")
    eligible = sorted(ids[1:-1])
    context = json.dumps([sequence, target, seed], separators=(",", ":"))
    sha = hashlib.sha256(context.encode()).hexdigest()
    chosen = eligible[int(sha, 16) % len(eligible)] if eligible else None
    return {"eligible_sources": eligible, "source_frame": chosen,
            "selection_sha256": sha, "selection_context": context}


class WithinFrameTokenRead(NativeRegionalTokenSelection):
    """Ablate one known source only; bypass first-read scoring and anchor caches.

    Inherits the existing alignment checks, four-layer receipt, private attention
    namespace, author K/V gather, original-coordinate RoPE and pointer checks.
    The initial PolypGen diagnostic uses the existing single-instance replay.
    No multi-object benchmark claim is made by this controller.
    """

    def __init__(self, model):
        super().__init__(model, "identity")
        self.arm = "native"
        self.source_frame = None

    def _before_selection(self, layer, module, args, kwargs):
        super()._before_selection(layer, module, args, kwargs)
        spans = self.context["spans"]
        if any(s["start"] != i*4096 or s["stop"] != (i+1)*4096
               for i, s in enumerate(spans)):
            raise RuntimeError("Noncontiguous or unexpected spatial memory spans")
        removed = removal_plan(self.arm)
        if self.arm != "native" and self.source_frame not in [s["frame"] for s in spans[1:-1]]:
            raise RuntimeError("Requested source absent or protected; never substitute another frame")
        masks, counts, hashes = [], [], []
        for i, span in enumerate(spans):
            keep = (~removed if self.arm != "native" and span["frame"] == self.source_frame
                    else torch.ones(4096, dtype=torch.bool))
            counts.append(int(keep.sum()))
            hashes.append(_tensor_sha256(torch.where(keep)[0]))
            if 0 < i < len(spans)-1:
                masks.append(keep)
        mask = torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)
        module._sparse_mask = mask.to(kwargs["v"].device)
        self.selection_layers[-1].update(
            mode=self.arm, source_frame=self.source_frame if self.arm != "native" else None,
            counts=counts, selected_indices_sha256=hashes, spatial_keys_after=sum(counts),
            removed_cell_indices_sha256=_tensor_sha256(torch.where(removed)[0]))

    def _capture_sdpa(self, q, k, v, **kwargs):
        expected = self.selection_layers[-1]["spatial_keys_after"] + self.current_pointers
        if k.shape[-2] != expected or v.shape[-2] != expected:
            raise RuntimeError("Actual gathered K/V length differs from intervention")
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
