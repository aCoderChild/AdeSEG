"""Diagnostic controls around the verified Lean/Efficient-SAM2 integration.

No new predictor or importance algorithm. Block interventions are experimental
controls, not an author method or a deployable ground-truth-informed selector.
"""
import hashlib

import torch
import torch.nn.functional as F

from modeling.complementary_memory.integration import _tensor_sha256
from modeling.regional_token_selection import NativeRegionalTokenSelection

ARMS = ("native", "rejected_a", "rejected_b", "rejected", "retained", "random")


def block_plan(retained, context, *, seed=20261107, block=4, groups=16):
    """Disjoint 4x4 cells; rank by retained fraction, with fixed random ties.

    'Retained/rejected' means enriched blocks, not that every constituent token
    has that selector label. Random controls match counts, square sizes and age,
    not global arrangement/connectivity. No GT enters the plan.
    """
    if retained.shape != (4096,) or retained.dtype != torch.bool:
        raise ValueError("Expected one 64x64 boolean selector mask")
    if 64 % block or groups <= 0 or groups % 2:
        raise ValueError("Require even positive group count and grid divisor")
    grid = torch.arange(4096).reshape(64, 64)
    cells = grid.unfold(0, block, block).unfold(1, block, block).reshape(-1, block * block)
    if 2 * groups > len(cells):
        raise ValueError("Retained and rejected groups must be disjoint")
    counts = retained.cpu()[cells].sum(1)
    token_seed = int.from_bytes(hashlib.sha256(f"{seed}|{context}".encode()).digest()[:8], "little") % (2**63-1)
    order = torch.randperm(len(cells), generator=torch.Generator().manual_seed(token_seed))
    ranked = order[torch.argsort(counts[order], stable=True)]
    rejected, kept = ranked[:groups], ranked[-groups:]
    random = torch.randperm(len(cells), generator=torch.Generator().manual_seed(token_seed+1))[:groups]
    ids = {"rejected_a": rejected[:groups//2], "rejected_b": rejected[groups//2:],
           "rejected": rejected, "retained": kept, "random": random}
    masks, metadata = {}, {}
    for arm, chosen in ids.items():
        removed = torch.zeros(4096, dtype=torch.bool)
        removed[cells[chosen].flatten()] = True
        masks[arm] = removed
        metadata[arm] = {"block_ids": chosen.tolist(), "removed": int(removed.sum()),
            "retained_fraction": float(retained.cpu()[removed].float().mean()),
            "mask_sha256": _tensor_sha256(removed)}
    return masks, metadata


class TokenContributionRead(NativeRegionalTokenSelection):
    """Collect selectors on native reads; ablate only private target replays."""

    def __init__(self, model):
        super().__init__(model, "regional")
        self.arm = "collect"
        self.usage = {}
        original = self.author_attention

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            self._usage_sum += result[1][2].sum((0, 1, 2)).double()
            return result
        self.author_attention = observed

    def _before_selection(self, layer, module, args, kwargs):
        super()._before_selection(layer, module, args, kwargs)
        if self.arm not in (*ARMS, "collect"):
            raise ValueError("Unknown diagnostic arm")
        c = self.context
        masks, details = [], []
        counts, ids = [], []
        for i, span in enumerate(c["spans"]):
            removed = torch.zeros(4096, dtype=torch.bool)
            if 0 < i < len(c["spans"])-1 and self.arm not in ("collect", "native"):
                key = (c["object_id"], layer, span["frame"])
                plan, info = block_plan(self.masks[key], f"{layer}|{span['frame']}|{self.sequence}")
                removed = plan[self.arm]
                usage = self.usage[key]
                occupancy = c["probabilities"][0, i].detach().cpu().flatten()
                vectors = c["outputs"][i]["maskmem_features"][0].float().flatten(1).T.cpu()
                details.append({"memory_frame": span["frame"], **info[self.arm],
                    "first_read_usage_mass": float(usage[removed].sum()),
                    "mean_soft_foreground": float(occupancy[removed].mean()),
                    "mean_vector_norm": float(vectors[removed].norm(dim=-1).mean())})
            keep = ~removed
            counts.append(int(keep.sum()))
            ids.append(_tensor_sha256(torch.where(keep)[0]))
            if 0 < i < len(c["spans"])-1:
                masks.append(keep.to(kwargs["v"].device))
        module._sparse_mask = torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)
        self.selection_layers[-1].update(counts=counts, selected_indices_sha256=ids,
            spatial_keys_after=sum(counts), mode=self.arm, intervention_groups=details)

    def _capture_sdpa(self, q, k, v, **kwargs):
        if k.shape[-2] != self.selection_layers[-1]["spatial_keys_after"] + self.current_pointers:
            raise RuntimeError("Actual key count differs from diagnostic plan")
        if self.arm != "collect":
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        self._usage_sum = torch.zeros(4096, dtype=torch.float64, device=q.device)
        result = super()._capture_sdpa(q, k, v, **kwargs)
        key = (self.context["object_id"], self.current_layer, self.context["spans"][-1]["frame"])
        self.usage[key] = (self._usage_sum / self._usage_sum.sum()).float().cpu()
        if abs(float(self.usage[key].sum())-1) > 1e-5:
            raise RuntimeError("Collected attention mass does not sum to one")
        return result

    def expire(self, frame):
        """First-read caches older than the native window cannot be used again."""
        for mapping in (self.masks, self.usage):
            for key in list(mapping):
                if key[-1] < frame-6:
                    del mapping[key]
