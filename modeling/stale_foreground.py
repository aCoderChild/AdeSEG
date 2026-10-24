"""Causal foreground exclusion around native SAM2/SAMURAI video propagation.

Reuses NativeRegionBias's raw-memory matching, hooks and native PyTorch SDPA.
This experimental policy is project glue, not an author-published algorithm.
"""

import torch

from modeling.region_bias import NativeRegionBias, feature_key
from modeling.region_definitions import mixed_cell_regions


def absent_from_score(score):
    """Complement of the author's existing `object_score_logits > 0` rule."""
    score = torch.as_tensor(score).detach().float().cpu()
    if score.numel() != 1 or not torch.isfinite(score).all():
        raise ValueError("Expected one finite native object-score logit")
    return bool(score.item() <= 0)


class StaleForegroundGate(NativeRegionBias):
    """Previous-frame score controls the next read; normal memory writes continue."""

    def __init__(self, model, sequence, prompt, *, enabled):
        super().__init__(model, "zero", sequence, prompt)
        self.enabled = enabled
        self.previous_score = None

    def _before(self, layer, module, args, kwargs):
        super()._before(layer, module, args, kwargs)
        if self.previous_score is None:
            raise RuntimeError("Capture the conditioning frame before a memory read")
        gate = self.enabled and absent_from_score(self.previous_score)
        # Base class assembles the actual author-selected spatial memories in
        # order. Pointers are appended separately and never receive an exclusion.
        memory = kwargs["v"]
        selected = torch.zeros(len(self.bias), dtype=torch.bool)
        if gate:
            for start in range(0, len(self.bias) - self.pointers, 4096):
                key = feature_key(memory[:, start : start + 4096])
                selected[start : start + 4096] = self.bank[key][1].flatten() == 1
        if selected.all():
            raise RuntimeError("All-key exclusion is undefined; no fallback is permitted")
        self.bias = torch.zeros_like(self.bias)
        self.bias[selected] = -torch.inf
        self.mask = (
            self.bias.to(memory.device, memory.dtype).reshape(1, 1, 1, -1)
            if selected.any() else None
        )
        self.trace[-1].update(
            biased_keys=int(selected.sum()),
            gate_active=gate,
            previous_object_score=self.previous_score,
            excluded_tokens=int(selected.sum()),
            excluded_region="foreground_interior",
        )

    def capture(self, state, index, logits):
        if index != self.frame:
            raise RuntimeError("Nonchronological full propagation")
        if index > self.prompt and [r["layer"] for r in self.trace] != list(range(4)):
            raise RuntimeError("Expected all four author memory-attention layers")
        outputs = state["output_dict_per_obj"][0]
        output = outputs["cond_frame_outputs"].get(
            index, outputs["non_cond_frame_outputs"].get(index)
        )
        if output is None or output["maskmem_features"] is None:
            raise RuntimeError("Actual stored native memory is unavailable")
        features = output["maskmem_features"]
        if tuple(features.shape) != (1, 64, 64, 64):
            raise RuntimeError("Expected native B+ memory shape")
        low = output["pred_masks"][0, 0]
        labels = torch.from_numpy(mixed_cell_regions(low)["labels"])
        key = feature_key(features.flatten(2).transpose(1, 2))
        if key in self.bank and not torch.equal(self.bank[key][1], labels):
            raise RuntimeError("Identical raw memory has ambiguous spatial labels")
        self.bank[key] = (index, labels)
        score = output["object_score_logits"]
        absent_from_score(score)  # Reject missing/nonfinite/multi-object scores.
        record = {
            "frame": index,
            "previous_object_score": self.previous_score,
            "current_object_score": float(score.detach().cpu().item()),
            "layers": self.trace,
            "region_counts": torch.bincount(labels.flatten(), minlength=3).tolist(),
            "memory_sha256": key,
            "label_source": "this frame's stored native pred_masks",
        }
        self.previous_score = record["current_object_score"]
        self.trace = []
        self.frame = index + 1
        self.cached = None
        return record
