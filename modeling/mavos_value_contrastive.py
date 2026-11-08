"""Training-only foreground/background contrast on SAM2 memory value tokens.

The focal value adapter remains in the inference path. This auxiliary loss
uses per-frame object/background prototypes pooled from the adapter's actual
pre-v_proj memory values, with fractional GT occupancy used only after the
forward pass. It reuses this repository's pinned SupContrast and soft-region
pooling helpers; it adds no projection head, queue, token pruning, or inference
decision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling.complementary_memory.contrastive import (
    author_supcon,
    region_prototypes,
)


class MAVOSValueContrastive:
    """Apply supervised contrastive loss to causal, aligned memory values.

    Args:
        integration: active ``MAVOSMemoryValueModulation`` with value capture
            enabled.
        observer: active native memory observer that records selected frame
            spans and verifies memory/position alignment.
        targets: map from local frame index to ``[1,1,H,W]`` GT mask tensor.
        weight: auxiliary loss coefficient; set to zero for the matched control
            while retaining identical computations and diagnostics.
    """

    def __init__(self, integration, observer, targets, weight=0.1, temperature=0.07):
        if weight < 0 or temperature <= 0:
            raise ValueError("weight must be nonnegative and temperature positive")
        if not integration.capture_adapted_values:
            raise ValueError("value capture must be enabled for training")
        self.integration = integration
        self.observer = observer
        self.targets = {}
        for frame, target in targets.items():
            if not isinstance(target, torch.Tensor) or target.ndim != 4 or target.shape[:2] != (1, 1):
                raise ValueError(f"Target {frame} must be a [1,1,H,W] mask tensor")
            if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
                raise ValueError(f"Target {frame} must contain finite soft occupancy in [0,1]")
            self.targets[int(frame)] = target.detach()
        self.weight = float(weight)
        self.loss_fn = author_supcon()(
            temperature=temperature, base_temperature=0.07
        )
        self.records = []

    def __call__(self, sample):
        values = self.integration.adapted_values
        self.integration.adapted_values = []
        if len(values) != 4 or [row["layer"] for row in values] != [0, 1, 2, 3]:
            raise RuntimeError("Expected captured adapted values from all four layers")
        if not self.observer.calls:
            raise RuntimeError("No native memory-read alignment receipt for this frame")

        trace = self.observer.calls[-1]
        if trace["frame"] != sample["frame"] or trace["layers"] != [0, 1, 2, 3]:
            raise RuntimeError("Auxiliary loss frame/layer trace is misaligned")
        spans = trace["memory_frames"]
        if any(span["frame"] >= sample["frame"] for span in spans):
            raise RuntimeError("Current or future frame entered contrastive memory")

        per_layer = []
        for row in values:
            sequence = row["adapted"]
            pointer_count = row["pointer_tokens"]
            spatial_count = sequence.shape[0] - pointer_count
            if sequence.ndim != 3 or sequence.shape[1] != 1:
                raise RuntimeError("Expected one-object [tokens,1,channels] memory")
            if spatial_count != sum(span["stop"] - span["start"] for span in spans):
                raise RuntimeError("Captured value tokens differ from selected memory spans")

            embeddings, labels = [], []
            covered = 0
            for span in spans:
                start, stop = int(span["start"]), int(span["stop"])
                frame = int(span["frame"])
                token_count = stop - start
                if start != covered or token_count <= 0:
                    raise RuntimeError("Spatial memory spans are not contiguous and ordered")
                covered = stop
                target = self.targets.get(frame)
                if target is None:
                    raise RuntimeError(f"Missing loss-only target for stored frame {frame}")
                adapter = self.integration.adapters[row["layer"]]
                grid = adapter.grid_shape
                if token_count != grid[0] * grid[1]:
                    raise RuntimeError("Stored frame span does not match adapter token grid")
                features = sequence[start:stop, 0]
                occupancy = F.adaptive_avg_pool2d(
                    target.detach().float().to(features.device), grid
                ).reshape(-1)
                prototypes, roles = region_prototypes(features.float(), occupancy)
                embeddings.extend(prototypes)
                labels.extend(roles)

            if covered != spatial_count:
                raise RuntimeError("Not all captured spatial value tokens have a frame label")

            # SupCon requires at least one same-role positive per anchor. With
            # one selected memory frame there is no temporal positive pair, so
            # this frame contributes zero auxiliary loss rather than inventing
            # a positive. It still contributes the ordinary mask/presence loss.
            counts = {role: labels.count(role) for role in (0, 1)}
            if min(counts.values()) < 2:
                layer_loss = sequence.sum() * 0.0
                active = False
            else:
                features = F.normalize(torch.stack(embeddings).float(), dim=-1, eps=1e-6)
                if not torch.isfinite(features).all():
                    raise RuntimeError("Nonfinite foreground/background memory prototype")
                role_labels = torch.tensor(labels, device=features.device, dtype=torch.long)
                layer_loss = self.loss_fn(features[:, None], labels=role_labels)
                active = True
            if not torch.isfinite(layer_loss):
                raise RuntimeError("Nonfinite memory-value contrastive loss")
            per_layer.append(layer_loss)
            self.records.append({
                "frame": int(sample["frame"]),
                "layer": int(row["layer"]),
                "memory_frames": [int(span["frame"]) for span in spans],
                "foreground_prototypes": counts[1],
                "background_prototypes": counts[0],
                "contrastive_active": active,
                "contrastive_loss": float(layer_loss.detach()),
                "weighted_contrastive_loss": float(layer_loss.detach()) * self.weight,
                "gt_used_only_after_forward": True,
            })

        loss = torch.stack(per_layer).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite four-layer contrastive loss")
        return self.weight * loss
