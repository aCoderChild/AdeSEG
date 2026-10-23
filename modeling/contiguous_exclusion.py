"""Rigid spatial nulls and native SDPA masks; no replacement model/attention.

The null is project experimental-control logic using PyTorch transforms and
OpenCV component measurements, not an algorithm attributed to an author paper.
"""

import hashlib

import cv2
import numpy as np
import torch

from modeling.region_bias import NativeRegionBias


def topology(mask):
    value = np.asarray(mask, np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=4)
    padded = np.pad(value.astype(np.int16), 1)
    perimeter = int(
        abs(np.diff(padded, axis=0)).sum() + abs(np.diff(padded, axis=1)).sum()
    )
    return (
        int(value.sum()),
        tuple(sorted(stats[1:, cv2.CC_STAT_AREA].tolist())),
        perimeter,
    )


def rigid_control(mask, seed, context):
    """Uniform unique nonidentity rigid placements, no clipping or wraparound."""
    mask = torch.as_tensor(mask).bool().cpu()
    if mask.shape != (64, 64):
        raise ValueError("Rigid controls require memory64 masks")
    positions = mask.nonzero()
    if not len(positions):
        return mask.clone(), {"candidates": 0, "informative": False, "reason": "empty"}
    origin = positions.min(0).values.tolist()
    end = (positions.max(0).values + 1).tolist()
    crop = mask[origin[0] : end[0], origin[1] : end[1]]
    variants, seen, total, identity = [], set(), 0, None
    for reflected in (False, True):
        for rotation in range(4):
            item = torch.rot90(
                torch.flip(crop, [1]) if reflected else crop, rotation, [0, 1]
            ).contiguous()
            key = (tuple(item.shape), item.numpy().tobytes())
            if key in seen:
                continue
            seen.add(key)
            height, width = item.shape
            columns = 65 - width
            count = (65 - height) * columns
            if item.shape == crop.shape and torch.equal(item, crop):
                identity = total + origin[0] * columns + origin[1]
            variants.append((item, reflected, rotation, total, count))
            total += count
    if identity is None:
        raise RuntimeError("Original rigid placement lost")
    candidates = total - 1
    if not candidates:
        return mask.clone(), {
            "candidates": 0,
            "informative": False,
            "reason": "rigidly_invariant",
        }
    token_seed = int.from_bytes(
        hashlib.sha256(f"{seed}|{context}".encode()).digest()[:8], "little"
    ) % (2**63 - 1)
    rank = int(
        torch.randint(
            candidates, (1,), generator=torch.Generator().manual_seed(token_seed)
        )
    )
    rank += rank >= identity
    for item, reflected, rotation, offset, count in variants:
        if offset <= rank < offset + count:
            height, width = item.shape
            y, x = divmod(rank - offset, 65 - width)
            selected = torch.zeros_like(mask)
            selected[y : y + height, x : x + width] = item
            if torch.equal(selected, mask) or topology(selected) != topology(mask):
                raise RuntimeError(
                    "Rigid control failed shape/count/connectivity/perimeter matching"
                )
            return selected, {
                "candidates": candidates,
                "informative": True,
                "rotation_quarters": rotation,
                "reflected": reflected,
                "top_left": [y, x],
                "overlap_tokens": int((selected & mask).sum()),
            }
    raise RuntimeError("Invalid rigid placement rank")


def exclusion_plan(schemes, units, pointers, definition, region, seed, context):
    pieces, metadata = [], []
    for unit in units:
        frame = unit["frame"]
        mask = torch.as_tensor(schemes[frame][definition]["labels"]) == region
        shape_hash = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
        _, eligibility = rigid_control(mask, 0, "eligibility")
        if seed is None:
            selected = mask
            info = {key: eligibility[key] for key in ("candidates", "informative")}
            info["role"] = "regional"
        else:
            selected, info = rigid_control(
                mask, seed, f"{context}|{frame}|{shape_hash}"
            )
        if int(selected.sum()) != int(mask.sum()):
            raise RuntimeError("Exclusion count mismatch")
        pieces.append(selected.flatten())
        metadata.append({"frame": frame, "excluded": int(mask.sum()), **info})
    selected = torch.cat(pieces + [torch.zeros(pointers, dtype=torch.bool)])
    return {
        "selected": selected,
        "units": metadata,
        "frames": [u["frame"] for u in units],
        "pointers": pointers,
        "informative": any(u["informative"] for u in metadata),
        "undefined": bool(selected.all()),
        "sha256": hashlib.sha256(selected.numpy().tobytes()).hexdigest(),
    }


class NativeExclusion(NativeRegionBias):
    """Reuse official raw-memory alignment and hook lifecycle, preserve Q/K/V."""

    def __init__(self, model, sequence, prompt):
        super().__init__(model, "zero", sequence, prompt)
        self.plan = None
        self.layout = []
        self.last_units = None

    def _before(self, *args):
        super()._before(*args)
        self.last_units = self.units
        if self.plan is not None and (
            self.plan["frames"] != [u["frame"] for u in self.units]
            or self.plan["pointers"] != self.pointers
            or self.plan["selected"].numel() != len(self.bias)
        ):
            raise RuntimeError("Counterfactual memory/pointer layout changed")

    def _sdpa(self, q, k, v, *args, **kwargs):
        if self.active:
            if args or kwargs.get("attn_mask") is not None:
                raise RuntimeError("Unexpected native mask/SDPA arguments")
            self.layout.append(
                tuple(
                    (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                    for t in (q, k, v)
                )
            )
            if any(
                t.dtype != torch.float32 or t.device.type != "cpu" for t in (q, k, v)
            ):
                raise RuntimeError("Expected CPU FP32 projected Q/K/V")
            if self.plan is not None:
                if self.plan["undefined"]:
                    raise RuntimeError("All-key exclusion is undefined")
                bias = torch.zeros(
                    (1, 1, 1, k.shape[-2]), dtype=q.dtype, device=q.device
                )
                bias[..., self.plan["selected"]] = -torch.inf
                kwargs["attn_mask"] = bias
        return self.native_sdpa(q, k, v, *args, **kwargs)
