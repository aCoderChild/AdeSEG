"""Small integration around author SAM2 and PyTorch SDPA; no new attention.

Match actual raw memory values to previously captured native memory frames.
Boundary IoU supplies predicted-mask spatial labels. Native selection, RoPE,
image encoder, memory writer and decoder are untouched.
"""

import hashlib
import math
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from modeling.region_tokens import reference_token_regions


def feature_key(value):
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def region_log_bias(labels, pointers, mode, seed):
    """One fixed hypothesis, not a tuned policy: boundary weight 1.1; others 1."""
    labels = torch.as_tensor(labels).long().flatten().cpu()
    if not ((labels >= 0) & (labels <= 2)).all() or pointers < 0:
        raise ValueError("Invalid three-region labels or pointer count")
    if mode not in ("zero", "boundary", "shuffled"):
        raise ValueError("Unknown regional intervention")
    if mode == "shuffled":
        labels = labels[
            torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        ]
    bias = torch.zeros(len(labels) + pointers)
    if mode != "zero":
        bias[: len(labels)][labels == 2] = math.log(1.1)
    return bias


class NativeRegionBias:
    """Temporary cross-attention-only SDPA bias with strict memory alignment."""

    def __init__(self, model, mode, sequence, prompt):
        self.model, self.mode, self.sequence, self.prompt = (
            model,
            mode,
            sequence,
            prompt,
        )
        self.frame = prompt
        self.bank, self.handles, self.trace = {}, [], []
        self.active, self.cached, self.mask = False, None, None
        self.native_sdpa = F.scaled_dot_product_attention

    def _before(self, layer, module, args, kwargs):
        if self.active or module.training or args:
            raise RuntimeError(
                "Unexpected nested/training/positional cross-attention call"
            )
        memory = kwargs["v"]
        pointers = int(kwargs.get("num_k_exclude_rope", 0))
        spatial = memory.shape[1] - pointers
        if memory.shape[0] != 1 or spatial <= 0 or spatial % 4096:
            raise RuntimeError("Unsupported native memory-key layout")
        identity = (
            self.frame,
            memory.data_ptr(),
            tuple(memory.shape),
            tuple(memory.stride()),
        )
        if identity != self.cached:
            biases, units = [], []
            for start in range(0, spatial, 4096):
                key = feature_key(memory[:, start : start + 4096])
                if key not in self.bank:
                    raise RuntimeError(
                        "Unmatched raw spatial memory; never guess a frame ID"
                    )
                frame, labels = self.bank[key]
                if frame >= self.frame:
                    raise RuntimeError("Current/future memory accessed")
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"20261021|{self.sequence}|{frame}".encode()
                    ).digest()[:8],
                    "little",
                ) % (2**63 - 1)
                biases.append(region_log_bias(labels, 0, self.mode, seed))
                units.append(
                    {
                        "frame": frame,
                        "counts": torch.bincount(
                            labels.flatten().long(), minlength=3
                        ).tolist(),
                    }
                )
            self.bias = torch.cat(biases + [torch.zeros(pointers)])
            self.units, self.pointers = units, pointers
            self.cached = identity
        if len(self.bias) != memory.shape[1] or pointers != self.pointers:
            raise RuntimeError("Pointer/key alignment changed between layers")
        self.mask = (
            None
            if not self.bias.any()
            else self.bias.to(memory.device, memory.dtype).reshape(1, 1, 1, -1)
        )
        self.trace.append(
            {
                "layer": layer,
                "memory_units": self.units,
                "pointer_tokens": pointers,
                "biased_keys": int((self.bias != 0).sum()),
            }
        )
        self.active = True

    def _after(self, module, args, kwargs, result):
        self.active = False

    def _sdpa(self, q, k, v, *args, **kwargs):
        if self.active:
            if (
                args
                or kwargs.get("attn_mask") is not None
                or k.shape[-2] != len(self.bias)
            ):
                raise RuntimeError("Unexpected SDPA mask/argument/key layout")
            if self.mask is not None:
                kwargs["attn_mask"] = self.mask
        return self.native_sdpa(q, k, v, *args, **kwargs)

    def capture(self, state, index, logits):
        """Read just-written native memory; grouping is never written back."""
        if index != self.frame:
            raise RuntimeError("Nonchronological regional rollout")
        if index > self.prompt and [r["layer"] for r in self.trace] != list(range(4)):
            raise RuntimeError("Expected all four memory cross-attention layers")
        outputs = state["output_dict_per_obj"][0]
        output = outputs["cond_frame_outputs"].get(
            index, outputs["non_cond_frame_outputs"].get(index)
        )
        if output is None or output["maskmem_features"] is None:
            raise RuntimeError("Native predicted memory unavailable")
        features = output["maskmem_features"]
        if tuple(features.shape) != (1, 64, 64, 64):
            raise RuntimeError("Unexpected B+ native memory shape")
        mask = logits[0, 0].detach().cpu().numpy() > 0
        labels = torch.from_numpy(
            np.asarray(reference_token_regions(mask, (64, 64), 0.02)["labels"])
        )
        key = feature_key(features.flatten(2).transpose(1, 2))
        if key in self.bank and self.bank[key][0] != index:
            raise RuntimeError("Ambiguous identical native memory frames")
        self.bank[key] = (index, labels)
        record = {
            "frame": index,
            "mode": self.mode,
            "weights_background_foreground_boundary": [1, 1, 1.1]
            if self.mode != "zero"
            else [1, 1, 1],
            "layers": self.trace,
        }
        self.trace = []
        self.frame = index + 1
        self.cached = None
        return record

    def __enter__(self):
        for layer, block in enumerate(self.model.memory_attention.layers):
            attention = block.cross_attn_image
            self.handles.append(
                attention.register_forward_pre_hook(
                    lambda module, args, kwargs, layer=layer: self._before(
                        layer, module, args, kwargs
                    ),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._after, with_kwargs=True, always_call=True
                )
            )
        self.patch = patch.object(F, "scaled_dot_product_attention", self._sdpa)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        for handle in self.handles:
            handle.remove()
        self.active = False
