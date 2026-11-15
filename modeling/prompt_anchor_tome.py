"""Prompt-anchored ToMe matching in SAM2's projected memory-key space.

The initial implementation compared raw ``maskmem_features`` with arbitrary
64-channel chunks of ``obj_ptr``. Although SAM2 splits pointers into memory-width
tokens for attention, those raw tensors are not established as a shared
similarity space. This version instead uses the prompt pointer's actual K token
as it enters each frozen cross-attention layer, so both sides of the comparison
have passed through that layer's learned key projection.

This remains an exploratory read-only memory-attention intervention. It uses
the original ToMe matching/weighted merge, preserves every memory frame and
pointer token, and does not alter SAM2 weights, writes, presence handling, or
decoder behavior.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from modeling.pointer_conditioned_tome import (
    NativePointerConditionedToMe,
    compress_pointer_conditioned_kv,
)


def projected_prompt_anchor_affinity(keys, pointers, pointer_tokens):
    """Return [B,N] cosine affinity in the actual projected K space.

    ``keys`` are the tensors received by SDPA: [B,H,N+P,D], after the frozen
    K projection and SAM2's native positional transform. The first prompt
    pointer occupies the first ``pointer_tokens`` positions of the pointer
    tail under this study's single-initial-prompt protocol.
    """
    if keys.ndim != 4 or not torch.is_floating_point(keys):
        raise ValueError("Expected projected attention keys [B,H,N+P,D]")
    if type(pointers) is not int or pointers < 1 or pointers >= keys.shape[-2]:
        raise ValueError("Invalid pointer-token tail length")
    spatial = keys.shape[-2] - pointers
    if type(pointer_tokens) is not int or not 1 <= pointer_tokens <= pointers:
        raise ValueError("Prompt pointer-token count must fit inside pointer tail")
    if not torch.isfinite(keys).all():
        raise ValueError("Projected attention keys must be finite")

    spatial_keys = keys[..., :spatial, :].float()
    anchor_keys = keys[..., spatial:spatial + pointer_tokens, :].float()
    spatial_keys = F.normalize(spatial_keys, dim=-1)
    anchor_keys = F.normalize(anchor_keys, dim=-1)
    # Average compatibility over heads and the prompt pointer's native split
    # tokens; no channel reshaping or cross-space raw-feature comparison.
    affinity = torch.einsum("bhnd,bhpd->bhnp", spatial_keys, anchor_keys)
    affinity = affinity.mean(dim=(1, 3))
    if affinity.shape != (keys.shape[0], spatial):
        raise RuntimeError("Projected prompt affinity has an invalid token layout")
    return affinity


class NativePromptAnchorToMe(NativePointerConditionedToMe):
    """Match each spatial key to the initial prompt pointer in projected K space."""

    def _compress_memory(self, keys, values, memberships, spans, pointers, merge_plan):
        if self.current_output_dict is None:
            raise RuntimeError("Prompt-anchored ToMe has no active source bank")
        cond = self.current_output_dict["cond_frame_outputs"]
        if len(cond) != 1:
            raise RuntimeError(
                "Prompt-anchor protocol requires exactly one conditioning frame"
            )
        anchor_frame = int(next(iter(cond)))
        anchor_pointer = cond[next(iter(cond))].get("obj_ptr")
        if not isinstance(anchor_pointer, torch.Tensor) or anchor_pointer.ndim != 2:
            raise RuntimeError("Initial conditioning output has no [B,C] object pointer")

        memory = self.model.memory_attention
        layers = getattr(memory, "layers", None)
        if layers is None:
            raise RuntimeError("Cannot locate frozen SAM2 memory-attention layers")
        cross = layers[0].cross_attn_image
        mem_dim = int(getattr(cross, "kv_in_dim", 0))
        pointer_dim = int(anchor_pointer.shape[-1])
        if mem_dim <= 0 or pointer_dim % mem_dim:
            raise RuntimeError("SAM2 pointer cannot be split at its configured memory width")
        pointer_tokens = pointer_dim // mem_dim
        if pointers < pointer_tokens:
            raise RuntimeError("Attention pointer tail omits the initial prompt pointer")

        affinities = projected_prompt_anchor_affinity(keys, pointers, pointer_tokens)
        out_keys, out_values, bias, audit = compress_pointer_conditioned_kv(
            keys,
            values,
            spans,
            affinities,
            pointers=pointers,
            window_size=self.window_size,
            merge_divisor=self.merge_divisor,
            merge_plan=merge_plan,
        )
        anchor_bytes = anchor_pointer.detach().float().contiguous().cpu().numpy().tobytes()
        pointer_key_bytes = (
            keys[..., keys.shape[-2] - pointers:
                 keys.shape[-2] - pointers + pointer_tokens, :]
            .detach().float().contiguous().cpu().numpy().tobytes()
        )
        audit.update({
            "mode": "prompt_anchor_projected_key",
            "pointer_reference_frame": anchor_frame,
            "pointer_reference_sha256": hashlib.sha256(anchor_bytes).hexdigest(),
            "projected_prompt_pointer_key_sha256": hashlib.sha256(pointer_key_bytes).hexdigest(),
            "prompt_pointer_tokens": pointer_tokens,
            "affinity_space": "frozen_cross_attention_projected_keys",
        })
        return out_keys, out_values, bias, audit
