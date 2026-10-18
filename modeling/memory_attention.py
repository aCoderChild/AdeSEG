"""Frozen experimental masks/substitutions; no trainable model or readout."""

from __future__ import annotations

import torch
from torch import nn

class UtilityController(nn.Module):
    """Build per-token attention biases, leave object pointers untouched."""

    def __init__(self, bias_scale: float = 1.0, learn_scale=True):
        super().__init__()
        initial = torch.tensor(float(bias_scale)).log()
        # Keep the old call signature for diagnostic snapshots, not training.
        self.register_buffer("log_bias_scale", initial)
        self.enabled = False
        self.excluded_frame_indices: set[int] = set()
        self.excluded_token_masks: dict[int, torch.Tensor] = {}
        self.token_biases: dict[int, torch.Tensor] = {}
        # E2 substitution controls replace values only.  Each entry is a
        # ``(selected_mask, replacement_values)`` pair for one spatial-memory
        # frame; keys, positions, and object pointers stay untouched.
        self.token_value_replacements: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.token_memory_replacements: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.exclude_pointers = False
        self.token_mask_policy = None
        self.token_bias_policy = None
        self.token_value_replacement_policy = None
        self.token_memory_replacement_policy = None

    @property
    def bias_scale(self) -> torch.Tensor:
        return self.log_bias_scale.exp()


    def replace_memory_values(self, memory, units, replacements=None):
        """Apply explicit spatial-value substitutions, leaving keys unchanged.

        ``memory`` is batch-first ``[B, K, D]``.  A replacement provides a
        boolean mask and one replacement vector per spatial key in the named
        memory frame.  Returning the original tensor for an empty mapping keeps
        the upstream cross-attention path exactly unchanged.
        """
        if replacements is None:
            replacements = self.token_value_replacements
            if self.token_value_replacement_policy is not None:
                replacements = self.token_value_replacement_policy(memory, units)
        if not replacements:
            return memory
        result = memory.clone()
        for unit in units:
            replacement = replacements.get(unit["frame_idx"])
            if replacement is None:
                continue
            selected, values = replacement
            start, end = unit["start"], unit["end"]
            selected = torch.as_tensor(
                selected, device=memory.device, dtype=torch.bool
            ).flatten()
            values = torch.as_tensor(values, device=memory.device, dtype=memory.dtype)
            length = end - start
            if selected.numel() != length:
                raise ValueError(
                    "Value-replacement mask length does not match spatial memory span"
                )
            if values.shape != (length, memory.shape[-1]):
                raise ValueError(
                    "Value replacements must have shape [spatial_tokens, memory_dim]"
                )
            result[:, start:end][:, selected] = values[selected].unsqueeze(0)
        return result

    def forward(self, curr, memory, units, num_obj_ptr_tokens, query_context):
        if self.token_mask_policy is not None:
            self.excluded_token_masks = self.token_mask_policy(units)
        if self.token_bias_policy is not None:
            self.token_biases = self.token_bias_policy(units)
        token_bias = memory.new_zeros(memory.shape[0], memory.shape[1])
        if self.enabled:
            raise ValueError("Learned utility scoring was removed; use explicit diagnostic masks/biases")
        for unit in units:
            start, end = unit["start"], unit["end"]
            if unit["frame_idx"] in self.token_biases:
                bias = torch.as_tensor(
                    self.token_biases[unit["frame_idx"]],
                    device=memory.device,
                    dtype=memory.dtype,
                ).flatten()
                if bias.numel() != end - start:
                    raise ValueError(
                        "Token bias length does not match spatial memory span"
                    )
                token_bias[:, start:end] += bias
            if unit["frame_idx"] in self.excluded_token_masks:
                mask = torch.as_tensor(
                    self.excluded_token_masks[unit["frame_idx"]],
                    device=memory.device,
                    dtype=torch.bool,
                ).flatten()
                if mask.numel() != end - start:
                    raise ValueError(
                        "Token mask length does not match spatial memory span"
                    )
                token_bias[:, start:end].masked_fill_(mask, -torch.inf)
            if unit["frame_idx"] in self.excluded_frame_indices:
                token_bias[:, unit["start"] : unit["end"]] = -torch.inf
        if self.exclude_pointers and num_obj_ptr_tokens:
            token_bias[:, -num_obj_ptr_tokens:] = -torch.inf
        # SDPA accepts [B, 1, 1, K], broadcast over heads and queries.
        attention_bias = token_bias[:, None, None, :]
        if (
            not self.enabled
            and not self.excluded_frame_indices
            and not any(
                torch.as_tensor(mask).any().item()
                for mask in self.excluded_token_masks.values()
            )
            and not self.token_biases
            and not self.exclude_pointers
        ):
            attention_bias = (
                None  # preserve the uninstrumented SDPA kernel path exactly
            )
        return attention_bias, None, None


def attach_utility_controller(
    model, controller: UtilityController, capture_attention=False
):
    """Attach to a built predictor and return the same model for fluent setup."""
    model.memory_attention.utility_controller = controller.to(model.device)
    model.memory_attention.capture_attention = capture_attention
    return model
