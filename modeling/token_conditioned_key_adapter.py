"""Token-local key adaptation for frozen SAM2 memory cross-attention.

The low-rank residual and rank-gating pattern follows the official MoGA
implementation (RobustPVOS, CVPR 2026; pinned in the experiment protocol).
This SAM2-specific extension conditions each spatial memory key's rank gate on
that key and its object's pointer context. It edits only the key input passed to
native memory cross-attention: values, pointer keys, positional encodings,
attention projections, memory writes, and the decoder are untouched.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn


class TokenConditionedKeyResidual(nn.Module):
    """Low-rank key residual with token-local or pooled rank gates.

    ``mode='token'`` is the candidate. ``mode='global'`` is the matched
    control: it uses the same parameterization but broadcasts one gate per
    object across its spatial memory tokens. Both modes still compute the
    low-rank residual from each token's own content.
    """

    def __init__(
        self,
        channels: int = 64,
        rank: int = 16,
        alpha: float = 16.0,
        gate_hidden: int = 128,
    ) -> None:
        super().__init__()
        if channels <= 0 or rank <= 0 or rank > channels:
            raise ValueError("Expected positive channels and 0 < rank <= channels")
        if alpha <= 0 or gate_hidden <= 0:
            raise ValueError("Expected positive alpha and gate_hidden")
        self.channels = int(channels)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        # Like MoGA's gate MLP, predict one activation per rank component. The
        # input is [memory-key content, same-object pointer context].
        self.gate = nn.Sequential(
            nn.Linear(2 * channels, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, rank),
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def _pointer_context(self, memory: Tensor, spatial_count: int, pointer_count: int) -> Tensor:
        if pointer_count:
            # SAM2 stores object pointers in the memory tail. The batch axis is
            # object-aligned, so this summary cannot mix object identities.
            return memory[:, spatial_count:, :].mean(dim=1)
        # SAM2 may have no pointer on the first read; mirror MoGA's fallback to
        # a memory summary, here formed only from spatial memory tokens.
        return memory[:, :spatial_count, :].mean(dim=1)

    def forward(self, memory_values: Tensor, pointer_count: int, mode: str = "token") -> Tensor:
        """Return a key residual shaped [B, spatial_tokens, channels].

        ``memory_values`` are SAM2's unpositioned key/value inputs. Positional
        and temporal encodings enter through ``k`` separately and are therefore
        not consumed or changed by this module.
        """
        if mode not in ("token", "global"):
            raise ValueError("mode must be 'token' or 'global'")
        if memory_values.ndim != 3:
            raise ValueError("Expected batch-first memory [B, tokens, channels]")
        if memory_values.shape[-1] != self.channels:
            raise ValueError(
                f"Expected {self.channels}-channel memory, got {memory_values.shape[-1]}"
            )
        if not torch.isfinite(memory_values).all():
            raise ValueError("Memory inputs must be finite")
        if not 0 <= pointer_count < memory_values.shape[1]:
            raise ValueError("pointer_count must leave at least one spatial token")

        spatial_count = memory_values.shape[1] - pointer_count
        spatial = memory_values[:, :spatial_count]
        pointer_context = self._pointer_context(memory_values, spatial_count, pointer_count)

        # Keep adapter math in parameter dtype, then cast the correction back
        # before adding it to the native attention key tensor.
        spatial_for_adapter = spatial.to(dtype=self.down.weight.dtype)
        pointer_for_adapter = pointer_context.to(dtype=self.down.weight.dtype)
        if mode == "token":
            context = pointer_for_adapter[:, None, :].expand(-1, spatial_count, -1)
            gate_input = torch.cat((spatial_for_adapter, context), dim=-1)
            gate_logits = self.gate(gate_input)
        else:
            pooled = spatial_for_adapter.mean(dim=1)
            gate_input = torch.cat((pooled, pointer_for_adapter), dim=-1)
            gate_logits = self.gate(gate_input)[:, None, :].expand(-1, spatial_count, -1)

        low_rank = self.down(spatial_for_adapter)
        correction = self.up(low_rank * torch.sigmoid(gate_logits))
        correction = correction * (self.alpha / self.rank)
        return correction.to(dtype=memory_values.dtype)


class TokenConditionedKeyHeads(nn.Module):
    """One independent low-rank key adapter for each native SAM2 layer."""

    def __init__(self, layers: int = 4, channels: int = 64, rank: int = 16) -> None:
        super().__init__()
        if layers != 4:
            raise ValueError("The frozen protocol requires all four SAM2 layers")
        self.layers = nn.ModuleList(
            TokenConditionedKeyResidual(channels=channels, rank=rank) for _ in range(layers)
        )


class NativeTokenConditionedKeyAdapter:
    """Inject key-only residuals through hooks on native SAM2 memory attention.

    The layer hook captures the memory and position tensors used by the native
    layer. The cross-attention hook verifies ``k == memory + pos`` before
    applying the correction to spatial keys only. Pointer keys and the value
    argument are returned unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        heads: TokenConditionedKeyHeads,
        mode: str = "token",
        *,
        inference_state: dict | None = None,
    ) -> None:
        if mode not in ("token", "global"):
            raise ValueError("mode must be 'token' or 'global'")
        self.model = model
        self.heads = heads
        self.mode = mode
        self.inference_state = inference_state
        self.calls: list[dict[str, Any]] = []
        self.handles = []
        self._pending: dict[int, tuple[Tensor, Tensor | None, int]] = {}
        self._active_object_id = None
        self._active_frame = None
        self._original_prepare = None
        self._prepare_signature = None

    def _wrap_memory_prepare(self):
        if self.inference_state is None:
            return
        self._original_prepare = self.model._prepare_memory_conditioned_features
        self._prepare_signature = inspect.signature(self._original_prepare)

        def prepare(*args, **kwargs):
            bound = self._prepare_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            output_dict = values.get("output_dict")
            frame_index = values.get("frame_idx")
            owners = [
                index for index, bank in self.inference_state["output_dict_per_obj"].items()
                if bank is output_dict
            ]
            if len(owners) != 1:
                raise RuntimeError("Active SAM2 memory bank does not map to one persistent object ID")
            index = owners[0]
            self._active_object_id = self.inference_state["obj_idx_to_id"][index]
            self._active_frame = int(frame_index)
            return self._original_prepare(*args, **kwargs)

        self.model._prepare_memory_conditioned_features = prepare

    @staticmethod
    def _get_tensor(name: str, args: tuple, kwargs: dict, index: int) -> Tensor:
        value = kwargs.get(name, args[index] if len(args) > index else None)
        if not isinstance(value, Tensor):
            raise RuntimeError(f"Native SAM2 memory attention did not provide tensor {name!r}")
        return value

    def _layer_pre_hook(self, layer_index: int):
        def before(module, args, kwargs):
            memory = self._get_tensor("memory", args, kwargs, 1)
            position = kwargs.get("pos", args[2] if len(args) > 2 else None)
            pointer_count = int(kwargs.get("num_k_exclude_rope", args[4] if len(args) > 4 else 0))
            if position is not None and not isinstance(position, Tensor):
                raise RuntimeError("Expected tensor memory position encoding")
            self._pending[layer_index] = (memory, position, pointer_count)
        return before

    def _cross_attention_pre_hook(self, layer_index: int):
        def before(module, args, kwargs):
            key = self._get_tensor("k", args, kwargs, 1)
            value = self._get_tensor("v", args, kwargs, 2)
            pointer_count = int(kwargs.get("num_k_exclude_rope", args[3] if len(args) > 3 else 0))
            pending = self._pending.pop(layer_index, None)
            if pending is None:
                raise RuntimeError(f"Layer {layer_index} cross-attention has no matching memory input")
            memory, position, layer_pointer_count = pending
            if pointer_count != layer_pointer_count:
                raise RuntimeError("Pointer-key count changed between the layer and attention call")
            if memory.shape != value.shape or key.shape != value.shape:
                raise RuntimeError("Memory, key, and value tensors are not aligned")
            if position is None:
                if not torch.equal(key, memory):
                    raise RuntimeError("Native key differs from memory without a position tensor")
            elif position.shape != memory.shape or not torch.equal(key, memory + position):
                raise RuntimeError("Native key is not aligned with memory plus its position encoding")

            spatial_count = memory.shape[1] - pointer_count
            if spatial_count <= 0:
                raise RuntimeError("Memory read contains no spatial tokens")
            # Each spatial frame must remain a complete native 64x64 grid.
            if spatial_count % 4096:
                raise RuntimeError(f"Spatial memory count {spatial_count} is not whole 4096-token frames")
            delta = self.heads.layers[layer_index](value, pointer_count, self.mode)
            if delta.shape != value[:, :spatial_count].shape:
                raise RuntimeError("Key adapter correction does not align with spatial tokens")
            new_key = torch.cat(
                (key[:, :spatial_count] + delta, key[:, spatial_count:]), dim=1
            )
            if not torch.equal(new_key[:, spatial_count:], key[:, spatial_count:]):
                raise RuntimeError("Key adapter modified object-pointer keys")
            if not torch.equal(value, memory):
                raise RuntimeError("Key adapter modified the native value input")

            self.calls.append({
                "layer": layer_index,
                "batch_objects": int(value.shape[0]),
                "query_tokens": int(self._get_tensor("q", args, kwargs, 0).shape[1]),
                "spatial_keys": int(spatial_count),
                "spatial_frames": int(spatial_count // 4096),
                "pointer_keys": int(pointer_count),
                "position_aligned_before_adapter": True,
                "values_unchanged": True,
                "pointer_keys_unchanged": True,
                "key_delta_l1": float(delta.detach().float().abs().sum().cpu()),
                "key_delta_nonzero": int(torch.count_nonzero(delta).item()),
                "object_id": self._active_object_id,
                "frame": self._active_frame,
            })
            if "k" in kwargs:
                return args, {**kwargs, "k": new_key}
            args = list(args)
            args[1] = new_key
            return tuple(args), kwargs
        return before

    def __enter__(self):
        layers = getattr(getattr(self.model, "memory_attention", None), "layers", None)
        if layers is None or len(layers) != 4:
            raise RuntimeError("Expected the four native SAM2 memory-attention layers")
        if len(self.heads.layers) != len(layers):
            raise RuntimeError("Adapter heads do not cover all native attention layers")
        if self.inference_state is not None:
            self._wrap_memory_prepare()
        for index, layer in enumerate(layers):
            attention = getattr(layer, "cross_attn_image", None)
            if attention is None:
                raise RuntimeError(f"Native layer {index} has no cross-attention module")
            self.handles.append(layer.register_forward_pre_hook(
                self._layer_pre_hook(index), with_kwargs=True
            ))
            self.handles.append(attention.register_forward_pre_hook(
                self._cross_attention_pre_hook(index), with_kwargs=True
            ))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._original_prepare is not None:
            self.model._prepare_memory_conditioned_features = self._original_prepare
            self._original_prepare = None
        if exc_type is None and self._pending:
            raise RuntimeError(f"Native layer inputs were not consumed: {sorted(self._pending)}")
        self._pending.clear()
