"""MAVOS-style focal context residual on SAM2 spatial memory values.

The context projection, gated depthwise focal levels, global context, and
projection follow the official MAVOS JointFocalAttention implementation
(``external/MAVOS/networks/layers/externel_utils.py``, commit
``e30876fbd248ebf91c7fff6b9606a70235f38944``). This SAM2 adaptation applies
that context path independently to each stored spatial frame and adds it to
the original memory value input before SAM2's frozen ``v_proj``. SAM2 query,
key, RoPE, attention, pointer values, and decoder are not changed.

This is a research candidate, not the full MAVOS architecture or a claim that
MAVOS's reported scores transfer to SAM2.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import torch
from torch import Tensor, nn


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode())
    hasher.update(str(tuple(tensor.shape)).encode())
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


class MAVOSFocalValueResidual(nn.Module):
    """Add a learned local/global focal residual to each memory token.

    ``memory`` is SAM2's sequence-first ``[tokens, batch, channels]`` value
    input. Spatial tokens are concatenated as complete ``height*width`` frame
    blocks and object pointers are the unchanged tail.
    """

    def __init__(
        self,
        channels: int,
        grid_shape: tuple[int, int] = (64, 64),
        focal_window: int = 3,
        focal_level: int = 2,
        focal_factor: int = 2,
        bias: bool = True,
        spatial_permutation: Tensor | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0 or min(grid_shape) <= 0:
            raise ValueError("channels and grid dimensions must be positive")
        if focal_window <= 0 or focal_level <= 0 or focal_factor <= 0:
            raise ValueError("focal configuration values must be positive")
        self.channels = int(channels)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.focal_level = int(focal_level)
        token_count = self.grid_shape[0] * self.grid_shape[1]
        if spatial_permutation is None:
            self.register_buffer("spatial_permutation", None)
            self.register_buffer("inverse_spatial_permutation", None)
        else:
            permutation = torch.as_tensor(spatial_permutation, dtype=torch.long)
            expected = torch.arange(token_count, dtype=torch.long)
            if permutation.shape != (token_count,) or not torch.equal(permutation.sort().values, expected):
                raise ValueError("spatial_permutation must permute every grid position once")
            inverse = torch.empty_like(permutation)
            inverse[permutation] = expected
            self.register_buffer("spatial_permutation", permutation)
            self.register_buffer("inverse_spatial_permutation", inverse)

        # Direct adaptation of MAVOS JointFocalAttention's focal path.
        self.f_context = nn.Linear(channels, channels + focal_level + 1, bias=bias)
        self.focal_layers = nn.ModuleList()
        for level in range(focal_level):
            kernel = focal_factor * level + focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        groups=channels,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
        self.act = nn.GELU()
        self.h = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

        # Identity start preserves exact native SAM2 outputs before training.
        nn.init.zeros_(self.h.weight)
        if self.h.bias is not None:
            nn.init.zeros_(self.h.bias)

    def _modulate_grid(self, grids: Tensor) -> Tensor:
        """Apply upstream MAVOS focal-context equations to ``[N,C,H,W]``."""
        _, channels, _, _ = grids.shape
        projected = self.f_context(grids.permute(0, 2, 3, 1))
        projected = projected.permute(0, 3, 1, 2).contiguous()
        context, gates = torch.split(
            projected, (channels, self.focal_level + 1), dim=1
        )

        focal_context = context
        aggregate = torch.zeros_like(context)
        for level, focal_layer in enumerate(self.focal_layers):
            # MAVOS chains its focal levels: each larger receptive field sees
            # the previous level's representation, rather than reprocessing
            # the original context independently.
            focal_context = focal_layer(focal_context)
            aggregate = aggregate + focal_context * gates[:, level:level + 1]
        global_context = self.act(focal_context.mean(dim=(2, 3), keepdim=True))
        aggregate = aggregate + global_context * gates[:, self.focal_level:]
        return grids + self.h(aggregate)

    def forward(self, memory: Tensor, pointer_tokens: int = 0) -> Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must have shape [tokens, batch, channels]")
        tokens, batch, channels = memory.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if tokens == 0 or not 0 <= pointer_tokens <= tokens:
            raise ValueError("pointer_tokens must be within the memory sequence")
        # SAM2 can read pointer tokens before any spatial memory frame exists.
        # With no spatial grid to process, preserve this pointer-only read.
        if pointer_tokens == tokens:
            return memory

        height, width = self.grid_shape
        frame_tokens = height * width
        spatial_tokens = tokens - pointer_tokens
        if spatial_tokens <= 0 or spatial_tokens % frame_tokens:
            raise ValueError(
                f"{spatial_tokens} spatial tokens do not form complete "
                f"{height}x{width} frame blocks"
            )

        frame_count = spatial_tokens // frame_tokens
        spatial = memory[:spatial_tokens]
        # [F*HW,B,C] -> [F*B,C,H,W], keeping each frame/object independent.
        grids = (
            spatial.reshape(frame_count, height, width, batch, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(frame_count * batch, channels, height, width)
        )
        if self.spatial_permutation is not None:
            flat = grids.flatten(2).index_select(2, self.spatial_permutation)
            shuffled = flat.reshape(-1, channels, height, width)
            modulated = self._modulate_grid(shuffled).flatten(2)
            updated = modulated.index_select(2, self.inverse_spatial_permutation)
            updated = updated.reshape(-1, channels, height, width)
        else:
            updated = self._modulate_grid(grids)
        updated = (
            updated.reshape(frame_count, batch, channels, height, width)
            .permute(0, 3, 4, 1, 2)
            .reshape(spatial_tokens, batch, channels)
        )
        if pointer_tokens:
            return torch.cat((updated, memory[spatial_tokens:]), dim=0)
        return updated


class MAVOSMemoryValueModulation:
    """Install trainable residuals on all four SAM2 memory cross-attention layers.

    The hook observes SAM2's own ``num_k_exclude_rope`` argument so pointers
    are kept outside spatial modulation. Adapter parameters live in the
    supplied module sequence, not in the frozen SAM2 model.
    """

    def __init__(
        self,
        model: nn.Module,
        adapters: Sequence[MAVOSFocalValueResidual],
        *,
        record_value_receipts: bool = False,
        capture_adapted_values: bool = False,
    ):
        self.model = model
        self.adapters = tuple(adapters)
        self.record_value_receipts = bool(record_value_receipts)
        self.capture_adapted_values = bool(capture_adapted_values)
        self.value_calls: list[dict] = []
        # Training-only live tensors, consumed by the auxiliary objective
        # immediately after each model.track_step. Never serialized or used at
        # inference. Values are stored in SAM2's sequence-first layout.
        self.adapted_values: list[dict] = []
        self.handles = []
        self.context: dict[int, tuple[int, Tensor]] = {}
        self.batch_first = False

    def _attention_pre_hook(self, index: int):
        def hook(module, args, kwargs):
            if args or "v" not in kwargs:
                raise RuntimeError("SAM2 memory cross-attention must use keyword inputs")
            if index in self.context:
                raise RuntimeError("Nested cross-attention call for one memory layer")
            memory = kwargs["v"]
            pointers = int(kwargs.get("num_k_exclude_rope", 0))
            self.context[index] = (pointers, memory)
        return hook

    def _value_pre_hook(self, index: int):
        def hook(module, args, kwargs):
            active = self.context.pop(index, None)
            if active is None or len(args) != 1 or kwargs:
                raise RuntimeError("Value projection was not paired with SAM2 cross-attention")
            pointers, memory = active
            if args[0] is not memory:
                raise RuntimeError("Value projection input differs from SAM2 memory values")
            memory_sequence_first = (
                memory.transpose(0, 1) if self.batch_first else memory
            )
            adapted_sequence_first = self.adapters[index](
                memory_sequence_first, pointer_tokens=pointers
            )
            if self.capture_adapted_values:
                self.adapted_values.append({
                    "layer": index,
                    "adapted": adapted_sequence_first,
                    "pointer_tokens": pointers,
                })
            adapted = (
                adapted_sequence_first.transpose(0, 1)
                if self.batch_first else adapted_sequence_first
            )
            if self.record_value_receipts:
                spatial_tokens = memory_sequence_first.shape[0] - pointers
                frame_tokens = self.adapters[index].grid_shape[0] * self.adapters[index].grid_shape[1]
                before = memory_sequence_first[spatial_tokens:]
                after = adapted_sequence_first[spatial_tokens:]
                self.value_calls.append({
                    "layer": index,
                    "input_shape": list(memory.shape),
                    "spatial_tokens": spatial_tokens,
                    "frame_blocks": spatial_tokens // frame_tokens,
                    "pointer_tokens": pointers,
                    "pointer_tail_bit_identical": _tensor_sha256(before) == _tensor_sha256(after),
                    "pointer_input_sha256": _tensor_sha256(before),
                    "pointer_output_sha256": _tensor_sha256(after),
                })
            return (adapted,), kwargs
        return hook

    def _attention_post_hook(self, index: int):
        def hook(module, args, kwargs, output):
            if index in self.context:
                self.context.pop(index)
                raise RuntimeError("SAM2 cross-attention skipped its native value projection")
        return hook

    def __enter__(self):
        # A context may be reused by a caller; never let a prior training
        # chunk's autograd tensors leak into a later chunk.
        self.adapted_values.clear()
        self.value_calls.clear()
        self.context.clear()
        memory_attention = getattr(self.model, "memory_attention", None)
        layers = getattr(memory_attention, "layers", None)
        if layers is None or len(layers) != 4 or len(self.adapters) != 4:
            raise RuntimeError("Expected four SAM2 memory-attention layers and adapters")
        self.batch_first = bool(getattr(memory_attention, "batch_first", False))
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Freeze all pretrained SAM2 parameters before installation")
        attentions = []
        for index, layer in enumerate(layers):
            attention = layer.cross_attn_image
            if not hasattr(attention, "v_proj"):
                raise RuntimeError("Cross-attention has no native value projection")
            if self.adapters[index].channels != attention.kv_in_dim:
                raise RuntimeError("Adapter channels differ from SAM2 memory-value input")
            grid_tokens = self.adapters[index].grid_shape[0] * self.adapters[index].grid_shape[1]
            if attention.freqs_cis.shape[0] != grid_tokens:
                raise RuntimeError("Adapter grid differs from SAM2 rotary spatial grid")
            adapter_devices = {parameter.device for parameter in self.adapters[index].parameters()}
            if adapter_devices != {attention.v_proj.weight.device}:
                raise RuntimeError("Move each adapter to its SAM2 attention device before installation")
            attentions.append(attention)
        for index, attention in enumerate(attentions):
            self.handles.extend((
                attention.register_forward_pre_hook(
                    self._attention_pre_hook(index), with_kwargs=True
                ),
                attention.v_proj.register_forward_pre_hook(
                    self._value_pre_hook(index), with_kwargs=True
                ),
                attention.register_forward_hook(
                    self._attention_post_hook(index), with_kwargs=True
                ),
            ))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.context.clear()
        self.adapted_values.clear()
