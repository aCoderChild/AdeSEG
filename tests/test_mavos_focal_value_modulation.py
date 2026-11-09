import torch
from torch import nn

from modeling.mavos_focal_value_modulation import (
    MAVOSFocalValueResidual,
    MAVOSMemoryValueModulation,
)
from modeling.mavos_value_contrastive import MAVOSValueContrastive


def test_zero_initialized_residual_is_exact_identity_and_preserves_pointers():
    adapter = MAVOSFocalValueResidual(8, (4, 4))
    memory = torch.randn(3 * 16 + 2, 2, 8)

    output = adapter(memory, pointer_tokens=2)

    assert torch.equal(output, memory)


def test_pointer_only_memory_is_passed_through_unchanged():
    adapter = MAVOSFocalValueResidual(8, (4, 4))
    pointers = torch.randn(3, 2, 8)
    assert torch.equal(adapter(pointers, pointer_tokens=3), pointers)


def test_context_runs_per_frame_and_backpropagates_only_through_adapter():
    adapter = MAVOSFocalValueResidual(8, (4, 4))
    with torch.no_grad():
        adapter.h.weight.normal_(0, 0.05)
        adapter.h.bias.normal_(0, 0.05)
    memory = torch.randn(2 * 16 + 3, 2, 8, requires_grad=True)

    output = adapter(memory, pointer_tokens=3)
    loss = output[:32].square().mean()
    loss.backward()

    assert output.shape == memory.shape
    assert torch.equal(output[-3:].detach(), memory[-3:].detach())
    assert torch.isfinite(output).all()
    assert adapter.f_context.weight.grad is not None
    assert torch.isfinite(adapter.f_context.weight.grad).all()
    assert memory.grad is not None


def test_focal_levels_are_chained_like_upstream_mavos():
    adapter = MAVOSFocalValueResidual(
        4, (3, 3), focal_window=3, focal_level=2, focal_factor=2
    )
    with torch.no_grad():
        adapter.h.weight.normal_(0, 0.1)
        adapter.h.bias.normal_(0, 0.1)
    memory = torch.randn(9, 1, 4)

    output = adapter(memory)

    grid = memory.permute(1, 2, 0).reshape(1, 4, 3, 3)
    projected = adapter.f_context(grid.permute(0, 2, 3, 1))
    projected = projected.permute(0, 3, 1, 2).contiguous()
    focal_context, gates = torch.split(projected, (4, 3), dim=1)
    aggregate = torch.zeros_like(focal_context)
    for level, focal_layer in enumerate(adapter.focal_layers):
        focal_context = focal_layer(focal_context)
        aggregate = aggregate + focal_context * gates[:, level:level + 1]
    global_context = adapter.act(focal_context.mean((2, 3), keepdim=True))
    aggregate = aggregate + global_context * gates[:, 2:3]
    expected = (grid + adapter.h(aggregate)).flatten(2).permute(2, 0, 1)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_spatially_shuffled_control_preserves_layout_and_pointer_tail():
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(7))
    adapter = MAVOSFocalValueResidual(8, (4, 4), spatial_permutation=permutation)
    memory = torch.randn(2 * 16 + 2, 1, 8)
    assert torch.equal(adapter(memory, pointer_tokens=2), memory)

    with torch.no_grad():
        adapter.h.weight.normal_(0, 0.1)
    output = adapter(memory, pointer_tokens=2)
    assert output.shape == memory.shape
    assert torch.equal(output[-2:], memory[-2:])
    assert torch.isfinite(output).all()


class _FakeCrossAttention(nn.Module):
    def __init__(self, channels=8, grid_tokens=16):
        super().__init__()
        self.kv_in_dim = channels
        self.freqs_cis = torch.zeros(grid_tokens, 1, dtype=torch.complex64)
        self.v_proj = nn.Linear(channels, channels)

    def forward(self, *, q, k, v, num_k_exclude_rope=0):
        del q, k, num_k_exclude_rope
        return self.v_proj(v)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = _FakeCrossAttention()


class _FakeModel(nn.Module):
    def __init__(self, batch_first=False):
        super().__init__()
        self.memory_attention = nn.Module()
        self.memory_attention.layers = nn.ModuleList(_FakeLayer() for _ in range(4))
        self.memory_attention.batch_first = batch_first


def test_native_value_hook_accepts_fixed_reference_buffers_without_parameters():
    from modeling.memory_value_oracle import FixedSpatialValues

    model = _FakeModel().requires_grad_(False)
    spatial = torch.randn(16, 1, 8)
    memory = torch.cat((spatial, torch.randn(2, 1, 8)))
    adapters = [FixedSpatialValues(spatial, spatial + 1, (4, 4)) for _ in range(4)]
    with MAVOSMemoryValueModulation(model, adapters, record_value_receipts=True) as integration:
        for layer in model.memory_attention.layers:
            layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=2)
    assert len(integration.value_calls) == 4
    assert all(v['pointer_tail_bit_identical'] for v in integration.value_calls)


def test_attention_hooks_keep_pointer_tail_and_match_identity_start():
    model = _FakeModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = [MAVOSFocalValueResidual(8, (4, 4)) for _ in range(4)]
    for adapter in adapters:
        with torch.no_grad():
            adapter.h.weight.normal_(0, 0.05)
            adapter.h.bias.normal_(0, 0.05)
    memory = torch.randn(2 * 16 + 3, 1, 8)
    expected = [
        layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=3)
        for layer in model.memory_attention.layers
    ]

    integration = MAVOSMemoryValueModulation(model, adapters, record_value_receipts=True)
    with integration:
        actual = [
            layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=3)
            for layer in model.memory_attention.layers
        ]
        sum(value[:-3].square().mean() for value in actual).backward()

    assert all(torch.equal(a[-3:], e[-3:]) for a, e in zip(actual, expected))
    assert not integration.context
    assert all(adapter.h.weight.grad is not None for adapter in adapters)
    assert all(torch.isfinite(adapter.h.weight.grad).all() for adapter in adapters)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [call["layer"] for call in integration.value_calls] == [0, 1, 2, 3]
    assert all(call["pointer_tail_bit_identical"] for call in integration.value_calls)


def test_batch_first_attention_layout_and_pointer_tail_are_supported():
    model = _FakeModel(batch_first=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = [MAVOSFocalValueResidual(8, (4, 4)) for _ in range(4)]
    for adapter in adapters:
        with torch.no_grad():
            adapter.h.weight.normal_(0, 0.05)
            adapter.h.bias.normal_(0, 0.05)
    memory = torch.randn(1, 2 * 16 + 3, 8)
    expected = [
        layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=3)
        for layer in model.memory_attention.layers
    ]

    integration = MAVOSMemoryValueModulation(model, adapters, record_value_receipts=True)
    with integration:
        actual = [
            layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=3)
            for layer in model.memory_attention.layers
        ]
        sum(value[:, :-3].square().mean() for value in actual).backward()

    assert all(torch.equal(a[:, -3:], e[:, -3:]) for a, e in zip(actual, expected))
    assert all(adapter.h.weight.grad is not None for adapter in adapters)
    assert all(torch.isfinite(adapter.h.weight.grad).all() for adapter in adapters)
    assert all(call["pointer_tail_bit_identical"] for call in integration.value_calls)


def test_batch_first_pointer_only_attention_is_unchanged():
    model = _FakeModel(batch_first=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = [MAVOSFocalValueResidual(8, (4, 4)) for _ in range(4)]
    memory = torch.randn(1, 3, 8)

    integration = MAVOSMemoryValueModulation(model, adapters, record_value_receipts=True)
    with integration:
        outputs = [
            layer.cross_attn_image(q=memory, k=memory, v=memory, num_k_exclude_rope=3)
            for layer in model.memory_attention.layers
        ]

    assert all(torch.equal(output, layer.cross_attn_image(q=memory, k=memory, v=memory,
                                                          num_k_exclude_rope=3))
               for output, layer in zip(outputs, model.memory_attention.layers))
    assert len(integration.value_calls) == 4
    assert all(call["spatial_tokens"] == 0 for call in integration.value_calls)
    assert all(call["pointer_tail_bit_identical"] for call in integration.value_calls)


def test_foreground_background_contrast_uses_aligned_memory_values_and_reaches_adapter():
    model = _FakeModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = [MAVOSFocalValueResidual(8, (4, 4)) for _ in range(4)]
    for adapter in adapters:
        with torch.no_grad():
            adapter.h.weight.normal_(0, 0.02)
            adapter.h.bias.normal_(0, 0.02)
    integration = MAVOSMemoryValueModulation(
        model, adapters, capture_adapted_values=True
    )
    memory = torch.randn(2 * 16 + 2, 1, 8)
    with integration:
        for layer in model.memory_attention.layers:
            layer.cross_attn_image(q=memory, k=memory, v=memory,
                                   num_k_exclude_rope=2)

        # Both masks have fractional object/background occupancy on a 4x4 grid.
        target_a = torch.zeros(1, 1, 16, 16)
        target_a[:, :, :, :8] = 1
        target_b = torch.zeros_like(target_a)
        target_b[:, :, 4:12, :8] = 1
        observer = type("Trace", (), {"calls": [{
            "frame": 3,
            "layers": [0, 1, 2, 3],
            "memory_frames": [
                {"frame": 0, "start": 0, "stop": 16},
                {"frame": 2, "start": 16, "stop": 32},
            ],
        }]})()
        auxiliary = MAVOSValueContrastive(
            integration, observer, {0: target_a, 2: target_b}, weight=0.1
        )
        loss = auxiliary({"frame": 3})
        assert torch.isfinite(loss)
        assert len(auxiliary.records) == 4
        assert all(record["contrastive_active"] for record in auxiliary.records)
        assert all(record["gt_used_only_after_forward"] for record in auxiliary.records)
        loss.backward()

    assert all(adapter.h.weight.grad is not None for adapter in adapters)
    assert all(torch.isfinite(adapter.h.weight.grad).all() for adapter in adapters)
    assert all(float(adapter.h.weight.grad.abs().sum()) > 0 for adapter in adapters)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not integration.adapted_values


def test_contrastive_term_is_zero_when_memory_has_no_cross_frame_positive():
    model = _FakeModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = [MAVOSFocalValueResidual(8, (4, 4)) for _ in range(4)]
    integration = MAVOSMemoryValueModulation(
        model, adapters, capture_adapted_values=True
    )
    memory = torch.randn(16 + 1, 1, 8)
    with integration:
        for layer in model.memory_attention.layers:
            layer.cross_attn_image(q=memory, k=memory, v=memory,
                                   num_k_exclude_rope=1)
        target = torch.zeros(1, 1, 16, 16)
        target[..., :, :8] = 1
        observer = type("Trace", (), {"calls": [{
            "frame": 2, "layers": [0, 1, 2, 3],
            "memory_frames": [{"frame": 0, "start": 0, "stop": 16}],
        }]})()
        auxiliary = MAVOSValueContrastive(integration, observer, {0: target})
        loss = auxiliary({"frame": 2})
    assert torch.equal(loss, torch.zeros_like(loss))
    assert len(auxiliary.records) == 4
    assert not any(record["contrastive_active"] for record in auxiliary.records)
