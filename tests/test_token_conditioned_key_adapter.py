import math

import torch
from torch import nn

from modeling.token_conditioned_key_adapter import (
    NativeTokenConditionedKeyAdapter,
    TokenConditionedKeyHeads,
    TokenConditionedKeyResidual,
)


def test_zero_initialized_residual_is_exact_zero_and_preserves_pointer_tail():
    memory = torch.randn(2, 8192 + 3, 64)
    adapter = TokenConditionedKeyResidual()
    delta = adapter(memory, pointer_count=3, mode="token")
    assert delta.shape == (2, 8192, 64)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_global_control_broadcasts_gate_but_keeps_token_specific_residual():
    torch.manual_seed(4)
    memory = torch.randn(2, 4096 + 2, 64)
    adapter = TokenConditionedKeyResidual()
    with torch.no_grad():
        adapter.up.weight.normal_(mean=0, std=0.05)
    token = adapter(memory, pointer_count=2, mode="token")
    global_gate = adapter(memory, pointer_count=2, mode="global")
    assert torch.isfinite(token).all() and torch.isfinite(global_gate).all()
    assert torch.count_nonzero(token) > 0
    assert not torch.equal(token, global_gate)
    # Both modes condition their token transform on the individual key.
    assert not torch.equal(token[:, 0], token[:, 1])
    assert not torch.equal(global_gate[:, 0], global_gate[:, 1])


def test_zero_up_projection_receives_first_gradient_then_gate_and_down_receive_gradient():
    torch.manual_seed(5)
    memory = torch.randn(1, 4096 + 2, 64)
    adapter = TokenConditionedKeyResidual()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
    target = torch.randn_like(memory[:, :-2])

    # A segmentation loss has a nonzero incoming gradient at the zero adapter;
    # delta**2 alone would have a zero derivative at exact initialization.
    first = (adapter(memory, pointer_count=2) * target).mean()
    first.backward()
    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.up.weight.grad).all()
    assert adapter.up.weight.grad.abs().sum() > 0
    assert adapter.down.weight.grad is None or adapter.down.weight.grad.abs().sum() == 0
    assert all(parameter.grad is None or parameter.grad.abs().sum() == 0
               for parameter in adapter.gate.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = adapter(memory, pointer_count=2).square().mean()
    second.backward()
    assert adapter.down.weight.grad is not None and adapter.down.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in adapter.gate.parameters())


class _FakeCrossAttention(nn.Module):
    def forward(self, q, k, v, num_k_exclude_rope=0):
        scores = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])
        weights = scores.softmax(dim=-1)
        return weights @ v


class _FakeMemoryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = _FakeCrossAttention()

    def forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
        key = memory if pos is None else memory + pos
        query = tgt if query_pos is None else tgt + query_pos
        return self.cross_attn_image(
            q=query, k=key, v=memory, num_k_exclude_rope=num_k_exclude_rope
        )


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_attention = nn.Module()
        self.memory_attention.layers = nn.ModuleList(_FakeMemoryLayer() for _ in range(4))

    def forward(self, memory, position, pointer_count):
        query = torch.randn(memory.shape[0], 3, memory.shape[-1], device=memory.device)
        return [layer(query, memory, position, num_k_exclude_rope=pointer_count)
                for layer in self.memory_attention.layers]


def test_native_hooks_preserve_zero_residual_parity_positions_values_and_pointer_keys():
    torch.manual_seed(6)
    model = _FakeModel()
    memory = torch.randn(2, 8192 + 3, 64)
    position = torch.randn_like(memory)
    heads = TokenConditionedKeyHeads()
    query = torch.randn(2, 3, 64)
    expected = []
    for layer in model.memory_attention.layers:
        expected.append(layer(query, memory, position, num_k_exclude_rope=3))

    with NativeTokenConditionedKeyAdapter(model, heads) as adapter:
        actual = [layer(query, memory, position, num_k_exclude_rope=3)
                  for layer in model.memory_attention.layers]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert [row["layer"] for row in adapter.calls] == [0, 1, 2, 3]
    assert all(row["spatial_keys"] == 8192 and row["spatial_frames"] == 2
               and row["pointer_keys"] == 3
               and row["position_aligned_before_adapter"]
               and row["values_unchanged"] and row["pointer_keys_unchanged"]
               and row["key_delta_l1"] == 0 and row["key_delta_nonzero"] == 0
               for row in adapter.calls)

    with torch.no_grad():
        for layer in heads.layers:
            layer.up.weight.normal_(mean=0, std=0.02)
    with NativeTokenConditionedKeyAdapter(model, heads) as adapter:
        _ = model(memory, position, pointer_count=3)
    assert all(row["key_delta_l1"] > 0 and row["key_delta_nonzero"] > 0
               for row in adapter.calls)


def test_batch_objects_do_not_share_pointer_context():
    torch.manual_seed(7)
    memory = torch.randn(2, 4096 + 1, 64)
    # Make the per-object pointer contexts clearly different.
    memory[0, -1].fill_(-2.0)
    memory[1, -1].fill_(2.0)
    adapter = TokenConditionedKeyResidual()
    with torch.no_grad():
        adapter.up.weight.normal_(mean=0, std=0.1)
    batched = adapter(memory, pointer_count=1)
    individually = torch.cat([
        adapter(memory[i:i + 1], pointer_count=1) for i in range(memory.shape[0])
    ], dim=0)
    assert torch.equal(batched, individually)
    assert not torch.equal(batched[0], batched[1])
