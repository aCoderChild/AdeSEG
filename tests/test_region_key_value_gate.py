import torch

from modeling.region_key_value_gate import RegionKeyValueGate, RegionKeyValueGateHeads


def test_zero_gain_is_exact_native_memory():
    memory = torch.randn(1, 4096 + 2, 8, dtype=torch.bfloat16)
    labels = torch.randint(0, 3, (1, 4096))
    gate = RegionKeyValueGate(channels=8)
    adapted = gate(memory, labels, 2)
    assert torch.equal(adapted, memory)


def test_gate_modulates_spatial_memory_but_not_pointers_or_positions():
    memory = torch.randn(1, 8, 8, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 1, 2, 0]])
    gate = RegionKeyValueGate(channels=8)
    with torch.no_grad():
        gate.gain_logits.fill_(0.5)
    adapted = gate(memory, labels, 2)
    assert torch.equal(adapted[:, -2:], memory[:, -2:])
    gains = adapted[:, :6] / memory[:, :6]
    assert torch.all((gains >= 0.75) & (gains <= 1.25))
    adapted.square().mean().backward()
    assert gate.gain_logits.grad is not None
    assert torch.isfinite(gate.gain_logits.grad).all()
    assert gate.gain_logits.grad.abs().sum() > 0


def test_joint_key_value_heads_cover_all_four_layers():
    heads = RegionKeyValueGateHeads(channels=8)
    assert len(heads.layers) == 4
    assert all(torch.count_nonzero(layer.gain_logits) == 0 for layer in heads.layers)
