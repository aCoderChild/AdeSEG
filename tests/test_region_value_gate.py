import torch
import torch.nn.functional as F

from modeling.region_value_gate import RegionValueGate, RegionValueGateHeads


def test_zero_gate_is_exact_native_attention():
    torch.manual_seed(7)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 0, 2]])
    native = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    output, _ = RegionValueGate(heads=2, head_dim=8)(
        native, query, key, value, labels, 2, sdpa=F.scaled_dot_product_attention)
    assert torch.equal(output, native)


def test_gate_scales_only_spatial_values_with_bounded_region_gains():
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 6, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 1]])
    gate = RegionValueGate(heads=2, head_dim=8)
    with torch.no_grad():
        gate.gain_logits.fill_(0.5)
    calls = []

    def sdpa(q, k, v, **kwargs):
        calls.append((k.detach().clone(), v.detach().clone()))
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    native = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    output, _ = gate(native, query, key, value, labels, 2, sdpa=sdpa)
    adapted_key, adapted_value = calls[-1]
    assert torch.equal(adapted_key, key)
    assert torch.equal(adapted_value[..., 4:, :], value[..., 4:, :])
    ratios = adapted_value[..., :4, :] / value[..., :4, :]
    assert torch.all((ratios >= 0.75) & (ratios <= 1.25))
    output.square().mean().backward()
    assert gate.gain_logits.grad is not None
    assert torch.isfinite(gate.gain_logits.grad).all()
    assert gate.gain_logits.grad.abs().sum() > 0


def test_gate_heads_cover_all_four_layers():
    heads = RegionValueGateHeads()
    assert len(heads.layers) == 4
    assert all(torch.count_nonzero(layer.gain_logits) == 0 for layer in heads.layers)
