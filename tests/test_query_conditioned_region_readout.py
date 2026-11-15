import pytest
import torch
import torch.nn.functional as F

from modeling.query_conditioned_region_readout import QueryConditionedRegionReadout


def inputs():
    torch.manual_seed(17)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 7, 8)
    v = torch.randn_like(k)
    labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    return q, k, v, labels


def test_zero_query_gate_is_exact_native_sdpa_with_pointer_keys():
    q, k, v, labels = inputs()
    native = F.scaled_dot_product_attention(q, k, v)
    head = QueryConditionedRegionReadout(heads=1, head_dim=8)
    output, valid = head(native, q, k, v, labels, 1, query_content=q)
    assert torch.equal(output, native)
    assert valid.tolist() == [[True, True, True]]


def test_query_gate_has_finite_gradients_and_changes_regional_mass():
    q, k, v, labels = inputs()
    native = F.scaled_dot_product_attention(q, k, v)
    head = QueryConditionedRegionReadout(heads=1, head_dim=8)
    head.query_gate.data.normal_(std=0.15)
    output, _ = head(native, q, k, v, labels, 1, query_content=q)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, native)
    output.square().mean().backward()
    assert head.query_gate.grad is not None
    assert torch.isfinite(head.query_gate.grad).all()
    assert head.query_gate.grad.abs().sum() > 0


def test_query_gate_outputs_are_position_dependent():
    q, k, v, labels = inputs()
    native = F.scaled_dot_product_attention(q, k, v)
    head = QueryConditionedRegionReadout(heads=1, head_dim=8)
    head.query_gate.data.copy_(torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) / 10)
    query_content = q.clone()
    query_content[:, :, 1] *= -3
    output, _ = head(native, q, k, v, labels, 1, query_content=query_content)
    assert not torch.equal(output[:, :, :1], output[:, :, 1:2])


def test_empty_region_is_masked_without_nan():
    q, k, v, _ = inputs()
    labels = torch.tensor([[0, 0, 0, 1, 1, 1]])
    native = F.scaled_dot_product_attention(q, k, v)
    head = QueryConditionedRegionReadout(heads=1, head_dim=8)
    output, valid = head(native, q, k, v, labels, 1, query_content=q)
    assert valid.tolist() == [[True, True, False]]
    assert torch.isfinite(output).all()
    assert torch.equal(output, native)


def test_query_content_shape_must_match_projected_attention_query():
    q, k, v, labels = inputs()
    native = F.scaled_dot_product_attention(q, k, v)
    head = QueryConditionedRegionReadout(heads=1, head_dim=8)
    with pytest.raises(ValueError, match='projected query'):
        head(native, q, k, v, labels, 1, query_content=q[..., :-1])
