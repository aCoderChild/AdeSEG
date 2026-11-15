import torch
import torch.nn.functional as F

from modeling.region_readout import RegionalReadoutResidual, regional_readouts


def test_regions_match_individual_sdpa_and_ignore_pointers():
    torch.manual_seed(9)
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k = torch.randn(1, 2, 6, 4, requires_grad=True)
    v = torch.randn(1, 2, 6, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 0, 2]])
    outputs, valid = regional_readouts(q, k, v, labels, pointer_tokens=2)
    assert valid.tolist() == [[True, True, True]]
    for i in range(3):
        selected = labels[0].eq(i).nonzero().flatten()
        expected = F.scaled_dot_product_attention(q, k[:, :, selected], v[:, :, selected])
        assert torch.allclose(outputs[:, i], expected, atol=1e-6)
    outputs.sum().backward()
    assert q.grad.abs().sum() > 0
    assert v.grad[:, :, -2:].count_nonzero() == 0


def test_empty_region_is_zero_and_finite_per_batch():
    q = torch.randn(2, 1, 2, 3)
    k = torch.randn(2, 1, 3, 3)
    v = torch.randn_like(k)
    outputs, valid = regional_readouts(q, k, v, torch.tensor([[0, 0, 0], [1, 2, 2]]), 0)
    assert valid.tolist() == [[True, False, False], [False, True, True]]
    assert torch.isfinite(outputs).all()
    assert outputs[0, 1:].count_nonzero() == 0
    assert outputs[1, 0].count_nonzero() == 0


def test_zero_residual_preserves_native_attention_and_has_gradient():
    torch.manual_seed(12)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn_like(k)
    labels = torch.tensor([[0, 1, 0, 2]])
    native = F.scaled_dot_product_attention(q, k, v)
    residual = RegionalReadoutResidual(4)
    out, valid = residual(native, q, k, v, labels, 2)
    assert torch.equal(out, native)
    assert valid.tolist() == [[True, True, True]]
    out.sum().backward()
    assert torch.isfinite(residual.mix.weight.grad).all()
    assert residual.mix.weight.grad.abs().sum() > 0


def test_generic_context_does_not_depend_on_region_assignment():
    torch.manual_seed(17)
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 6, 4)
    v = torch.randn_like(k)
    native = F.scaled_dot_product_attention(q, k, v)
    residual = RegionalReadoutResidual(4)
    torch.nn.init.normal_(residual.mix.weight)
    first, _ = residual(native, q, k, v, torch.tensor([[0, 1, 0, 2]]), 2, 'generic')
    second, _ = residual(native, q, k, v, torch.tensor([[2, 2, 1, 1]]), 2, 'generic')
    assert torch.equal(first, second)
