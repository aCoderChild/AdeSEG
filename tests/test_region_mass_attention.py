import torch
import torch.nn.functional as F

from modeling.region_mass_attention import RegionMassBias, redistribute_region_mass


def test_zero_bias_is_bit_identical_and_has_nonzero_bias_gradient():
    torch.manual_seed(3)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 1, 0, 2, 1]])
    native = F.scaled_dot_product_attention(query, key, value)
    head = RegionMassBias()
    output, valid = head(native, query, key, value, labels, 0)
    assert torch.equal(output, native)
    assert valid.tolist() == [[True, True, True]]
    output.square().sum().backward()
    assert torch.isfinite(head.log_bias.grad).all()
    assert head.log_bias.grad.abs().sum() > 0


def test_pointer_value_gradient_is_unchanged_by_regional_bias():
    torch.manual_seed(7)
    query = torch.randn(1, 2, 3, 6)
    key = torch.randn(1, 2, 8, 6)
    value = torch.randn(1, 2, 8, 6, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 1, 0, 2]])
    bias = torch.tensor([0.4, -0.2, 0.7])
    native = F.scaled_dot_product_attention(query, key, value)
    native_grad = torch.autograd.grad(native.sum(), value, retain_graph=True)[0][:, :, -2:]
    regional, _ = redistribute_region_mass(query, key, value, labels, 2, bias)
    regional_grad = torch.autograd.grad(regional.sum(), value)[0][:, :, -2:]
    assert torch.allclose(regional_grad, native_grad, atol=1e-6, rtol=1e-6)


def test_missing_regions_are_finite_and_zero_bias_without_pointers_is_native():
    torch.manual_seed(11)
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn_like(key)
    labels = torch.zeros(1, 4, dtype=torch.long)
    native = F.scaled_dot_product_attention(query, key, value)
    output, valid = redistribute_region_mass(query, key, value, labels, 0, torch.zeros(3))
    assert torch.equal(output, native)
    assert valid.tolist() == [[True, False, False]]
    assert torch.isfinite(output).all()


def test_region_bias_redistributes_only_when_regions_are_present():
    torch.manual_seed(17)
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 0, 1, 1]])
    output, _ = redistribute_region_mass(query, key, value, labels, 0, torch.tensor([1., -1., 0.]))
    native = F.scaled_dot_product_attention(query, key, value)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, native)


def test_per_query_region_bias_is_supported():
    torch.manual_seed(23)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 0]])
    bias = torch.nn.Parameter(torch.randn(1, 2, 3, 3) * 0.1)
    output, valid = redistribute_region_mass(query, key, value, labels, 1, bias)
    assert output.shape == query.shape
    assert valid.tolist() == [[True, True, True]]
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert bias.grad is not None
    assert torch.isfinite(bias.grad).all()
    assert bias.grad.abs().sum() > 0


def test_region_bias_has_a_fixed_twofold_bound():
    head = RegionMassBias()
    with torch.no_grad():
        head.log_bias[:] = torch.tensor([-100., 0., 100.])
    bounded = head.max_log_scale * torch.tanh(head.log_bias)
    assert bounded.abs().max() <= torch.log(torch.tensor(2.0))
