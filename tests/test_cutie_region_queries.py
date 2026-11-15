import torch

from modeling.cutie_region_queries import CutieRegionQueryCore, CutieRegionQueryResidual


def inputs():
    torch.manual_seed(9)
    query = torch.randn(1, 4, 16, 8, requires_grad=True)
    key = torch.randn(1, 4, 20, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
    native = torch.randn_like(query)
    return native, query, key, value, labels


def test_zero_scale_is_exact_identity_with_cutie_spatial_encoding():
    native, query, key, value, labels = inputs()
    module = CutieRegionQueryResidual(CutieRegionQueryCore(channels=32, heads=4, feedforward=64))
    output, valid = module(native, query, key, value, labels, 4)
    assert torch.equal(output, native)
    assert valid.tolist() == [[True, True, True]]


def test_nonzero_residual_trains_region_queries_and_author_block():
    native, query, key, value, labels = inputs()
    module = CutieRegionQueryResidual(CutieRegionQueryCore(channels=32, heads=4, feedforward=64))
    with torch.no_grad():
        module.scale.fill_(0.1)
    output, _ = module(native, query, key, value, labels, 4)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert module.scale.grad is not None and module.scale.grad.abs() > 0
    assert module.region_queries.grad is not None
    assert module.region_queries.grad.abs().sum() > 0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in module._core.block.parameters())


def test_generic_control_ignores_region_assignments():
    native, query, key, value, labels = inputs()
    module = CutieRegionQueryResidual(CutieRegionQueryCore(channels=32, heads=4, feedforward=64))
    with torch.no_grad():
        module.scale.fill_(0.1)
        first, _ = module(native, query, key, value, labels, 4, 'generic')
        second, _ = module(native, query, key, value, labels.roll(3, -1), 4, 'generic')
    assert torch.equal(first, second)


def test_region_summary_excludes_pointer_values():
    native, query, key, value, labels = inputs()
    module = CutieRegionQueryResidual(CutieRegionQueryCore(channels=32, heads=4, feedforward=64))
    with torch.no_grad():
        module.scale.fill_(0.1)
        first, _ = module(native, query, key, value, labels, 4)
        value[..., -4:, :] = 1e6
        second, _ = module(native, query, key, value, labels, 4)
    assert torch.equal(first, second)
