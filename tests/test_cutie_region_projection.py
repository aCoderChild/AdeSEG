import torch

from modeling.cutie_region_projection import CutieRegionProjectionHead


def _inputs():
    query = torch.randn((1, 8, 4, 32))
    key = torch.randn((1, 8, 4098, 32))
    value = torch.randn_like(key)
    native = torch.randn((1, 8, 4, 32))
    labels = torch.zeros((1, 4096), dtype=torch.long)
    labels[:, 1024:1536] = 1
    labels[:, 1536:2048] = 2
    return native, query, key, value, labels


def test_zero_projection_is_exact_native_identity_and_projection_learns_first():
    torch.set_num_threads(1)
    module = CutieRegionProjectionHead(layer_index=0).train()
    native, query, key, value, labels = _inputs()
    output, valid = module(native, query, key, value, labels, 2)
    assert torch.equal(output, native)
    assert valid.all()
    output.square().mean().backward()
    assert module.output_projection.weight.grad is not None
    assert torch.isfinite(module.output_projection.weight.grad).all()
    assert module.output_projection.weight.grad.abs().sum() > 0


def test_reader_learns_after_zero_projection_has_updated():
    torch.set_num_threads(1)
    module = CutieRegionProjectionHead(layer_index=0).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    native, query, key, value, labels = _inputs()
    output, _ = module(native, query, key, value, labels, 2)
    output.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output, _ = module(native, query, key, value, labels, 2)
    output.square().mean().backward()
    reader_grad = sum(p.grad.abs().sum() for p in module.reader.parameters() if p.grad is not None)
    assert torch.isfinite(reader_grad)
    assert reader_grad > 0
    assert module.scale_logit.grad is None
