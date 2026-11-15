import torch

from modeling.cutie_region_prototypes import CutieRegionMessageHeads, CutieRegionPrototypes


def test_official_cutie_summarizer_emits_spatial_region_prototypes():
    torch.set_num_threads(1)
    model = CutieRegionPrototypes().eval()
    labels = torch.zeros((1, 64, 64), dtype=torch.long)
    labels[:, 20:24, 20:24] = 1
    labels[:, 24:28, 20:24] = 2
    values = torch.randn((1, 256, 64, 64))
    features, areas = model(values, labels)
    assert features.shape == (1, 3, 8, 256)
    assert areas.shape == (1, 3, 8)
    assert torch.isfinite(features).all() and torch.isfinite(areas).all()
    assert (areas > 0).all()
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_zero_scale_cutie_message_is_native_identity_and_scale_trains():
    torch.set_num_threads(1)
    module = CutieRegionMessageHeads(layer_index=0).train()
    query = torch.randn((1, 8, 4, 32))
    key = torch.randn((1, 8, 4098, 32))
    value = torch.randn_like(key)
    native = torch.randn((1, 8, 4, 32))
    labels = torch.zeros((1, 4096), dtype=torch.long)
    labels[:, 1024:1536] = 1
    labels[:, 1536:2048] = 2
    output, valid = module(native, query, key, value, labels, 2)
    assert torch.equal(output, native)
    assert valid.all()
    output.square().mean().backward()
    assert module.scale_logit.grad is not None
    assert torch.isfinite(module.scale_logit.grad)
    assert module.scale_logit.grad.abs() > 0
