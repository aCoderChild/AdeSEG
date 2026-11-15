import torch

from modeling.temporal_region_fusion_change import (
    ChangeSupervisedRegionFusion,
    ChangeSupervisedRegionHeads,
    temporal_change_loss,
)


def _inputs(channels=8, batch=2):
    native = torch.randn(batch, channels, 8, 8)
    regions = torch.randn(batch, 3, channels, 8, 8)
    features = [torch.randn_like(native) for _ in range(3)]
    mask = torch.randint(0, 2, (batch, 1, 8, 8)).float()
    return native, regions, *features, mask


def test_zero_mix_is_exact_native_output():
    head = ChangeSupervisedRegionFusion(channels=8).train()
    native, regions, recent, current, difference, mask = _inputs()
    output, probability = head(native, regions, recent, current, mask, difference)
    assert torch.equal(output, native)
    assert probability.shape == mask.shape


def test_temporal_change_supervision_trains_official_predictor():
    heads = ChangeSupervisedRegionHeads(channels=8, layers=1).train()
    samples = []
    prompt_target = torch.zeros(1, 1, 32, 32)
    for shift in (2, 5):
        native, regions, recent, current, difference, mask = _inputs(batch=1)
        heads.layers[0].mix_logit.data.fill_(0.1)
        output, _ = heads.layers[0](native, regions, recent, current, mask, difference)
        samples.append({"target": torch.roll(prompt_target, shift, dims=-1)})
    loss = temporal_change_loss(heads, samples, prompt_target)
    loss = loss + output.square().mean()
    loss.backward()
    uncertainty_grad = heads.layers[0].uncertainty.conv1x1_v2.weight.grad
    assert uncertainty_grad is not None and torch.isfinite(uncertainty_grad).all()
    assert uncertainty_grad.abs().sum() > 0
    assert heads.layers[0].regional_residual.weight.grad is not None
    assert heads.layers[0].regional_residual.weight.grad.abs().sum() > 0
