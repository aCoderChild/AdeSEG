import torch

from modeling.temporal_region_fusion import TemporalRegionFusion


def inputs(size=8):
    return (
        torch.randn(1, 256, size, size),
        torch.randn(1, 3, 256, size, size),
        torch.randn(1, 256, size, size),
        torch.randn(1, 256, size, size),
        torch.rand(1, 1, size, size),
        torch.randn(1, 256, size, size),
    )


def test_zero_initialized_fusion_is_exact_identity():
    module = TemporalRegionFusion().eval()
    native, reads, last, current, mask, difference = inputs()
    output, gate = module(native, reads, last, current, mask, difference)
    assert torch.equal(output, native)
    assert gate.shape == (1, 1, 8, 8)
    assert torch.isfinite(gate).all()


def test_fusion_training_reaches_regional_and_uncertainty_modules():
    torch.manual_seed(13)
    module = TemporalRegionFusion().train()
    values = inputs()
    output, _ = module(*values)
    (output.square().mean()).backward()
    assert module.regional_residual.weight.grad is not None
    assert torch.isfinite(module.regional_residual.weight.grad).all()
    initial_uncertainty_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in module.uncertainty.parameters()
        if parameter.grad is not None
    )
    assert initial_uncertainty_gradient == 0

    with torch.no_grad():
        module.regional_residual.weight.add_(module.regional_residual.weight.grad, alpha=-0.01)
    module.zero_grad(set_to_none=True)
    output, _ = module(*values)
    output.square().mean().backward()
    gradients = [p.grad for p in module.uncertainty.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_regional_residual_is_bounded_relative_to_native_read():
    module = TemporalRegionFusion(max_residual_rms_ratio=0.25).eval()
    with torch.no_grad():
        module.regional_residual.weight.fill_(100)
    native, reads, last, current, mask, difference = inputs()
    output, gate = module(native, reads, last, current, mask, difference)
    bound = 0.25 * native.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
    assert ((output - native).abs() <= bound + 1e-6).all()
