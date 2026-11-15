import pytest
import torch
import torch.nn.functional as F

from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss


def dense_reference(prediction, target):
    weight = 1 + 5 * (F.avg_pool2d(target, 31, stride=1, padding=15) - target).abs()
    bce = (weight * F.binary_cross_entropy_with_logits(prediction, target, reduction='none')).sum((-2, -1)) / weight.sum((-2, -1))
    p = prediction.sigmoid()
    inter = (weight * p * target).sum((-2, -1))
    union = (weight * (p + target - p * target)).sum((-2, -1))
    iou = 1 - (inter + 1) / (union + 1)
    dice = 1 - (2 * (p * target).sum((-2, -1)) + 1) / (p.sum((-2, -1)) + target.sum((-2, -1)) + 1)
    return (bce + iou + dice).mean()


@pytest.mark.parametrize('kind', ['empty', 'present', 'full'])
def test_matches_dense_author_adaptation_and_gradient(kind):
    torch.manual_seed(11)
    pred = torch.randn(2, 1, 40, 51, requires_grad=True)
    target = torch.zeros_like(pred)
    if kind == 'present':
        target[:, :, 8:29, 13:39] = 1
    if kind == 'full':
        target.fill_(1)
    actual = cmsa_boundary_loss(pred, target)
    expected = dense_reference(pred, target)
    grad = torch.autograd.grad(actual, pred, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad, expected_grad, atol=1e-7, rtol=1e-6)


def test_finite_suppressed_logits_and_empty_targets():
    pred = torch.full((1, 1, 32, 32), -1024.0, requires_grad=True)
    target = torch.zeros_like(pred)
    target[..., 10:20, 10:20] = 1
    loss = cmsa_boundary_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_rejects_invalid_geometry_and_target():
    pred = torch.zeros(1, 1, 32, 32)
    with pytest.raises(ValueError):
        cmsa_boundary_loss(pred, torch.zeros(1, 32, 32))
    with pytest.raises(ValueError):
        cmsa_boundary_loss(pred, torch.full_like(pred, 2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_loss_and_gradient_on_train_synthetic():
    torch.manual_seed(9)
    cpu = torch.randn(1, 1, 48, 53, requires_grad=True)
    tgt = torch.zeros_like(cpu)
    tgt[..., 8:35, 11:34] = 1
    gpu = cpu.detach().to('mps').requires_grad_()
    cpu_loss = cmsa_boundary_loss(cpu, tgt)
    gpu_loss = cmsa_boundary_loss(gpu, tgt.to('mps'))
    cpu_loss.backward()
    gpu_loss.backward()
    torch.testing.assert_close(cpu_loss, gpu_loss.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cpu.grad, gpu.grad.cpu(), rtol=1e-3, atol=1e-4)
