import pytest
import torch
import torch.nn.functional as F

from modeling.soft_role_affinity import (
    SoftRoleAffinityLayer,
    mask_logits_to_soft_roles,
    soft_mixed_cell_roles,
    target_to_soft_roles,
)
from modeling.soft_role_attention_training import balanced_soft_role_loss


def test_mixed_cell_roles_distinguish_pure_mixed_and_uniform_uncertain_cells():
    probability = torch.zeros(256, 256)
    probability[:, 130:] = 1
    probability[:, :4] = 0.5
    roles = soft_mixed_cell_roles(probability)

    assert roles.shape == (64, 64, 3)
    assert torch.allclose(roles.sum(-1), torch.ones(64, 64))
    assert torch.equal(roles[0, 8], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(roles[0, 40], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(roles[0, 0], torch.tensor([0.5, 0.5, 0.0]))
    assert torch.equal(roles[0, 32], torch.tensor([0.0, 0.0, 1.0]))


def test_prediction_and_training_role_encoders_share_the_same_geometry():
    logits = torch.full((1, 1, 256, 256), -12.0)
    logits[:, :, :, 128:] = 12
    predicted = mask_logits_to_soft_roles(logits)
    target = torch.zeros(1080, 1350)
    target[:, 675:] = 1
    supervised = target_to_soft_roles(target)

    assert predicted.shape == supervised.shape == (64, 64, 3)
    assert torch.allclose(predicted.sum(-1), torch.ones(64, 64))
    assert torch.allclose(supervised.sum(-1), torch.ones(64, 64))
    assert predicted[:, :32, 0].mean() > predicted[:, :32, 1].mean()
    assert predicted[:, 32:, 1].mean() > predicted[:, 32:, 0].mean()


@pytest.mark.parametrize("mode", ["regional", "global"])
def test_zero_bias_is_exact_native_and_keeps_gradient_path(mode):
    torch.manual_seed(4)
    batch, heads, queries, spatial, pointers, channels = 1, 2, 7, 11, 3, 8
    q = torch.randn(batch, heads, queries, channels, requires_grad=True)
    k = torch.randn(batch, heads, spatial + pointers, channels, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    memberships = torch.rand(batch, spatial, 3)
    memberships /= memberships.sum(-1, keepdim=True)
    head = SoftRoleAffinityLayer(heads, channels)
    sdpa = F.scaled_dot_product_attention
    native = sdpa(q, k, v, dropout_p=0.0)

    output, valid = head(
        native, q, k, v, memberships, pointers, mode,
        query_content=q, sdpa=sdpa,
    )
    assert torch.equal(output, native)
    assert valid.shape == (batch, 3)
    assert torch.allclose(head.last_query_roles.sum(-1), torch.ones(batch, heads, queries))

    output.square().mean().backward()
    assert head.role_bias.grad is not None
    assert torch.isfinite(head.role_bias.grad).all()
    assert head.role_bias.grad.abs().sum() > 0


def test_soft_role_attention_uses_all_keys_and_has_finite_gradients():
    torch.manual_seed(8)
    batch, heads, queries, spatial, pointers, channels = 1, 2, 9, 13, 2, 8
    q = torch.randn(batch, heads, queries, channels)
    k = torch.randn(batch, heads, spatial + pointers, channels)
    v = torch.randn_like(k)
    memberships = torch.rand(batch, spatial, 3)
    memberships /= memberships.sum(-1, keepdim=True)
    head = SoftRoleAffinityLayer(heads, channels)
    with torch.no_grad():
        head.role_bias.fill_(0.15)
    output, valid = head(
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.0),
        q, k, v, memberships, pointers, "regional", query_content=q,
        sdpa=F.scaled_dot_product_attention,
    )
    assert output.shape == q.shape
    assert valid.all()
    assert torch.isfinite(output).all()
    assert not torch.equal(output, F.scaled_dot_product_attention(q, k, v, dropout_p=0.0))


def test_invalid_or_incomplete_role_maps_fail_closed():
    with pytest.raises(ValueError):
        soft_mixed_cell_roles(torch.full((256, 256), 2.0))
    with pytest.raises(ValueError):
        mask_logits_to_soft_roles(torch.full((256, 256), float("nan")))
    with pytest.raises(ValueError):
        target_to_soft_roles(torch.full((16, 16), -1.0))


def test_role_loss_compensates_for_rare_boundary_without_thresholds():
    target = torch.zeros(1, 1, 100, 3)
    target[..., 0] = 1
    target[..., 1] = 0
    target[..., 2] = 0
    target[..., 1, :] = torch.tensor([0.0, 1.0, 0.0])
    target[..., 2, :] = torch.tensor([0.0, 0.0, 1.0])
    target[..., 3, :] = torch.tensor([0.0, 0.0, 1.0])
    logits = torch.zeros(1, 2, 100, 3, requires_grad=True)

    loss, frequency, weight = balanced_soft_role_loss(logits, target)
    assert torch.allclose(frequency, torch.tensor([0.97, 0.01, 0.02]))
    assert weight[1] > weight[2] > weight[0]
    assert torch.allclose(loss, torch.log(torch.tensor(3.0)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
