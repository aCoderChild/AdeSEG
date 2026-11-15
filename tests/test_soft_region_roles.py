import torch
import torch.nn.functional as F

from modeling.soft_region_roles import predicted_soft_regions
from modeling.soft_region_value_adapter import SoftRegionValueAdapter


def test_soft_roles_are_exhaustive_and_continuous():
    logits = torch.full((256, 256), -100.0)
    logits[:, 126:] = 100.0
    roles = predicted_soft_regions(logits)
    assert roles.shape == (64, 64, 3)
    assert torch.allclose(roles.sum(-1), torch.ones(64, 64))
    assert torch.allclose(roles[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(roles[0, -1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(roles[0, 31], torch.tensor([0.25, 0.25, 0.5]))


def test_uncertain_mask_logits_produce_soft_roles_without_thresholding():
    logits = torch.zeros((256, 256))
    roles = predicted_soft_regions(logits)
    assert torch.allclose(roles, torch.tensor([0.25, 0.25, 0.5]).expand_as(roles))


def test_zero_role_embeddings_reproduce_native_attention_exactly():
    torch.manual_seed(4)
    adapter = SoftRegionValueAdapter(heads=2, head_dim=8)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 11, 8)
    value = torch.randn(1, 2, 11, 8)
    memberships = torch.softmax(torch.randn(1, 9, 3), dim=-1)
    native = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    result, valid = adapter(native, query, key, value, memberships, 2, sdpa=F.scaled_dot_product_attention)
    assert torch.equal(result, native)
    assert valid.all()


def test_soft_role_embeddings_are_differentiable_and_leave_pointers_unchanged():
    torch.manual_seed(5)
    adapter = SoftRegionValueAdapter(heads=2, head_dim=8)
    with torch.no_grad():
        adapter.role_embeddings.fill_(0.2)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 11, 8)
    value = torch.randn(1, 2, 11, 8)
    original = value.clone()
    memberships = torch.softmax(torch.randn(1, 9, 3), dim=-1)
    output, _ = adapter(None, query, key, value, memberships, 2, sdpa=F.scaled_dot_product_attention)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(adapter.role_embeddings.grad).all()
    assert adapter.role_embeddings.grad.abs().sum() > 0
    assert torch.equal(value, original)


def test_shared_embedding_component_cannot_act_as_generic_residual():
    torch.manual_seed(6)
    adapter = SoftRegionValueAdapter(heads=2, head_dim=8)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 11, 8)
    value = torch.randn(1, 2, 11, 8)
    memberships = torch.softmax(torch.randn(1, 9, 3), dim=-1)
    with torch.no_grad():
        adapter.role_embeddings.copy_(torch.randn_like(adapter.role_embeddings))
    first, _ = adapter(None, query, key, value, memberships, 2,
                       sdpa=F.scaled_dot_product_attention)
    with torch.no_grad():
        adapter.role_embeddings.add_(torch.randn(2, 1, 8))
    second, _ = adapter(None, query, key, value, memberships, 2,
                        sdpa=F.scaled_dot_product_attention)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
