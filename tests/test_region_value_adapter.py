import torch
import torch.nn.functional as F

from modeling.region_value_adapter import RegionValueAdapter, RegionValueHeads


def test_zero_adapter_reproduces_native_attention_exactly():
    torch.manual_seed(3)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 0, 1]])
    native = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    output, valid = RegionValueAdapter(heads=2, head_dim=8)(
        native, query, key, value, labels, 2,
        sdpa=F.scaled_dot_product_attention)
    assert torch.equal(output, native)
    assert valid.all()


def test_only_spatial_values_change_and_role_embeddings_get_gradients():
    torch.manual_seed(5)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 6, 8)
    value = torch.randn_like(key)
    labels = torch.tensor([[0, 1, 2, 1]])
    adapter = RegionValueAdapter(heads=2, head_dim=8)
    with torch.no_grad():
        adapter.role_embeddings.fill_(0.4)
    calls = []

    def sdpa(q, k, v, **kwargs):
        calls.append((k.detach().clone(), v.detach().clone()))
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    native = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    output, _ = adapter(native, query, key, value, labels, 2, sdpa=sdpa)
    adapted_key, adapted_value = calls[-1]
    assert torch.equal(adapted_key, key)
    assert torch.equal(adapted_value[..., 4:, :], value[..., 4:, :])
    assert torch.all((adapted_value[..., :4, :] - value[..., :4, :]).abs() <=
                     0.1 * value[..., :4, :].float().square().mean(dim=-2, keepdim=True).sqrt() + 1e-6)
    output.square().mean().backward()
    assert adapter.role_embeddings.grad is not None
    assert torch.isfinite(adapter.role_embeddings.grad).all()
    assert adapter.role_embeddings.grad.abs().sum() > 0


def test_region_heads_cover_all_four_layers():
    heads = RegionValueHeads()
    assert len(heads.layers) == 4
    assert all(torch.count_nonzero(layer.role_embeddings) == 0 for layer in heads.layers)
