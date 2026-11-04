import pytest
import torch

from modeling.regional_memory_prompts import NativeRegionalMemoryPrompts, RegionalMemoryPrompts


def fixture():
    torch.manual_seed(17)
    heads = RegionalMemoryPrompts(channels=4, tokens_per_region=2)
    value = torch.randn(2, 8, 4)
    key = value + torch.randn_like(value)
    roles = torch.softmax(torch.randn(2, 8, 3), -1)
    spans = [{"start": 0, "stop": 4}, {"start": 4, "stop": 8}]
    return heads, key, value, roles, spans


def test_author_pooling_is_continuous_finite_and_instance_independent():
    heads, key, value, roles, spans = fixture()
    k, v = heads(key, value, roles, spans)
    assert k.shape == v.shape == (2, 12, 4)
    changed = value.clone()
    changed[1] += 10
    k2, v2 = heads(key, changed, roles, spans)
    assert torch.equal(k[0], k2[0]) and torch.equal(v[0], v2[0])
    empty = torch.zeros_like(roles)
    empty[..., 0] = 1
    assert all(torch.isfinite(x).all() for x in heads(key, value, empty, spans))
    perturbed = roles * 0.999 + 0.001 / 3
    assert max((a-b).abs().max().item() for a,b in zip((k,v),heads(key,value,perturbed,spans))) < 0.02


def test_only_added_author_learners_receive_gradients():
    heads, key, value, roles, spans = fixture()
    k, v = heads(key, value, roles, spans)
    (k.square().mean() + (v * torch.randn_like(v)).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
    assert sum(p.grad.abs().sum().item() for p in heads.parameters()) > 0
    assert key.grad is None and value.grad is None


def test_invalid_roles_and_incomplete_frame_spans_fail_closed():
    heads, key, value, roles, spans = fixture()
    with pytest.raises(ValueError, match="sum to one"):
        heads(key, value, roles * 0, spans)
    with pytest.raises(ValueError, match="cover every"):
        heads(key, value, roles, spans[:1])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from scripts.run_cfbi_region_pilot import activate_author_repo
        activate_author_repo()
        from sam2.modeling.sam.transformer import RoPEAttention
        self.memory_attention = torch.nn.Module()
        self.memory_attention.layers = torch.nn.ModuleList()
        for _ in range(4):
            layer = torch.nn.Module()
            layer.cross_attn_image = RoPEAttention(
                embedding_dim=8, num_heads=1, kv_in_dim=4, feat_sizes=(2, 2),
                rope_k_repeat=True, dropout=0.0)
            self.memory_attention.layers.append(layer)
        self.requires_grad_(False)
        self.eval()

    def _prepare_memory_conditioned_features(self, **kwargs):
        return None


def test_native_rope_prefix_pointers_and_all_layers_are_preserved():
    heads, key, value, roles, spans = fixture()
    model = Model()
    ptr = torch.randn(2, 2, 4)
    key = torch.cat((key, ptr), 1)
    value = torch.cat((value, ptr), 1)
    query = torch.randn(2, 4, 8)
    before = {n: p.clone() for n,p in model.named_parameters()}
    with NativeRegionalMemoryPrompts(model, heads) as adapter:
        adapter.memberships, adapter.spans = roles, spans
        for index, layer in enumerate(model.memory_attention.layers):
            args, augmented = adapter._before(index)(layer.cross_attn_image, (),
                {"q": query, "k": key, "v": value, "num_k_exclude_rope": 2})
            assert torch.equal(augmented["k"][:, :10], key)
            assert torch.equal(augmented["v"][:, :10], value)
            assert augmented["num_k_exclude_rope"] == 14
            # Call forward directly to avoid adding prompts twice in this inspection.
            output = layer.cross_attn_image.forward(**augmented)
            assert output.shape == query.shape and torch.isfinite(output).all()
    assert [x["layer"] for x in adapter.calls] == [0,1,2,3]
    assert all(torch.equal(p, before[n]) for n,p in model.named_parameters())


def test_disabled_hooks_are_bit_identical_and_trainable_sam2_is_rejected():
    heads, key, value, roles, spans = fixture()
    model = Model()
    q = torch.randn(2, 4, 8)
    layer = model.memory_attention.layers[0].cross_attn_image
    native = layer(q=q, k=key, v=value)
    with NativeRegionalMemoryPrompts(model, heads, enabled=False):
        observed = layer(q=q, k=key, v=value)
    assert torch.equal(native, observed)
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="must be frozen"):
        with NativeRegionalMemoryPrompts(model, heads):
            pass
