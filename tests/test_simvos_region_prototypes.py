import torch
import torch.nn.functional as F
from types import SimpleNamespace

from modeling.native_simvos_prototype_read import NativeSimVOSPrototypeRead
from modeling.region_bias import feature_key
from modeling.simvos_region_prototypes import RegionPrototypeHeads


def test_author_assignment_is_region_constrained_and_exhaustive():
    torch.manual_seed(5)
    heads = RegionPrototypeHeads(channels=64, prototypes=4)
    features = torch.randn(1, 64, 64, 64)
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[12:30, 12:30] = 1
    labels[30:42, 30:42] = 2

    pools = heads.assign(features, labels, "regional")
    global_pools = heads.assign(features, labels, "global")

    assert [region for region, _, _ in pools] == [0, 1, 2]
    assert len(global_pools) == len(pools)
    for region, prototypes, assignment in pools:
        assert prototypes.shape == (1, 4, 64)
        assert assignment.shape == (1, 4, 4096)
        assert torch.allclose(assignment.sum(-1), torch.ones(1, 4), atol=1e-6)
        outside = assignment[..., labels.flatten() != region]
        assert torch.count_nonzero(outside) == 0


def test_empty_region_is_omitted_in_both_arms():
    torch.manual_seed(9)
    heads = RegionPrototypeHeads(channels=64, prototypes=4)
    features = torch.randn(1, 64, 64, 64)
    labels = torch.zeros(64, 64, dtype=torch.long)

    regional = heads.assign(features, labels, "regional")
    global_pools = heads.assign(features, labels, "global")

    assert [region for region, _, _ in regional] == [0]
    assert [region for region, _, _ in global_pools] == [0]
    assert all(torch.isfinite(assignment).all() for _, _, assignment in global_pools)


def test_author_prototype_modules_receive_finite_gradients():
    torch.manual_seed(13)
    heads = RegionPrototypeHeads(channels=64, prototypes=4)
    features = torch.randn(1, 64, 64, 64)
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[10:28, 10:28] = 1
    labels[28:40, 28:40] = 2

    loss = sum(prototypes.square().mean() for _, prototypes, _ in heads.assign(features, labels, "regional"))
    loss.backward()

    for learner in heads.learners:
        for parameter in learner.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_native_memory_blocks_are_matched_by_exact_stored_features_and_keep_pointers():
    class Model:
        def _prepare_memory_conditioned_features(self, **kwargs):
            return None

    class Attention:
        def __init__(self):
            self.v_proj = torch.nn.Linear(64, 64)
            self.num_heads = 4

        @staticmethod
        def _separate_heads(value, num_heads):
            batch, tokens, channels = value.shape
            return value.reshape(batch, tokens, num_heads, channels // num_heads).transpose(1, 2)

    torch.manual_seed(21)
    heads = RegionPrototypeHeads(channels=64, prototypes=2)
    model = Model()
    model.memory_attention = SimpleNamespace(
        layers=[SimpleNamespace(cross_attn_image=Attention())]
    )
    reader = NativeSimVOSPrototypeRead(model, heads, "regional", "seq2")
    reader.frame = 3
    features = torch.randn(1, 4096, 64)
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[10:32, 10:32] = 1
    labels[32:40, 32:40] = 2
    reader.bank = {feature_key(features): (2, labels)}
    memory = torch.cat((features, torch.randn(1, 2, 64)), dim=1).transpose(0, 1)
    memory_pos = torch.zeros_like(memory)

    reader._capture_memory(
        None,
        (),
        {"memory": memory, "memory_pos": memory_pos, "num_obj_ptr_tokens": 2},
    )

    assert reader.current["memory_frames"] == [2]
    assert reader.current["spatial_keys"] == 4096
    assert reader.current["pointer_keys"] == 2
    assert [item["region"] for item in reader.current["items"]] == [0, 1, 2]
    assert all(item["end"] - item["start"] == 4096 for item in reader.current["items"])

    reader.sdpa = F.scaled_dot_product_attention
    reader.active = 0
    q = torch.randn(1, 4, 16, 16)
    k = torch.randn(1, 4, 4098, 16)
    v = torch.randn_like(k)
    actual = reader._attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.equal(actual, expected)
    assert reader.calls[-1]["spatial_keys"] == 4096
    assert reader.calls[-1]["pointer_keys"] == 2
    assert reader.calls[-1]["prototype_keys"] == 6
