import pytest
import torch
from torch import nn
import torch.nn.functional as F

from modeling.region_bias import NativeRegionBias, region_log_bias
from scripts.run_author_baseline import (
    parameter_fingerprints,
    assert_parameter_integrity,
)


def test_fixed_weights_and_untouched_pointers():
    labels = torch.tensor([0, 1, 2, 2])
    bias = region_log_bias(labels, 4, "boundary", 1)
    assert torch.allclose(
        bias.exp(), torch.tensor([1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0])
    )
    assert not region_log_bias(labels, 4, "zero", 1).any()


def test_shuffled_control_is_deterministic_and_count_preserving():
    labels = torch.arange(4096) % 3
    original = region_log_bias(labels, 4, "boundary", 1)
    shuffled = region_log_bias(labels, 4, "shuffled", 20)
    assert torch.equal(shuffled, region_log_bias(labels, 4, "shuffled", 20))
    assert torch.equal(original.sort().values, shuffled.sort().values)
    assert not torch.equal(original, shuffled)
    assert not shuffled[-4:].any()


class Attention(nn.Module):
    def forward(self, *, q, k, v, num_k_exclude_rope=4):
        return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]


def fake_model():
    model = nn.Module()
    model.memory_attention = nn.Module()
    model.memory_attention.layers = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.cross_attn_image = Attention()
        model.memory_attention.layers.append(block)
    return model.eval()


def setup():
    model = fake_model()
    features = torch.randn(1, 64, 64, 64)
    state = {
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: {"maskmem_features": features}},
                "non_cond_frame_outputs": {},
            }
        }
    }
    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[:, :, 8:24, 8:24] = 10
    v = torch.cat([features.flatten(2).transpose(1, 2), torch.randn(1, 4, 64)], 1)
    q = torch.randn(1, 4, 64)
    return model, state, logits, q, v


def test_zero_bias_native_exact_and_all_four_layers_observed():
    model, state, logits, q, v = setup()
    native = F.scaled_dot_product_attention(q[:, None], v[:, None], v[:, None])[:, 0]
    original = F.scaled_dot_product_attention
    with NativeRegionBias(model, "zero", "seq2", 0) as control:
        control.capture(state, 0, logits)
        for block in model.memory_attention.layers:
            actual = block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
            assert torch.equal(actual, native)
        assert [r["layer"] for r in control.trace] == [0, 1, 2, 3]
        assert not control.bias.any()
    assert F.scaled_dot_product_attention is original


def test_unknown_memory_fails_and_native_function_is_restored():
    model, state, logits, q, v = setup()
    original = F.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="Unmatched"):
        with NativeRegionBias(model, "boundary", "seq2", 0) as control:
            control.capture(state, 0, logits)
            model.memory_attention.layers[0].cross_attn_image(
                q=q, k=v + 1, v=v + 1, num_k_exclude_rope=4
            )
    assert F.scaled_dot_product_attention is original


def test_nonzero_bias_matches_native_sdpa_with_explicit_additive_mask():
    model, state, logits, q, v = setup()
    original = F.scaled_dot_product_attention
    with NativeRegionBias(model, "boundary", "seq2", 0) as control:
        control.capture(state, 0, logits)
        for block in model.memory_attention.layers:
            actual = block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
            expected = original(
                q[:, None], v[:, None], v[:, None], attn_mask=control.mask
            )[:, 0]
            assert torch.equal(actual, expected)
        assert control.bias.any() and not control.bias[-4:].any()
        assert control.trace[0]["memory_units"][0]["frame"] == 0


def test_named_integrity_check_catches_real_mutation_and_bf16(tmp_path):
    model = nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
    before = parameter_fingerprints(model)
    assert_parameter_integrity(model, before, tmp_path, "unchanged")
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(RuntimeError, match="weight"):
        assert_parameter_integrity(model, before, tmp_path, "changed")
    assert (tmp_path / "parameter_integrity_changed.json").exists()


def test_integrity_is_named_value_based_not_registration_order(tmp_path):
    from collections import OrderedDict

    model = nn.Linear(2, 2).requires_grad_(False)
    before = parameter_fingerprints(model)
    model._parameters = OrderedDict(reversed(list(model._parameters.items())))
    assert_parameter_integrity(model, before, tmp_path, "reordered")
    model.register_parameter("extra", nn.Parameter(torch.ones(1), requires_grad=False))
    with pytest.raises(RuntimeError, match="extra"):
        assert_parameter_integrity(model, before, tmp_path, "added")
