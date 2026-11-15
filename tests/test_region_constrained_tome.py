import pytest
import torch
import torch.nn.functional as F

from modeling.region_constrained_tome import compress_memory_kv


def roles_for_four_by_four():
    labels = torch.tensor([
        [0, 0, 0, 1],
        [0, 2, 2, 1],
        [0, 2, 1, 1],
        [0, 0, 1, 1],
    ]).reshape(1, 16)
    return F.one_hot(labels, num_classes=3).float()


def test_region_and_global_tome_match_key_budget_and_preserve_pointers():
    torch.manual_seed(7)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn(1, 2, 18, 8)
    original_pointer_keys, original_pointer_values = keys[..., 16:, :].clone(), values[..., 16:, :].clone()
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]

    regional = compress_memory_kv(keys, values, roles_for_four_by_four(), spans,
                                  mode="regional", pointers=2, window_size=2)
    r_keys, r_values, r_bias, r_audit = regional
    plan = {
        frame["frame"]: {
            window["window"]: window["merges"]
            for window in frame["windows"]
        }
        for frame in r_audit["frames"]
    }
    globalized = compress_memory_kv(keys, values, roles_for_four_by_four(), spans,
                                    mode="global", pointers=2, window_size=2,
                                    merge_plan=plan)

    g_keys, g_values, g_bias, g_audit = globalized
    assert r_keys.shape == g_keys.shape
    assert r_values.shape == g_values.shape
    assert r_audit["spatial_keys_after"] == g_audit["spatial_keys_after"]
    assert [frame["windows"] for frame in r_audit["frames"]] == [
        frame["windows"] for frame in g_audit["frames"]
    ]
    assert r_audit["spatial_keys_after"] < 16
    assert r_audit["frames"][0]["t_pos"] == 2
    assert all(window["original_spatial_keys"] == 4 for window in r_audit["frames"][0]["windows"])
    assert r_audit["pointer_keys"] == g_audit["pointer_keys"] == 2
    assert torch.equal(r_keys[..., -2:, :], original_pointer_keys)
    assert torch.equal(r_values[..., -2:, :], original_pointer_values)
    assert torch.equal(g_keys[..., -2:, :], original_pointer_keys)
    assert torch.equal(g_values[..., -2:, :], original_pointer_values)
    assert torch.isfinite(r_bias).all() and torch.isfinite(g_bias).all()
    assert torch.allclose(r_bias.exp().sum(), torch.tensor(18.0), atol=1e-5)
    assert torch.allclose(g_bias.exp().sum(), torch.tensor(18.0), atol=1e-5)


def test_generic_control_reuses_region_budget_when_its_roles_differ():
    torch.manual_seed(11)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn(1, 2, 18, 8)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]
    regional = compress_memory_kv(
        keys, values, roles_for_four_by_four(), spans,
        mode="regional", pointers=2, window_size=2,
    )
    plan = {
        3: {
            window["window"]: window["merges"]
            for window in regional[3]["frames"][0]["windows"]
        }
    }
    different_roles = F.one_hot(torch.zeros(1, 16, dtype=torch.long), 3).float()
    generic = compress_memory_kv(
        keys, values, different_roles, spans,
        mode="global", pointers=2, window_size=2, merge_plan=plan,
    )
    assert generic[0].shape[-2] == regional[0].shape[-2]
    assert generic[3]["frames"][0]["windows"] == regional[3]["frames"][0]["windows"]


def test_mass_corrected_merge_preserves_attention_for_identical_tokens():
    keys = torch.ones(1, 1, 18, 8)
    values = torch.ones(1, 1, 18, 8) * 3
    query = torch.randn(1, 1, 5, 8)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]

    reduced = compress_memory_kv(keys, values, roles_for_four_by_four(), spans,
                                 mode="regional", pointers=2, window_size=2)
    compressed_keys, compressed_values, bias, _ = reduced
    expected = F.scaled_dot_product_attention(query, keys, values)
    actual = F.scaled_dot_product_attention(query, compressed_keys, compressed_values,
                                             attn_mask=bias)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["regional", "global"])
def test_rejects_non_square_or_incomplete_memory_spans(mode):
    keys = torch.randn(1, 1, 18, 8)
    roles = F.one_hot(torch.zeros(1, 15, dtype=torch.long), num_classes=3).float()
    with pytest.raises(ValueError, match="contiguous square grids"):
        compress_memory_kv(keys, keys, roles,
                           [{"frame": 3, "t_pos": 2, "start": 0, "stop": 15}],
                           mode=mode, pointers=3, window_size=1)
