import torch

from modeling.prompt_anchor_tome import (
    projected_prompt_anchor_affinity,
)
from modeling.pointer_conditioned_tome import compress_pointer_conditioned_kv


def test_prompt_anchor_affinity_uses_projected_keys_and_pointer_tail():
    # Two heads, three spatial keys and two projected prompt-pointer tokens.
    keys = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    ])
    affinity = projected_prompt_anchor_affinity(keys, pointers=2, pointer_tokens=2)
    expected = torch.tensor([[0.5, 0.5, -0.5]])
    assert torch.allclose(affinity, expected)


def test_prompt_anchor_affinity_rejects_invalid_layout_or_nonfinite_keys():
    keys = torch.ones(1, 2, 6, 4)
    try:
        projected_prompt_anchor_affinity(keys, pointers=2, pointer_tokens=3)
    except ValueError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Oversized prompt pointer tail must be rejected")

    keys[0, 0, 0, 0] = float("nan")
    try:
        projected_prompt_anchor_affinity(keys, pointers=2, pointer_tokens=2)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Nonfinite projected keys must be rejected")


def test_prompt_anchor_reuses_tome_merge_and_preserves_pointer_tail():
    torch.manual_seed(7)
    keys = torch.randn(1, 4, 17, 64)
    values = torch.randn_like(keys)
    spans = [{"frame": 0, "start": 0, "stop": 16, "t_pos": 0}]
    affinities = torch.linspace(-1.0, 1.0, 16).reshape(1, 16)
    out_keys, out_values, bias, audit = compress_pointer_conditioned_kv(
        keys, values, spans, affinities, pointers=1, window_size=4,
        merge_divisor=2,
    )

    assert out_keys.shape == out_values.shape == (1, 4, 9, 64)
    assert bias.shape == (1, 1, 1, 9)
    assert torch.equal(out_keys[..., -1:, :], keys[..., -1:, :])
    assert torch.equal(out_values[..., -1:, :], values[..., -1:, :])
    assert audit["spatial_keys_before"] == 16
    assert audit["spatial_keys_after"] == 8
    assert audit["spatial_key_mass_sum"] == 16.0
