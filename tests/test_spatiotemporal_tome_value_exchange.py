import pytest
import torch

from modeling.spatiotemporal_tome_value_exchange import exchange_memory_values
from scripts.run_spatiotemporal_value_exchange_gate import verify_source_native_inputs


def _fixture(batch=1, *, structured_keys=False):
    generator = torch.Generator().manual_seed(41)
    keys = torch.randn(batch, 2, 50, 16, generator=generator)
    values = torch.randn(batch, 2, 50, 16, generator=generator)
    if structured_keys:
        # Same spatial cell has matching content across the two past maps;
        # distinct cells within each 2x2 tile are orthogonal.
        keys.zero_()
        for frame_start in (16, 32):
            for y in range(4):
                for x in range(4):
                    token = frame_start + y * 4 + x
                    keys[:, :, token, y * 4 + x] = 1.0
    spans = [
        {"frame": 0, "t_pos": 0, "start": 0, "stop": 16},
        {"frame": 1, "t_pos": 1, "start": 16, "stop": 32},
        {"frame": 2, "t_pos": 2, "start": 32, "stop": 48},
    ]
    return keys, values, spans


def test_noop_is_exact_and_does_not_load_merge_source():
    keys, values, spans = _fixture()
    out_keys, out_values, audit = exchange_memory_values(
        keys, values, spans, pointers=2, mode="noop", window_size=2
    )
    assert out_keys is keys
    assert out_values is values
    assert audit["merged_pairs"] == 0
    assert audit["eligible_frames"] == [1, 2]
    assert audit["eligible_t_pos"] == [1, 2]
    assert [frame["t_pos"] for frame in audit["frames"]] == [0, 1, 2]
    assert [frame["source_tokens"] for frame in audit["frames"]] == [16, 16, 16]
    assert [frame["target_tokens"] for frame in audit["frames"]] == [16, 16, 16]
    assert audit["spatial_key_mass_sum"] == 48.0
    assert audit["key_mass_sum"] == 50.0


def test_cross_frame_value_exchange_preserves_slots_keys_and_pointers():
    keys, values, spans = _fixture()
    out_keys, out_values, audit = exchange_memory_values(
        keys, values, spans, pointers=2, mode="cross_frame", window_size=2
    )
    assert out_keys is keys
    assert out_values.shape == values.shape
    assert torch.equal(out_values[..., :16, :], values[..., :16, :])
    assert torch.equal(out_values[..., -2:, :], values[..., -2:, :])
    assert torch.equal(out_keys, keys)
    assert not torch.equal(out_values[..., 16:48, :], values[..., 16:48, :])
    assert audit["eligible_tokens"] == 32
    assert audit["merged_pairs"] == 8
    assert audit["spatial_keys_before"] == audit["spatial_keys_after"] == 48


def test_cross_frame_mode_uses_temporal_matches_when_content_aligns():
    keys, values, spans = _fixture(structured_keys=True)
    _, _, audit = exchange_memory_values(
        keys, values, spans, pointers=2, mode="cross_frame", window_size=2
    )
    assert audit["scope"] == "cross_frame"
    assert audit["cross_frame_groups"] > 0


def test_cross_frame_and_local_control_have_equal_merge_budgets():
    keys, values, spans = _fixture()
    _, _, local = exchange_memory_values(
        keys, values, spans, pointers=2, mode="local", window_size=2
    )
    _, _, temporal = exchange_memory_values(
        keys, values, spans, pointers=2, mode="cross_frame", window_size=2
    )
    assert local["merged_pairs"] == temporal["merged_pairs"] == 8
    assert local["spatial_keys_after"] == temporal["spatial_keys_after"] == 48


def test_batch_items_do_not_exchange_values_with_each_other():
    keys, values, spans = _fixture(batch=1)
    batched_keys = torch.cat((keys, keys), dim=0)
    batched_values = torch.cat((values, values + 100), dim=0)
    _, out, _ = exchange_memory_values(
        batched_keys, batched_values, spans, pointers=2,
        mode="cross_frame", window_size=2
    )
    assert torch.allclose(out[1] - out[0], torch.full_like(out[0], 100))


def test_rejects_bad_memory_spans_and_nonfinite_keys():
    keys, values, spans = _fixture()
    bad_spans = [dict(spans[0]), dict(spans[1]), dict(spans[2])]
    bad_spans[1]["start"] += 1
    with pytest.raises(ValueError, match="contiguous"):
        exchange_memory_values(keys, values, bad_spans, pointers=2, window_size=2)
    keys[0, 0, 20, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        exchange_memory_values(keys, values, spans, pointers=2, window_size=2)


def _gate_audits(cross_frame_groups=1):
    audit = {
        "eligible_frames": [1, 2],
        "eligible_t_pos": [1, 2],
        "frames": [
            {"frame": 0, "t_pos": 0, "source_tokens": 16, "target_tokens": 16,
             "merged_tokens": 0, "windows": []},
            {"frame": 1, "t_pos": 1, "source_tokens": 16, "target_tokens": 16,
             "merged_tokens": 0, "windows": []},
            {"frame": 2, "t_pos": 2, "source_tokens": 16, "target_tokens": 16,
             "merged_tokens": 0, "windows": []},
        ],
        "spatial_keys_before": 48,
        "spatial_keys_after": 48,
        "spatial_key_mass_sum": 48.0,
        "key_mass_sum": 50.0,
        "pointer_keys": 2,
        "merged_pairs": 8,
        "qkv_input_sha256": ["q", "k", "v"],
    }
    return {
        "noop": [{**audit, "scope": "noop", "cross_frame_groups": 0} for _ in range(4)],
        "local": [{**audit, "scope": "local", "cross_frame_groups": 0} for _ in range(4)],
        "cross_frame": [
            {**audit, "scope": "cross_frame", "cross_frame_groups": cross_frame_groups}
            for _ in range(4)
        ],
    }


def test_engineering_gate_requires_real_cross_frame_matches():
    verify_source_native_inputs(_gate_audits(cross_frame_groups=1))
    with pytest.raises(RuntimeError, match="No cross-frame token pair"):
        verify_source_native_inputs(_gate_audits(cross_frame_groups=0))


def test_engineering_gate_rejects_cross_frame_fallback_to_local():
    audits = _gate_audits()
    audits["cross_frame"][0]["scope"] = "local"
    with pytest.raises(RuntimeError, match="did not execute its declared matching scope"):
        verify_source_native_inputs(audits)

