import json
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

from modeling.attention_distortion_observer import (
    AttentionDistortionObserver,
    DEFAULT_CONDITIONS,
)
from modeling.region_constrained_tome import compress_memory_kv

ROOT = Path(__file__).resolve().parents[1]
PITOME_REPOSITORY = Path(json.loads(
    (ROOT / "protocols/pitome_protection_engineering_v1.json").read_text()
)["pitome_repository"])


class _DummyModel:
    training = False
    memory_attention = SimpleNamespace(layers=[])

    def _prepare_memory_conditioned_features(self, *args, **kwargs):
        raise AssertionError("The unit fixture does not run the model")

    def parameters(self):
        return iter(())


def _fixture(*, identical_keys=False):
    torch.manual_seed(113)
    query = torch.randn(1, 1, 5, 8)
    spatial_keys = torch.ones(1, 1, 16, 8) if identical_keys else torch.randn(1, 1, 16, 8)
    pointer_keys = torch.randn(1, 1, 2, 8)
    keys = torch.cat((spatial_keys, pointer_keys), dim=-2)
    values = torch.randn_like(keys)
    labels = torch.zeros(1, 16, dtype=torch.long)
    roles = F.one_hot(labels, num_classes=3).float()
    spans = [{"frame": 4, "t_pos": 2, "start": 0, "stop": 16}]
    return query, keys, values, roles, spans


def test_observer_returns_exact_native_output_and_preserves_inputs():
    query, key, value, roles, spans = _fixture()
    snapshots = [tensor.clone() for tensor in (query, key, value)]
    observer = AttentionDistortionObserver(
        _DummyModel(), pitome_repository=PITOME_REPOSITORY
    )
    observer.original_sdpa = F.scaled_dot_product_attention
    observer.memberships, observer.spans = roles, spans
    observer.frame = 9
    observer.current_object_id = 1
    observer.active = (0, 2)

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = observer._sdpa(query, key, value)

    assert torch.equal(actual, expected)
    assert all(torch.equal(before, after) for before, after in zip(
        snapshots, (query, key, value), strict=True
    ))
    assert len(observer.records) == 1
    record = observer.records[0]
    assert record["layer"] == 0
    assert record["source_frames"] == [{"frame": 4, "t_pos": 2, "start": 0, "stop": 16}]
    assert len(record["conditions"]) == len(DEFAULT_CONDITIONS) == 6
    assert all(row["pointer_keys"] == 2 for row in record["conditions"])
    assert all(row["spatial_keys_after"] < row["spatial_keys_before"]
               for row in record["conditions"])
    assert all(row["native_pointer_key_sha256"] == row["shadow_pointer_key_sha256"]
               for row in record["conditions"])
    assert all(row["native_pointer_value_sha256"] == row["shadow_pointer_value_sha256"]
               for row in record["conditions"])


def test_mass_corrected_merge_is_exact_when_spatial_keys_are_identical():
    query, key, value, roles, spans = _fixture(identical_keys=True)
    native = F.scaled_dot_product_attention(query, key, value)
    observer = AttentionDistortionObserver(
        _DummyModel(), pitome_repository=PITOME_REPOSITORY
    )
    for condition in DEFAULT_CONDITIONS:
        with observer._backend_context(condition["matcher"]):
            merged_key, merged_value, mass_bias, _ = compress_memory_kv(
                key, value, roles, spans,
                mode=condition["mode"], pointers=2,
                merge_divisor=condition["merge_divisor"],
            )
        shadow = observer._shadow_sdpa(
            query, merged_key, merged_value, mass_bias,
            F.scaled_dot_product_attention, {},
        )
        torch.testing.assert_close(shadow, native, rtol=1e-6, atol=1e-6)


def test_zero_removal_preserves_original_key_value_order_and_attention():
    query, key, value, roles, spans = _fixture()
    merged_key, merged_value, mass_bias, audit = compress_memory_kv(
        key, value, roles, spans, mode="regional", pointers=2,
        merge_divisor=4097,
    )
    assert audit["spatial_keys_after"] == audit["spatial_keys_before"] == 16
    assert torch.equal(merged_key, key)
    assert torch.equal(merged_value, value)
    assert torch.equal(mass_bias, torch.zeros_like(mass_bias))
    assert torch.equal(
        F.scaled_dot_product_attention(query, merged_key, merged_value),
        F.scaled_dot_product_attention(query, key, value),
    )


@pytest.mark.parametrize("merge_divisor", [4, 4097])
def test_observer_norm_weights_include_singletons_and_actual_cluster_mass(merge_divisor):
    query, key, value, roles, spans = _fixture()
    observer = AttentionDistortionObserver(
        _DummyModel(), pitome_repository=PITOME_REPOSITORY,
        conditions=({"matcher": "tome", "mode": "global",
                     "merge_divisor": merge_divisor},),
    )
    observer.original_sdpa = F.scaled_dot_product_attention
    observer.memberships, observer.spans = roles, spans
    observer.frame, observer.active = 9, (0, 2)
    observer._sdpa(query, key, value)
    row = observer.records[0]["conditions"][0]
    merged_key, merged_value, log_mass, _ = compress_memory_kv(
        key, value, roles, spans, mode="global", pointers=2,
        merge_divisor=merge_divisor,
    )
    mass = log_mass.flatten()[:-2].exp()
    assert (mass == 1).any()  # Singletons must contribute to the statistic.
    for name, merged in (("key", merged_key), ("value", merged_value)):
        norms = merged[..., :-2, :].float().norm(dim=-1).flatten()
        expected = float((norms * mass).sum() / mass.sum())
        observed = row[f"{name}_norm_after_mass_weighted"]["mass_weighted_mean_l2"]
        assert observed == pytest.approx(expected, rel=1e-6)
        if merge_divisor == 4097:
            assert observed == pytest.approx(row[f"{name}_norm_before"]["mean_l2"], rel=1e-6)


def test_output_distortion_preserves_float64_sub_float32_difference():
    from modeling.attention_distortion_observer import _output_distortion
    native = torch.ones(1, 1, 2, 3, dtype=torch.float64)
    shadow = native + 2.0**-40
    result = _output_distortion(native, shadow)
    assert result['max_absolute_difference'] == 2.0**-40
    assert result['relative_squared_error'] == 2.0**-80


def test_calibration_shadow_reuses_one_merge_and_returns_native(monkeypatch):
    import modeling.attention_distortion_observer as module

    query = torch.zeros(1, 1, 5, 8, dtype=torch.float64)
    key = torch.ones(1, 1, 18, 8, dtype=torch.float64)
    value = torch.ones_like(key)
    roles = F.one_hot(torch.zeros(1, 16, dtype=torch.long), 3).double()
    calls = []
    original = module.compress_memory_kv

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'compress_memory_kv', counted)
    observer = AttentionDistortionObserver(
        _DummyModel(), pitome_repository=PITOME_REPOSITORY,
        conditions=({'matcher': 'tome', 'mode': 'global', 'merge_divisor': 2},),
        calibration_shadow=True,
    )
    observer.original_sdpa = F.scaled_dot_product_attention
    observer.memberships = roles
    observer.spans = [{'frame': 4, 't_pos': 2, 'start': 0, 'stop': 16}]
    observer.frame, observer.active = 9, (0, 2)
    before = [t.clone() for t in (query, key, value)]
    native = F.scaled_dot_product_attention(query, key, value)
    assert torch.equal(observer._sdpa(query, key, value), native)
    assert calls == [True]
    assert all(torch.equal(a, b) for a, b in zip(before, (query, key, value)))
    row = observer.records[0]['conditions'][0]
    assert row['output_distortion']['relative_squared_error'] < 1e-28
    assert row['calibrated_shadow']['output_distortion']['relative_squared_error'] > 0
    assert len(row['calibrated_shadow']['shared_merge_input_sha256']) == 3
    assert row['calibrated_shadow']['returns_native_output'] is True


def test_calibration_runner_restricts_conditions_without_reviving_pitome():
    from scripts.run_attention_distortion_diagnostic import observer_settings

    ordinary = [dict(c) for c in DEFAULT_CONDITIONS]
    assert observer_settings({'conditions': ordinary})['calibration_shadow'] is False
    calibrated = {'conditions': ordinary[:2], 'calibration_shadow': True}
    assert observer_settings(calibrated)['conditions'] == DEFAULT_CONDITIONS[:2]
    with pytest.raises(RuntimeError):
        observer_settings({'conditions': ordinary, 'calibration_shadow': True})
    with pytest.raises(RuntimeError):
        observer_settings({'conditions': ordinary[:2], 'calibration_shadow': 'true'})
