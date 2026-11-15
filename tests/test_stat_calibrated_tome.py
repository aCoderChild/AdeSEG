import math

import pytest
import torch

from modeling.stat_calibrated_tome import calibrate_merged_kv


def test_calibration_applies_retention_scaling_and_log_mass_shrinkage():
    keys = torch.arange(1, 1 + 1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    values = keys + 100
    original_keys, original_values = keys.clone(), values.clone()
    log_mass = torch.tensor([[[[math.log(2), math.log(1), math.log(3), 0.0, 0.0]]]])
    original_bias = log_mass.clone()

    calibrated_k, calibrated_v, calibrated_bias, coefficient = calibrate_merged_kv(
        keys, values, log_mass, spatial_before=8, spatial_after=3
    )

    expected = math.sqrt(3 / 8)
    assert coefficient == pytest.approx(expected)
    torch.testing.assert_close(calibrated_k[..., :3, :], original_keys[..., :3, :] * expected)
    torch.testing.assert_close(calibrated_v[..., :3, :], original_values[..., :3, :] * expected)
    torch.testing.assert_close(calibrated_k[..., 3:, :], original_keys[..., 3:, :])
    torch.testing.assert_close(calibrated_v[..., 3:, :], original_values[..., 3:, :])
    torch.testing.assert_close(calibrated_bias, original_bias * (1 - expected))
    torch.testing.assert_close(keys, original_keys)
    torch.testing.assert_close(values, original_values)
    torch.testing.assert_close(log_mass, original_bias)


def test_no_merge_is_exact_identity():
    keys = torch.randn(1, 2, 5, 4)
    values = torch.randn_like(keys)
    bias = torch.zeros(1, 1, 1, 5)

    out_k, out_v, out_bias, coefficient = calibrate_merged_kv(
        keys, values, bias, spatial_before=4, spatial_after=4
    )

    assert out_k is keys
    assert out_v is values
    assert out_bias is bias
    assert coefficient == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_before": 0, "spatial_after": 0},
        {"spatial_before": 4, "spatial_after": 5},
    ],
)
def test_invalid_token_counts_are_rejected(kwargs):
    keys = torch.randn(1, 1, 4, 2)
    bias = torch.zeros(1, 1, 1, 4)
    with pytest.raises(ValueError):
        calibrate_merged_kv(keys, keys, bias, **kwargs)
