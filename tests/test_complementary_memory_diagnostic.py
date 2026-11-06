from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.diagnose_complementary_memory import (
    EXPERIMENT_NAME,
    _area_target,
    _bootstrap,
    _bounded_predict,
    _fit,
    _sequence_scores,
    _stored_inputs,
    check_output_path,
)


def test_ground_truth_is_fractional_square_grid_occupancy(tmp_path: Path):
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:64] = 1
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)

    occupancy = _area_target(path).reshape(64, 64)

    assert occupancy.shape == (64, 64)
    assert np.allclose(occupancy[:32], 1.0)
    assert np.allclose(occupancy[32:], 0.0)
    assert np.isclose(occupancy.mean(), 0.5)


def test_stored_mask_and_feature_channels_align_on_64_grid(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.ones((64, 64), dtype=np.uint8)).save(mask_path)
    features = torch.arange(64 * 64 * 64, dtype=torch.float32).reshape(1, 64, 64, 64)
    logits = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    logits[:, :, :4, :4] = 2.0

    feature_grid, probability, target = _stored_inputs(
        {"maskmem_features": features, "pred_masks": logits}, mask_path
    )

    assert feature_grid.shape == (64, 64, 64)
    assert np.array_equal(feature_grid[3, 5], features[0, :, 3, 5].numpy())
    assert probability.shape == (64, 64)
    assert np.isclose(probability[0, 0], torch.sigmoid(torch.tensor(2.0)).item())
    assert np.isclose(probability[0, 1], torch.sigmoid(torch.tensor(0.0)).item())
    assert np.allclose(target, 1.0)


def test_weighted_probe_outputs_are_finite_and_bounded():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, 64)).astype(np.float32)
    probabilities = rng.uniform(size=128).astype(np.float32)
    target = np.clip(0.7 * probabilities + 0.15 * features[:, 0] + 0.1, 0, 1)
    weights = np.full(128, 1 / 128, dtype=np.float32)

    scaler, probe = _fit("combined", features, probabilities, target, weights)
    predictions = _bounded_predict(
        scaler, probe, np.column_stack((features[:16], probabilities[:16]))
    )

    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_sequence_metrics_and_bootstrap_are_deterministic():
    rows = [
        {"target_present": True, "brier": 0.2, "mae": 0.4,
         "baseline_error_weighted_brier": 0.3},
        {"target_present": False, "brier": 0.6, "mae": 0.8,
         "baseline_error_weighted_brier": 0.5},
    ]
    scores = _sequence_scores(rows)
    assert scores["all_brier"] == pytest.approx(0.4)
    assert scores["present_brier"] == pytest.approx(0.2)
    assert scores["empty_brier"] == pytest.approx(0.6)

    values = np.asarray([0.1, 0.2, -0.1, 0.3])
    assert _bootstrap(values, seed=20261106) == _bootstrap(values, seed=20261106)


def test_output_is_confined_to_unique_drive_attempt(tmp_path: Path):
    drive = tmp_path / "My Drive/AdeSEG"
    attempt = drive / "outputs/experiment" / EXPERIMENT_NAME / "attempt_01"

    assert check_output_path(attempt, drive) == attempt.resolve()
    with pytest.raises(ValueError, match="attempt_NN"):
        check_output_path(drive / "outputs/local/attempt_01", drive)
    attempt.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        check_output_path(attempt, drive)
