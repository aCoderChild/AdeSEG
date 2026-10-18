"""Checks for the historical frame-box adapter; GPU exactness is separate."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_medsam2_region_boxes import box_points, bias_for_bank


def test_native_box_corner_scaling_and_labels():
    result = box_points([20, 10, 180, 90], 100, 200, 512, "cpu")
    assert result["point_coords"].shape == (1, 2, 2)
    assert torch.allclose(result["point_coords"], torch.tensor([[[51.2, 51.2], [460.8, 460.8]]]))
    assert result["point_labels"].tolist() == [[2, 3]]


def test_region_bias_uses_predicted_mask_at_actual_tiny_grid(monkeypatch):
    import scripts.run_medsam2_region_boxes as module
    monkeypatch.setattr(module.cv2, "imread", lambda path: np.zeros((512, 512, 3), np.uint8))
    mask = np.zeros((512, 512), bool)
    mask[100:400, 100:400] = True
    unit = dict(frame_idx=0, temporal_distance=1, predicted_iou=0.8, object_score=3.)
    result = bias_for_bank([unit], {0: mask}, [SimpleNamespace(image_path="unused")],
                           32, [1.2, 1., 1., .8], 1., "full", 0)
    assert result[0].shape == (1024,)
    assert np.isfinite(result[0]).all()
    reliability = bias_for_bank([unit], {0: mask}, [SimpleNamespace(image_path="unused")],
                               32, [1.2, 1., 1., .8], 1., "reliability_only", 0)
    assert np.ptp(reliability[0]) == pytest.approx(0.)
    assert np.ptp(result[0]) > 0
    assert bias_for_bank([unit], {0: mask}, [SimpleNamespace(image_path="unused")],
                         32, [1.2, 1., 1., .8], 0., "full", 0) == {}


def test_development_gate_rejects_failed_validation(tmp_path):
    import json
    from scripts.run_medsam2_region_boxes import check_development_gate
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    (tmp_path / "setup.json").write_text(json.dumps({"scale": 0}))
    with pytest.raises(ValueError, match="zero-scale"):
        check_development_gate(tmp_path, tmp_path / "absent.pt", "mps")
