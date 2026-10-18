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


def test_memory_disabled_video_control_never_builds_or_reads_a_bank(tmp_path, monkeypatch):
    from PIL import Image
    import scripts.run_medsam2_region_boxes as module
    import external.MedSAM2.sam2.sam2_image_predictor as upstream
    monkeypatch.setattr(upstream, "SAM2ImagePredictor", lambda predictor: None)
    monkeypatch.setattr(module, "stage_frames", lambda *args: None)
    monkeypatch.setattr(module, "load_label_mask", lambda *args: np.ones((8, 8), np.uint8))
    image = tmp_path / "frame.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image)
    frames = [SimpleNamespace(image_path=image, mask_path=image, video_id="seq2") for _ in range(3)]
    calls = []

    def inference(**kwargs):
        calls.append(kwargs)
        assert kwargs["output_dict"] == {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        return {"object_score_logits": torch.tensor([[2.]]),
                "iou_predictions": torch.tensor([[.8]])}, torch.ones(1, 1, 2, 2)

    predictor = SimpleNamespace(image_size=512, backbone_stride=16, device="cpu",
        memory_attention=torch.nn.Identity(), init_state=lambda **kw: {},
        reset_state=lambda state: None, run_single_frame_inference=inference)
    detections = [{"box": [0, 0, 8, 8]}, None, {"box": [0, 0, 8, 8]}]
    rows, reads = module.run_video(predictor, SimpleNamespace(), frames, detections,
        tmp_path, ["video_no_memory"], [1, 1, 1, 1], 0, save_masks=False)
    assert len(calls) == 2
    assert all(c["is_init_cond_frame"] and not c["run_mem_encoder"] for c in calls)
    assert reads["video_no_memory"] == 0
    assert rows[1]["prediction_area"] == 0
    assert rows[0]["object_score_logit"] == 2.


def test_detector_refresh_keeps_memory_but_reads_it_only_without_detection(tmp_path, monkeypatch):
    from PIL import Image
    import scripts.run_medsam2_region_boxes as module
    import external.MedSAM2.sam2.sam2_image_predictor as upstream
    monkeypatch.setattr(upstream, "SAM2ImagePredictor", lambda predictor: None)
    monkeypatch.setattr(module, "stage_frames", lambda *args: None)
    monkeypatch.setattr(module, "load_label_mask", lambda *args: np.ones((8, 8), np.uint8))
    image = tmp_path / "frame.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image)
    frames = [SimpleNamespace(image_path=image, mask_path=image, video_id="seq2") for _ in range(3)]
    attention = torch.nn.Identity()
    calls = []

    def inference(**kwargs):
        calls.append(kwargs["is_init_cond_frame"])
        assert kwargs["run_mem_encoder"]
        if not kwargs["is_init_cond_frame"]:
            assert kwargs["output_dict"]["cond_frame_outputs"]
            attention(torch.ones(1))
        return {"object_score_logits": torch.tensor([[2.]]),
                "iou_predictions": torch.tensor([[.8]])}, torch.ones(1, 1, 2, 2)

    predictor = SimpleNamespace(image_size=512, backbone_stride=16, device="cpu",
        memory_attention=attention, init_state=lambda **kw: {},
        reset_state=lambda state: None, run_single_frame_inference=inference)
    detections = [{"box": [0, 0, 8, 8]}, None, {"box": [0, 0, 8, 8]}]
    rows, reads = module.run_video(predictor, SimpleNamespace(), frames, detections,
        tmp_path, ["detector_refresh"], [1, 1, 1, 1], 0, save_masks=False)
    assert calls == [True, False, True]
    assert reads["detector_refresh"] == 1
    assert [r["detector_refresh"] for r in rows] == [True, False, True]
