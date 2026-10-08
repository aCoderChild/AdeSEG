"""Guards for the Kalman/RDE training loss."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from training.kalman_trainer import KalmanLoss  # noqa: E402


def frame(score, size=8):
    masks = torch.zeros(1, 1, size, size)
    return {
        "pred_masks_high_res": masks,
        "multistep_pred_multimasks_high_res": [masks],
        "multistep_pred_ious": [torch.zeros(1, 1)],
        "multistep_object_score_logits": [torch.tensor([[score]], requires_grad=True)],
    }


def test_object_score_bce_weights_every_frame_equally():
    # Prompt frame, then 3 present frames and 1 empty frame, all at logit 0 (BCE = ln 2 each).
    masks = torch.zeros(5, 8, 8)
    masks[1:4, 2:4, 2:4] = 1
    student = [frame(0.0) for _ in range(5)]
    _, parts = KalmanLoss(distill_weight=0.0)(student, None, masks)
    assert math.isclose(parts["presence"], 3 * math.log(2) / 4, rel_tol=1e-5)
    assert math.isclose(parts["absence"], math.log(2) / 4, rel_tol=1e-5)


def test_a_closed_gate_on_a_present_frame_is_charged_as_an_empty_mask():
    masks = torch.zeros(2, 8, 8)
    masks[1, 2:4, 2:4] = 1
    closed, opened = [frame(0.0), frame(-3.0)], [frame(0.0), frame(3.0)]
    opened[1]["pred_masks_high_res"] = torch.full((1, 1, 8, 8), -10.0)
    _, closed_parts = KalmanLoss(distill_weight=0.0)(closed, None, masks)
    _, open_parts = KalmanLoss(distill_weight=0.0)(opened, None, masks)
    assert closed_parts["segmentation"] > 0
    assert math.isclose(closed_parts["segmentation"], open_parts["segmentation"], rel_tol=1e-5)


def test_no_worse_penalises_only_frames_where_the_student_is_worse_than_the_teacher():
    masks = torch.zeros(3, 8, 8)
    masks[1:, 2:4, 2:4] = 1
    teacher = [frame(2.0) for _ in range(3)]
    for item in teacher:
        item["pix_feat_with_mem"] = torch.ones(1, 4, 2, 2)
    student = [dict(item) for item in teacher]
    _, same = KalmanLoss()(student, teacher, masks)
    assert same["no_worse"] == 0.0
    student[1] = {**frame(1.0), "pix_feat_with_mem": torch.ones(1, 4, 2, 2)}  # lower object score
    student[2] = {**frame(5.0), "pix_feat_with_mem": torch.ones(1, 4, 2, 2)}  # higher: no reward, no penalty
    _, worse = KalmanLoss()(student, teacher, masks)
    assert math.isclose(worse["no_worse"], (2.0 - 1.0) / 2, rel_tol=1e-5)
