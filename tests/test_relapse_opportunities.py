import pytest
import io
import zipfile

from PIL import Image

from scripts.false_alarm_statistics import (
    native_opportunities, prediction_nonempty, relapse_risk_summary,
)
from scripts.audit_relapse_opportunities import verify_archive_masks


def row(frame, present, nonempty, score=1):
    return dict(frame=frame, source_frame_index=frame * 3, present=present,
                prediction_nonempty=nonempty, absent_fp=nonempty if not present else None,
                previous_object_score=score,
                scores=dict(dice=1 if present and nonempty else 0,
                            precision=1 if not nonempty or present else 0))


def test_persistent_false_alarms_are_not_empty_prediction_opportunities():
    rows = [row(1, False, True), row(2, False, True), row(3, False, False),
            row(4, False, True), row(5, True, False), row(6, True, True)]
    result = native_opportunities(rows)
    assert result["gt_empty"] == 4
    assert result["gt_empty_after_empty_prediction"] == 1
    assert result["after_correct_empty"] == 1
    assert result["false_alarms_after_empty_prediction"] == 1
    assert result["false_alarms_after_nonempty_prediction"] == 2
    assert result["consecutive_gt_empty_false_alarms"] == 1
    assert result["present_after_empty_prediction"] == 1


def test_missed_present_predecessor_is_reported_separately():
    result = native_opportunities([row(1, True, False), row(2, False, False)])
    assert result["after_missed_present"] == 1
    assert result["after_correct_empty"] == 0


@pytest.mark.parametrize("dice,precision,nonempty", [(0, 1, False), (0, 0, True), (.4, 1, True)])
def test_legacy_binary_metric_convention_recovers_presence_not_score(dice, precision, nonempty):
    record = dict(present=True, scores=dict(dice=dice, precision=precision))
    assert prediction_nonempty(record) is nonempty


def test_empty_mask_does_not_imply_nonpositive_object_score():
    rows = [row(1, False, False, score=1), row(2, False, True, score=.1)]
    assert native_opportunities(rows)["gt_empty_after_empty_prediction"] == 1
    assert relapse_risk_summary(rows)["relapse_risk_targets"] == 0


def test_fixed_native_risk_set_is_not_changed_by_intervention_scores():
    baseline = [row(1, False, True, score=-1), row(2, False, True, score=1)]
    gated = [row(1, False, False, score=1), row(2, False, True, score=-1)]
    fixed = {r["source_frame_index"] for r in baseline if r["previous_object_score"] <= 0}
    before = relapse_risk_summary(baseline, fixed)
    after = relapse_risk_summary(gated, fixed)
    assert before["relapse_false_alarm_rate"] == 1
    assert after["relapse_false_alarm_rate"] == 0
    assert after["relapse_risk_targets"] == 1


def test_unknown_score_is_not_fabricated_and_missing_targets_fail():
    record = row(1, False, False)
    record.pop("previous_object_score")
    assert relapse_risk_summary([record])["relapse_risk_targets"] is None
    with pytest.raises(ValueError, match="coverage"):
        relapse_risk_summary([record], {999})


def test_saved_png_independently_checks_the_metric_convention():
    image = io.BytesIO()
    Image.new("L", (2, 2), 0).save(image, format="PNG")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        archive.writestr("seq4/frame_000001.png", image.getvalue())
    with zipfile.ZipFile(packed) as archive:
        assert verify_archive_masks(archive, "seq4", [row(1, True, False)]) == 1
        with pytest.raises(RuntimeError, match="disagrees"):
            verify_archive_masks(archive, "seq4", [row(1, True, True)])
