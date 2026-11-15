import pytest

from scripts.run_spatiotemporal_tome_value_exchange_cohort import (
    CONDITIONS,
    SEQUENCES,
    cohort_comparison,
)


def _arm(value, *, false_alarms=2, undetected_returns=1):
    return {
        "present_dice": value,
        "iou": value - 0.1,
        "boundary_f": 0.7,
        "gt_empty_false_alarm_count": false_alarms,
        "gt_empty_targets": 10,
        "return_count": 3,
        "undetected_returns": undetected_returns,
    }


def test_cohort_comparison_uses_all_sequences_and_matched_controls():
    summaries = {
        "noop": {sequence: _arm(0.70) for sequence in SEQUENCES},
        "local": {sequence: _arm(0.701) for sequence in SEQUENCES},
        "cross_frame": {sequence: _arm(0.72) for sequence in SEQUENCES},
    }

    result = cohort_comparison(
        summaries,
        {
            "bootstrap_seed": 7,
            "bootstrap_resamples": 200,
            "confidence": 0.95,
        },
    )

    assert result["primary_metric"] == "equal-sequence mean present-frame Dice"
    assert result["cohort_summary"]["cross_frame"]["present_dice"] == pytest.approx(0.72)
    assert result["contrasts"]["cross_frame_minus_noop"]["mean_delta"] == pytest.approx(0.02)
    assert result["contrasts"]["cross_frame_minus_local"]["mean_delta"] == pytest.approx(0.019)
    assert result["contrasts"]["cross_frame_minus_noop"]["win_tie_loss"] == [21, 0, 0]
    assert result["screen_passed"]


def test_cohort_comparison_rejects_incomplete_conditions():
    summaries = {condition: {} for condition in CONDITIONS}
    with pytest.raises(ValueError, match="21-sequence cohort"):
        cohort_comparison(summaries, {})
