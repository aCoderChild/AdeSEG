from scripts.run_wider_history_tome_full import compare_sequences
import pytest


def test_full_cohort_summary_uses_paired_sequence_dice_and_declared_screen():
    names = [f"seq{i}" for i in range(2, 24) if i not in (7,)]
    assert len(names) == 21
    native = {}
    generic = {}
    regional = {}
    for index, name in enumerate(names):
        score = 0.60 + 0.005 * index
        common = {
            "gt_empty_false_alarm_count": 1,
            "gt_empty_targets": 3,
            "return_count": 1,
            "undetected_returns": 0,
        }
        native[name] = {**common, "present_dice": score}
        generic[name] = {**common, "present_dice": score + 0.002}
        regional[name] = {**common, "present_dice": score + 0.012}

    result = compare_sequences(regional, generic, native)
    assert result["cohort_summary"]["regional"]["equal_sequence_mean_present_dice"] == pytest.approx(0.662)
    assert result["planned_contrasts"]["regional_minus_native"]["mean_sequence_delta"] == pytest.approx(0.012)
    assert result["predeclared_meaningful_screen"]["passed"] is True
    assert result["cohort_summary"]["regional"]["sequence_win_tie_loss_vs_native"] == [21, 0, 0]
