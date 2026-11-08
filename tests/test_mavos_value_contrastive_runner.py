import json

from scripts.run_mavos_value_contrastive import summarize


def _row(dice, false_alarm_rate=0.0):
    return {
        "present_dice": dice,
        "present_iou": dice,
        "present_boundary_f": dice,
        "gt_empty_false_alarm_rate": false_alarm_rate,
    }


def test_validation_release_screen_compares_c2_against_both_controls(tmp_path):
    names = ["seq13", "seq14", "seq15"]
    groups = {
        "native": {name: _row(0.70 + i * .01) for i, name in enumerate(names)},
        "segmentation_only": {name: _row(0.71 + i * .01) for i, name in enumerate(names)},
        "segmentation_plus_contrastive": {name: _row(0.73 + i * .01) for i, name in enumerate(names)},
    }
    protocol = {"training": {"seed": 17}, "evaluation": {"release_rule": "fixed"}}

    result = summarize(groups, names, tmp_path, protocol, apply_release_rule=True)

    assert result["validation_release_screen"]["passed"] is True
    assert result["contrasts"]["segmentation_plus_contrastive_minus_native"]["wins"] == 3
    assert result["contrasts"]["segmentation_plus_contrastive_minus_segmentation_only"]["wins"] == 3
    assert json.loads((tmp_path / "results.json").read_text())["validation_release_screen"]["passed"]


def test_test_summary_does_not_reapply_validation_release_rule(tmp_path):
    names = ["seq16", "seq17"]
    groups = {
        "native": {name: _row(.70) for name in names},
        "segmentation_only": {name: _row(.71) for name in names},
        "segmentation_plus_contrastive": {name: _row(.72) for name in names},
    }
    protocol = {"training": {"seed": 17}, "evaluation": {"release_rule": "fixed"}}

    result = summarize(groups, names, tmp_path, protocol)

    assert result["validation_release_screen"] is None


def test_validation_screen_fails_closed_when_false_alarm_rate_is_unavailable(tmp_path):
    names = ["seq13", "seq14"]
    groups = {
        "native": {name: _row(.70, None) for name in names},
        "segmentation_only": {name: _row(.70, None) for name in names},
        "segmentation_plus_contrastive": {name: _row(.73, None) for name in names},
    }
    protocol = {"training": {"seed": 17}, "evaluation": {"release_rule": "fixed"}}

    result = summarize(groups, names, tmp_path, protocol, apply_release_rule=True)

    assert result["validation_release_screen"]["passed"] is False
    assert result["validation_release_screen"]["false_alarm_increase_max"] is None
