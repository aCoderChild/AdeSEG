import json
from pathlib import Path

import pytest

from scripts import summarize_iterative_memory_pilot as summary


def protocol():
    return json.loads(summary.PROTOCOL.read_text())


def fixture_results():
    p = protocol()
    names = p["data"]["test"]
    arms = {}
    for arm in summary.ARMS:
        arms[arm] = {}
        for index, name in enumerate(names):
            baseline = 0.70 + index * 0.001
            dice = {
                "native": baseline,
                "single_regional": baseline + 0.004,
                "iterative_generic": baseline + 0.006,
                "iterative_regional": baseline + (0.02 if index < 7 else -0.05),
            }[arm]
            arms[arm][name] = {
                "present_dice": dice,
                "present_iou": dice - 0.1,
                "present_boundary_f": dice - 0.2,
                "all_frame_dice": dice - 0.01,
                "target_frames": 20,
                "elapsed_seconds": 10.0 + index,
                "fps": 20 / (10.0 + index),
                "gt_empty_false_alarm_count": index % 2,
                "gt_empty_frames": 2,
                "propagation": {
                    "returns": [
                        {"detected": True, "delay_sampled_frames": 2, "censored": False},
                        {"detected": False, "delay_sampled_frames": None, "censored": True},
                    ]
                },
            }
    return {"per_sequence": arms}


def test_sequence_level_summary_includes_controls_outliers_and_recovery():
    result = summary.analyze_results(fixture_results(), protocol())
    contrast = result["iterative_regional_present_dice_contrasts"]["native"]
    assert contrast["n_sequences"] == 8
    assert contrast["wins"] == 7
    assert contrast["losses"] == 1
    assert len(contrast["leave_one_sequence_out_mean_deltas"]) == 8
    assert contrast["leave_one_out_max"] > contrast["mean_delta"]

    regional = result["arms"]["iterative_regional"]
    assert regional["false_alarms"]["count"] == 4
    assert regional["false_alarms"]["ground_truth_empty_frames"] == 16
    assert regional["redetection"]["return_events"] == 16
    assert regional["redetection"]["detected_returns"] == 8
    assert regional["redetection"]["undetected_or_censored_returns"] == 8
    assert regional["redetection"]["median_detected_delay_sampled_frames"] == 2
    assert regional["compute"]["added_reader_parameters"] == protocol()["parameters"]["iterative"]


def test_missing_or_extra_test_sequence_fails_closed():
    records = fixture_results()
    records["per_sequence"]["native"].pop("seq23")
    with pytest.raises(ValueError, match="frozen test cohort"):
        summary.analyze_results(records, protocol())


def test_markdown_reports_cohort_and_leave_one_out_sensitivity():
    report = summary.markdown_summary(summary.analyze_results(fixture_results(), protocol()))
    assert "Per-sequence diagnostics" in report
    assert "Leave-one-out mean range" in report
    assert "iterative regional minus control" in report
    assert "False alarms / GT-empty frames" in report
    assert "| seq23 | iterative_regional |" in report
    assert "seq23" in report


def test_writer_records_input_and_protocol_hashes(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(fixture_results()))
    analysis_protocol_path = summary.ANALYSIS_PROTOCOL
    json_result = summary.write_report(
        results_path,
        tmp_path / "out",
        summary.PROTOCOL,
        analysis_protocol_path,
    )
    saved = json.loads(Path(json_result["analysis_json"]).read_text())
    assert saved["input_results_sha256"] == summary.digest(results_path)
    assert saved["protocol_sha256"] == summary.digest(summary.PROTOCOL)
    assert saved["analysis_protocol_sha256"] == summary.digest(analysis_protocol_path)
    assert saved["analysis_script_sha256"] == summary.digest(Path(summary.__file__))

