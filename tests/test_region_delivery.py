import json

from scripts.analyze_region_token_pilot import analyze
from scripts.collect_region_results import validate


def test_missing_or_failed_results_never_validate(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    receipt = validate(tmp_path)
    assert not receipt["delivery_valid"]
    assert not receipt["full_study_complete"]
    assert "No E1 records" in receipt["problems"]


def test_pilot_empty_or_incomplete_sequence_coverage_never_passes():
    result = analyze([], [f"seq{i}" for i in range(1, 11)])
    assert result["decision"] == "stop_or_pivot"
    assert not result["gate_pass"]
    assert not result["full_study_complete"]


