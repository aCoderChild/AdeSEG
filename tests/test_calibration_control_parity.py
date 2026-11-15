"""Regression coverage for the cross-calibration control missed by the v1 gate."""

from copy import deepcopy

import pytest

from scripts import run_region_constrained_tome_gate as gate


def records(monkeypatch):
    monkeypatch.setattr(gate, "SEQUENCES", ("seq_fixture",))
    reads = [
        {
            "layer": layer,
            "pointer_keys": 4,
            "spatial_keys_after": 8,
            "merge_assignment_sha256": "a" * 64,
            "frames": [{
                "frame": 0,
                "retained_spatial_keys": 8,
                "windows": [{"window": 0, "retained_spatial_keys": 8}],
            }],
        }
        for layer in range(4)
    ]
    return {"seq_fixture": {
        arm: [{"frame": 1, "attention_reads": deepcopy(reads)}]
        for arm in gate.CALIBRATED_CONDITIONS
    }}


def test_calibration_gate_checks_both_cross_calibration_pairs(monkeypatch):
    checks = gate.compare_matched_budgets(records(monkeypatch), gate.CALIBRATED_CONDITIONS)
    assert len(checks) == 16  # Four comparisons across four layers.


def test_matching_within_setting_is_insufficient_without_assignment_receipts(monkeypatch):
    data = records(monkeypatch)
    for rows in data["seq_fixture"].values():
        for read in rows[0]["attention_reads"]:
            read.pop("merge_assignment_sha256")
    # The original three-arm gate has no exact-assignment promise.
    assert len(gate.compare_matched_budgets(data, gate.CONDITIONS)) == 4
    with pytest.raises(RuntimeError, match="Missing or unequal exact calibration"):
        gate.compare_matched_budgets(data, gate.CALIBRATED_CONDITIONS)


def test_matching_token_counts_do_not_prove_matching_assignments(monkeypatch):
    data = records(monkeypatch)
    data["seq_fixture"]["calibrated_region_constrained_tome"][0]["attention_reads"][0]["merge_assignment_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="Missing or unequal exact calibration"):
        gate.compare_matched_budgets(data, gate.CALIBRATED_CONDITIONS)


def test_matched_pairs_can_still_violate_cross_calibration_budget(monkeypatch):
    data = records(monkeypatch)
    for arm in ("calibrated_region_constrained_tome", "calibrated_generic_local_tome"):
        for read in data["seq_fixture"][arm][0]["attention_reads"]:
            read["frames"][0]["windows"][0]["retained_spatial_keys"] = 9
    with pytest.raises(RuntimeError, match="Spatial key budgets differ by local window"):
        gate.compare_matched_budgets(data, gate.CALIBRATED_CONDITIONS)
