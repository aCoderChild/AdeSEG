import copy
import json
from pathlib import Path

import pytest

from scripts.run_prediction_conditioned_affinity_cv import SEQUENCES
from scripts.memory_token_io import digest
from scripts.run_regional_low_rank_cv import matched_schedules, result_table, verified_history_cache


def test_schedule_check_catches_pointer_or_object_identity_mismatch():
    rows = [{"frame": 3, "attention_layers": [{"layer": 0, "object_id": 7, "pointer_keys": 4}]}]
    matched_schedules(rows, copy.deepcopy(rows))
    for field, changed in (("object_id", 42), ("pointer_keys", 8)):
        other = copy.deepcopy(rows)
        other[0]["attention_layers"][0][field] = changed
        with pytest.raises(RuntimeError, match="schedules differ"):
            matched_schedules(rows, other)


def test_complete_report_keeps_unavailable_native_runtime_explicit():
    summary = {"present_dice": .5, "iou": .4, "boundary_f": .3,
               "gt_empty_false_alarm_count": 10, "inference_wall_seconds": None}
    result = {mode: {name: {"present_dice": .5} for name in SEQUENCES}
              for mode in ("native", "global", "regional")}
    result["summary"] = {mode: summary for mode in result}
    for key in ("regional_vs_native", "regional_vs_global"):
        result[key] = {"mean_delta": 0, "ci95": [0, 0], "wins": 0, "losses": 0, "ties": 21}
    result.update(holm_p={"regional_vs_native": 1, "regional_vs_global": 1}, meaningful_signal_screen=False)
    text = result_table(result)
    assert "| native | 0.500000 | 0.400000 | 0.300000 | 10 | N/A |" in text
    assert "seq23" in text and "| seq1 |" not in text and "| seq7 |" not in text
    assert "screen passed: False" in text


def cache_fixture(directory):
    provenance = {"checkpoint_sha256": "checkpoint", "manifest_sha256": "manifest", "model_config": "config"}
    sources = {"source.py": "frozen"}
    common = {**provenance, "sources": sources}
    def write(name, value):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    write("setup.json", {**common, "capacity_completion_sha256": "capacity"})
    write("gate_completion.json", {**common, "passed": True})
    index = {}
    for i, name in enumerate(SEQUENCES):
        count = 2012 if i == 0 else 0
        receipt = {**provenance, "bank_sha256": name, "native_frames_checked": count}
        write(f"{name}/receipt.json", receipt)
        index[name] = {"bank": f"{name}/bank.pt", "receipt": f"{name}/receipt.json",
                       "bank_sha256": name, "receipt_sha256": digest(directory / name / "receipt.json"),
                       "native_frames_checked": count}
    write("cache_index.json", index)
    write("completion.json", {**common, "complete": True, "sequences": len(SEQUENCES), "native_targets": 2012,
        "cache_index_sha256": digest(directory / "cache_index.json"),
        "gate_completion_sha256": digest(directory / "gate_completion.json")})
    return provenance, index, sources


def test_native_history_cache_requires_complete_unchanged_provenance(tmp_path):
    provenance, index, sources = cache_fixture(tmp_path)
    assert verified_history_cache(tmp_path, provenance, "capacity") == (index, sources)
    with pytest.raises(RuntimeError, match="provenance"):
        verified_history_cache(tmp_path, {**provenance, "checkpoint_sha256": "other"}, "capacity")
    with pytest.raises(RuntimeError, match="prerequisites"):
        verified_history_cache(tmp_path, provenance, "different_capacity")
    (tmp_path / "seq2/receipt.json").write_text("{}")
    with pytest.raises(RuntimeError, match="receipt changed"):
        verified_history_cache(tmp_path, provenance, "capacity")


def test_native_history_cache_rejects_unfinished_cohort_and_changed_index(tmp_path):
    provenance, _, _ = cache_fixture(tmp_path)
    path = tmp_path / "completion.json"
    completed = json.loads(path.read_text())
    path.write_text(json.dumps({**completed, "native_targets": 73}))
    with pytest.raises(RuntimeError, match="incomplete"):
        verified_history_cache(tmp_path, provenance, "capacity")
    path.write_text(json.dumps(completed))
    (tmp_path / "cache_index.json").write_text("{}")
    with pytest.raises(RuntimeError, match="index or gate changed"):
        verified_history_cache(tmp_path, provenance, "capacity")


def test_native_history_protocol_keeps_folds_optimizer_and_success_screen():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "protocols/regional_low_rank_polypgen_cv_v1.json").read_text())
    new = json.loads((root / "protocols/regional_low_rank_native_history_cv_v1.json").read_text())
    for field in ("folds", "training", "primary", "secondary", "inference", "meaningful_screen",
                  "method", "global_control", "native_reference", "manifest", "model_config"):
        assert old[field] == new[field]
    assert new["context"] == "native_history"
