import hashlib
import json

import pytest

from scripts import run_memsam_residual_writer as run


def _write_completion(run_dir, *, continuation_passed=False):
    run_dir.mkdir(parents=True)
    results = run_dir / "results.json"
    results.write_text('{"continuation_passed": false}\n')
    completion = {
        "complete": True,
        "continuation_passed": continuation_passed,
        "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
    }
    (run_dir / "completion.json").write_text(json.dumps(completion))


def test_run174_hold_requires_integral_completed_negative_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "current_command", lambda _pid: "")
    run_dir = tmp_path / "run174"
    _write_completion(run_dir, continuation_passed=False)

    released = run.run174_hold(run_dir)

    assert released["released"] is True
    assert released["reason"] == "Run174 completed and failed its frozen continuation rule"


@pytest.mark.parametrize("continuation_passed", [True, None])
def test_run174_hold_refuses_pass_or_unspecified_continuation(tmp_path, monkeypatch,
                                                              continuation_passed):
    monkeypatch.setattr(run, "current_command", lambda _pid: "")
    run_dir = tmp_path / "run174"
    _write_completion(run_dir, continuation_passed=continuation_passed)
    if continuation_passed is None:
        completion = json.loads((run_dir / "completion.json").read_text())
        del completion["continuation_passed"]
        (run_dir / "completion.json").write_text(json.dumps(completion))

    assert run.run174_hold(run_dir)["released"] is False


def _row(frame):
    return {
        "frame": frame,
        "prediction_sha256": f"prediction-{frame}",
        "stored_sha256": f"stored-{frame}",
        "pointer_sha256": f"pointer-{frame}",
        "position_sha256": f"position-{frame}",
    }


def _read(frame, layers=(0, 1, 2, 3)):
    return {
        "frame": frame,
        "object_id": 1,
        "memory_frames": [{"frame": 0, "start": 0, "stop": 4096, "t_pos": 0}],
        "spatial_keys": 4096,
        "pointer_keys": 1,
        "pointer_input_sha256": "pointer-input",
        "pointer_position_sha256": "pointer-position",
        "layers": list(layers),
        "stored_mask_alignment": True,
        "native_positions_verified": True,
    }


def _rollout(writer_records, layers=(0, 1, 2, 3)):
    rows = [_row(0), _row(1)]
    return {
        "prompt_frame": 0,
        "rows": rows,
        "reader_calls": [_read(1, layers)],
        "memory_writer_calls": writer_records,
    }


def test_zero_writer_identity_requires_all_layers_and_unchanged_memory():
    writer_rows = [
        {
            "frame": frame,
            "input_feature_sha256": f"feature-{frame}",
            "adapted_feature_sha256": f"feature-{frame}",
            "memory_mask_scale": 20.0,
            "memory_mask_bias": -10.0,
        }
        for frame in (0, 1)
    ]
    native = _rollout([])
    adapted = _rollout(writer_rows)

    result = run.compare_identity(native, adapted, "seq2")

    assert result["native_logits_memory_positions_pointers_bit_identical"] is True
    assert result["all_four_attention_layers_verified"] is True


def test_zero_writer_identity_rejects_missing_layer():
    writer_rows = [
        {
            "frame": frame,
            "input_feature_sha256": f"feature-{frame}",
            "adapted_feature_sha256": f"feature-{frame}",
            "memory_mask_scale": 20.0,
            "memory_mask_bias": -10.0,
        }
        for frame in (0, 1)
    ]

    with pytest.raises(RuntimeError, match="all native attention layers"):
        run.compare_identity(
            _rollout([], layers=(0, 1, 2)),
            _rollout(writer_rows, layers=(0, 1, 2)),
            "seq2",
        )


def test_frozen_source_inventory_verifies_uncommitted_snapshot_contents(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n")
    second.write_text("value = 2\n")
    expected = {
        "first.py": hashlib.sha256(first.read_bytes()).hexdigest(),
        "second.py": hashlib.sha256(second.read_bytes()).hexdigest(),
    }

    assert run.verify_source_inventory(expected, expected, root=tmp_path) == expected

    second.write_text("value = 3\n")
    with pytest.raises(RuntimeError, match="Frozen study source changed: second.py"):
        run.verify_source_inventory(expected, expected, root=tmp_path)


def test_frozen_source_inventory_requires_exact_file_set(tmp_path):
    source = tmp_path / "one.py"
    source.write_text("value = 1\n")

    with pytest.raises(RuntimeError, match="Frozen source inventory differs"):
        run.verify_source_inventory(["one.py"], {}, root=tmp_path)


def _validation_summary(dice, false_alarms=1):
    return {"present_dice": dice, "gt_empty_frames": 10,
            "gt_empty_false_alarm_count": false_alarms}


def test_validation_release_requires_gain_over_both_matched_arms_and_false_alarm_safety():
    protocol = {"data": {"validation_sequences": ["a", "b", "c"]}}
    summaries = {
        "native_sam2": {"a": _validation_summary(.695), "b": _validation_summary(.700),
                        "c": _validation_summary(.690)},
        "mask_neutral_writer": {"a": _validation_summary(.690), "b": _validation_summary(.695),
                                "c": _validation_summary(.685)},
        "soft_mask_writer": {"a": _validation_summary(.700), "b": _validation_summary(.710),
                             "c": _validation_summary(.700)},
    }

    result = run.validation_release(protocol, summaries)

    assert result["passed"] is True

    summaries["soft_mask_writer"]["a"]["gt_empty_false_alarm_count"] = 2
    assert run.validation_release(protocol, summaries)["passed"] is False
