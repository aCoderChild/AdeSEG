import json

import pytest

from modeling.wider_history_tome import allocate_history_budget
import scripts.run_ten_map_allocation_tome_full as full_run
from scripts.run_ten_map_allocation_tome_full import compare_arms
from scripts.run_ten_map_allocation_tome_gate import _assert_matched_design


SEQUENCES = [f"seq{i}" for i in range(2, 24) if i != 7]


def test_recent_dense_real_grid_quotas_fit_per_window_tome_limit():
    for count in range(7, 11):
        spans = [{"frame": 0, "t_pos": 0, "start": 0, "stop": 4096}]
        for index, age in enumerate(range(count, 0, -1), start=1):
            spans.append({
                "frame": index,
                "t_pos": 13 - age,
                "start": index * 4096,
                "stop": (index + 1) * 4096,
            })
        quotas = allocate_history_budget(
            spans, max_noncond_maps=10, allocation="recent_dense"
        )
        history = [quotas[index] for index in range(1, count + 1)]
        assert sum(history) == 24576
        assert min(history) >= 2048
        assert max(history) <= 4096
        assert max((4096 - tokens + 255) // 256 for tokens in history) <= 8


def test_gate_requires_matching_arms_to_share_exact_quotas_and_source_spans():
    spans = [{"frame": 0, "t_pos": 0, "target_tokens": 4096}]
    spans.extend(
        {"frame": frame, "t_pos": 13 - age, "target_tokens": 3000}
        for frame, age in enumerate(range(7, 0, -1), start=1)
    )
    # Use a valid fixed-budget vector for seven maps: six 3,511 and one 3,510.
    for index, row in enumerate(spans[1:]):
        row["target_tokens"] = 3510 + int(index < 6)
    read = {
        "layer": 0,
        "spatial_keys_before": 4096 + 7 * 4096,
        "spatial_keys_after": 28672,
        "pointer_keys": 4,
        "spatial_key_mass_sum": float(4096 + 7 * 4096),
        "key_mass_sum": float(4096 + 7 * 4096 + 4),
        "frames": spans,
    }
    rows = [{"frame": 8, "attention_reads": [read]}]
    _assert_matched_design({
        "uniform_generic": rows,
        "uniform_regional": rows,
    }, "uniform")

    mismatch_frames = [dict(row) for row in spans]
    mismatch_frames[-2]["target_tokens"] = 3510
    mismatch_frames[-1]["target_tokens"] = 3511
    mismatch = [{"frame": 8, "attention_reads": [{**read, "frames": mismatch_frames}]}]
    with pytest.raises(RuntimeError, match="different local quotas"):
        _assert_matched_design({
            "uniform_generic": rows,
            "uniform_regional": mismatch,
        }, "uniform")


def test_comparison_uses_sequence_unit_and_reports_all_five_fixed_contrasts():
    summaries = {arm: {} for arm in (
        "native", "uniform_generic", "uniform_regional",
        "recent_dense_generic", "recent_dense_regional",
    )}
    offsets = {
        "native": 0.0,
        "uniform_generic": 0.001,
        "uniform_regional": 0.002,
        "recent_dense_generic": 0.011,
        "recent_dense_regional": 0.012,
    }
    for index, sequence in enumerate(SEQUENCES):
        base = 0.60 + index * 0.001
        for arm, offset in offsets.items():
            summaries[arm][sequence] = {
                "present_dice": base + offset,
                "iou": 0.5 + offset,
                "boundary_f": 0.6 + offset,
                "gt_empty_false_alarm_count": 1,
                "gt_empty_targets": 3,
                "return_count": 2,
                "undetected_returns": 1,
            }

    result = compare_arms(summaries)
    contrasts = result["planned_contrasts"]
    assert len(contrasts) == 5
    assert contrasts["recent_dense_regional_minus_native"]["mean_sequence_delta"] == pytest.approx(0.012)
    assert contrasts["recent_dense_regional_minus_recent_dense_generic"]["mean_sequence_delta"] == pytest.approx(0.001)
    assert contrasts["recent_dense_regional_minus_native"]["sequence_win_tie_loss"] == [21, 0, 0]
    assert result["cohort_summary"]["recent_dense_regional"]["equal_sequence_present_dice"] < 0.74
    assert result["screen"]["passed"] is False


def test_resume_loader_reuses_only_complete_audited_arm_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(full_run, "summarize", lambda rows: {"targets": len(rows)})
    monkeypatch.setattr(full_run, "_assert_read_integrity", lambda *args, **kwargs: None)
    source = tmp_path / "attempt01"
    frames = source / "uniform_generic" / "seq2" / "frames.json"
    frames.parent.mkdir(parents=True)
    rows = [{
        "frame": 4,
        "condition_name": "uniform_generic",
        "allocation": "uniform",
        "attention_reads": [{"layer": layer} for layer in range(4)],
    }]
    frames.write_text(json.dumps(rows))
    (source / "sequence_summaries.json").write_text(json.dumps({
        "uniform_generic": {"seq2": {"targets": 1}}
    }))
    (source / "runtime.json").write_text(json.dumps({
        "uniform_generic": {"seq2": {
            "sequence": "seq2",
            "targets": 1,
            "attention_layer_calls": 4,
            "wall_seconds": 12.5,
            "parameter_hash_unchanged": True,
            "temporal_slots_restored": True,
        }}
    }))

    restored = full_run._load_resumable_arm(
        source, "uniform_generic", "seq2", range(4, 5), "uniform"
    )
    assert restored == (rows, {
        "sequence": "seq2",
        "targets": 1,
        "attention_layer_calls": 4,
        "wall_seconds": 12.5,
        "parameter_hash_unchanged": True,
        "temporal_slots_restored": True,
    })

    assert full_run._load_resumable_arm(
        source, "uniform_generic", "seq3", range(4, 5), "uniform"
    ) is None
    with pytest.raises(RuntimeError, match="incomplete or misordered"):
        full_run._load_resumable_arm(
            source, "uniform_generic", "seq2", range(4, 6), "uniform"
        )


def test_resume_source_requires_same_protocol_and_method_hashes(tmp_path):
    source = tmp_path / "attempt01"
    source.mkdir()
    setup = {
        "protocol_sha256": "protocol",
        "source_hashes": {
            "scripts/run_ten_map_allocation_tome_full.py": "old-runner",
            "modeling/wider_history_tome.py": "same-method",
        },
    }
    setup_path = source / "setup.json"
    setup_path.write_text(json.dumps(setup))
    identity = {"protocol_sha256": "protocol"}
    current = {
        "scripts/run_ten_map_allocation_tome_full.py": "new-runner",
        "modeling/wider_history_tome.py": "same-method",
    }
    assert full_run._validate_resume_source(source, identity, current)[
        "original_full_runner_sha256"
    ] == "old-runner"

    current["modeling/wider_history_tome.py"] = "changed-method"
    with pytest.raises(RuntimeError, match="source-code hash differs"):
        full_run._validate_resume_source(source, identity, current)
