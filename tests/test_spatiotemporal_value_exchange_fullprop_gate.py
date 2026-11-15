import pytest

from scripts.run_spatiotemporal_value_exchange_fullprop_gate import (
    CONDITIONS,
    count_fullprop_outputs,
    verify_fullprop_audits,
)


def _read(frame, layer, mode):
    history = [] if frame == 0 else ([frame - 1] if frame == 1 else [frame - 2, frame - 1])
    t_pos = list(range(1, len(history) + 1))
    eligible = 4096 * len(history)
    cross_scope = mode == "cross_frame" and len(history) > 1
    scope = "noop" if mode == "noop" or not history else (
        "cross_frame" if cross_scope else "local"
    )
    merged = 0 if mode == "noop" else eligible // 4
    return {
        "frame": frame,
        "layer": layer,
        "key_unchanged": True,
        "scope": scope,
        "eligible_frames": history,
        "eligible_t_pos": t_pos,
        "eligible_tokens": 0 if mode == "noop" else eligible,
        "merged_pairs": merged,
        "cross_frame_groups": 1 if cross_scope else 0,
        "frames": [
            {"frame": 0, "t_pos": 0, "source_tokens": 4096,
             "target_tokens": 4096, "spatial_keys_before": 0},
            *[
                {"frame": past, "t_pos": position, "source_tokens": 4096,
                 "target_tokens": 4096, "spatial_keys_before": 0}
                for position, past in enumerate(history, start=1)
            ],
        ],
        "spatial_keys_before": 4096 * (1 + len(history)),
        "spatial_keys_after": 4096 * (1 + len(history)),
        "spatial_key_mass_sum": float(4096 * (1 + len(history))),
        "key_mass_sum": float(4096 * (1 + len(history)) + 2),
        "pointer_keys": 2,
    }


def _audits():
    return {
        mode: [_read(frame, layer, mode) for frame in (1, 2) for layer in range(4)]
        for mode in CONDITIONS
    }


def test_fullprop_gate_accepts_matched_full_video_audits():
    receipt = verify_fullprop_audits(_audits(), [1, 2])
    assert receipt["attention_calls_per_arm"] == 8
    assert receipt["layout_and_budget_match"]
    assert receipt["cross_frame_groups"] == 4


def test_fullprop_gate_accepts_empty_history_before_any_memory_frame_is_written():
    audits = _audits()
    for mode in CONDITIONS:
        audits[mode] = [_read(0, layer, mode) for layer in range(4)] + audits[mode]
    receipt = verify_fullprop_audits(audits, [0, 1, 2])
    assert receipt["attention_calls_per_arm"] == 12
    assert receipt["cross_frame_groups"] == 4


def test_fullprop_gate_requires_cross_frame_matches_on_every_eligible_read():
    audits = _audits()
    audits["cross_frame"][6]["cross_frame_groups"] = 0
    with pytest.raises(RuntimeError, match="No cross-frame ToMe group"):
        verify_fullprop_audits(audits, [1, 2])


def test_fullprop_gate_rejects_wrong_scope_and_merge_budget():
    audits = _audits()
    audits["local"][0]["scope"] = "cross_frame"
    with pytest.raises(RuntimeError, match="Local control used the wrong scope"):
        verify_fullprop_audits(audits, [1, 2])

    audits = _audits()
    audits["cross_frame"][4]["merged_pairs"] -= 1
    with pytest.raises(RuntimeError, match="Matched merge counts differ"):
        verify_fullprop_audits(audits, [1, 2])


def test_fullprop_gate_rejects_noop_with_any_merge():
    audits = _audits()
    audits["noop"][0]["merged_pairs"] = 1
    with pytest.raises(RuntimeError, match="No-op arm recorded ToMe merges"):
        verify_fullprop_audits(audits, [1, 2])


def test_completion_counts_unique_targets_and_condition_outputs_separately():
    rows = {
        "seq6": {mode: [None] * 66 for mode in CONDITIONS},
        "seq8": {mode: [None] * 72 for mode in CONDITIONS},
    }
    assert count_fullprop_outputs(rows) == {
        "sequence_targets": {"seq6": 66, "seq8": 72},
        "unique_post_prompt_targets": 138,
        "condition_frame_outputs": 414,
    }


def test_completion_counts_reject_incomplete_condition_coverage():
    rows = {"seq6": {"noop": [None], "local": [None]}}
    with pytest.raises(RuntimeError, match="Incomplete conditions"):
        count_fullprop_outputs(rows)

    rows = {"seq6": {mode: [None] * (2 if mode == "cross_frame" else 1)
                      for mode in CONDITIONS}}
    with pytest.raises(RuntimeError, match="Condition frame counts differ"):
        count_fullprop_outputs(rows)
