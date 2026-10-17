import json

from scripts.validate_kaggle_results import EXPECTED_STAGES, inspect


def make_results(root, *, online=True):
    (root / "status.json").write_text('{"stage":"complete","state":"complete"}')
    (root / "events.jsonl").write_text('{}\n')
    if online:
        (root / "tracking.json").write_text(
            '{"mode":"online","run_id":"orchestrator1234"}'
        )
    for name in ("E1_E2/taxonomy.json", "E3/mechanism.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for name in EXPECTED_STAGES:
        stage = root / name
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "setup.json").write_text("{}")
        (stage / "frames.jsonl").write_text('{"frame":1}\n')
        (stage / "utility_records.pt").write_bytes(b"data")
        (stage / "summary.json").write_text('{"analyzed_frames":1,"utility_units":2}')
        (stage / "status.json").write_text(
            '{"state":"complete","completed_rows":1,"utility_units":2}'
        )
        if online:
            (stage / "tracking.json").write_text(json.dumps(
                {"mode": "online", "run_id": "live1234"}
            ))


def test_complete_phase_a_requires_all_results_and_live_tracking(tmp_path):
    make_results(tmp_path)
    result = inspect(tmp_path, require_online=True)
    assert result["complete"]
    assert result["all_online"]
    assert len(result["stages"]) == len(EXPECTED_STAGES)


def test_missing_stage_or_tracking_fails_validation(tmp_path):
    make_results(tmp_path, online=False)
    missing = tmp_path / "E3" / "c4" / "setup.json"
    missing.unlink()
    result = inspect(tmp_path, require_online=True)
    assert not result["complete"]
    assert any("Missing experiment stage: E3/c4" in problem for problem in result["problems"])
    assert any("live W&B run not recorded" in problem for problem in result["problems"])


def test_mismatched_frame_counts_fail_validation(tmp_path):
    make_results(tmp_path)
    (tmp_path / "E3" / "natural" / "frames.jsonl").write_text("")
    result = inspect(tmp_path)
    assert not result["complete"]
    assert any("recorded frame counts disagree" in problem for problem in result["problems"])
