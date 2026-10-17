import json

from scripts.analyze_region_token_pilot import analyze
from scripts.collect_region_results import validate
from scripts.collect_region_live import download_verified
import hashlib
import base64
from types import SimpleNamespace
import pytest
from training.tracking import latest_results_artifact


def test_results_selection_ignores_newer_automatic_history_and_incomplete_gates():
    def artifact(name, kind, date, files):
        return SimpleNamespace(
            name=name,
            type=kind,
            created_at=date,
            manifest=SimpleNamespace(entries=dict.fromkeys(files)),
        )

    complete = artifact(
        "results:v1", "experiment-results", "2026-10-17T13:00:00", ["E1/decision.json"]
    )
    incomplete = artifact(
        "results:v2", "experiment-results", "2026-10-17T14:00:00", ["status.json"]
    )
    history = artifact(
        "history:v0", "wandb-history", "2026-10-17T15:00:00", ["history.parquet"]
    )
    source = SimpleNamespace(logged_artifacts=lambda: [complete, incomplete, history])
    assert latest_results_artifact(source).name == "results:v2"
    assert latest_results_artifact(source, ("decision.json",)).name == "results:v1"
    with pytest.raises(ValueError, match="required evidence"):
        latest_results_artifact(source, ("nonexistent.json",))


def test_live_download_replaces_stale_same_length_file_and_verifies(tmp_path):
    path = tmp_path / "tracking.json"
    path.write_bytes(b"old")

    class Entry:
        ref = None
        digest = base64.b64encode(hashlib.md5(b"new").digest()).decode()

        def download(self, root, skip_cache):
            assert skip_cache and not path.exists()
            path.write_bytes(b"new")

    entry = Entry()
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(entries={"tracking.json": entry}),
        get_entry=lambda name: entry,
    )
    download_verified(artifact, tmp_path)
    assert path.read_bytes() == b"new"


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


def test_random_controls_match_feasible_category_frames_before_averaging():
    rows = [
        {
            "state": "complete",
            "video": "seq1",
            "frame": 1,
            "bank": "recent",
            "category": "near_background",
            "seed": 0,
            "harm": {"dice": 0.02, "boundary_f": 0.03},
            "attention": [],
        }
    ]
    for seed in range(5):
        for frame, harm in ((1, 0.0), (2, 0.9)):
            rows.append(
                {
                    "state": "complete",
                    "video": "seq1",
                    "frame": frame,
                    "bank": "recent",
                    "category": "random",
                    "seed": seed,
                    "harm": {"dice": harm, "boundary_f": harm},
                }
            )
    result = analyze(rows, ["seq1"])
    effect = result["effects"]["recent/near_background/dice"]
    assert effect["random_seed_range"] == [0, 0]
    assert effect["excess_over_max_random"] == 0.02
    assert not result["gate_pass"]  # one sequence cannot pass the minimum-n gate
