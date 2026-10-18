import json

from scripts.analyze_region_token_pilot import analyze
from scripts.collect_region_results import validate, validate_e0_delivery
from scripts.collect_region_live import download_verified
import hashlib
import base64
from types import SimpleNamespace
import pytest
from training.tracking import latest_results_artifact
from training.tracking import download_forge_run_file


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


def test_forge_run_file_requires_origin_auth_and_verified_payload(
    tmp_path, monkeypatch
):
    import requests

    payload = b'{"gate_pass":true}'
    file = SimpleNamespace(
        url="https://forge.coreweave.com/files/entity/project/run/E1/decision_verified.json",
        name="E1/decision_verified.json",
        size=len(payload),
        md5=base64.b64encode(hashlib.md5(payload).digest()).decode(),
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-secret")

    def get(url, **kwargs):
        assert kwargs["headers"] == {"Authorization": "Bearer test-secret"}
        assert kwargs["allow_redirects"] is True and kwargs["timeout"] == 30
        return SimpleNamespace(
            content=payload, status_code=200, raise_for_status=lambda: None
        )

    monkeypatch.setattr(requests, "get", get)
    redirected = requests.Request(
        "GET",
        "https://storage.googleapis.com/signed-file",
        headers={"Authorization": "Bearer test-secret"},
    ).prepare()
    requests.Session().rebuild_auth(
        redirected, SimpleNamespace(request=SimpleNamespace(url=file.url))
    )
    assert "Authorization" not in redirected.headers
    assert download_forge_run_file(file, tmp_path).read_bytes() == payload
    file.md5 = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        download_forge_run_file(file, tmp_path)
    file.url = "https://example.com/files/secret"
    with pytest.raises(ValueError, match="another origin"):
        download_forge_run_file(file, tmp_path)
    file.url = "https://forge.coreweave.com/files/test"
    file.name = "../outside.json"
    with pytest.raises(ValueError, match="Unsafe"):
        download_forge_run_file(file, tmp_path)


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


def test_known_orchestrator_still_collects_child_inference_artifacts(
    tmp_path, monkeypatch
):
    import sys
    import wandb
    from scripts import collect_region_live as collector

    root = SimpleNamespace(
        id="root",
        name="E2 | pipeline",
        state="running",
        config={},
        url="https://example.com/root",
    )
    child = SimpleNamespace(
        id="child",
        name="E2 | measurements",
        state="running",
        config={"stage": "E2"},
        url="https://example.com/child",
    )
    api = SimpleNamespace(
        run=lambda _: root, runs=lambda *args, **kwargs: [root, child]
    )
    artifacts = {r.id: SimpleNamespace(digest=r.id, name=r.id) for r in (root, child)}
    delivered = []
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: api)
    monkeypatch.setattr(collector, "latest_results_artifact", lambda r: artifacts[r.id])
    monkeypatch.setattr(
        collector,
        "download_verified",
        lambda artifact, destination: delivered.append((artifact.name, destination)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--group",
            "group",
            "--root-run-id",
            "root",
            "--output-root",
            str(tmp_path),
            "--folder-name",
            "delivery",
        ],
    )
    for name in ("WANDB_BASE_URL", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(name, "test")
    collector.main()
    folder = tmp_path / "delivery"
    assert delivered == [("root", folder), ("child", folder / "study")]
    snapshot = json.loads((folder / "live_tracking.json").read_text())
    assert [r["id"] for r in snapshot["runs"]] == ["root", "child"]


def test_missing_or_failed_results_never_validate(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    receipt = validate(tmp_path)
    assert not receipt["delivery_valid"]
    assert not receipt["full_study_complete"]
    assert "No E1 records" in receipt["problems"]


def test_standalone_e0_delivery_requires_independent_evidence_and_online_provenance(
    tmp_path,
):
    assert not validate_e0_delivery(tmp_path)["delivery_valid"]
    (tmp_path / "E0").mkdir()
    values = {
        "status.json": {"state": "complete", "stage": "E0_complete"},
        "tracking.json": {"mode": "online", "run_id": "root", "git_revision": "abc"},
        "E0/tracking.json": {"mode": "online", "run_id": "child"},
        "E0/status.json": {"state": "complete", "e0_pass": True, "sequences": ["seq2"]},
        "E0/setup.json": {
            "git_revision": "abc",
            "sequences": ["seq2"],
            "checkpoint_sha256": "a" * 64,
            "protocol_sha256": "b" * 64,
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value))
    check = dict.fromkeys(
        (
            "passed",
            "detached_noop_exact",
            "empty_token_mask_exact",
            "zero_bias_exact",
            "value_identity_exact",
            "memory_identity_exact",
            "independent_restore_exact",
            "mutation_changed_prediction",
            "full_token_equals_frame",
        ),
        True,
    )
    check.update(video="seq2", attention_reconstruction_max_abs=0.0)
    sanity = tmp_path / "E0/sanity.jsonl"
    sanity.write_text(json.dumps(check) + "\n")
    receipt = validate_e0_delivery(tmp_path)
    assert receipt["delivery_valid"] and not receipt["full_study_complete"]
    assert len(receipt["sha256"]) == 6
    for key, value in (
        ("independent_restore_exact", False),
        ("memory_identity_exact", False),
        ("attention_reconstruction_max_abs", float("nan")),
        ("video", "seq3"),
    ):
        invalid = {**check, key: value}
        sanity.write_text(json.dumps(invalid) + "\n")
        assert not validate_e0_delivery(tmp_path)["delivery_valid"]
    sanity.write_text(json.dumps(check) + "\n")
    (tmp_path / "E0/tracking.json").write_text(json.dumps({"mode": "disabled"}))
    assert not validate_e0_delivery(tmp_path)["delivery_valid"]


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
