import json
import sys
from types import SimpleNamespace

import pytest

from scripts import verify_wandb_connection
from training.tracking import log_results_artifact, start_wandb


def test_finalization_uses_mapping_api_and_waits_for_upload(tmp_path, monkeypatch):
    from training import tracking

    class StrictSummary(dict):
        def update(self, mapping):
            super().update(mapping)

    run = SimpleNamespace(id="existing", summary=StrictSummary())

    def upload(*args):
        assert not (tmp_path / "delivery.json").exists()

    monkeypatch.setattr(tracking, "log_results_artifact", upload)
    tracking.finalize_tracking(run, tmp_path, {"completed": True}, ["summary.json"])
    assert run.summary["completed"] and run.summary["wandb_artifact_upload_complete"]
    assert json.loads((tmp_path / "delivery.json").read_text())["completed"]


def test_failed_upload_never_claims_delivery(tmp_path, monkeypatch):
    from training import tracking

    def fail(*args):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(tracking, "log_results_artifact", fail)
    with pytest.raises(RuntimeError, match="upload failed"):
        tracking.finalize_tracking(
            SimpleNamespace(id="existing", summary={}),
            tmp_path,
            {"completed": True},
            [],
        )
    assert not (tmp_path / "delivery.json").exists()


def test_online_tracking_fails_before_wandb_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        start_wandb(
            "project",
            "entity",
            "name",
            "online",
            "https://example.invalid",
            {},
            tmp_path,
            (),
        )


def test_connection_check_does_not_create_run(monkeypatch, capsys):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"project": {"name": "ade-seg"}}}

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_BASE_URL", "https://example.invalid/api/wandb")
    monkeypatch.setattr("sys.argv", ["verify_wandb_connection.py", "--check-only"])
    monkeypatch.setattr(
        verify_wandb_connection.requests, "post", lambda *args, **kwargs: Response()
    )
    verify_wandb_connection.main()
    assert '"state": "connected"' in capsys.readouterr().out


def test_online_tracking_records_run_and_uploads_results(monkeypatch, tmp_path):
    uploaded = []

    class Artifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class Run:
        id = "live1234"
        url = "https://forge.coreweave.com/wandb/example/live1234"

        def log_artifact(self, artifact):
            uploaded.append(artifact)
            return SimpleNamespace(wait=lambda: uploaded.append("confirmed"))

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: Run(),
            Artifact=Artifact,
        ),
    )
    run = start_wandb(
        "project",
        "entity",
        "name",
        "online",
        "https://example.invalid",
        {},
        tmp_path,
        (),
    )
    receipt = json.loads((tmp_path / "tracking.json").read_text())
    assert receipt["mode"] == "online"
    assert receipt["run_id"] == run.id
    (tmp_path / "summary.json").write_text("{}")
    log_results_artifact(run, tmp_path, ("summary.json",))
    assert uploaded[0].files == [(str(tmp_path / "summary.json"), "summary.json")]
    assert uploaded[1] == "confirmed"
