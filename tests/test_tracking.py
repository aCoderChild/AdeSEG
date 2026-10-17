import pytest

from scripts.backfill_kaggle_wandb import collect
from scripts import verify_wandb_connection
from training.tracking import start_wandb


def test_online_tracking_fails_before_wandb_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        start_wandb(
            "project", "entity", "name", "online", "https://example.invalid",
            {}, tmp_path, (),
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
    monkeypatch.setattr(verify_wandb_connection.requests, "post", lambda *args, **kwargs: Response())
    verify_wandb_connection.main()
    assert '"state": "connected"' in capsys.readouterr().out


def test_backfill_collects_only_saved_results(tmp_path):
    stage = tmp_path / "E1_E2" / "natural_failures"
    stage.mkdir(parents=True)
    (stage / "summary.json").write_text('{"analyzed_frames": 2, "mean_baseline_dice": 0.4, "note": null}')
    (stage / "frames.jsonl").write_text('{"frame": 1}\n{"frame": 2}\n')
    (stage / "utility_records.pt").write_bytes(b"saved labels and frozen features")
    (stage / "utility_records.pt.tmp").write_bytes(b"incomplete")
    metrics, files = collect(tmp_path)
    assert metrics == {
        "E1_E2/natural_failures/analyzed_frames": 2,
        "E1_E2/natural_failures/mean_baseline_dice": 0.4,
        "E1_E2/natural_failures/recorded_frames": 2,
    }
    assert {path.name for path in files} == {
        "summary.json", "frames.jsonl", "utility_records.pt"
    }

