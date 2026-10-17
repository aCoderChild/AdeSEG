import pytest

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

