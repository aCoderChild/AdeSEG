import pytest

from training.tracking import start_wandb


def test_online_tracking_fails_before_wandb_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        start_wandb(
            "project", "entity", "name", "online", "https://example.invalid",
            {}, tmp_path, (),
        )

