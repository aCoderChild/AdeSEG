import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_author_baseline_batch as batch


def configure(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "validate_output", lambda path: Path(path))
    monkeypatch.setattr(batch, "METHODS", ("sam2", "sam2long"))
    monkeypatch.setattr(batch, "summarize", lambda output: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "--device",
            "cpu",
            "--checkpoint",
            "test.pt",
            "--output-dir",
            str(tmp_path / "batch"),
        ],
    )


def test_all_validations_and_deliveries_precede_any_full_run(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command[command.index("--method") + 1], "--smoke" in command))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "delivery.json").write_text(
            json.dumps({"completed": True, "wandb_artifact_upload_complete": True})
        )

    monkeypatch.setattr(batch.subprocess, "run", run)
    batch.main()
    assert calls == [
        ("sam2", True),
        ("sam2long", True),
        ("sam2", False),
        ("sam2long", False),
    ]


def test_failed_validation_blocks_all_full_inference(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(batch.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        batch.main()
    assert len(calls) == 1 and "--smoke" in calls[0]
    assert (
        json.loads((tmp_path / "batch/queue_progress.json").read_text())["state"]
        == "blocked"
    )


def test_comparison_rejects_mismatched_source_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "METHODS", ("sam2", "sam2long"))
    for method, frame in [("sam2", 4), ("sam2long", 5)]:
        folder = tmp_path / f"{method}_full"
        (folder / "seq2").mkdir(parents=True)
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "sequences": [
                        {"sequence": "seq2", "status": "completed", "present_dice": 0.7}
                    ]
                }
            )
        )
        (folder / "seq2/frames.jsonl").write_text(
            json.dumps({"source_frame_index": frame, "present": True}) + "\n"
        )
    # Undo the queue-test stub: this test exercises the real comparison function.
    with pytest.raises(RuntimeError, match="coverage differs"):
        batch.summarize(tmp_path)
