import json

import torch

from scripts.generate_utility_labels import save_progress


def test_utility_progress_is_recoverable_after_each_video(tmp_path):
    records = [{"features": torch.tensor([1.0]), "utility": 0.25}]
    rows = [{"video": "bear:1", "frame": 20}]

    save_progress(
        tmp_path, records, rows, state="running", video="bear", label_id=1
    )

    restored = torch.load(tmp_path / "utility_records.pt", weights_only=False)
    assert restored[0]["utility"] == 0.25
    assert (tmp_path / "frames.jsonl").read_text() == json.dumps(rows[0]) + "\n"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "running"
    assert status["completed_rows"] == 1
    assert status["utility_units"] == 1
    assert status["last_video"] == "bear"
    assert status["last_label_id"] == 1
    assert isinstance(status["updated_unix"], float)
    assert not list(tmp_path.glob("*.tmp"))
