"""Exercise the full local queue with fake inference, never real GPU work."""
import json
from pathlib import Path
from types import SimpleNamespace

import scripts.run_local_region_boxes as module


def test_dev_selection_precedes_single_heldout_evaluation(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "checkpoints").mkdir(parents=True)
    (repo / "checkpoints/MedSAM2_latest.pt").write_bytes(b"checkpoint")
    nonzero = tmp_path / "nonzero"
    nonzero.mkdir()
    (nonzero / "status.json").write_text('{"state":"complete"}')
    monkeypatch.setattr(module, "ROOT", repo)
    monkeypatch.setattr(module, "check_development_gate", lambda *a: None)
    order = []

    def inference(command, **kwargs):
        dest = Path(command[command.index("--output-dir") + 1])
        dest.mkdir()
        order.append(dest.name)
        if "--evaluation" in command:
            frozen = json.loads((dest.parent / "frozen_selection.json").read_text())
            assert frozen["weights"][0] == 1.2
            ids = range(16, 24)
        else:
            assert not (dest.parent / "frozen_selection.json").exists()
            ids = range(1, 16)
        summary = {f"seq{i}": {"full": .8 if "down" not in dest.name else .7,
                              "framewise": .75, "memory": .76, "reliability_only": .77} for i in ids}
        (dest / "summary.json").write_text(json.dumps(summary))
        (dest / "status.json").write_text('{"state":"complete"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", inference)
    monkeypatch.setattr(module, "start_wandb", lambda *a: SimpleNamespace(log=lambda *a: None, finish=lambda: None))
    monkeypatch.setattr(module, "log_results_artifact", lambda *a: None)
    monkeypatch.setattr(module.sys, "argv", ["local", "--output-root", str(tmp_path),
                        "--zero-validation", str(tmp_path / "zero"), "--nonzero-validation", str(nonzero)])
    module.main()
    assert order == ["dev_boundary_up", "dev_boundary_down", "heldout_once"]
    result = json.loads((tmp_path / "region_boxes_local_method_v1/results_summary.json").read_text())
    assert result["test_score_directed_retries"] is False
    assert len(result["heldout"]["full"]["by_sequence"]) == 8
    assert set(result["holm_adjusted_p"]) == {"framewise", "memory", "reliability_only"}
