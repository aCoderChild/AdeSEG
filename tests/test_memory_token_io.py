"""Current entry points and shared native memory-token I/O contracts."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from datasets.video import ManifestFrame
from scripts.memory_token_io import prompt_index, validate_output


def test_prompt_uses_first_visible_mask_and_rejects_unpromptable_sequence(tmp_path):
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    frames = [ManifestFrame("seq", i, path, path) for i, path in enumerate(paths)]
    for path in paths:
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(path)
    with pytest.raises(ValueError, match="No eligible"):
        prompt_index(frames, {"initialization": "first_labeled_frame"})
    Image.fromarray(np.ones((8, 8), np.uint8)).save(paths[1])
    assert prompt_index(frames, {"initialization": "first_labeled_frame"}) == 1


def test_outputs_cannot_be_written_to_project_or_parent_directory():
    with pytest.raises(ValueError, match="outputs must"):
        validate_output(Path(__file__).resolve().parents[1])
    assert validate_output("/kaggle/working/current").name == "current"


@pytest.mark.parametrize(
    "script",
    [
        "audit_memory_token_regions.py",
        "build_polypgen_manifest.py",
        "build_polypgen_kaggle_manifest.py",
        "verify_wandb_connection.py",
    ],
)
def test_current_cli_imports_without_retired_experiment_modules(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in result.stdout
    if script == "audit_memory_token_regions.py":
        assert "--checkpoint" in result.stdout and "--output-dir" in result.stdout


def test_current_protocol_and_author_revisions_are_explicit():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "protocols/memory_token_regions.json").read_text())
    assert (
        protocol["model"]["checkpoint_sha256"]
        == "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
    )
    assert protocol["assignment_audit"]["sensitivity_ratios"] == [0.01, 0.02, 0.04]
    assert protocol["assignment_audit"]["classifier_used"] is False
    assert protocol["assignment_audit"]["regions"] == [
        "background",
        "foreground_interior",
        "boundary",
    ]


def test_bundled_predictor_project_imports_survive_cleanup():
    import ast
    import importlib

    root = Path(__file__).resolve().parents[1]
    for path in (root / "external/MedSAM2/sam2").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "modeling."
            ):
                module = importlib.import_module(node.module)
                assert all(hasattr(module, alias.name) for alias in node.names)


def test_predictor_loader_sets_its_own_hydra_config_path(monkeypatch):
    from types import SimpleNamespace
    from modeling.medsam2 import build_video_predictor

    native = str(Path(__file__).resolve().parents[1] / "external/MedSAM2")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != native])
    monkeypatch.setitem(
        sys.modules,
        "external.MedSAM2.sam2.build_sam",
        SimpleNamespace(
            build_sam2_video_predictor=lambda *args, **kwargs: SimpleNamespace(
                fill_hole_area=8
            )
        ),
    )
    model = build_video_predictor("unused", "unused", "cpu")
    assert sys.path[0] == native and model.fill_hole_area == 0


def test_native_memory_capture_is_read_only_and_excludes_prompt(tmp_path, monkeypatch):
    from scripts import memory_token_io as runner

    memory = torch.arange(64 * 4 * 4).reshape(1, 64, 4, 4).float()
    original = memory.clone()
    frames = [
        SimpleNamespace(video_id="seq2", frame_index=i, mask_path=i) for i in range(3)
    ]
    monkeypatch.setattr(runner, "prompt_index", lambda *args: 0)
    monkeypatch.setattr(
        runner, "load_label_mask", lambda *args: np.ones((16, 16), np.uint8)
    )
    monkeypatch.setattr(runner, "stage_frames", lambda *args: None)
    monkeypatch.setattr(runner, "synchronize_storage", lambda *args: None)

    class Predictor:
        reset = False

        def init_state(self, **kwargs):
            assert kwargs["offload_state_to_cpu"] is False
            return {"output_dict": {"non_cond_frame_outputs": {}}}

        def add_new_mask(self, state, **kwargs):
            assert kwargs["frame_idx"] == 0 and kwargs["obj_id"] == 1

        def propagate_in_video(self, state, **kwargs):
            for i in range(3):
                state["output_dict"]["non_cond_frame_outputs"][i] = {
                    "maskmem_features": memory
                }
                yield i, [1], torch.full((1, 1, 16, 16), float(i))

        def reset_state(self, state):
            self.reset = True

    predictor = Predictor()
    features, records = runner.extract_memory_features(
        predictor, frames[1:], {"seq2": frames}, "cpu", tmp_path
    )
    assert features.shape == (2, 64, 4, 4)
    assert [r["frame"] for r in records] == [1, 2]
    assert predictor.reset and torch.equal(memory, original)
    features.zero_()
    assert torch.equal(memory, original)
    with pytest.raises(ValueError, match="follow the prompt"):
        runner.extract_memory_features(
            predictor, frames[:1], {"seq2": frames}, "cpu", tmp_path
        )
