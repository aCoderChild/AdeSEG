"""Public training commands must pass --help and enforce source checks."""
import hashlib
import json

import pytest

from training.safe_train import run_training


def case(tmp_path, file_field):
    path = tmp_path / "sample.pt"
    path.write_bytes(b"fake test capture")
    manifest = tmp_path / "manifest.json"
    split = tmp_path / "groups.json"
    split.write_text(json.dumps({
        "train": ["subject_A"], "dev": ["subject_B"], "test": ["subject_C"]
    }))
    manifest.write_text(json.dumps([{
        "split": "train", "source_group": "subject_A",
        file_field: "sample.pt",
        "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }]))
    return manifest, split, tmp_path / "model.pt"


@pytest.mark.parametrize("mode,file_field", [("lean", "tensor_file"), ("med", "file")])
def test_cli_forwards_settings_and_writes_provenance(tmp_path, mode, file_field):
    manifest, split, output = case(tmp_path, file_field)
    calls = []
    def trainer(args):
        calls.append(args)
        assert "--manifest" in args
        assert "--output" in args
        assert "--epochs" in args
        output.write_bytes(b"model checkpoint")
    run_training(trainer, mode=mode, argv=[
        "--manifest", str(manifest), "--split-file", str(split),
        "--output", str(output), "--epochs", "1"
    ])
    assert len(calls) == 1
    receipt = json.loads((tmp_path / "model.pt.split.json").read_text())
    assert receipt["training_method"] == mode
    assert receipt["trained_checkpoint_sha256"] == hashlib.sha256(b"model checkpoint").hexdigest()


def test_help_runs_without_split_and_does_not_launch_training():
    called = []
    run_training(lambda a: called.append(a), mode="med", argv=["--help"])
    assert called == [["--help"]]


def test_missing_split_rejected(tmp_path):
    manifest, _, output = case(tmp_path, "tensor_file")
    with pytest.raises(SystemExit):
        run_training(lambda a: None, mode="lean", argv=[
            "--manifest", str(manifest), "--output", str(output)
        ])
