import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import run_cmsa_seq23_viewport_v1 as runner
from scripts import verify_cmsa_seq23_viewport_gate_v1 as verifier
from scripts import audit_cmsa_seq23_viewport_predictions_v1 as full_auditor


def staged_fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    out = tmp_path / "drive"
    data = root / "data"
    source_scripts = root / "scripts"
    models = root / "modeling"
    images = out / "stage/inputs/seq23"
    for directory in (data, source_scripts, models, images):
        directory.mkdir(parents=True)
    (source_scripts / "stage_cmsa_seq23_viewport_v1.py").write_text("stager")
    (models / "seq23_viewport.py").write_text("crop")
    manifest = data / "manifest.jsonl"
    manifest.write_text("manifest")
    dummy = out / "stage/viewport_dummy_mask.png"
    Image.new("L", (834, 720), 0).save(dummy)
    source = data / "source.png"
    Image.new("RGB", (1280, 720), (3, 4, 5)).save(source)
    crop = images / "000000.png"
    Image.new("RGB", (834, 720), (3, 4, 5)).save(crop)
    for i in range(1, 56):
        (images / f"{i:06d}.png").hardlink_to(crop)
    records = [{"frame": i, "source_frame_index": i, "source": str(source),
                "crop": str(images / f"{i:06d}.png"), "source_sha256": runner.sha(source),
                "crop_sha256": runner.sha(crop)} for i in range(56)]
    stage = out / "stage"
    receipt = {"passed": True, "sequence": "seq23", "viewport_xyxy": list(runner.VIEWPORT),
               "stager_sha256": runner.sha(source_scripts / "stage_cmsa_seq23_viewport_v1.py"),
               "crop_helper_sha256": runner.sha(models / "seq23_viewport.py"),
               "manifest": str(manifest), "manifest_sha256": runner.sha(manifest),
               "dummy_mask": str(dummy), "dummy_mask_sha256": runner.sha(dummy), "records": records}
    (stage / "stage.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", out)
    monkeypatch.setattr(runner, "STAGE_SHA", runner.sha(stage / "stage.json"))
    monkeypatch.setattr(runner, "load_manifest", lambda _path, split:
                        {"seq23": [SimpleNamespace(frame_index=i, image_path=source) for i in range(56)]}
                        if split == "dev" else {})
    return stage, source, crop, dummy


def test_stage_verification_rejects_altered_crop(tmp_path, monkeypatch):
    stage, _, crop, _ = staged_fixture(tmp_path, monkeypatch)
    receipt, frames = runner.verify_stage(stage)
    assert receipt["passed"] and len(frames) == 56
    Image.new("RGB", (834, 720), (3, 4, 6)).save(crop)
    with pytest.raises(ValueError, match="binding"):
        runner.verify_stage(stage)


def test_stage_verification_rejects_nonzero_dummy(tmp_path, monkeypatch):
    stage, _, _, dummy = staged_fixture(tmp_path, monkeypatch)
    Image.new("L", (834, 720), 1).save(dummy)
    with pytest.raises(ValueError, match="mask"):
        runner.verify_stage(stage)


def test_smoke_audit_rejects_prediction_changed_after_gate(tmp_path, monkeypatch):
    stage, smoke = tmp_path / "stage", tmp_path / "smoke"
    stage.mkdir()
    predictions = smoke / "predictions/seq23"
    predictions.mkdir(parents=True)
    records = []
    for i in range(4):
        path = predictions / f"{i:06d}.png"
        Image.new("L", (834, 720), i).save(path)
        records.append({"frame": i, "prediction_sha256": runner.sha(path)})
    identity = {"stage_sha256": verifier.STAGE_SHA,
                "runner_sha256": runner.sha(verifier.ROOT / "scripts/run_cmsa_seq23_viewport_v1.py"),
                "verifier_sha256": runner.sha(Path(verifier.__file__))}
    receipt = {"passed": True, "smoke": True, "finite_clip_calls": 1,
               "label_guard": {"open_attempts": 0}, "identity": identity,
               "state_sha256": "a" * 64, "records": records}
    (smoke / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(verifier, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(verifier, "verify_stage", lambda _stage: None)
    monkeypatch.setattr(verifier, "install_mask_open_guard", lambda _manifest: {"open_attempts": 0})
    assert verifier.audit(stage, smoke)["passed"]
    Image.new("L", (834, 720), 9).save(predictions / "000001.png")
    with pytest.raises(ValueError, match="prediction hash"):
        verifier.audit(stage, smoke)


def test_full_audit_rejects_changed_prediction_before_scoring(tmp_path, monkeypatch):
    stage, checkpoint, run = tmp_path / "stage", tmp_path / "checkpoint", tmp_path / "run"
    stage.mkdir()
    (stage / "stage.json").write_text("stage")
    checkpoint.write_text("checkpoint")
    outputs = run / "predictions/seq23"
    outputs.mkdir(parents=True)
    first = outputs / "000000.png"
    Image.new("L", (834, 720), 2).save(first)
    for i in range(1, 56):
        (outputs / f"{i:06d}.png").hardlink_to(first)
    records = [{"frame": i, "prediction_sha256": runner.sha(first)} for i in range(56)]
    identity = {"stage_sha256": runner.sha(stage / "stage.json"), "checkpoint_sha256": runner.sha(checkpoint),
                "runner_sha256": runner.sha(full_auditor.ROOT / "scripts/run_cmsa_seq23_viewport_v1.py"),
                "verifier_sha256": runner.sha(full_auditor.ROOT / "scripts/verify_cmsa_seq23_viewport_gate_v1.py"),
                "statistics_sha256": runner.sha(full_auditor.AUTHOR / "lib/dataloader/statistics.pth"),
                "author_test_sha256": runner.sha(full_auditor.AUTHOR / "scripts/my_test.py"),
                "author_revision": full_auditor.REVISION, "normalization": "released",
                "device": "mps", "dtype": "float32", "hash_seed": "0", "fallback": "0", "fast_math": "0",
                "dependencies": {}}
    smoke_receipt = tmp_path / "smoke/receipt.json"
    smoke_receipt.parent.mkdir()
    smoke_receipt.write_text("smoke")
    gate = {"identity": identity, "state_sha256": "a" * 64, "smoke_receipt": str(smoke_receipt)}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    receipt = {"passed": True, "smoke": False, "label_guard": {"open_attempts": 0},
               "finite_clip_calls": 53, "identity": identity, "state_sha256": "a" * 64,
               "gate": str(gate_path), "gate_sha256": runner.sha(gate_path), "records": records}
    (run / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(full_auditor, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(full_auditor, "CHECKPOINT_SHA", runner.sha(checkpoint))
    monkeypatch.setattr(full_auditor, "verify_stage", lambda _stage: None)
    monkeypatch.setattr(full_auditor, "install_mask_open_guard", lambda _manifest: {"open_attempts": 0})
    monkeypatch.setattr(full_auditor, "audit_smoke", lambda *_args: gate)
    monkeypatch.setattr(full_auditor, "dependency_hashes", lambda: {})
    monkeypatch.setattr(full_auditor.subprocess, "check_output", lambda *_a, **_kw: full_auditor.REVISION)
    monkeypatch.setattr(full_auditor.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=0))
    assert full_auditor.audit(stage, checkpoint, run)["passed"]
    Image.new("L", (834, 720), 3).save(first)
    with pytest.raises(ValueError, match="Prediction changed"):
        full_auditor.audit(stage, checkpoint, run)
