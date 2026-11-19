"""Fast hermetic regression tests for research experiment contracts."""
import hashlib
import json
import sys
from pathlib import Path
import pytest

from adeseg.research.protocol import load_protocol, verify_protocol, comparison_key
from adeseg.research.lifecycle import run_stage
from adeseg.research.data import validate_video_manifest
from adeseg.research.ablation import paired_comparison

def make_protocol(tmp_path, experiment_id="baseline", prompt="oracle_mask"):
    manifest = tmp_path / "manifest.jsonl"
    checkpoint = tmp_path / "backbone.pt"
    manifest.write_text('{"patient_id":"p1","video_id":"v1","frame_index":0,"split":"test","image":"a.jpg","mask":"a.png"}\n')
    checkpoint.write_bytes(b"checkpoint")
    spec = {
        "schema_version": 1, "experiment_id": experiment_id,
        "track": "medsam2_highres", "dataset": {
            "name": "toy", "manifest": {
                "path": "manifest.jsonl", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}},
        "checkpoint": {"path": "backbone.pt", "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "split": "test", "initialization": {"source": prompt, "prompt_budget": 1},
        "evaluation": {
            "resolution_hw": [512, 512], "mask_threshold": 0.5,
            "postprocessing": "none", "frame_policy": "post_prompt_present",
            "aggregation": "sequence_macro",
            "prompt_frame_manifest_sha256": "a" * 64
        },
        "stages": {s: {"command": [], "artifacts": []}
                   for s in ("data", "training", "inference", "evaluation")}
    }
    p = tmp_path / f"{experiment_id}.json"
    p.write_text(json.dumps(spec))
    return p

def test_protocol_detects_asset_tampering(tmp_path):
    path = make_protocol(tmp_path)
    assert verify_protocol(path)[1]["track"] == "medsam2_highres"
    (tmp_path / "backbone.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        verify_protocol(path)

def test_invalid_prompt_budget(tmp_path):
    path = make_protocol(tmp_path)
    value = json.loads(path.read_text())
    value["initialization"]["prompt_budget"] = -1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prompt_budget"):
        load_protocol(path)

def test_stage_runs_and_records_artifact(tmp_path):
    spec = {"experiment_id": "smoke", "stages": {
        "data": {"command": [sys.executable, "-c",
                 "from pathlib import Path; Path('OUTPUT').write_text('ok')"],
                 "artifacts": ["result.txt"]}}}
    # The command writes to an absolute path to avoid depending on stage cwd.
    workdir = tmp_path / "run"
    spec["stages"]["data"]["command"][-1] = (
        f"from pathlib import Path; Path({str(workdir / 'result.txt')!r}).parent.mkdir(parents=True,exist_ok=True); "
        f"Path({str(workdir / 'result.txt')!r}).write_text('ok')")
    receipt = run_stage("data", spec, tmp_path, workdir)
    assert receipt["status"] == "completed"
    assert len(receipt["artifacts"][0]["sha256"]) == 64

def test_missing_artifact_fails_closed(tmp_path):
    spec = {"experiment_id": "x", "stages": {"data": {
        "command": [sys.executable, "-c", "pass"], "artifacts": ["missing.txt"]}}}
    assert run_stage("data", spec, tmp_path, tmp_path / "work")["status"] == "missing_artifacts"

def test_subject_leakage_is_rejected(tmp_path):
    for name in ("a.jpg", "a.png", "b.jpg", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    rows = [
        {"patient_id":"same", "video_id":"v1", "frame_index":0, "split":"train", "image":"a.jpg","mask":"a.png"},
        {"patient_id":"same", "video_id":"v2", "frame_index":0, "split":"test", "image":"b.jpg","mask":"b.png"}]
    path = tmp_path / "video.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError, match="leakage"):
        validate_video_manifest(path)

def test_paired_comparison_checks_prompt_source(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate", "automatic_mask")
    for name in ("base.csv", "candidate.csv"):
        (tmp_path / name).write_text("sequence,dice\nv1,0.5\n")
    with pytest.raises(ValueError, match="Protocol mismatch"):
        paired_comparison(base, candidate, tmp_path / "base.csv", tmp_path / "candidate.csv")

def test_paired_comparison_numeric(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate")
    (tmp_path / "base.csv").write_text("sequence,dice\nv1,0.4\nv2,0.6\n")
    (tmp_path / "candidate.csv").write_text("sequence,dice\nv1,0.5\nv2,0.7\n")
    result = paired_comparison(base, candidate, tmp_path / "base.csv", tmp_path / "candidate.csv",
                               n_bootstrap=100)
    assert result["paired_mean_difference"] == pytest.approx(0.1)

def test_rejects_different_backbone_for_paired_ablation(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate")
    value = json.loads(candidate.read_text())
    value["checkpoint"]["sha256"] = "f" * 64
    candidate.write_text(json.dumps(value))
    (tmp_path / "base.csv").write_text("sequence,dice\nv1,0.5\n")
    (tmp_path / "candidate.csv").write_text("sequence,dice\nv1,0.6\n")
    with pytest.raises(ValueError, match="backbone checkpoint"):
        paired_comparison(base, candidate, tmp_path / "base.csv",
                          tmp_path / "candidate.csv")

def test_executable_stage_requires_artifacts(tmp_path):
    protocol = make_protocol(tmp_path)
    value = json.loads(protocol.read_text())
    value["stages"]["inference"]["command"] = [sys.executable, "-c", "pass"]
    protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="declare artifacts"):
        load_protocol(protocol)

def test_rejects_unsafe_experiment_id(tmp_path):
    protocol = make_protocol(tmp_path)
    value = json.loads(protocol.read_text())
    value["experiment_id"] = "../../outside"
    protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="experiment_id"):
        load_protocol(protocol)

def test_video_cannot_change_patient_within_split(tmp_path):
    for filename in ("a.jpg", "a.png", "b.jpg", "b.png"):
        (tmp_path / filename).write_bytes(b"x")
    rows = [
        {"patient_id":"p1","video_id":"v1","frame_index":0,"split":"train","image":"a.jpg","mask":"a.png"},
        {"patient_id":"p2","video_id":"v1","frame_index":1,"split":"train","image":"b.jpg","mask":"b.png"}]
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError, match="multiple patients"):
        validate_video_manifest(path)

def test_ablation_rejects_different_frame_policy(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate")
    config = json.loads(candidate.read_text())
    config["evaluation"]["frame_policy"] = "all_frames"
    candidate.write_text(json.dumps(config))
    (tmp_path / "base.csv").write_text("sequence,dice\\nv1,0.5\\n")
    (tmp_path / "candidate.csv").write_text("sequence,dice\\nv1,0.6\\n")
    with pytest.raises(ValueError, match="Protocol mismatch"):
        paired_comparison(base, candidate, tmp_path / "base.csv",
                          tmp_path / "candidate.csv")

def test_ablation_fails_if_evaluation_unspecified(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate")
    value = json.loads(base.read_text())
    del value["evaluation"]
    base.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fully specified evaluation"):
        comparison_key(load_protocol(base)[1])

def test_ablation_rejects_modified_checkpoint_file(tmp_path):
    base = make_protocol(tmp_path, "baseline")
    candidate = make_protocol(tmp_path, "candidate")
    (tmp_path / "base.csv").write_text("sequence,dice\nv1,0.5\n")
    (tmp_path / "candidate.csv").write_text("sequence,dice\nv1,0.6\n")
    (tmp_path / "backbone.pt").write_bytes(b"changed after protocol creation")
    with pytest.raises(ValueError, match="SHA256"):
        paired_comparison(base, candidate, tmp_path / "base.csv",
                          tmp_path / "candidate.csv")
