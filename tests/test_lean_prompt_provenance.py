"""Lean generated prompt masks require authenticated producer provenance."""
import json
from pathlib import Path
import pytest

from scripts.export_lean_fgpi_prompts import PROTOCOL
from scripts.run_lean_memory_fusion_ablation import file_sha256
from scripts.verify_lean_prompt_provenance import verify_prompt_producer


def artifacts(tmp_path):
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("image-only sealed input\n")
    model_digest = "a" * 64
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompts = prompts_dir / "frozen_prompts.jsonl"
    prompts.write_text("sealed automatic output\n")
    receipt = {
        "protocol": PROTOCOL,
        "producer_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1] /
            "scripts/export_lean_fgpi_prompts.py"),
        "image_only_manifest_sha256": file_sha256(manifest),
        "lean_checkpoint_sha256": model_digest,
        "lean_source_revision": "b" * 40,
        "frozen_prompt_manifest_sha256": file_sha256(prompts),
        "source_video_count": 2,
        "promptable_video_count": 1,
        "mode": "fgpi", "split": "dev",
        "per_video": {
            "with_prompt": {"status": "completed"},
            "missing_detection": {"status": "no_detection"},
        },
    }
    path = prompts_dir / "receipt.json"
    path.write_text(json.dumps(receipt))
    return manifest, prompts, path


def check(manifest, prompts, *, strict=True, count=1):
    return verify_prompt_producer(
        prompts, image_manifest=manifest, backbone_sha256="a"*64,
        lean_source_revision="b"*40, require_verified=strict,
        expected_source_videos=2, expected_prompt_count=count,
        split="dev",
    )


def test_verified_prompt_receipt_rejects_forged_or_mismatched_bytes(tmp_path):
    manifest, prompts, receipt = artifacts(tmp_path)
    assert check(manifest, prompts)["status"] == (
        "verified_same_backbone_automatic_mask")
    prompts.write_text("modified prompt bytes\n")
    with pytest.raises(ValueError, match="inconsistent"):
        check(manifest, prompts)


def test_missing_producer_receipt_is_visible_and_strictly_rejected(tmp_path):
    manifest, prompts, receipt = artifacts(tmp_path)
    receipt.unlink()
    report = check(manifest, prompts, strict=False)
    assert report["status"] == "external_unverified"
    with pytest.raises(ValueError, match="Missing authenticated"):
        check(manifest, prompts, strict=True)


def test_producer_coverage_and_method_are_authenticated(tmp_path):
    manifest, prompts, receipt = artifacts(tmp_path)
    with pytest.raises(ValueError, match="coverage receipt"):
        check(manifest, prompts, count=2)
    row = json.loads(receipt.read_text())
    row["per_video"]["missing_detection"]["status"] = "completed"
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="coverage receipt"):
        check(manifest, prompts)
