"""Unit-contract checks for held-out Lean DEV-selection lock."""
import json

import pytest

from scripts.freeze_lean_test_protocol import (
    LOCK_FORMAT, verify_test_lock,
)
from scripts.run_lean_memory_fusion_ablation import file_sha256


def setup_lock(tmp_path):
    image_manifest = tmp_path / "images.jsonl"
    image_manifest.write_text("test images sealed\n")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("automatic prompts sealed\n")
    lock_path = tmp_path / "test_lock.json"
    lock = {
        "format": LOCK_FORMAT,
        "adeseg_head": "a" * 40,
        "lean_source_revision": "b" * 40,
        "test_image_manifest_sha256": file_sha256(image_manifest),
        "test_prompt_manifest_sha256": file_sha256(prompts),
        "backbone_sha256": "c" * 64,
        "selected_fusion_input_mode": "full",
        "initialization_mode": "mask_reprompt",
        "selected_fusion_checkpoint_sha256": "d" * 64,
        "dev_score_sha256": "e" * 64,
        "dev_candidate_receipt_sha256": "f" * 64,
    }
    lock_path.write_text(json.dumps(lock))
    return lock_path, image_manifest, prompts


def verify(lock, images, prompts, *, arm="fusion", mode="full", digest="d"*64):
    return verify_test_lock(
        lock, image_manifest=images, prompt_manifest=prompts,
        backbone_sha256="c"*64, source_head="a"*40,
        arm=arm, input_mode=mode, fusion_sha256=digest)


def test_heldout_lock_accepts_only_dev_selected_model_and_immutable_inputs(tmp_path):
    lock, images, prompts = setup_lock(tmp_path)
    assert verify(lock, images, prompts)["selected_fusion_input_mode"] == "full"
    assert verify(lock, images, prompts, arm="native")
    assert verify(lock, images, prompts, arm="identity")
    with pytest.raises(ValueError, match="DEV selection"):
        verify(lock, images, prompts, mode="no_early")
    with pytest.raises(ValueError, match="DEV selection"):
        verify(lock, images, prompts, digest="0"*64)
    prompts.write_text("new prompt bytes")
    with pytest.raises(ValueError, match="TEST lock mismatches"):
        verify(lock, images, prompts)


def test_heldout_lock_rejects_source_or_missing_dev_evidence(tmp_path):
    lock, images, prompts = setup_lock(tmp_path)
    data = json.loads(lock.read_text())
    data["adeseg_head"] = "0" * 40
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="TEST lock mismatches"):
        verify(lock, images, prompts)
    data["adeseg_head"] = "a" * 40
    data["dev_score_sha256"] = None
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete DEV evidence"):
        verify(lock, images, prompts)
