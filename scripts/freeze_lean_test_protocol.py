"""Seal a selected DEV fusion checkpoint before once-only held-out TEST runs.

This is an audit/provenance mechanism, NOT a cryptographic signature or a
substitute for researcher discipline. It binds source commit, DEV scoring,
chosen model, TEST image-only frames, automatic prompt masks and checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from datasets.video import load_manifest
from scripts.run_lean_memory_fusion_ablation import (
    file_sha256, load_frozen_prompts, require_digest,
)


LOCK_FORMAT = "adeseg_lean_fusion_test_lock_v1"


def verify_test_lock(lock_path, *, image_manifest, prompt_manifest,
                     backbone_sha256, source_head, arm, input_mode,
                     fusion_sha256=None):
    if arm not in ("native", "identity", "fusion"):
        raise ValueError("Invalid held-out arm")
    lock = json.loads(Path(lock_path).resolve(strict=True).read_text("utf8"))
    required = {
        "format": LOCK_FORMAT,
        "adeseg_head": source_head,
        "test_image_manifest_sha256": file_sha256(image_manifest),
        "test_prompt_manifest_sha256": file_sha256(prompt_manifest),
        "backbone_sha256": require_digest(backbone_sha256),
    }
    if (not isinstance(lock, dict)
            or any(lock.get(k) != v for k, v in required.items())):
        raise ValueError(
            "TEST lock mismatches source commit, images, prompts, or backbone")
    selected_mode = lock.get("selected_fusion_input_mode")
    if selected_mode not in ("full", "no_early", "early0_only",
                             "early1_only", "no_mask"):
        raise ValueError("Malformed TEST-selected fusion input mode")
    digest = require_digest(lock.get("selected_fusion_checkpoint_sha256"))
    if (arm == "fusion" and
            (input_mode != selected_mode
             or fusion_sha256 != digest)):
        raise ValueError("TEST learned fusion differs from frozen DEV selection")
    for key in ("dev_score_sha256", "dev_candidate_receipt_sha256",
                "lean_source_revision"):
        if not isinstance(lock.get(key), str) or not lock[key]:
            raise ValueError("Incomplete DEV evidence in TEST lock")
    return lock


def freeze_test_protocol(*, dev_score, dev_candidate_receipt,
                         test_image_manifest, test_prompt_manifest,
                         backbone_checkpoint, fusion_checkpoint,
                         selected_run_name, output):
    from scripts.run_author_baseline import source_identity
    score_path = Path(dev_score).resolve(strict=True)
    candidate_path = Path(dev_candidate_receipt).resolve(strict=True)
    image_manifest = Path(test_image_manifest).resolve(strict=True)
    prompts = Path(test_prompt_manifest).resolve(strict=True)
    backbone = Path(backbone_checkpoint).resolve(strict=True)
    fusion = Path(fusion_checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite sealed held-out TEST lock")
    scores = json.loads(score_path.read_text("utf8"))
    candidate = json.loads(candidate_path.read_text("utf8"))
    if (scores.get("format") != "adeseg_paired_lean_memory_fusion_dice_v1"
            or scores.get("run_arm_mapping", {}).get(selected_run_name) != "fusion"
            or candidate.get("arm") != "fusion"
            or candidate.get("split") != "dev"
            or candidate.get("format") != "adeseg_lean_memory_fusion_video_v3"):
        raise ValueError("DEV scorer and selected fusion receipt mismatch")
    selected_mode = candidate.get("fusion_input_mode")
    if (scores.get("run_input_modes", {}).get(selected_run_name) != selected_mode
            or not isinstance(selected_mode, str)):
        raise ValueError("Chosen DEV fusion input mode is not in paired score")
    if (not isinstance(candidate.get("fusion_checkpoint_sha256"), str)
            or file_sha256(fusion) != candidate["fusion_checkpoint_sha256"]
            or file_sha256(backbone) != candidate.get("backbone_sha256")):
        raise ValueError("DEV selected backbone/fusion checkpoint bytes differ")
    repo, lean_revision = source_identity("lean")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Do not freeze TEST with uncommitted code")
    evidence = candidate.get("runtime_signature", {})
    scored_runtime = scores.get("provenance", {}).get("runtime_signature", {})
    if (evidence.get("adeseg_head") != head
            or scored_runtime.get("adeseg_head") != head
            or evidence.get("lean_config_sha256") !=
                    scored_runtime.get("lean_config_sha256")
            or candidate.get("lean_source_revision") != lean_revision
            or scores.get("provenance", {}).get("lean_source_revision") != lean_revision):
        raise ValueError("DEV evidence is not pinned to the current Lean/AdeSEG source")
    videos = load_manifest(
        image_manifest, "test", require_masks=False,
        require_contiguous=True, require_jpeg=True, safe_output_ids=True)
    selections = load_frozen_prompts(prompts, videos)
    if not selections:
        raise ValueError("No frozen automatic TEST prompts")
    lock = {
        "format": LOCK_FORMAT,
        "adeseg_head": head,
        "lean_source_revision": lean_revision,
        "selected_dev_run_name": selected_run_name,
        "selected_fusion_input_mode": selected_mode,
        "selected_fusion_checkpoint_sha256": candidate[
            "fusion_checkpoint_sha256"],
        "backbone_sha256": file_sha256(backbone),
        "test_image_manifest_sha256": file_sha256(image_manifest),
        "test_prompt_manifest_sha256": file_sha256(prompts),
        "test_video_count": len(videos),
        "test_initialized_video_count": len(selections),
        "dev_score_sha256": file_sha256(score_path),
        "dev_candidate_receipt_sha256": file_sha256(candidate_path),
        "warning": (
            "This locks declared choices/hashes but cannot prevent "
            "intentional inspection/retesting of held-out labels. "
            "Publish TEST only after source/prompt/model freeze."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", "utf8")
    return lock


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-score", type=Path, required=True)
    p.add_argument("--dev-candidate-receipt", type=Path, required=True)
    p.add_argument("--test-image-manifest", type=Path, required=True)
    p.add_argument("--test-prompt-manifest", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--fusion-checkpoint", type=Path, required=True)
    p.add_argument("--selected-run-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    result = freeze_test_protocol(
        dev_score=a.dev_score,
        dev_candidate_receipt=a.dev_candidate_receipt,
        test_image_manifest=a.test_image_manifest,
        test_prompt_manifest=a.test_prompt_manifest,
        backbone_checkpoint=a.backbone_checkpoint,
        fusion_checkpoint=a.fusion_checkpoint,
        selected_run_name=a.selected_run_name,
        output=a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
