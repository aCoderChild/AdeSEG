"""Canonical, authenticated contract for oracle-initialized MedSAM2 video runs.

The legacy research protocol schema is retained. This strict specialization
requires explicit model_config, label_id, and a frozen prompt-manifest file.
It is used by inference, evaluation, comparison and GPU preflight.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adeseg.research.protocol import verify_protocol, digest
from datasets.io import load_label_mask
from datasets.video import load_manifest

EVAL_FIXED = {
    "resolution_hw": [512, 512],
    "mask_threshold": 0.5,
    "postprocessing": "none",
    "frame_policy": "post_prompt_present",
    "aggregation": "sequence_macro",
}
MODEL_CONFIG = "sam2.1_hiera_t512.yaml"


def prompt_records(videos, label_id):
    if type(label_id) is not int or label_id <= 0:
        raise ValueError("label_id must be a positive integer")
    rows = []
    for video_id, frames in sorted(videos.items()):
        position = None
        for index, frame in enumerate(frames):
            if (load_label_mask(frame.mask_path, [label_id]) == label_id).any():
                position = index
                break
        frame = frames[position] if position is not None else None
        rows.append({
            "video_id": video_id,
            "prompt_position": position,
            "prompt_frame_index": frame.frame_index if frame else None,
            "prompt_image_sha256": digest(frame.image_path) if frame else None,
            "prompt_mask_sha256": digest(frame.mask_path) if frame else None,
        })
    return rows


def encode_prompt_records(rows):
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def load_med_video_contract(path, *, expected_backbone_sha256=None):
    """Verify source file bytes, selected oracle frames and fixed evaluation rules."""
    path, spec, evidence = verify_protocol(path)
    if spec["track"] != "medsam2_highres":
        raise ValueError("Med video contract requires medsam2_highres track")
    if spec["split"] not in ("dev", "test"):
        raise ValueError("Video evaluation requires dev/test split")
    if spec["initialization"] != {"source": "oracle_mask", "prompt_budget": 1}:
        raise ValueError("Med video contract requires one oracle-mask initialization")
    evaluation = spec.get("evaluation", {})
    for key, expected in EVAL_FIXED.items():
        if evaluation.get(key) != expected or type(evaluation.get(key)) != type(expected):
            raise ValueError(f"Med video evaluation contract mismatch: {key}")
    if evaluation.get("model_config") != MODEL_CONFIG:
        raise ValueError("Med video contract requires pinned Hiera-T/512 configuration")
    label_id = evaluation.get("label_id")
    if type(label_id) is not int or label_id < 1:
        raise ValueError("Med video contract must declare positive label_id")
    if "prompt_frame_manifest_path" not in evaluation:
        raise ValueError("Med video contract requires a real prompt-frame manifest file")
    if expected_backbone_sha256 is None:
        from modeling.medsam2_latest import MEDSAM2_LATEST_SHA256
        expected_backbone_sha256 = MEDSAM2_LATEST_SHA256
    if spec["checkpoint"]["sha256"] != expected_backbone_sha256:
        raise ValueError("Med video contract backbone differs from pinned MedSAM2")
    root = path.parent
    manifest = (root / spec["dataset"]["manifest"]["path"]).resolve(strict=True)
    checkpoint = (root / spec["checkpoint"]["path"]).resolve(strict=True)
    videos = load_manifest(manifest, spec["split"], require_contiguous=True,
                           require_jpeg=True, safe_output_ids=True)
    expected = encode_prompt_records(prompt_records(videos, label_id))
    prompt_path = (root / evaluation["prompt_frame_manifest_path"]).resolve(strict=True)
    if prompt_path.read_bytes() != expected:
        raise ValueError("Prompt-frame manifest does not match actual first-GT-positive frames")
    return {
        "protocol_path": path,
        "protocol_sha256": digest(path),
        "prompt_manifest_sha256": digest(prompt_path),
        "spec": spec, "manifest": manifest, "checkpoint": checkpoint,
        "model_config": MODEL_CONFIG, "label_id": label_id,
        "videos": videos, "evidence": evidence,
    }


def verify_med_run_receipt(contract, receipt):
    """Reject receipt/config drift before any metric is interpreted."""
    spec = contract["spec"]
    expected = {
        "research_protocol_sha256": contract["protocol_sha256"],
        "prompt_manifest_sha256": contract["prompt_manifest_sha256"],
        "manifest_sha256": spec["dataset"]["manifest"]["sha256"],
        "backbone_sha256": spec["checkpoint"]["sha256"],
        "model_config": contract["model_config"], "label_id": contract["label_id"],
        "split": spec["split"], "postprocessing": "no_hole_filling",
        "protocol": "oracle_first_present_mask_post_prompt_v1",
    }
    bad = [key for key, value in expected.items() if receipt.get(key) != value]
    if bad:
        raise ValueError(f"Med video receipt differs from sealed contract: {bad}")
