"""Verify that frozen Lean initial masks came from the pinned GT-free exporter.

A filename or SHA256 on a mask cannot prove how it was produced. This checks
the exporter receipt, source code hash, declared no-GT code path, image
manifest, pinned backbone, and the exact prompt bytes. It does not attest
honesty of external input data or independently verify patient identity.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.export_lean_fgpi_prompts import PROTOCOL
from scripts.run_lean_memory_fusion_ablation import file_sha256


def verify_prompt_producer(prompts_file, *, image_manifest,
                           backbone_sha256, lean_source_revision,
                           require_verified=False):
    prompts_file = Path(prompts_file).resolve(strict=True)
    image_manifest = Path(image_manifest).resolve(strict=True)
    receipt_path = prompts_file.parent / "receipt.json"
    if not receipt_path.is_file():
        if require_verified:
            raise ValueError(
                "Missing authenticated GT-free Lean prompt producer receipt")
        return {
            "status": "external_unverified",
            "receipt_sha256": None,
            "note": "Frozen prompt mask SHA256 does not prove it was generated without GT",
        }
    data = json.loads(receipt_path.read_text("utf8"))
    producer = Path(__file__).resolve().with_name("export_lean_fgpi_prompts.py")
    needed = {
        "protocol": PROTOCOL,
        "producer_source_sha256": file_sha256(producer),
        "image_only_manifest_sha256": file_sha256(image_manifest),
        "lean_checkpoint_sha256": backbone_sha256,
        "lean_source_revision": lean_source_revision,
        "frozen_prompt_manifest_sha256": file_sha256(prompts_file),
    }
    if not isinstance(data, dict) or any(data.get(k) != v for k, v in needed.items()):
        raise ValueError(
            "Lean automatic prompt producer receipt is inconsistent "
            "with current source, inputs, backbone, or frozen prompt bytes")
    completed = sum(
        row.get("status") == "completed"
        for row in data.get("per_video", {}).values())
    if (not isinstance(data.get("per_video"), dict)
            or type(data.get("source_video_count")) is not int
            or type(data.get("promptable_video_count")) is not int
            or completed != data["promptable_video_count"]
            or not 0 < completed <= data["source_video_count"]
            or data.get("mode") not in ("fgpi", "box_only")):
        raise ValueError("Malformed automatic prompt producer coverage receipt")
    return {
        "status": "verified_same_backbone_automatic_mask",
        "receipt_sha256": file_sha256(receipt_path),
        "mode": data["mode"],
        "promptable_video_count": completed,
        "source_video_count": data["source_video_count"],
        "note": "Proves a byte-consistent pinned producer receipt, not absence of external data fabrication",
    }
