"""Strict inference loader for NEW reconstructed MedSAM2 adapter checkpoints.

This is not a historical forward-parity loader. It verifies serialized weight
bytes, current adapter implementation, expected backbone and model config,
and the training delta before authorizing inference.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from .adapter import MedicalHighResolutionAdapter

FORMAT = "reconstructed_medsam2_highres_teacher_v1"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_reconstructed_medical_adapter(checkpoint, *, expected_sha256,
                                      backbone_sha256, model_config):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    actual = sha256(checkpoint)
    if actual != expected_sha256:
        raise ValueError("Adapter checkpoint SHA256 mismatch")
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("Expected reconstructed medical adapter checkpoint")
    if record.get("reconstruction_not_historical_forward") is not True:
        raise ValueError("Missing reconstruction provenance")
    if record.get("backbone_sha256") != backbone_sha256:
        raise ValueError("Medical backbone SHA256 mismatch")
    if record.get("model_config") != model_config:
        raise ValueError("MedSAM2 configuration mismatch")
    source = Path(__file__).resolve().with_name("adapter.py")
    if record.get("source_sha256") != sha256(source):
        raise ValueError("Medical adapter source SHA256 mismatch")
    if type(record.get("training_steps")) is not int or record["training_steps"] <= 0:
        raise ValueError("No verified adapter training steps")
    delta = record.get("maximum_delta")
    if type(delta) not in (float, int) or not 0 < delta <= 1:
        raise ValueError("Missing or invalid maximum_delta")
    model = MedicalHighResolutionAdapter(maximum_delta=float(delta))
    model.load_historical_state(checkpoint, expected_sha256=actual,
                                allowed_formats={FORMAT})
    return model.eval(), record
