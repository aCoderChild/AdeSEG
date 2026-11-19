"""Fingerprint the complete active capture-training pipeline for safe resumption."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "medical": (
        "training/medical.py",
        "training/medical_highres_training.py",
        "training/restart_state.py",
        "training/training_provenance.py",
        "training/split_guard.py",
        "adeseg/losses/segmentation.py",
        "modeling/medical_highres/teacher_loss.py",
        "modeling/positive_teacher_preservation_loss.py",
    ),
    "lean": (
        "training/lean.py",
        "training/restart_state.py",
        "training/training_provenance.py",
        "training/split_guard.py",
        "modeling/cmsa_boundary_training_loss.py",
        "adeseg/losses/foreground_preservation.py",
    ),
}


def training_source_digest(track):
    if track not in FILES:
        raise ValueError("Unknown training track")
    h = hashlib.sha256()
    for name in FILES[track]:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()
