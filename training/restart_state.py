"""Authenticated epoch-boundary restart state for frozen-capture training.

Snapshots are complete only at epoch boundaries. A resumed run must use the
same model source, data bytes, optimizer settings and execution device.
This is NOT arbitrary mid-batch restart or cross-hardware bitwise parity.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def capture_rng():
    numpy_state = np.random.get_state()
    result = {
        "torch_cpu": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(),
                  int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "cuda": None, "mps": None,
    }
    if torch.cuda.is_available():
        result["cuda"] = [x.clone() for x in torch.cuda.get_rng_state_all()]
    if (hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state")
            and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        result["mps"] = torch.mps.get_rng_state().clone()
    return result


def restore_rng(result):
    if not isinstance(result, dict) or set(result) != {
        "torch_cpu", "python", "numpy", "cuda", "mps"
    }:
        raise ValueError("Incomplete RNG restoration record")
    if result["cuda"] is not None:
        if not torch.cuda.is_available() or len(result["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device RNG configuration changed")
    if result["mps"] is not None:
        if (not hasattr(torch, "mps") or not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()):
            raise ValueError("MPS device RNG not available for resume")
    torch.set_rng_state(result["torch_cpu"])
    random.setstate(result["python"])
    x = result["numpy"]
    np.random.set_state((x[0], np.asarray(x[1], dtype=np.uint32), x[2], x[3], x[4]))
    if result["cuda"] is not None:
        torch.cuda.set_rng_state_all(result["cuda"])
    if result["mps"] is not None:
        torch.mps.set_rng_state(result["mps"])


def save_epoch_snapshot(path, record):
    """Publish a complete checkpoint atomically without overwriting any file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".partial")
    if scratch.exists():
        raise FileExistsError(scratch)
    try:
        with scratch.open("xb") as stream:
            torch.save(record, stream)
        # Atomic hardlink refuses an existing destination, even under races.
        os.link(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def load_epoch_snapshot(path, expected_sha256, *, expected):
    path = Path(path).resolve(strict=True)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or digest(path) != expected_sha256):
        raise ValueError("Resume checkpoint SHA256 mismatch")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Resume checkpoint provenance/hyperparameters mismatch")
    if (type(record.get("completed_epochs")) is not int or record["completed_epochs"] < 1
            or type(record.get("training_steps")) is not int or record["training_steps"] < 1
            or not isinstance(record.get("state_dict"), dict)
            or not isinstance(record.get("optimizer_state_dict"), dict)
            or not isinstance(record.get("losses"), list)):
        raise ValueError("Resume checkpoint missing epoch/model/optimizer state")
    if not isinstance(record.get("rng_state"), dict):
        raise ValueError("Resume checkpoint missing complete RNG state")
    return record
