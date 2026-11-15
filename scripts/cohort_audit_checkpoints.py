"""Durable evidence checkpoints for the frozen cohort runner."""

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_condition_checkpoint(folder, sequence, condition, rows, calls, runtime):
    """Write attention evidence and a hash-bound receipt after a complete arm."""
    folder = Path(folder)
    frame_path = folder / "frames.json"
    audit_path = folder / "attention_audit.json"
    receipt_path = folder / "condition_checkpoint.json"
    if audit_path.exists() or receipt_path.exists():
        raise FileExistsError("Condition checkpoint already exists")
    if json.loads(frame_path.read_text()) != rows:
        raise ValueError("Saved frame rows differ from the completed condition")
    expected = [(row["frame"], layer) for row in rows for layer in range(4)]
    observed = [(call["frame"], call["layer"]) for call in calls]
    if observed != expected:
        raise ValueError("Attention audit does not contain all four layers per frame")
    _atomic_json(audit_path, calls)
    receipt = {
        "sequence": sequence,
        "condition": condition,
        "targets": len(rows),
        "attention_calls": len(calls),
        "frames_sha256": _digest(frame_path),
        "attention_audit_sha256": _digest(audit_path),
        "runtime": runtime,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def save_sequence_checkpoint(folder, sequence, summary):
    """Persist the original cross-condition audit only after it succeeds."""
    path = Path(folder) / "sequence_audit.json"
    if path.exists():
        raise FileExistsError("Sequence audit already exists")
    _atomic_json(path, {"sequence": sequence, "audit": summary})
    return _digest(path)


def load_condition_checkpoint(folder, sequence, condition):
    """Reject missing, mismatched or altered evidence before any reuse."""
    folder = Path(folder)
    receipt = json.loads((folder / "condition_checkpoint.json").read_text())
    if receipt["sequence"] != sequence or receipt["condition"] != condition:
        raise ValueError("Condition checkpoint identity mismatch")
    frames = folder / "frames.json"
    audit = folder / "attention_audit.json"
    if _digest(frames) != receipt["frames_sha256"] or _digest(audit) != receipt["attention_audit_sha256"]:
        raise ValueError("Condition checkpoint content hash mismatch")
    rows = json.loads(frames.read_text())
    calls = json.loads(audit.read_text())
    expected = [(row["frame"], layer) for row in rows for layer in range(4)]
    if len(rows) != receipt["targets"] or len(calls) != receipt["attention_calls"]:
        raise ValueError("Condition checkpoint count mismatch")
    if [(call["frame"], call["layer"]) for call in calls] != expected:
        raise ValueError("Condition checkpoint attention order mismatch")
    return rows, calls, receipt["runtime"]


def archive_incomplete_condition(output, sequence, condition):
    """Preserve an incomplete arm before restarting that arm from its prompt."""
    output = Path(output)
    folder = output / condition / sequence
    if not folder.exists():
        return None
    if (folder / "condition_checkpoint.json").exists():
        raise ValueError("A completed condition must be verified, not archived")
    archive = output / "interrupted_conditions"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / f"{sequence}_{condition}_{uuid.uuid4().hex}"
    folder.rename(destination)
    return destination
