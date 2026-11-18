"""Strict, independent experiment specification and provenance verification.

No model is loaded here. This module validates an experiment before expensive
training/inference and records the exact bytes used to define an execution.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

STAGES = ("data", "training", "inference", "evaluation")
TRACKS = ("medsam2_highres", "lean_sam2_aggregation", "reference_baseline")
PROMPTS = ("oracle_mask", "oracle_box", "automatic_box", "automatic_mask", "text", "none")
REQUIRED = {"schema_version", "experiment_id", "track", "dataset", "split",
            "initialization", "checkpoint", "stages"}
OPTIONAL = {"hypothesis", "seed", "notes"}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _require_keys(obj, required, optional=(), label="object"):
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing, extra = set(required)-set(obj), set(obj)-set(required)-set(optional)
    if missing or extra:
        raise ValueError(f"{label}: missing {sorted(missing)}, unknown {sorted(extra)}")

def _file_record(record, root, label):
    _require_keys(record, {"path", "sha256"}, label=label)
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError(f"{label}.path must be a nonempty string")
    if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(
        c not in "0123456789abcdef" for c in record["sha256"]
    ):
        raise ValueError(f"{label}.sha256 must be lowercase SHA256")
    path = (root / record["path"]).resolve(strict=True)
    if not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError(f"{label}: file bytes do not match pinned SHA256: {path}")
    return path

def load_protocol(path):
    path = Path(path).resolve(strict=True)
    spec = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(spec, REQUIRED, OPTIONAL, "protocol")
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported protocol schema_version")
    if (not isinstance(spec["experiment_id"], str) or not spec["experiment_id"].strip()
            or "/" in spec["experiment_id"] or "\\" in spec["experiment_id"]
            or spec["experiment_id"] in (".", "..")):
        raise ValueError("experiment_id must be nonempty")
    if spec["track"] not in TRACKS:
        raise ValueError(f"Unknown research track: {spec['track']}")
    if "seed" in spec and type(spec["seed"]) is not int:
        raise ValueError("seed must be an integer")
    _require_keys(spec["dataset"], {"name", "manifest"}, {"patient_key"}, "dataset")
    if not isinstance(spec["dataset"]["name"], str) or not spec["dataset"]["name"]:
        raise ValueError("dataset.name must be nonempty")
    if spec["split"] not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    _require_keys(spec["initialization"], {"source", "prompt_budget"}, label="initialization")
    if spec["initialization"]["source"] not in PROMPTS:
        raise ValueError("Invalid initialization source")
    if (type(spec["initialization"]["prompt_budget"]) is not int
            or spec["initialization"]["prompt_budget"] < 0):
        raise ValueError("prompt_budget must be a nonnegative integer")
    if not isinstance(spec["stages"], dict) or set(spec["stages"]) != set(STAGES):
        raise ValueError("stages must declare exactly data/training/inference/evaluation")
    for stage, cfg in spec["stages"].items():
        _require_keys(cfg, {"command", "artifacts"}, label=f"stages.{stage}")
        command = cfg["command"]
        if not isinstance(command, list) or any(
            not isinstance(token, str) or not token for token in command
        ):
            raise ValueError(f"stages.{stage}.command must be a list of nonempty strings")
        if not isinstance(cfg["artifacts"], list) or any(
            not isinstance(item, str) or not item for item in cfg["artifacts"]
        ):
            raise ValueError(f"stages.{stage}.artifacts must be a string list")
        if command and not cfg["artifacts"]:
            raise ValueError(f"stages.{stage}: executable stages must declare artifacts")
    return path, spec

def verify_protocol(path):
    path, spec = load_protocol(path)
    root = path.parent
    checked = {"protocol": {"path": str(path), "sha256": digest(path)}}
    checked["dataset_manifest"] = str(_file_record(spec["dataset"]["manifest"], root, "dataset.manifest"))
    checked["checkpoint"] = str(_file_record(spec["checkpoint"], root, "checkpoint"))
    return path, spec, checked

def comparison_key(spec):
    """Conditions that must match for a causal paired comparison."""
    return (
        spec["dataset"]["manifest"]["sha256"],
        spec["checkpoint"]["sha256"], spec["split"],
        spec["initialization"]["source"],
        spec["initialization"]["prompt_budget"],
    )
