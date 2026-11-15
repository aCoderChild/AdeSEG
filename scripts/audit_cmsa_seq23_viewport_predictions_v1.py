#!/usr/bin/env python3
"""Independently verify all CMSA viewport predictions before opening labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cmsa_seq23_viewport_v1 import (
    AUTHOR, CHECKPOINT_SHA, OUTPUT_ROOT, REVISION, dependency_hashes, read, sha, verify_stage,
)
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_cmsa_seq23_viewport_gate_v1 import audit as audit_smoke


def audit(stage: Path, checkpoint: Path, run: Path) -> dict:
    stage, checkpoint, run = stage.resolve(strict=True), checkpoint.resolve(strict=True), run.resolve(strict=True)
    if not all(p.is_relative_to(OUTPUT_ROOT) for p in (stage, checkpoint, run)) or sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Input location or released checkpoint differs")
    guard = install_mask_open_guard(ROOT / "data/polypgen_sequence.jsonl")
    verify_stage(stage)
    receipt_path = run / "receipt.json"
    receipt_sha = sha(receipt_path)
    receipt = read(receipt_path)
    identity = receipt["identity"]
    gate_path = Path(receipt["gate"]).resolve(strict=True)
    if not gate_path.is_relative_to(OUTPUT_ROOT) or sha(gate_path) != receipt["gate_sha256"]:
        raise ValueError("Full-run smoke gate differs")
    gate = read(gate_path)
    if (receipt.get("passed") is not True or receipt.get("smoke") is not False
            or receipt.get("label_guard", {}).get("open_attempts") != 0
            or receipt.get("finite_clip_calls") != 53 or gate != audit_smoke(stage, Path(gate["smoke_receipt"]).parent)
            or gate["identity"] != identity or gate["state_sha256"] != receipt["state_sha256"]
            or identity.get("stage_sha256") != sha(stage / "stage.json")
            or identity.get("checkpoint_sha256") != sha(checkpoint)
            or identity.get("runner_sha256") != sha(ROOT / "scripts/run_cmsa_seq23_viewport_v1.py")
            or identity.get("verifier_sha256") != sha(ROOT / "scripts/verify_cmsa_seq23_viewport_gate_v1.py")
            or identity.get("statistics_sha256") != sha(AUTHOR / "lib/dataloader/statistics.pth")
            or identity.get("author_test_sha256") != sha(AUTHOR / "scripts/my_test.py")
            or identity.get("author_revision") != REVISION or identity.get("normalization") != "released"
            or identity.get("device") != "mps" or identity.get("dtype") != "float32"
            or identity.get("hash_seed") != "0" or identity.get("fallback") != "0"
            or identity.get("fast_math") != "0"
            or identity.get("dependencies") != dependency_hashes()):
        raise ValueError("Full-run identity, guard, smoke or clip coverage differs")
    revision = subprocess.check_output(['git', '-C', str(AUTHOR.parent), 'rev-parse', 'HEAD'], text=True).strip()
    clean = subprocess.run(['git', '-C', str(AUTHOR.parent), 'diff', '--quiet', 'HEAD', '--', 'cmsa_project'])
    if revision != REVISION or clean.returncode != 0:
        raise ValueError("CMSA author checkout differs")
    records = receipt["records"]
    if [r.get("frame") for r in records] != list(range(56)):
        raise ValueError("Full-run prediction coverage differs")
    for row in records:
        prediction = run / "predictions" / "seq23" / f"{row['frame']:06d}.png"
        if sha(prediction) != row["prediction_sha256"]:
            raise ValueError("Prediction changed after inference")
        with Image.open(prediction) as image:
            if image.format != "PNG" or image.size != (834, 720):
                raise ValueError("Prediction geometry differs")
        if sha(prediction) != row["prediction_sha256"]:
            raise ValueError("Prediction changed during verification")
    if (sha(receipt_path) != receipt_sha or sha(gate_path) != receipt["gate_sha256"]
            or sha(stage / "stage.json") != identity["stage_sha256"]
            or sha(AUTHOR / "lib/dataloader/statistics.pth") != identity["statistics_sha256"]
            or sha(AUTHOR / "scripts/my_test.py") != identity["author_test_sha256"] or guard["open_attempts"]):
        raise ValueError("Receipt changed or annotation open was attempted")
    return {"passed": True, "run": str(run), "receipt_sha256": receipt_sha,
            "stage": str(stage), "stage_sha256": sha(stage / "stage.json"),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "verifier_sha256": sha(Path(__file__)), "label_guard": guard,
            "prediction_sha256": [r["prediction_sha256"] for r in records],
            "scope": "No-label output verification; not a score or a repeated-inference claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists() or output.suffix != ".json":
        raise ValueError("Output must be a new JSON file under Drive experiment storage")
    result = audit(args.stage, args.checkpoint, args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "verified_predictions": len(result["prediction_sha256"])}))


if __name__ == "__main__":
    main()
