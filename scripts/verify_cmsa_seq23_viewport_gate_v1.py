#!/usr/bin/env python3
"""Verify saved label-free CMSA seq23 smoke artifacts before a full run."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cmsa_seq23_viewport_v1 import OUTPUT_ROOT, STAGE_SHA, read, sha, verify_stage
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def audit(stage: Path, smoke: Path) -> dict:
    stage, smoke = stage.resolve(strict=True), smoke.resolve(strict=True)
    if not smoke.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Smoke must be in Drive experiment storage")
    guard = install_mask_open_guard(ROOT / "data/polypgen_sequence.jsonl")
    verify_stage(stage)
    receipt_path = smoke / "receipt.json"
    receipt_sha = sha(receipt_path)
    receipt = read(receipt_path)
    identity = receipt["identity"]
    if (receipt.get("passed") is not True or receipt.get("smoke") is not True
            or receipt.get("finite_clip_calls") != 1 or receipt.get("label_guard", {}).get("open_attempts") != 0
            or identity.get("stage_sha256") != STAGE_SHA
            or identity.get("runner_sha256") != sha(ROOT / "scripts/run_cmsa_seq23_viewport_v1.py")
            or identity.get("verifier_sha256") != sha(Path(__file__))
            or not isinstance(receipt.get("state_sha256"), str) or len(receipt["state_sha256"]) != 64):
        raise ValueError("Smoke receipt identity or execution differs")
    records = receipt["records"]
    if [r.get("frame") for r in records] != list(range(4)):
        raise ValueError("Smoke output coverage differs")
    for row in records:
        prediction = smoke / "predictions" / "seq23" / f"{row['frame']:06d}.png"
        if sha(prediction) != row["prediction_sha256"]:
            raise ValueError("Smoke prediction hash differs")
        with Image.open(prediction) as image:
            if image.format != "PNG" or image.size != (834, 720):
                raise ValueError("Smoke prediction geometry differs")
        if sha(prediction) != row["prediction_sha256"]:
            raise ValueError("Smoke prediction changed during verification")
    if sha(receipt_path) != receipt_sha or guard["open_attempts"]:
        raise ValueError("Smoke receipt changed or annotation open attempted")
    return {"passed": True, "identity": identity, "state_sha256": receipt["state_sha256"],
              "smoke_receipt": str(receipt_path), "smoke_receipt_sha256": receipt_sha,
              "verifier_sha256": sha(Path(__file__)), "label_guard": guard,
              "scope": "One targeted MPS smoke; not a repeated-run or efficacy claim."}


def verify(stage: Path, smoke: Path, output: Path) -> None:
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists() or output.suffix != ".json":
        raise ValueError("New gate must be in Drive experiment storage")
    result = audit(stage, smoke)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "gate": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.stage, args.smoke, args.output)
