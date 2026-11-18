#!/usr/bin/env python3
"""Gate the pinned Lean author runner and compare its full run with native B+."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity  # noqa: E402

SEQUENCES = [f"seq{i}" for i in range(2, 24) if i != 7]
ASSETS = (
    "configs/runtime/lvos.yaml",
    "configs/lean_sam2/lean_sam2.yaml",
    "checkpoints/bypass_bottleneck_base.pth",
    "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
)


def read(path):
    return json.loads(path.read_text())


def check_gate(folder, sequence, expected_count):
    validation, delivery = read(folder / "validation.json"), read(folder / "delivery.json")
    summary = read(folder / "summary.json")
    completed = summary["sequences"]
    if not (validation["passed"] and delivery["completed"] and summary["completed"]):
        raise ValueError(f"Incomplete gate: {folder}")
    if len(completed) != 1 or completed[0]["sequence"] != sequence or completed[0]["frames"] != expected_count:
        raise ValueError(f"Incorrect gate coverage: {folder}")
    replay = read(folder / sequence / "replay.json")
    if len(replay) != expected_count or not all(row["exact"] for row in replay):
        raise ValueError(f"Nonexact or incomplete replay: {folder}")
    for stage in ("first", "replay"):
        if not read(folder / sequence / f"parameter_integrity_{stage}.json")["unchanged"]:
            raise ValueError(f"Changed parameters: {folder}")
    return validation["signature"]


def compare(lean, native):
    result = []
    for name in SEQUENCES:
        files = [base / name / "frames.jsonl" for base in (lean, native)]
        left, right = [[json.loads(line) for line in p.read_text().splitlines()] for p in files]
        inputs = [read(base / name / "inputs.json") for base in (lean, native)]
        fingerprint = lambda records: [(r["frame"], r["image_sha256"], r["mask_sha256"]) for r in records]
        if fingerprint(inputs[0]) != fingerprint(inputs[1]):
            raise ValueError(f"Unmatched input content: {name}")
        keys = lambda rows: [(r["source_frame_index"], r["present"]) for r in rows]
        if not left or keys(left) != keys(right):
            raise ValueError(f"Unmatched frame IDs or presence: {name}")
        present = [(a["scores"]["dice"], b["scores"]["dice"]) for a, b in zip(left, right) if a["present"]]
        empty = [(a["absent_fp"], b["absent_fp"]) for a, b in zip(left, right) if not a["present"]]
        if not present or any(type(x) is not bool or type(y) is not bool for x, y in empty):
            raise ValueError(f"Invalid present/absence records: {name}")
        lm, nm = np.mean(present, axis=0)
        result.append({"sequence": name, "targets": len(left), "present": len(present),
                       "empty": len(empty), "lean_dice": float(lm), "native_dice": float(nm),
                       "delta": float(lm - nm), "lean_false_alarms": sum(x for x, _ in empty),
                       "native_false_alarms": sum(y for _, y in empty)})
    differences = {r["sequence"]: r["delta"] for r in result}
    return {"scope": "exploratory matched PolypGen; full Lean system versus native B+",
            "sequences": result, "target_frames": sum(r["targets"] for r in result),
            "empty_frames": sum(r["empty"] for r in result),
            "lean_false_alarms": sum(r["lean_false_alarms"] for r in result),
            "native_false_alarms": sum(r["native_false_alarms"] for r in result),
            "lean_dice": float(np.mean([r["lean_dice"] for r in result])),
            "native_dice": float(np.mean([r["native_dice"] for r in result])),
            "mean_delta": float(np.mean(list(differences.values()))),
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(differences)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "seq2-gate", "seq3-gate", "author-parity", "native-dir", "output-dir"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    args = p.parse_args()
    repo, revision = source_identity("lean")
    identity = check_gate(args.seq2_gate, "seq2", 59)
    if identity != check_gate(args.seq3_gate, "seq3", 14):
        raise ValueError("Validation source/device/checkpoint mismatch")
    parity = read(args.author_parity)
    if not parity["passed"] or len(parity["frames"]) != 14 or any(x["changed_pixels"] for x in parity["frames"]):
        raise ValueError("Author CLI parity failed")
    if parity["author_source_sha256"] != digest(repo / "tools/vos_inference.py"):
        raise ValueError("Author parity was run against different source")
    if digest(args.checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from validation")
    if digest(ROOT / "scripts/run_author_baseline.py") != identity["runner_sha256"]:
        raise ValueError("Author runner differs from validation")
    if digest(ROOT / "data/polypgen_sequence.jsonl") != identity["manifest_sha256"]:
        raise ValueError("Manifest differs from validation")
    native_setup = read(args.native_dir / "setup.json")
    if not read(args.native_dir / "delivery.json")["completed"]:
        raise ValueError("Native baseline incomplete")
    for key in ("checkpoint_sha256", "manifest_sha256", "device", "torch", "precision", "config", "memory_stride", "apply_postprocessing"):
        if identity[key] != native_setup[key]:
            raise ValueError(f"Native baseline mismatch: {key}")
    assets = {name: digest(repo / name) for name in ASSETS}
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"author_revision": revision, "lean_signature": identity, "asset_sha256": assets,
               "seq2_gate_sha256": digest(args.seq2_gate / "validation.json"),
               "seq3_gate_sha256": digest(args.seq3_gate / "validation.json"),
               "author_parity_sha256": digest(args.author_parity),
               "native_setup_sha256": digest(args.native_dir / "setup.json"),
               "native_frames_sha256": {n: digest(args.native_dir / n / "frames.jsonl") for n in SEQUENCES},
               "native_inputs_sha256": {n: digest(args.native_dir / n / "inputs.json") for n in SEQUENCES},
               "wrapper_sha256": digest(Path(__file__)),
               "comparison": "equal-sequence mean present Dice; paired sequence bootstrap; false alarms on empty frames"}
    (output / "preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    command = [sys.executable, str(ROOT / "scripts/run_author_baseline.py"), "--method", "lean",
               "--checkpoint", str(args.checkpoint), "--output-dir", str(output / "full"),
               "--device", "cpu", "--validation-dir", str(args.seq2_gate)]
    with (output / "launch.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if assets != {name: digest(repo / name) for name in ASSETS} or source_identity("lean")[1] != revision:
        raise RuntimeError("Lean author assets changed during run")
    if not read(output / "full/delivery.json")["completed"]:
        raise RuntimeError("Lean output incomplete")
    if receipt["native_frames_sha256"] != {n: digest(args.native_dir / n / "frames.jsonl") for n in SEQUENCES}:
        raise RuntimeError("Native reference changed during run")
    if receipt["native_inputs_sha256"] != {n: digest(args.native_dir / n / "inputs.json") for n in SEQUENCES}:
        raise RuntimeError("Native input provenance changed during run")
    comparison = compare(output / "full", args.native_dir)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
