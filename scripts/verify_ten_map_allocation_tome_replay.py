"""Fresh-process replay of the ten-map recent-dense regional gate trajectory."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_ten_map_allocation_tome_gate import ARMS, PROTOCOL
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    run_condition,
    verify_native_setup,
)


GATE = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "ten_map_allocation_tome_polypgen_v1/engineering_gate_attempt01"
)
REFERENCE_SEQUENCE = "seq3"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    gate_setup = json.loads((GATE / "setup.json").read_text())
    gate_results = json.loads((GATE / "results.json").read_text())
    gate_completion = json.loads((GATE / "completion.json").read_text())
    if not gate_completion.get("passed"):
        raise RuntimeError("The ten-map engineering gate did not pass")
    if digest(PROTOCOL) != gate_setup["protocol_sha256"]:
        raise RuntimeError("Protocol hash changed since the engineering gate")
    if digest(args.checkpoint) != gate_setup["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash changed since the engineering gate")
    for source, expected in gate_setup["source_hashes"].items():
        if digest(ROOT / source) != expected:
            raise RuntimeError(f"Source changed since the engineering gate: {source}")

    manifest = ROOT / "data/polypgen_sequence.jsonl"
    verify_native_setup(args.checkpoint, manifest)
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    prompt, native_rows, native_receipt = native_reference(
        NATIVE_RESULTS.parent,
        native_results,
        native_protocol,
        REFERENCE_SEQUENCE,
        videos[REFERENCE_SEQUENCE],
    )

    arm_receipts = {}
    compared_fields = (
        "frame",
        "present",
        "scores",
        "absent_fp",
        "prediction_sha256",
        "stored_sha256",
        "object_score",
        "condition_name",
        "allocation",
        "available_noncond_frames",
        "attention_reads",
    )
    for arm, matching, allocation in ARMS:
        replay_rows, summary = run_condition(
            matching,
            REFERENCE_SEQUENCE,
            videos[REFERENCE_SEQUENCE],
            prompt,
            native_rows,
            args.checkpoint,
            args.output_dir,
            max_noncond_maps=10,
            allocation=allocation,
            condition_name=arm,
        )
        expected_rows = gate_results["arms"][arm][REFERENCE_SEQUENCE]
        if len(replay_rows) != len(expected_rows):
            raise RuntimeError(
                f"{arm} replay target count differs: {len(replay_rows)} vs {len(expected_rows)}"
            )
        mismatches = []
        for actual, expected in zip(replay_rows, expected_rows, strict=True):
            different = [key for key in compared_fields if actual.get(key) != expected.get(key)]
            if different:
                mismatches.append({"frame": actual.get("frame"), "fields": different})
        arm_receipts[arm] = {
            "passed": not mismatches,
            "targets_compared": len(replay_rows),
            "mismatches": mismatches,
            "summary": summary,
        }
    passed = all(row["passed"] for row in arm_receipts.values())
    receipt = {
        "passed": passed,
        "classification": "independent-process engineering replay; not an efficacy result",
        "reference_gate": str(GATE),
        "reference_sequence": REFERENCE_SEQUENCE,
        "arms": list(arm_receipts),
        "targets_compared_per_arm": 14,
        "total_target_comparisons": sum(row["targets_compared"] for row in arm_receipts.values()),
        "compared_fields": list(compared_fields),
        "arm_results": arm_receipts,
        "gate_protocol_sha256": gate_setup["protocol_sha256"],
        "gate_source_hashes": gate_setup["source_hashes"],
        "native_reference_receipt": native_receipt,
        "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write(args.output_dir / "replay_receipt.json", receipt)
    write(args.output_dir / "progress.json", {
        "status": "complete" if passed else "failed",
        "passed": passed,
        "total_target_comparisons": receipt["total_target_comparisons"],
        "mismatch_count": sum(len(row["mismatches"]) for row in arm_receipts.values()),
    })
    if not passed:
        raise RuntimeError(f"Fresh-process replay mismatch by arm: {arm_receipts}")
    print(
        f"PASS: fresh-process replay matched all four arms, 14 seq3 targets each; "
        f"receipt: {args.output_dir / 'replay_receipt.json'}"
    )


if __name__ == "__main__":
    main()
