"""Audit native-prefix parity in completed and active ten-map sequence outputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_rows(rows, native_rows, expected_count, *, label):
    if len(rows) != expected_count:
        raise RuntimeError(f"{label}: got {len(rows)} rows, expected {expected_count}")
    if len(rows) > len(native_rows):
        raise RuntimeError(f"{label}: more rows than native reference")
    if [int(r["frame"]) for r in rows] != [
        int(r["frame"]) for r in native_rows[:len(rows)]
    ]:
        raise RuntimeError(f"{label}: target frame IDs/order differ from native")
    if [r["present"] for r in rows] != [
        r["present"] for r in native_rows[:len(rows)]
    ]:
        raise RuntimeError(f"{label}: GT-presence flags differ from native")

    prefix = [r for r in rows if int(r["available_noncond_frames"]) <= 6]
    mismatches = []
    for row in prefix:
        ref = native_rows[int(row["frame"]) - 1]
        fields = [
            key for key in ("prediction_sha256", "stored_sha256", "scores", "absent_fp")
            if row.get(key) != ref.get(key)
        ]
        reads = row.get("attention_reads", [])
        if (
            fields
            or [read.get("layer") for read in reads] != [0, 1, 2, 3]
            or any(
                read.get("spatial_keys_before") != read.get("spatial_keys_after")
                for read in reads
            )
        ):
            mismatches.append({"frame": row.get("frame"), "fields": fields})
    if mismatches:
        raise RuntimeError(f"{label}: native-prefix parity failed: {mismatches[:3]}")
    return {
        "targets": len(rows),
        "first_frame": rows[0]["frame"] if rows else None,
        "last_frame": rows[-1]["frame"] if rows else None,
        "native_prefix_targets": len(prefix),
        "native_prefix_hash_mismatches": 0,
        "all_four_layers_and_no_merges_checked": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-frames", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--completed-arm", required=True)
    parser.add_argument("--active-arm", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    native_rows = read_json(args.native_frames)
    progress_path = args.run_dir / "progress.json"
    completed_path = args.run_dir / args.completed_arm / args.sequence / "frames.json"
    active_path = args.run_dir / args.active_arm / args.sequence / "frames.json"

    # The active arm is being appended while this audit runs; accept only a
    # stable snapshot whose row count agrees with the runner's progress record.
    for attempt in range(5):
        progress_before = read_json(progress_path)
        completed_bytes = completed_path.read_bytes()
        active_bytes = active_path.read_bytes()
        progress_after = read_json(progress_path)
        if progress_before != progress_after:
            time.sleep(0.25)
            continue
        completed_rows = json.loads(completed_bytes)
        active_rows = json.loads(active_bytes)
        if (
            progress_after.get("sequence") != args.sequence
            or progress_after.get("condition") != args.active_arm
            or progress_after.get("status") != "running"
            or len(active_rows) != int(progress_after["targets_done"])
        ):
            time.sleep(0.25)
            continue
        completed_result = audit_rows(
            completed_rows, native_rows, len(native_rows), label=args.completed_arm
        )
        active_result = audit_rows(
            active_rows, native_rows, int(progress_after["targets_done"]),
            label=args.active_arm,
        )
        receipt = {
            "passed": True,
            "classification": "read-only per-sequence integrity audit; not an efficacy result",
            "sequence": args.sequence,
            "completed_arm": args.completed_arm,
            "active_arm": args.active_arm,
            "paths": {
                "run_dir": str(args.run_dir),
                "native_frames": str(args.native_frames),
                "completed_arm_frames": str(completed_path),
                "active_arm_frames": str(active_path),
                "progress": str(progress_path),
            },
            "progress_snapshot": progress_after,
            "checks": {
                args.completed_arm: completed_result,
                args.active_arm: active_result,
            },
            "input_sha256": {
                "audit_script": sha256(Path(__file__).resolve()),
                "native_frames": sha256(args.native_frames),
                "completed_arm_frames": hashlib.sha256(completed_bytes).hexdigest(),
                "active_arm_frames": hashlib.sha256(active_bytes).hexdigest(),
                "progress": sha256(progress_path),
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return
    raise RuntimeError("Could not capture a stable progress/output snapshot")


if __name__ == "__main__":
    main()
