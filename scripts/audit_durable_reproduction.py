"""Recheck completed durable cohort triplets and compare available v1 rows.

This reads saved outputs only. It does not rescore candidate masks or perform
inference, and comparisons with the interrupted run are not new-data evidence.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cohort_audit_checkpoints import _atomic_json, load_condition_checkpoint
from scripts.memory_token_io import digest, validate_output
from scripts.run_spatiotemporal_value_exchange_fullprop_gate import CONDITIONS, verify_fullprop_audits


def audit(current, prior, protocol_path):
    current, prior, protocol_path = Path(current), Path(prior), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if digest(protocol_path) != protocol_path.with_suffix(protocol_path.suffix + ".sha256").read_text().strip():
        raise ValueError("Frozen protocol checksum differs")
    setup = json.loads((current / "setup.json").read_text())
    if setup["protocol_sha256"] != digest(protocol_path) or setup["protocol"] != protocol:
        raise ValueError("Attempt setup does not match the frozen protocol")
    if any(digest(ROOT / name) != expected for name, expected in protocol["source_hashes"].items()):
        raise ValueError("Frozen source differs")
    snapshot = (current / "partial_results.json").read_bytes()
    partial = json.loads(snapshot)
    prior_partial = json.loads((prior / "partial_results.json").read_text())
    current_complete = set.intersection(*(set(partial["summaries"][c]) for c in CONDITIONS))
    prior_complete = set.intersection(*(set(prior_partial["summaries"][c]) for c in CONDITIONS))
    sequences = [s for s in protocol["sequences"] if s in current_complete and
                 (current / "sequence_audits" / s / "sequence_audit.json").is_file()]
    if not sequences:
        raise ValueError("No completed sequence audit is available")
    native_data = json.loads(Path(protocol["native_reference"]["results"]).read_text())
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "sequences": sequences,
              "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(), "per_sequence": {},
              "scope": "Saved score rows and raw attention calls; candidate masks are not independently rescored. Prior comparisons are same-sequence repeatability, not independent-data efficacy."}
    for seq in sequences:
        expected = protocol["sequence_targets"][seq]
        target_frames = list(range(expected["prompt"] + 1, expected["frames"]))
        native_ref = native_data["frame_evidence"][seq]
        if digest(native_ref["path"]) != native_ref["sha256"]:
            raise ValueError(f"Native reference hash mismatch: {seq}")
        native_rows = {r["frame"]: r for r in json.loads(Path(native_ref["path"]).read_text())}
        rows_by_condition, audits, entries = {}, {}, {}
        for condition in CONDITIONS:
            folder = current / condition / seq
            rows, calls, runtime = load_condition_checkpoint(folder, seq, condition)
            if [r["frame"] for r in rows] != target_frames:
                raise ValueError(f"Target coverage mismatch: {seq}/{condition}")
            if not runtime["parameter_hash_unchanged"] or not runtime["temporal_slots_restored"]:
                raise ValueError(f"Runtime invariants failed: {seq}/{condition}")
            if any(row["present"] != native_rows[row["frame"]]["present"] for row in rows):
                raise ValueError(f"Presence mismatch: {seq}/{condition}")
            if condition == "noop":
                for row in rows:
                    native = native_rows[row["frame"]]
                    if any(row[k] != native[k] for k in ("scores", "prediction_sha256", "stored_sha256", "absent_fp")):
                        raise ValueError(f"Native/no-op mismatch: {seq}/{row['frame']}")
            present = [r for r in rows if r["present"]]
            score = sum(r["scores"]["dice"] for r in present) / len(present)
            if abs(score - partial["summaries"][condition][seq]["present_dice"]) > 1e-12:
                raise ValueError(f"Saved mean Dice mismatch: {seq}/{condition}")
            prior_path = prior / condition / seq / "frames.json"
            prior_rows = json.loads(prior_path.read_text()) if seq in prior_complete else None
            entries[condition] = {"targets": len(rows), "attention_calls": len(calls),
                                  "present_dice": score, "runtime_seconds": runtime["wall_seconds"],
                                  "prior_frame_rows_identical": rows == prior_rows if prior_rows is not None else None,
                                  "current_frames_sha256": digest(folder / "frames.json"),
                                  "prior_frames_sha256": digest(prior_path) if prior_rows is not None else None}
            rows_by_condition[condition], audits[condition] = rows, calls
        recomputed = verify_fullprop_audits(audits, target_frames)
        saved = json.loads((current / "sequence_audits" / seq / "sequence_audit.json").read_text())
        if saved != {"sequence": seq, "audit": recomputed}:
            raise ValueError(f"Saved sequence audit differs: {seq}")
        result["per_sequence"][seq] = {"conditions": entries, "attention_audit": recomputed}
    result["condition_frame_records_checked"] = sum(
        sum(x["targets"] for x in item["conditions"].values()) for item in result["per_sequence"].values()
    )
    prior_checks = [item["prior_frame_rows_identical"]
                    for seq in result["per_sequence"].values()
                    for item in seq["conditions"].values()
                    if item["prior_frame_rows_identical"] is not None]
    result["prior_conditions_compared"] = len(prior_checks)
    result["prior_rows_all_identical"] = all(prior_checks) if prior_checks else None
    result["source_sha256"] = digest(Path(__file__))
    result["status"] = "passed" if all(prior_checks) else "prior_reproduction_mismatch"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.current, args.prior, args.protocol)
    _atomic_json(validate_output(args.output), result)
    print(json.dumps({"status": result["status"], "sequences": result["sequences"],
                      "condition_frame_records_checked": result["condition_frame_records_checked"]}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
