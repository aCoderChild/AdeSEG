#!/usr/bin/env python3
"""Read-only integrity audit for completed sequences in the five-arm ToMe run."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path


ARMS = (
    "native",
    "region_constrained_tome",
    "generic_local_tome",
    "calibrated_region_constrained_tome",
    "calibrated_generic_local_tome",
)
MATCHED_PAIRS = (
    ("region_constrained_tome", "generic_local_tome"),
    ("calibrated_region_constrained_tome", "calibrated_generic_local_tome"),
)
LAYER_FIELDS = (
    "spatial_keys_before",
    "spatial_keys_after",
    "pointer_keys",
    "memory_frames",
    "spatial_key_mass_sum",
    "key_mass_sum",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict:
    setup = read_json(run_dir / "setup.json")
    expected_arms = setup["conditions"]
    if tuple(expected_arms) != ARMS:
        raise ValueError(f"Unexpected condition order in setup.json: {expected_arms}")

    setup_hash = sha256(run_dir / "setup.json")
    completed = sorted((run_dir / "sequence_completion").glob("seq*.json"))
    if not completed:
        raise ValueError("No completed sequence receipts found")

    problems: list[str] = []
    sequence_receipts = []
    for completion_path in completed:
        completion = read_json(completion_path)
        sequence = completion["sequence"]
        targets = int(completion["targets"])
        if not completion.get("complete"):
            problems.append(f"{sequence}: completion receipt is not marked complete")
        if completion.get("conditions") != list(ARMS):
            problems.append(f"{sequence}: completion receipt has a different arm list")
        if completion.get("setup_sha256") != setup_hash:
            problems.append(f"{sequence}: completion receipt setup hash mismatch")

        summaries = {}
        rows = {}
        for arm in ARMS:
            arm_dir = run_dir / arm / sequence
            summary_path = arm_dir / "summary.json"
            frames_path = arm_dir / "frames.json"
            if not summary_path.exists() or not frames_path.exists():
                problems.append(f"{sequence}/{arm}: missing summary or frame records")
                continue
            summaries[arm] = read_json(summary_path)
            rows[arm] = read_json(frames_path)
            summary = summaries[arm]
            frame_ids = [row.get("frame") for row in rows[arm]]
            if not summary.get("complete"):
                problems.append(f"{sequence}/{arm}: summary is not marked complete")
            if int(summary.get("targets", -1)) != targets or len(rows[arm]) != targets:
                problems.append(f"{sequence}/{arm}: target count mismatch")
            if summary.get("setup_sha256") != setup_hash:
                problems.append(f"{sequence}/{arm}: setup hash mismatch")
            if arm != "native":
                if summary.get("parameters_unchanged") is not True:
                    problems.append(f"{sequence}/{arm}: SAM2 parameters were not certified unchanged")
                if int(summary.get("attention_layer_calls", -1)) != targets * 4:
                    problems.append(f"{sequence}/{arm}: expected four attention layers per target")
                for row in rows[arm]:
                    layers = row.get("attention_summary", [])
                    if [item.get("layer") for item in layers] != [0, 1, 2, 3]:
                        problems.append(f"{sequence}/{arm}/frame{row.get('frame')}: layer summary mismatch")
                    for item in layers:
                        before = item.get("spatial_keys_before")
                        mass = item.get("spatial_key_mass_sum")
                        pointer_count = item.get("pointer_keys")
                        all_mass = item.get("key_mass_sum")
                        if before is None or mass is None or pointer_count is None or all_mass is None:
                            problems.append(f"{sequence}/{arm}/frame{row.get('frame')}: incomplete mass audit")
                            continue
                        if abs(float(before) - float(mass)) > 1e-5:
                            problems.append(f"{sequence}/{arm}/frame{row.get('frame')}: spatial mass not conserved")
                        if abs(float(mass) + float(pointer_count) - float(all_mass)) > 1e-5:
                            problems.append(f"{sequence}/{arm}/frame{row.get('frame')}: total key mass mismatch")
            if not all(row.get("prediction_sha256") for row in rows[arm]):
                problems.append(f"{sequence}/{arm}: missing prediction hashes")

        if len(rows) != len(ARMS):
            continue
        native_ids = [row.get("frame") for row in rows["native"]]
        for arm in ARMS[1:]:
            if [row.get("frame") for row in rows[arm]] != native_ids:
                problems.append(f"{sequence}/{arm}: frame IDs differ from native")

        for treatment, control in MATCHED_PAIRS:
            treatment_rows = rows[treatment]
            control_rows = rows[control]
            for treatment_row, control_row in zip(treatment_rows, control_rows):
                for t_layer, c_layer in zip(
                    treatment_row.get("attention_summary", []),
                    control_row.get("attention_summary", []),
                ):
                    if any(t_layer.get(field) != c_layer.get(field) for field in LAYER_FIELDS):
                        problems.append(
                            f"{sequence}/{treatment}/{treatment_row.get('frame')}: "
                            f"matched budget differs from {control}"
                        )
            if int(summaries[control].get("matched_budget_checks", 0)) <= 0:
                problems.append(f"{sequence}/{control}: no matched-budget checks recorded")

        # The calibration experiment promises the same assignments and budgets
        # across scaled/unscaled arms as well as within each regional/control pair.
        for unscaled, scaled in (
            ("region_constrained_tome", "calibrated_region_constrained_tome"),
            ("generic_local_tome", "calibrated_generic_local_tome"),
        ):
            missing_or_different = False
            for unscaled_row, scaled_row in zip(rows[unscaled], rows[scaled]):
                for left, right in zip(
                    unscaled_row.get("attention_summary", []),
                    scaled_row.get("attention_summary", []),
                ):
                    assignment = left.get("merge_assignment_sha256")
                    if not assignment or assignment != right.get("merge_assignment_sha256"):
                        missing_or_different = True
            if missing_or_different:
                problems.append(f"{sequence}/{scaled}: exact assignments are missing or differ from {unscaled}")
        plan_paths = [
            run_dir / arm / sequence / "merge_plan.json.gz"
            for arm in ("region_constrained_tome", "calibrated_region_constrained_tome")
        ]
        if not all(path.exists() for path in plan_paths):
            problems.append(f"{sequence}: missing cross-calibration merge-budget plans")
        else:
            plans = [json.loads(gzip.decompress(path.read_bytes())) for path in plan_paths]
            if plans[0] != plans[1]:
                problems.append(f"{sequence}: calibrated/unscaled regional merge-budget plans differ")

        sequence_receipts.append({
            "sequence": sequence,
            "targets": targets,
            "prompt": completion.get("prompt"),
            "conditions": {
                arm: {
                    "targets": summaries[arm].get("targets"),
                    "attention_layer_calls": summaries[arm].get("attention_layer_calls"),
                    "matched_budget_checks": summaries[arm].get("matched_budget_checks", 0),
                    "parameters_unchanged": summaries[arm].get("parameters_unchanged"),
                }
                for arm in ARMS
            },
        })

    result = {
        "audit_time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_setup_sha256": setup_hash,
        "protocol_sha256": setup.get("protocol_sha256"),
        "checkpoint_sha256": setup.get("checkpoint_sha256"),
        "native_results_sha256": setup.get("native_results_sha256"),
        "runner_sha256": setup.get("full_runner_sha256"),
        "method_module_sha256": setup.get("method_module_sha256"),
        "calibration_module_sha256": setup.get("calibration_module_sha256"),
        "sam2_revision": setup.get("sam2_revision"),
        "tome_revision": setup.get("tome_revision"),
        "completed_sequences": sequence_receipts,
        "integrity_passed": not problems,
        "problems": problems,
        "scope_limit": (
            "Integrity audit only for currently completed sequences. It does not estimate or "
            "interpret full-cohort efficacy."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
