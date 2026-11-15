"""Audit one completed native/pointer-ToMe/generic-ToMe sequence from saved outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("native", "region_constrained_tome", "generic_local_tome")
TOL = 1e-10


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOL)


def load_arm(root: Path, arm: str, sequence: str):
    folder = root / arm / sequence
    summary = json.loads((folder / "summary.json").read_text())
    rows = json.loads((folder / "frames.json").read_text())
    if not summary.get("complete"):
        raise ValueError(f"Incomplete summary: {folder}")
    if len(rows) != summary.get("targets"):
        raise ValueError(f"Target count differs from summary: {folder}")
    return summary, rows


def recompute_summary(rows):
    present = [r for r in rows if r["present"]]
    empty = [r for r in rows if not r["present"]]
    return {
        "present_targets": len(present),
        "gt_empty_targets": len(empty),
        "present_dice": sum(float(r["scores"]["dice"]) for r in present) / len(present),
        "present_iou": sum(float(r["scores"]["iou"]) for r in present) / len(present),
        "present_boundary_f": sum(float(r["scores"]["boundary_f"]) for r in present) / len(present),
        "gt_empty_false_alarms": sum(bool(r["absent_fp"]) for r in empty),
    }


def audit_sequence(root: Path, sequence: str):
    loaded = {arm: load_arm(root, arm, sequence) for arm in ARMS}
    summaries = {arm: loaded[arm][0] for arm in ARMS}
    rows = {arm: loaded[arm][1] for arm in ARMS}
    errors = []

    native = rows["native"]
    native_ids = [int(r["frame"]) for r in native]
    if native_ids != sorted(native_ids) or len(native_ids) != len(set(native_ids)):
        errors.append("Native target frame IDs are not unique and ordered")

    for arm in ARMS:
        summary = summaries[arm]
        arm_rows = rows[arm]
        if [int(r["frame"]) for r in arm_rows] != native_ids:
            errors.append(f"{arm}: target frame IDs differ from native")
        if [bool(r["present"]) for r in arm_rows] != [bool(r["present"]) for r in native]:
            errors.append(f"{arm}: GT-presence pattern differs from native")
        if summary.get("parameters_unchanged") is not True:
            errors.append(f"{arm}: SAM2 parameters were not attested unchanged")
        calculated = recompute_summary(arm_rows)
        for key, value in calculated.items():
            if key in ("gt_empty_false_alarms", "present_targets", "gt_empty_targets"):
                if int(summary.get(key, -1)) != int(value):
                    errors.append(f"{arm}: summary {key} does not match per-frame rows")
            elif not close(summary.get(key, float("nan")), value):
                errors.append(f"{arm}: summary {key} does not match per-frame rows")

    digests = {summaries[a].get("parameter_digest") for a in ARMS}
    if len(digests) != 1:
        errors.append("Parameter digests differ across arms")
    refs = {
        summaries[a].get("reference_receipt", {}).get("frames_sha256") for a in ARMS
    }
    if len(refs) != 1 or None in refs:
        errors.append("Native-reference receipt hashes differ or are missing")

    attention_checks = {"regional": 0, "generic": 0, "matched_source_lists": 0, "mass": 0}
    for index, target in enumerate(native_ids):
        pointer_reads = rows["region_constrained_tome"][index].get("attention_summary", [])
        generic_reads = rows["generic_local_tome"][index].get("attention_summary", [])
        if [r.get("layer") for r in pointer_reads] != [0, 1, 2, 3]:
            errors.append(f"pointer: missing or duplicate attention layer at frame {target}")
        if [r.get("layer") for r in generic_reads] != [0, 1, 2, 3]:
            errors.append(f"generic: missing or duplicate attention layer at frame {target}")
        for p_read, g_read in zip(pointer_reads, generic_reads):
            attention_checks["regional"] += 1
            attention_checks["generic"] += 1
            if p_read["memory_frames"] != g_read["memory_frames"]:
                errors.append(f"memory source frames differ at target {target}")
            if p_read["spatial_keys_before"] != g_read["spatial_keys_before"]:
                errors.append(f"source token counts differ at target {target}")
            if p_read["pointer_keys"] != g_read["pointer_keys"]:
                errors.append(f"pointer token counts differ at target {target}")
            else:
                attention_checks["matched_source_lists"] += 1
            for read in (p_read, g_read):
                before = int(read["spatial_keys_before"])
                after = int(read["spatial_keys_after"])
                ptr = int(read["pointer_keys"])
                if after > before or not close(read["spatial_key_mass_sum"], before):
                    errors.append(f"invalid spatial mass/key reduction at target {target}")
                if not close(read["key_mass_sum"], before + ptr):
                    errors.append(f"invalid total attention mass at target {target}")
                attention_checks["mass"] += 1

    for arm in ("region_constrained_tome", "generic_local_tome"):
        if summaries[arm].get("attention_layer_calls") != 4 * len(native_ids):
            errors.append(f"{arm}: summary layer-call count is not 4 per target")
    budget_checks = int(summaries["generic_local_tome"].get("matched_budget_checks", 0))
    if budget_checks <= 0:
        errors.append("Generic control has no recorded matched-budget checks")

    metrics = {arm: recompute_summary(rows[arm]) for arm in ARMS}
    result = {
        "sequence": sequence,
        "scope": "saved-output integrity audit; no inference rerun; no mask pixels are stored in these compact records",
        "targets": len(native_ids),
        "first_target": native_ids[0] if native_ids else None,
        "last_target": native_ids[-1] if native_ids else None,
        "all_checks_passed": not errors,
        "errors": errors,
        "common_parameter_digest": next(iter(digests)) if len(digests) == 1 else None,
        "common_native_reference_frames_sha256": next(iter(refs)) if len(refs) == 1 else None,
        "generic_matched_budget_checks_recorded": budget_checks,
        "attention_integrity_checks": attention_checks,
        "per_frame_rows_recomputed_metrics": metrics,
        "run_summaries": summaries,
        "limitation": "Metric arithmetic is recomputed from saved per-frame scores; mask-level metrics cannot be independently recomputed because treatment masks are not stored, only their SHA256 digests.",
    }
    return result


def render(result):
    lines = [
        f"# {result['sequence']} pointer-ToMe saved-output audit",
        "",
        f"Scope: {result['scope']}.",
        f"Targets: {result['targets']} ({result['first_target']}–{result['last_target']}).",
        f"Integrity checks: {'PASS' if result['all_checks_passed'] else 'FAIL'}.",
        "",
        "| Arm | Present Dice | IoU | Boundary F | Empty false alarms / empty targets | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metric = result["per_frame_rows_recomputed_metrics"][arm]
        summary = result["run_summaries"][arm]
        lines.append(
            f"| {arm} | {metric['present_dice']:.6f} | {metric['present_iou']:.6f} | "
            f"{metric['present_boundary_f']:.6f} | {metric['gt_empty_false_alarms']} / "
            f"{metric['gt_empty_targets']} | {summary['wall_seconds']:.1f} |"
        )
    lines += [
        "",
        f"Generic matched-budget checks recorded by runner: {result['generic_matched_budget_checks_recorded']:,}.",
        f"Parameter digest: `{result['common_parameter_digest']}`.",
        f"Native reference frames SHA256: `{result['common_native_reference_frames_sha256']}`.",
        "",
        result["limitation"],
    ]
    if result["errors"]:
        lines += ["", "Errors:", *[f"- {item}" for item in result["errors"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit_sequence(args.run_dir, args.sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.sequence}_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / f"{args.sequence}_audit.md").write_text(render(result))
    print(render(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
