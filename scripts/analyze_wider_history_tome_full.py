"""Audit saved wider-history ToMe outputs without rerunning SAM2 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.statistics import sequence_bootstrap_interval
from scripts.false_alarm_statistics import propagation_summary


METRICS = ("dice", "iou", "boundary_f")
ARMS = ("native", "global", "regional")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def present_metric_means(rows):
    present = [row for row in rows if row.get("present") is True]
    if not present:
        raise ValueError("Every evaluated sequence must contain a GT-present target")
    output = {"present_targets": len(present)}
    for metric in METRICS:
        values = [float(row["scores"][metric]) for row in present]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {metric} in saved frame rows")
        output[metric] = sum(values) / len(values)
    return output


def validate_matched_rows(name, rows_by_arm, expected_targets):
    frame_lists = {
        arm: [int(row["frame"]) for row in rows_by_arm[arm]] for arm in ARMS
    }
    if any(len(frames) != expected_targets for frames in frame_lists.values()):
        raise ValueError(f"{name}: frame-row count differs from native receipt")
    if any(frames != frame_lists["native"] for frames in frame_lists.values()):
        raise ValueError(f"{name}: native and candidate target frame IDs differ")
    if any(right != left + 1 for left, right in zip(frame_lists["native"],
                                                    frame_lists["native"][1:])):
        raise ValueError(f"{name}: target source-frame indices are not consecutive")
    present_by_arm = {
        arm: [row.get("present") for row in rows_by_arm[arm]] for arm in ARMS
    }
    if any(value is not True and value is not False
           for values in present_by_arm.values() for value in values):
        raise ValueError(f"{name}: missing boolean GT-presence labels")
    if any(values != present_by_arm["native"] for values in present_by_arm.values()):
        raise ValueError(f"{name}: GT-presence labels differ across methods")
    for arm, rows in rows_by_arm.items():
        for row in rows:
            if row["present"]:
                if not isinstance(row.get("scores"), dict):
                    raise ValueError(f"{name}/{arm}: present frame has no metric scores")
                for metric in METRICS:
                    if metric not in row["scores"]:
                        raise ValueError(f"{name}/{arm}: missing {metric}")
            elif not isinstance(row.get("absent_fp"), bool):
                raise ValueError(f"{name}/{arm}: empty frame lacks false-alarm label")


def trajectory_diagnostic(rows_by_arm):
    by_frame = {
        arm: {int(row["frame"]): row for row in rows_by_arm[arm]}
        for arm in ARMS
    }
    history_count = {
        frame: int(by_frame["global"][frame]["available_noncond_frames"])
        for frame in by_frame["global"]
    }
    for frame in history_count:
        if int(by_frame["regional"][frame]["available_noncond_frames"]) != history_count[frame]:
            raise ValueError("Generic and regional arms read different history counts")

    first_hash_divergence = {}
    first_stored_hash_divergence = {}
    first_material_dice_drop = {}
    history_groups = {}
    for arm in ("global", "regional"):
        differences = []
        stored_differences = []
        for frame in sorted(history_count):
            candidate = by_frame[arm][frame]
            native = by_frame["native"][frame]
            if candidate["prediction_sha256"] != native["prediction_sha256"]:
                differences.append(frame)
            if candidate["stored_sha256"] != native["stored_sha256"]:
                stored_differences.append(frame)
            if candidate["present"]:
                delta = float(candidate["scores"]["dice"]) - float(native["scores"]["dice"])
                if delta <= -0.05 and arm not in first_material_dice_drop:
                    first_material_dice_drop[arm] = {
                        "frame": frame,
                        "history_frames_read": history_count[frame],
                        "native_dice": float(native["scores"]["dice"]),
                        "candidate_dice": float(candidate["scores"]["dice"]),
                        "delta": delta,
                    }
        first_hash_divergence[arm] = (
            {"frame": differences[0], "history_frames_read": history_count[differences[0]]}
            if differences else None
        )
        first_stored_hash_divergence[arm] = (
            {"frame": stored_differences[0],
             "history_frames_read": history_count[stored_differences[0]]}
            if stored_differences else None
        )
        for group, predicate in (
            ("up_to_6_history_maps", lambda count: count <= 6),
            ("more_than_6_history_maps", lambda count: count > 6),
        ):
            selected = [frame for frame, count in history_count.items() if predicate(count)]
            present = [frame for frame in selected if by_frame["native"][frame]["present"]]
            if not present:
                history_groups.setdefault(group, {})[arm] = {
                    "targets": len(selected), "present_targets": 0, "present_dice": None,
                }
                continue
            values = [float(by_frame[arm][frame]["scores"]["dice"]) for frame in present]
            history_groups.setdefault(group, {})[arm] = {
                "targets": len(selected),
                "present_targets": len(present),
                "present_dice": sum(values) / len(values),
            }

    for group in history_groups.values():
        group["native"] = {
            "targets": 0,
            "present_targets": 0,
            "present_dice": None,
        }
    for group_name, group in history_groups.items():
        selected = [frame for frame, count in history_count.items()
                    if (count <= 6 if group_name == "up_to_6_history_maps" else count > 6)]
        present = [frame for frame in selected if by_frame["native"][frame]["present"]]
        group["native"] = {
            "targets": len(selected),
            "present_targets": len(present),
            "present_dice": (
                sum(float(by_frame["native"][frame]["scores"]["dice"]) for frame in present)
                / len(present) if present else None
            ),
        }
    return {
        "history_groups": history_groups,
        "first_prediction_hash_divergence_vs_native": first_hash_divergence,
        "first_stored_memory_hash_divergence_vs_native": first_stored_hash_divergence,
        "first_present_dice_drop_at_least_0_05_vs_native": first_material_dice_drop,
    }


def analyze(run_dir: Path):
    setup = read_json(run_dir / "setup.json")
    runtime = read_json(run_dir / "runtime_summaries.json")
    native_receipts = setup["native_reference_receipts"]
    sequences = setup["sequences"]
    if len(sequences) != 21 or set(sequences) & {"seq1", "seq7"}:
        raise ValueError("Unexpected evaluation sequence scope")

    per_sequence = {}
    source_hashes = {
        "setup.json": sha256(run_dir / "setup.json"),
        "runtime_summaries.json": sha256(run_dir / "runtime_summaries.json"),
    }
    pending = []
    for name in sequences:
        receipt = native_receipts[name]
        native_path = Path(receipt["frames_path"])
        if sha256(native_path) != receipt["frames_sha256"]:
            raise ValueError(f"{name}: archived native frame hash mismatch")
        paths = {
            "native": native_path,
            "global": run_dir / "global" / name / "frames.json",
            "regional": run_dir / "regional" / name / "frames.json",
        }
        if any(not path.is_file() for arm, path in paths.items() if arm != "native"):
            pending.append(name)
            continue
        if name not in runtime.get("global", {}) or name not in runtime.get("regional", {}):
            pending.append(name)
            continue
        rows_by_arm = {arm: read_json(path) for arm, path in paths.items()}
        validate_matched_rows(name, rows_by_arm, int(receipt["targets"]))
        for arm, path in paths.items():
            source_hashes[f"{arm}/{name}/frames.json"] = sha256(path)

        sequence = {
            "targets": int(receipt["targets"]),
            "gt_present_targets": sum(row["present"] for row in rows_by_arm["native"]),
            "gt_empty_targets": sum(not row["present"] for row in rows_by_arm["native"]),
            "conditions": {},
            "trajectory_diagnostic": trajectory_diagnostic(rows_by_arm),
        }
        for arm in ARMS:
            rows = rows_by_arm[arm]
            metrics = present_metric_means(rows)
            empty_rows = [row for row in rows if not row["present"]]
            events = propagation_summary(rows)
            metrics.update({
                "gt_empty_false_alarms": sum(bool(row["absent_fp"]) for row in empty_rows),
                "return_events": events["return_count"],
                "detected_returns": events["return_count"] - events["undetected_returns"],
                "undetected_returns": events["undetected_returns"],
            })
            if arm != "native":
                timing = runtime[arm][name]
                if timing.get("targets") != len(rows):
                    raise ValueError(f"{name}/{arm}: runtime target count mismatch")
                metrics["wall_seconds"] = float(timing["wall_seconds"])
            sequence["conditions"][arm] = metrics
        per_sequence[name] = sequence

    complete_names = sorted(per_sequence)
    complete = len(complete_names) == len(sequences)
    cohort = {arm: {} for arm in ARMS}
    contrasts = {}
    for arm in ARMS:
        for metric in METRICS:
            cohort[arm][metric] = (
                sum(per_sequence[name]["conditions"][arm][metric] for name in complete_names)
                / len(complete_names) if complete_names else None
            )
        empty = sum(per_sequence[name]["gt_empty_targets"] for name in complete_names)
        false_alarms = sum(
            per_sequence[name]["conditions"][arm]["gt_empty_false_alarms"]
            for name in complete_names
        )
        returns = sum(
            per_sequence[name]["conditions"][arm]["return_events"]
            for name in complete_names
        )
        detected = sum(
            per_sequence[name]["conditions"][arm]["detected_returns"]
            for name in complete_names
        )
        cohort[arm].update({
            "sequences": len(complete_names),
            "gt_empty_false_alarms": false_alarms,
            "gt_empty_targets": empty,
            "gt_empty_false_alarm_rate": false_alarms / empty if empty else None,
            "detected_returns": detected,
            "return_events": returns,
            "total_wall_seconds": (
                sum(per_sequence[name]["conditions"][arm]["wall_seconds"]
                    for name in complete_names) if arm != "native" else None
            ),
        })

    for left, right in (("global", "native"), ("regional", "native"),
                        ("regional", "global")):
        contrasts[f"{left}_minus_{right}"] = {}
        for metric in METRICS:
            deltas = {
                name: per_sequence[name]["conditions"][left][metric]
                - per_sequence[name]["conditions"][right][metric]
                for name in complete_names
            }
            row = {"mean_sequence_delta": sum(deltas.values()) / len(deltas)
                   if deltas else None}
            # Interim intervals are omitted; the frozen primary Dice analysis is
            # produced by the cohort runner after all 21 sequences are complete.
            if complete and deltas:
                row["sequence_bootstrap_ci95_descriptive"] = sequence_bootstrap_interval(
                    deltas, seed=20261112 + METRICS.index(metric)
                )
            contrasts[f"{left}_minus_{right}"][metric] = row
        if complete_names:
            contrasts[f"{left}_minus_{right}"]["sequence_wins_ties_losses_dice"] = [
                sum(per_sequence[n]["conditions"][left]["dice"]
                    > per_sequence[n]["conditions"][right]["dice"] + 1e-12
                    for n in complete_names),
                sum(abs(per_sequence[n]["conditions"][left]["dice"]
                        - per_sequence[n]["conditions"][right]["dice"]) <= 1e-12
                    for n in complete_names),
                sum(per_sequence[n]["conditions"][left]["dice"]
                    < per_sequence[n]["conditions"][right]["dice"] - 1e-12
                    for n in complete_names),
            ]

    return {
        "status": "complete" if complete else "interim",
        "scope": "paired saved-frame analysis; equal-sequence means over GT-present post-prompt frames",
        "sequences_completed": complete_names,
        "sequences_pending": pending,
        "n_sequences_completed": len(complete_names),
        "n_sequences_planned": len(sequences),
        "all_target_rows_matched": True,
        "per_sequence": per_sequence,
        "equal_sequence_cohort_means": cohort,
        "paired_sequence_contrasts": contrasts,
        "runtime_note": "Candidate wall times are accumulated per-condition CPU propagation times; they exclude the archived native rerun, which was reused.",
        "source_sha256": source_hashes,
        "caveat": (
            "Completed saved-output summaries are descriptive; sequence-level patient "
            "independence is unverified. Boundary F and IoU are computed from saved "
            "per-frame scores; no inference was rerun."
            if complete else
            "Interim summaries are descriptive and exclude incomplete sequences. Boundary F "
            "and IoU are computed from saved per-frame scores; no inference was rerun."
        ),
    }


def render_markdown(result):
    names = result["sequences_completed"]
    lines = [
        "# Wider-history ToMe: saved-output analysis",
        "",
        f"Status: **{result['status']}**, {len(names)}/{result['n_sequences_planned']} sequences complete.",
        "",
        "Equal-sequence means of per-sequence means over GT-present post-prompt frames:",
        "",
        "| Condition | Dice | IoU | Boundary F | GT-empty false alarms | Returns detected | Total candidate wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = result["equal_sequence_cohort_means"][arm]
        def fmt(value): return "—" if value is None else f"{value:.6f}"
        runtime = row["total_wall_seconds"]
        lines.append(
            f"| {arm} | {fmt(row['dice'])} | {fmt(row['iou'])} | {fmt(row['boundary_f'])} | "
            f"{row['gt_empty_false_alarms']}/{row['gt_empty_targets']} | "
            f"{row['detected_returns']}/{row['return_events']} | "
            f"{fmt(runtime)} s |"
        )
    lines += ["", "Paired mean sequence deltas:", "", "| Contrast | Dice Δ | IoU Δ | Boundary F Δ | Dice sequence W/T/L |",
              "|---|---:|---:|---:|---:|"]
    for name, row in result["paired_sequence_contrasts"].items():
        wins = row.get("sequence_wins_ties_losses_dice", "—")
        dice_delta = row["dice"]["mean_sequence_delta"]
        iou_delta = row["iou"]["mean_sequence_delta"]
        boundary_delta = row["boundary_f"]["mean_sequence_delta"]
        def fmt_delta(value): return "—" if value is None else f"{value:+.6f}"
        lines.append(
            f"| {name.replace('_', ' ')} | {fmt_delta(dice_delta)} | "
            f"{fmt_delta(iou_delta)} | {fmt_delta(boundary_delta)} | {wins} |"
        )
    lines += [
        "",
        "First saved-trajectory divergence checks:",
        "",
        "| Sequence | First prediction hash divergence (generic / regional) | First Dice drop ≥0.05 (generic / regional) |",
        "|---|---|---|",
    ]
    for sequence in names:
        diagnostic = result["per_sequence"][sequence]["trajectory_diagnostic"]
        hash_rows = diagnostic["first_prediction_hash_divergence_vs_native"]
        drop_rows = diagnostic["first_present_dice_drop_at_least_0_05_vs_native"]
        def fmt_event(event):
            if event is None:
                return "—"
            return f"frame {event['frame']} ({event['history_frames_read']} history maps)"
        lines.append(
            f"| {sequence} | {fmt_event(hash_rows['global'])} / {fmt_event(hash_rows['regional'])} | "
            f"{fmt_event(drop_rows.get('global'))} / {fmt_event(drop_rows.get('regional'))} |"
        )
    if not result["status"] == "complete":
        lines += ["", "**Interim only:** do not treat this subset as cohort evidence."]
    lines += ["", result["runtime_note"], result["caveat"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "saved_output_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(run_dir)
    result["analysis_script_sha256"] = sha256(Path(__file__).resolve())
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "analysis.md").write_text(render_markdown(result))
    print(render_markdown(result))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
