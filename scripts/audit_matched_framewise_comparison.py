"""Audit saved frame-wise masks against a matched SAM2/ToMe sequence subset.

This is an offline saved-output comparison. It performs no model inference.
It checks manifest frame identities, current GT hashes, saved YOLO-mask hashes,
and equal-sequence present-frame Dice before reporting descriptive means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    videos: dict[str, dict[int, dict[str, Any]]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        row = json.loads(line)
        video = row["video_id"]
        index = int(row["frame_index"])
        if index in videos.setdefault(video, {}):
            raise ValueError(f"Duplicate manifest frame: {video}/{index} (line {line_number})")
        videos[video][index] = row
    return videos


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values) / len(values)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.candidate_run_dir
    baseline_audit = args.baseline_audit_dir
    setup_path = run_dir / "setup.json"
    analysis_path = run_dir / "saved_output_analysis" / "analysis.json"
    baseline_scores_path = baseline_audit / "saved_mask_recomputation.json"
    baseline_reaggregation_path = baseline_audit / "recorded_baseline_reaggregation.json"
    setup = read_json(setup_path)
    analysis = read_json(analysis_path)
    baseline_scores = read_json(baseline_scores_path)
    baseline_reaggregation = read_json(baseline_reaggregation_path)
    manifest = load_manifest(args.manifest)
    sequences = list(analysis["sequences_completed"])
    if not sequences or set(sequences) & {"seq1", "seq7"}:
        raise ValueError("Candidate analysis has an empty or ineligible sequence scope")

    candidate_arms = ("native", "global", "regional")
    yolo_by_sequence: dict[str, list[dict[str, Any]]] = {seq: [] for seq in sequences}
    for row in baseline_scores["frames"]:
        if row.get("post_prompt") is True and row["sequence"] in yolo_by_sequence:
            yolo_by_sequence[row["sequence"]].append(row)

    per_sequence: dict[str, Any] = {}
    identity_checks = 0
    yolo_prediction_hash_checks = 0
    gt_hash_checks = 0
    candidate_rows_checked = 0
    yolo_scores: list[float] = []
    candidate_scores = {arm: [] for arm in candidate_arms}

    # Both studies must use the same shared segmentation evaluator revision.
    candidate_evaluator_hash = setup["source_hashes"]["evaluation/segmentation.py"]
    baseline_evaluator_hash = baseline_scores["evaluator_sources"]["evaluation/segmentation.py"]
    if candidate_evaluator_hash != baseline_evaluator_hash:
        raise ValueError("Candidate and frame-wise baseline evaluator source hashes differ")

    for sequence in sequences:
        receipt = setup["native_reference_receipts"][sequence]
        native_path = Path(receipt["frames_path"])
        if sha256(native_path) != receipt["frames_sha256"]:
            raise ValueError(f"{sequence}: native-reference hash mismatch")
        native_rows = read_json(native_path)
        rows_by_arm = {
            "native": native_rows,
            "global": read_json(run_dir / "global" / sequence / "frames.json"),
            "regional": read_json(run_dir / "regional" / sequence / "frames.json"),
        }
        frame_ids = {arm: [int(row["frame"]) for row in rows]
                     for arm, rows in rows_by_arm.items()}
        if any(ids != frame_ids["native"] for ids in frame_ids.values()):
            raise ValueError(f"{sequence}: native and ToMe target IDs do not match")
        if len(native_rows) != int(receipt["targets"]):
            raise ValueError(f"{sequence}: native row count differs from receipt")
        for arm, rows in rows_by_arm.items():
            if any(row.get("present") not in (True, False) for row in rows):
                raise ValueError(f"{sequence}/{arm}: missing GT-presence label")
            if [bool(row["present"]) for row in rows] != [bool(row["present"]) for row in native_rows]:
                raise ValueError(f"{sequence}/{arm}: GT-presence labels differ")
            candidate_rows_checked += len(rows)

        yolo_rows = sorted(yolo_by_sequence[sequence], key=lambda row: int(row["frame_index"]))
        yolo_ids = [int(row["frame_index"]) for row in yolo_rows]
        if yolo_ids != frame_ids["native"]:
            raise ValueError(f"{sequence}: frame-wise baseline target IDs do not match SAM2")
        if len(yolo_rows) != int(receipt["targets"]):
            raise ValueError(f"{sequence}: baseline row count differs from native receipt")

        present_values = {arm: [] for arm in candidate_arms}
        yolo_present_values: list[float] = []
        for native_row, yolo_row in zip(native_rows, yolo_rows):
            frame_index = int(yolo_row["frame_index"])
            manifest_row = manifest[sequence][frame_index]
            if Path(manifest_row["image"]).stem != yolo_row["frame"]:
                raise ValueError(f"{sequence}/{frame_index}: filename differs from manifest")
            if bool(native_row["present"]) != bool(yolo_row["gt_present"]):
                raise ValueError(f"{sequence}/{frame_index}: baseline GT presence differs")
            identity_checks += 1

            gt_path = args.dataset_root / manifest_row["mask"]
            if sha256(gt_path) != yolo_row["gt_sha256"]:
                raise ValueError(f"{sequence}/{frame_index}: current GT hash mismatch")
            gt_hash_checks += 1

            mask_path = args.baseline_output_dir / "masks" / sequence / f"{yolo_row['frame']}.png"
            if sha256(mask_path) != yolo_row["prediction_sha256"]:
                raise ValueError(f"{sequence}/{frame_index}: saved baseline-mask hash mismatch")
            yolo_prediction_hash_checks += 1

            if yolo_row.get("recomputed_dice") is None:
                raise ValueError(f"{sequence}/{frame_index}: missing recomputed Dice")
            if abs(float(yolo_row["recomputed_dice"]) - float(yolo_row["recorded_dice"])) > 1e-8:
                raise ValueError(f"{sequence}/{frame_index}: recorded Dice does not reproduce")
            if native_row["present"]:
                yolo_present_values.append(float(yolo_row["recomputed_dice"]))
                for arm, rows in rows_by_arm.items():
                    present_values[arm].append(float(rows_by_arm[arm][frame_ids[arm].index(frame_index)]["scores"]["dice"]))

        if len(yolo_present_values) != int(analysis["per_sequence"][sequence]["gt_present_targets"]):
            raise ValueError(f"{sequence}: present-frame denominator differs")
        per_sequence[sequence] = {
            "targets": len(native_rows),
            "present_targets": len(yolo_present_values),
            "empty_targets": len(native_rows) - len(yolo_present_values),
            "dice": {
                "framewise_yolo_boxes": mean(yolo_present_values),
                "native_sam2": mean(present_values["native"]),
                "generic_tome": mean(present_values["global"]),
                "region_aware_tome": mean(present_values["regional"]),
            },
        }
        yolo_scores.append(per_sequence[sequence]["dice"]["framewise_yolo_boxes"])
        for arm, name in (("native", "native_sam2"), ("global", "generic_tome"), ("regional", "region_aware_tome")):
            candidate_scores[arm].append(per_sequence[sequence]["dice"][name])

    # Confirm direct recomputation agrees with the saved per-sequence baseline aggregate.
    for sequence in sequences:
        if abs(per_sequence[sequence]["dice"]["framewise_yolo_boxes"] -
               float(baseline_reaggregation["by_sequence"][sequence]["post_prompt_present_dice"])) > 1e-12:
            raise ValueError(f"{sequence}: saved baseline aggregation does not reproduce")

    all_21_present = float(baseline_reaggregation["eligible21_sequence_macro_post_prompt_present"])
    all_21_all_frames = float(baseline_reaggregation["eligible21_sequence_macro_all_frames"])
    summary = {
        "status": (
            "verified descriptive matched-full-cohort comparison; inference provenance incomplete"
            if len(sequences) == int(analysis["n_sequences_planned"])
            else "verified descriptive matched-subset comparison; inference provenance incomplete"
        ),
        "candidate_analysis_status": analysis["status"],
        "candidate_sequences_planned": int(analysis["n_sequences_planned"]),
        "inference_rerun": False,
        "comparison_scope": "same eligible sequences, exact post-prompt frame IDs, same GT-presence labels, equal-sequence mean of present-frame Dice",
        "sequences": sequences,
        "n_sequences": len(sequences),
        "post_prompt_targets": sum(row["targets"] for row in per_sequence.values()),
        "gt_present_targets": sum(row["present_targets"] for row in per_sequence.values()),
        "gt_empty_targets": sum(row["empty_targets"] for row in per_sequence.values()),
        "identity_checks_passed": identity_checks,
        "gt_raw_hash_checks_passed": gt_hash_checks,
        "saved_baseline_prediction_hash_checks_passed": yolo_prediction_hash_checks,
        "candidate_rows_checked": candidate_rows_checked,
        "candidate_and_baseline_shared_evaluator_sha256_match": candidate_evaluator_hash,
        "equal_sequence_mean_present_dice": {
            "framewise_yolo_boxes": mean(yolo_scores),
            "native_sam2": mean(candidate_scores["native"]),
            "generic_tome": mean(candidate_scores["global"]),
            "region_aware_tome": mean(candidate_scores["regional"]),
        },
        "delta_vs_framewise_yolo": {
            "native_sam2": mean(candidate_scores["native"]) - mean(yolo_scores),
            "generic_tome": mean(candidate_scores["global"]) - mean(yolo_scores),
            "region_aware_tome": mean(candidate_scores["regional"]) - mean(yolo_scores),
        },
        "framewise_yolo_all_21_references": {
            "post_prompt_present_only": all_21_present,
            "all_frames_including_gt_empty": all_21_all_frames,
            "post_prompt_present_targets": int(baseline_reaggregation["post_prompt_present_frames"]),
            "post_prompt_empty_targets": int(baseline_reaggregation["post_prompt_empty_frames"]),
            "false_alarms_on_post_prompt_gt_empty": int(baseline_reaggregation["post_prompt_false_alarms_from_recorded_empty_dice"]),
        },
        "per_sequence": per_sequence,
        "provenance_gaps": [
            "Historical SAM2 checkpoint/source revision is not identified.",
            "YOLO detector weights, confidence threshold, and box-selection policy are not recorded.",
            "Historical preprocessing/inference resolution, prompt/box frequency, and compute/runtime are not recorded.",
            "Therefore this is a same-frame saved-output comparison, not a fully matched system benchmark.",
        ],
        "inputs_sha256": {
            "candidate_setup.json": sha256(setup_path),
            "candidate_analysis.json": sha256(analysis_path),
            "baseline_saved_mask_recomputation.json": sha256(baseline_scores_path),
            "baseline_recorded_reaggregation.json": sha256(baseline_reaggregation_path),
            "manifest": sha256(args.manifest),
        },
    }
    return summary


def render(summary: dict[str, Any]) -> str:
    means = summary["equal_sequence_mean_present_dice"]
    deltas = summary["delta_vs_framewise_yolo"]
    scope_label = (
        "full eligible cohort"
        if summary["n_sequences"] == summary["candidate_sequences_planned"]
        else "matched subset"
    )
    lines = [
        f"# Saved frame-wise YOLO-box comparison on matched {scope_label}",
        "",
        f"Scope: {summary['n_sequences']} sequences; {summary['gt_present_targets']} GT-present targets and "
        f"{summary['gt_empty_targets']} GT-empty targets. This is a saved-output audit; no inference was rerun.",
        "Exact native/ToMe and YOLO frame IDs and GT-presence labels match for every target. "
        f"Verified hashes: {summary['identity_checks_passed']} target identities, "
        f"{summary['gt_raw_hash_checks_passed']} GT masks, and "
        f"{summary['saved_baseline_prediction_hash_checks_passed']} saved YOLO masks.",
        "",
        "Equal-sequence mean of per-sequence GT-present frame Dice:",
        "",
        "| Method | Dice | Difference vs saved frame-wise YOLO |",
        "|---|---:|---:|",
        f"| Frame-wise YOLO-box saved masks | {means['framewise_yolo_boxes']:.6f} | — |",
        f"| Native SAM2 | {means['native_sam2']:.6f} | {deltas['native_sam2']:+.6f} |",
        f"| Generic wider-history ToMe | {means['generic_tome']:.6f} | {deltas['generic_tome']:+.6f} |",
        f"| Region-aware wider-history ToMe | {means['region_aware_tome']:.6f} | {deltas['region_aware_tome']:+.6f} |",
        "",
        f"Frame-wise YOLO full-cohort references: {summary['framewise_yolo_all_21_references']['post_prompt_present_only']:.6f} "
        "on post-prompt GT-present frames; "
        f"{summary['framewise_yolo_all_21_references']['all_frames_including_gt_empty']:.6f} "
        "when including all post-prompt frames. The latter has a different denominator and must not be compared "
        "as if it were present-frame Dice.",
        "",
        "Per-sequence present-frame Dice:",
        "",
        "| Sequence | Present / empty | YOLO boxes | Native | Generic ToMe | Regional ToMe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sequence, row in summary["per_sequence"].items():
        d = row["dice"]
        lines.append(
            f"| {sequence} | {row['present_targets']} / {row['empty_targets']} | "
            f"{d['framewise_yolo_boxes']:.6f} | {d['native_sam2']:.6f} | "
            f"{d['generic_tome']:.6f} | {d['region_aware_tome']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"This comparison has exact matched sequence/frame coverage and uses the same present-frame, equal-sequence "
        "Dice aggregation. It is descriptive: the historical frame-wise run's SAM2 checkpoint, YOLO weights, box "
        "policy, preprocessing, prompt frequency, and runtime are undocumented. Do not claim a fully matched "
        f"system-level comparison until those are recovered. The wider-history analysis is "
        f"{summary['candidate_analysis_status']} and its Dice result remains seq8-sensitive.",
        "",
        "The regional-vs-native and regional-vs-generic efficacy decisions remain governed by the frozen full-cohort "
        "protocol; these YOLO comparisons were added as a descriptive benchmark reference, not a replacement screen.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-audit-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "framewise_yolo_matched_subset.json"
    md_path = args.output_dir / "framewise_yolo_matched_subset.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(render(summary))
    print(render(summary))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
