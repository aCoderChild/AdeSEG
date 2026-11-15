"""Summarize paired DAVIS results for regional-token and author methods."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.vos_benchmarks import inventory, layout
from evaluation.segmentation import calculate_scores
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.memory_token_io import digest, validate_output


LOCAL_CONDITIONS = ("native", "generic_local_tome", "soft_mask_aware_tome")
AUTHOR_METHODS = ("samurai", "sam2long")
COMPARISONS = {
    "soft_mask_aware_tome - native": ("soft_mask_aware_tome", "native"),
    "soft_mask_aware_tome - generic_local_tome": (
        "soft_mask_aware_tome", "generic_local_tome"
    ),
    "soft_mask_aware_tome - samurai": ("soft_mask_aware_tome", "samurai"),
    "soft_mask_aware_tome - sam2long": ("soft_mask_aware_tome", "sam2long"),
}
PROTOCOL = ROOT / "protocols/davis_region_vs_temporal_selection_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_predictions(prediction_dir, records, receipts):
    """Check every archived mask against its per-video hash receipt."""
    for record in records:
        video = record["video"]
        receipt = receipts[video]
        if receipt.get("video") != video:
            raise RuntimeError(f"Invalid completion receipt for {video}")
        if isinstance(receipt.get("frames"), int):
            if receipt["frames"] != len(record["frames"]):
                raise RuntimeError(f"Invalid frame count in completion receipt for {video}")
            expected_hashes = receipt.get("prediction_sha256", {})
        else:
            frame_rows = receipt.get("frames", [])
            if len(frame_rows) != len(record["frames"]):
                raise RuntimeError(f"Invalid frame list in completion receipt for {video}")
            expected_hashes = {
                Path(row["prediction_png"]).name: row.get("prediction_png_sha256")
                for row in frame_rows
            }
            if any(not value for value in expected_hashes.values()):
                raise RuntimeError(f"No saved-mask digest in local receipt for {video}")
        expected_names = {Path(frame).name for frame in record["frames"]}
        if set(expected_hashes) != expected_names:
            raise RuntimeError(f"Receipt has incomplete frame hashes for {video}")
        for filename, expected_hash in expected_hashes.items():
            path = prediction_dir / video / filename
            if not path.is_file() or digest(path) != expected_hash:
                raise RuntimeError(f"Missing or changed saved prediction: {video}/{filename}")


def framewise_dice(prediction_dir, root, records, split):
    """Compute supplementary per-object Dice over official scored frames only."""
    _, mask_root, _, _ = layout(root, "davis2017", split)
    per_object = {}
    per_video = {}
    for record in records:
        video_rows = []
        scored_frames = record["frames"][1:-1]
        for object_id in record["object_ids"]:
            values = []
            for frame_name in scored_frames:
                gt = np.asarray(Image.open(mask_root / record["video"] / frame_name))
                pred = np.asarray(Image.open(prediction_dir / record["video"] / frame_name))
                valid = gt != 255
                gt_binary = (gt[valid] == object_id).astype(np.uint8)
                pred_binary = (pred[valid] == object_id).astype(np.uint8)
                values.append(calculate_scores(pred_binary, gt_binary)["dice"])
            value = float(np.mean(values))
            key = f"{record['video']}_{object_id}"
            per_object[key] = value
            video_rows.append(value)
        per_video[record["video"]] = float(np.mean(video_rows))
    return {
        "definition": "Per-frame binary Dice for each object on frames 1..N-2, averaged per object, then objects averaged within video; supplementary to official J&F.",
        "global_object_macro_mean": float(np.mean(list(per_object.values()))),
        "per_object": per_object,
        "per_video": per_video,
    }


def collect_conditions(local_run, author_full, records):
    local_setup = read_json(local_run / "setup.json")
    local_results_path = local_run / "results.json"
    local_results = read_json(local_results_path)
    conditions = {}
    receipts_by_condition = {}
    for condition in LOCAL_CONDITIONS:
        if condition not in local_results.get("conditions", {}):
            raise RuntimeError(f"Missing local comparison condition: {condition}")
        row = local_results["conditions"][condition]
        predictions = local_run / "conditions" / condition / "predictions"
        receipts = {
            record["video"]: read_json(
                local_run / "conditions" / condition / f"{record['video']}.complete.json"
            )
            for record in records
        }
        verify_predictions(predictions, records, receipts)
        evaluation = row.get("official_evaluation")
        if not evaluation:
            raise RuntimeError(f"Missing official J/F evaluation for {condition}")
        conditions[condition] = {
            "global": evaluation["global"],
            "per_video": evaluation["per_video"],
            "runtime": {
                "inference_seconds": sum(
                    value["wall_seconds"] for value in row.values()
                    if isinstance(value, dict) and "wall_seconds" in value
                ),
                "model_load_seconds": None,
                "segmentation_pathways": 1,
            },
        }
        receipts_by_condition[condition] = receipts

    author_setups = {}
    for method in AUTHOR_METHODS:
        summary_path = author_full / f"{method}_full_summary.json"
        summary = read_json(summary_path)
        setup = summary.get("setup", {})
        if setup.get("method") != method or setup.get("stage") != "full" or setup.get("split") != "val":
            raise RuntimeError(f"Invalid full-run setup receipt for {method}")
        if setup.get("protocol_sha256") != digest(PROTOCOL):
            raise RuntimeError(f"{method} was not run under the frozen comparator protocol")
        video_receipts = {
            record["video"]: read_json(
                author_full / "conditions" / method / f"{record['video']}.complete.json"
            )
            for record in records
        }
        verify_predictions(
            author_full / "conditions" / method / "predictions", records, video_receipts
        )
        evaluation = summary.get("official_evaluation")
        if not evaluation:
            raise RuntimeError(f"Missing official J/F evaluation for {method}")
        conditions[method] = {
            "global": evaluation["global"],
            "per_video": evaluation["per_video"],
            "runtime": {
                "inference_seconds": summary["total_inference_seconds"],
                "model_load_seconds": summary["total_model_load_seconds"],
                "segmentation_pathways": 3 if method == "sam2long" else 1,
            },
        }
        author_setups[method] = {
            "setup": setup,
            "summary_sha256": digest(summary_path),
        }
    return conditions, local_setup, author_setups, local_results_path


def paired_contrasts(conditions):
    output = {}
    raw_p = {}
    for label, (left_name, right_name) in COMPARISONS.items():
        left = conditions[left_name]["per_video"]
        right = conditions[right_name]["per_video"]
        if set(left) != set(right):
            raise RuntimeError(f"Unpaired video sets in contrast: {label}")
        differences = {video: left[video]["J&F-Mean"] - right[video]["J&F-Mean"]
                       for video in sorted(left)}
        test = paired_wilcoxon(
            {video: left[video]["J&F-Mean"] for video in left},
            {video: right[video]["J&F-Mean"] for video in right},
        )
        output[label] = {
            "n_videos": len(differences),
            "mean_paired_delta_JF": float(np.mean(list(differences.values()))),
            "median_paired_delta_JF": float(np.median(list(differences.values()))),
            "wins_ties_losses": [
                sum(value > 0 for value in differences.values()),
                sum(value == 0 for value in differences.values()),
                sum(value < 0 for value in differences.values()),
            ],
            "bootstrap_95_percentile_ci": sequence_bootstrap_interval(
                differences, seed=20261105, resamples=10_000
            ),
            "wilcoxon_two_sided_p": test["p"],
        }
        raw_p[label] = test["p"]
    for label, adjusted in holm_adjust(raw_p).items():
        output[label]["holm_adjusted_p"] = adjusted
    return output


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--author-full-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    local_run = args.local_run_dir.resolve()
    author_full = args.author_full_dir.resolve()
    out = validate_output(args.output_dir)
    protocol = read_json(PROTOCOL)
    inv = inventory(root, "davis2017", "val")
    if inv["video_count"] != protocol["dataset"]["videos"]:
        raise RuntimeError("DAVIS validation sequence count differs from protocol")
    out.mkdir(parents=True, exist_ok=True)
    conditions, local_setup, author_setups, local_results_path = collect_conditions(
        local_run, author_full, inv["videos"]
    )
    for condition in conditions:
        if set(conditions[condition]["per_video"]) != {r["video"] for r in inv["videos"]}:
            raise RuntimeError(f"Incomplete per-video result map: {condition}")
    # Dice is evaluated from saved PNGs and the indexed masks; it is never inferred from J.
    for condition in LOCAL_CONDITIONS:
        conditions[condition]["dice"] = framewise_dice(
            local_run / "conditions" / condition / "predictions", root, inv["videos"], "val"
        )
    for method in AUTHOR_METHODS:
        conditions[method]["dice"] = framewise_dice(
            author_full / "conditions" / method / "predictions", root, inv["videos"], "val"
        )

    contrasts = paired_contrasts(conditions)
    global_rows = []
    for condition, values in conditions.items():
        global_rows.append({
            "condition": condition,
            "official_J_mean": values["global"]["J-Mean"],
            "official_F_mean": values["global"]["F-Mean"],
            "official_JF_mean": values["global"]["J&F-Mean"],
            "supplementary_framewise_Dice_object_macro": values["dice"]["global_object_macro_mean"],
            "inference_seconds": values["runtime"]["inference_seconds"],
            "model_load_seconds": values["runtime"]["model_load_seconds"],
            "segmentation_pathways": values["runtime"]["segmentation_pathways"],
        })
    video_rows = []
    for record in inv["videos"]:
        video = record["video"]
        row = {"video": video, "objects": len(record["object_ids"])}
        for condition, values in conditions.items():
            row[f"{condition}_JF"] = values["per_video"][video]["J&F-Mean"]
            row[f"{condition}_Dice"] = values["dice"]["per_video"][video]
        video_rows.append(row)
    contrast_rows = [{"contrast": name, **result} for name, result in contrasts.items()]
    write_csv(out / "global_comparison.csv", list(global_rows[0]), global_rows)
    write_csv(out / "per_video_comparison.csv", list(video_rows[0]), video_rows)
    write_csv(out / "paired_contrasts.csv", list(contrast_rows[0]), contrast_rows)
    summary = {
        "protocol_sha256": digest(PROTOCOL),
        "dataset": "DAVIS 2017 semi-supervised val",
        "videos": inv["video_count"],
        "objects": inv["object_count"],
        "local_reference_setup_sha256": digest(local_run / "setup.json"),
        "local_reference_results_sha256": digest(local_results_path),
        "author_setups": author_setups,
        "conditions": conditions,
        "paired_video_level_contrasts": contrasts,
        "statistics_note": "Two-sided Wilcoxon and 10,000-resample 95% paired video bootstrap; Holm correction across four prespecified contrasts. Inference unit is video (n=30), not frame or object.",
        "precision_note": "SAM2Long upstream uses BF16 mask-memory storage on CPU; system-level comparisons are not fully precision- or compute-matched.",
        "dice_note": "Supplementary framewise object Dice on the same frames the official semi-supervised evaluator scores (frames 1..N-2); object means then video means. DAVIS J&F remains the benchmark primary metric.",
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# DAVIS 2017: regional token use and temporal-selection comparators",
        "",
        "This is a paired evaluation on the same 30 DAVIS 2017 validation videos, frame-0 prompts, SAM2.1 Hiera-Tiny checkpoint, indexed-mask policy, and official evaluator. Local contrasts isolate the regional cue; SAMURAI and SAM2Long are system-level references, not pure frame-selection ablations.",
        "",
        "## Overall scores",
        "",
        "| Condition | Official J | Official F | Official J&F | Supplementary framewise Dice | Inference seconds | Model load seconds | Pathways |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['condition']} | {row['official_J_mean']:.4f} | {row['official_F_mean']:.4f} | {row['official_JF_mean']:.4f} | {row['supplementary_framewise_Dice_object_macro']:.4f} | {row['inference_seconds']:.1f} | {row['model_load_seconds'] if row['model_load_seconds'] is not None else 'n/a'} | {row['segmentation_pathways']} |"
        )
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "| Contrast (left − right) | Mean ΔJ&F | Median ΔJ&F | Wins/ties/losses | 95% video-bootstrap CI | Wilcoxon p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, result in contrasts.items():
        lo, hi = result["bootstrap_95_percentile_ci"]
        w, t, l = result["wins_ties_losses"]
        lines.append(
            f"| {name} | {result['mean_paired_delta_JF']:+.4f} | {result['median_paired_delta_JF']:+.4f} | {w}/{t}/{l} | [{lo:+.4f}, {hi:+.4f}] | {result['wilcoxon_two_sided_p']:.4g} | {result['holm_adjusted_p']:.4g} |"
        )
    lines.extend([
        "",
        "The local generic-ToMe comparison is the control for whether the region cue adds value beyond generic token merging. A win over SAMURAI or SAM2Long is descriptive system-level competitiveness; it does not show that region guidance alone caused the difference. Report compute and precision differences alongside scores.",
        "",
        "Full per-video values and unrounded statistics are in `per_video_comparison.csv`, `paired_contrasts.csv`, and `comparison.json`.",
    ])
    (out / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
