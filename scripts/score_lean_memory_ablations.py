"""Independent Dice ablations from frozen Lean native/identity/fusion mask PNGs.

Strictly matches the *same videos, prompt hashes, source frame IDs and labels*
across arms. Never silently intersects incomplete cohorts. Empty GT frames
are excluded from present-frame Dice and counted as false alarms instead.

Expected input: receipts from scripts.run_lean_memory_fusion_ablation.
Results are development evidence unless the split is independently sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores


FORMAT = "adeseg_lean_memory_fusion_video_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda: stream.read(1048576), b""):
            h.update(buf)
    return h.hexdigest()


def parse_run(value):
    if "=" not in value:
        raise ValueError("Expected NAME=/path/to/run")
    name, path = value.split("=", 1)
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Unsafe or empty run name")
    folder = Path(path).resolve(strict=True)
    payload = json.loads((folder / "receipt.json").read_text("utf8"))
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"Invalid Lean memory fusion receipt: {name}")
    return name, folder, payload


def verified_prediction(folder, sequence, frame, record):
    expected = f"{frame:06d}.png"
    if (not isinstance(record, dict)
            or record.get("frame_index") != frame
            or record.get("prediction_file") != expected):
        raise ValueError(f"Missing or reordered prediction at {sequence}:{frame}")
    # The exact path is derived, never read from arbitrary manifest text.
    path = folder / "predictions" / sequence / expected
    if path.is_symlink() or not path.is_file() or sha(path) != record.get("prediction_sha256"):
        raise ValueError(f"Prediction checksum or path mismatch: {sequence}:{frame}")
    with Image.open(path) as im:
        array = np.asarray(im.convert("L"))
    if not np.isin(array, (0, 255)).all():
        raise ValueError(f"Prediction must be an exact 0/255 binary PNG: {path}")
    return array > 0


def score_runs(arms, manifest, output, *, baseline="native", bootstrap=10000, seed=20261120):
    """Create paired per-video/per-frame Dice and IoU; no model inference."""
    manifest = Path(manifest).resolve(strict=True)
    if len(arms) < 2 or baseline not in arms:
        raise ValueError("At least baseline and one candidate arm are required")
    if output.exists():
        raise FileExistsError("Refusing to overwrite prior score output")
    if bootstrap < 100 or seed < 0:
        raise ValueError("Invalid bootstrap settings")
    receipts = {name: parse_run(f"{name}={path}") for name, path in arms.items()}
    roots = {name: value[1] for name, value in receipts.items()}
    runs = {name: value[2] for name, value in receipts.items()}
    reference = runs[baseline]
    keys = ("track", "split", "label_id", "manifest_sha256",
            "prompt_manifest_sha256", "backbone_sha256",
            "lean_source_revision")
    for name, data in runs.items():
        if any(data.get(k) != reference.get(k) for k in keys):
            raise ValueError(f"Run {name} has unmatched backbone, split, dataset or prompts")
        if set(data.get("sequences", {})) != set(reference.get("sequences", {})):
            raise ValueError(f"Run {name} omitted sequences")
        if name != baseline and data.get("arm") == "native" and reference.get("arm") == "native":
            raise ValueError("Candidate was not run with a module enabled")
    if sha(manifest) != reference["manifest_sha256"]:
        raise ValueError("Dataset manifest checksum does not match frozen inference")
    videos = load_manifest(
        manifest, reference["split"],
        require_contiguous=True, require_jpeg=True, safe_output_ids=True,
    )
    if set(videos) != set(reference["sequences"]):
        raise ValueError("Scoring manifest sequences differ from completed receipts")
    per_video, per_frame, summary = {}, [], {}
    for video, frames in sorted(videos.items()):
        records = {name: runs[name]["sequences"][video] for name in runs}
        status = reference["sequences"][video].get("status")
        if any(item.get("status") != status for item in records.values()):
            raise ValueError(f"Run status mismatch on {video}")
        if status == "no_initialization":
            if any(item.get("predictions") for item in records.values()):
                raise ValueError(f"Unexpected predictions in unpromptable video {video}")
            continue
        if status != "completed":
            raise ValueError(f"Unexpected run status {status!r} for {video}")
        prompt_fields = ("prompt_frame_index", "prompt_mask_sha256",
                         "prompt_image_sha256", "prompt_origin")
        if any(any(record.get(k) != records[baseline].get(k) for k in prompt_fields)
               for record in records.values()):
            raise ValueError(f"Prompt mismatch on {video}")
        prompt = records[baseline]["prompt_frame_index"]
        frame_ids = [frame.frame_index for frame in frames]
        if prompt not in frame_ids:
            raise ValueError(f"Prompt frame absent from {video}")
        start = frame_ids.index(prompt) + 1
        target_frames = frames[start:]
        if any(len(item.get("predictions", [])) != len(target_frames)
               for item in records.values()):
            raise ValueError(f"Propagation coverage mismatch on {video}")
        video_metrics = {name: [] for name in runs}
        for position, frame in enumerate(target_frames):
            # Labels are opened here, after every run's predictions are frozen.
            truth = load_label_mask(frame.mask_path, [reference["label_id"]]) > 0
            truth_present = bool(truth.any())
            image_hash = sha(frame.image_path)
            scores = {}
            for name in runs:
                record = records[name]["predictions"][position]
                if record.get("image_sha256") != image_hash:
                    raise ValueError(f"Source-image mismatch for {video}:{frame.frame_index}")
                prediction = verified_prediction(
                    roots[name], video, frame.frame_index, record
                )
                if prediction.shape != truth.shape:
                    raise ValueError(f"Mask size mismatch on {video}:{frame.frame_index}")
                metrics = calculate_scores(prediction, truth)
                item = {
                    "dice": float(metrics["dice"]) if truth_present else None,
                    "iou": float(metrics["iou"]) if truth_present else None,
                    "present": truth_present,
                    "false_alarm": bool(prediction.any()) if not truth_present else None,
                }
                video_metrics[name].append(item)
                scores[name] = item
            if "identity" in runs and reference.get("arm") == "native":
                if (records[baseline]["predictions"][position]["prediction_sha256"]
                        != records["identity"]["predictions"][position]["prediction_sha256"]):
                    raise RuntimeError(f"Zero-residual identity diverged at {video}:{frame.frame_index}")
            per_frame.append({"video_id": video, "frame_index": frame.frame_index,
                              "present": truth_present, "arms": scores})
        per_video[video] = {}
        for name, observed in video_metrics.items():
            present = [r for r in observed if r["present"]]
            empty = [r for r in observed if not r["present"]]
            per_video[video][name] = {
                "n_present": len(present),
                "n_empty": len(empty),
                "mean_present_dice": float(np.mean([r["dice"] for r in present])) if present else None,
                "mean_present_iou": float(np.mean([r["iou"] for r in present])) if present else None,
                "empty_false_alarms": sum(r["false_alarm"] for r in empty),
            }

    eligible = [v for v in per_video if per_video[v][baseline]["n_present"] > 0]
    if not eligible:
        raise ValueError("No labeled-present post-prompt target frames to score")
    rng = np.random.default_rng(seed)
    for name in runs:
        macro = float(np.mean([per_video[v][name]["mean_present_dice"] for v in eligible]))
        frame_dice = [x["arms"][name]["dice"] for x in per_frame if x["present"]]
        gt_empty = [x["arms"][name]["false_alarm"] for x in per_frame if not x["present"]]
        paired = np.array([
            per_video[v][name]["mean_present_dice"] -
            per_video[v][baseline]["mean_present_dice"] for v in eligible
        ])
        sampled = rng.integers(0, len(paired), size=(bootstrap, len(paired)))
        ci = np.quantile(paired[sampled].mean(axis=1), (0.025, 0.975)).tolist()
        summary[name] = {
            "sequence_macro_present_dice": macro,
            "frame_weighted_present_dice": float(np.mean(frame_dice)),
            "mean_paired_delta_vs_baseline": float(paired.mean()),
            "paired_sequence_bootstrap_95_ci": [float(x) for x in ci],
            "present_sequences": len(eligible),
            "present_frames": len(frame_dice),
            "empty_frames": len(gt_empty),
            "empty_false_alarms": sum(gt_empty),
            "empty_false_alarm_rate": float(np.mean(gt_empty)) if gt_empty else None,
        }
    report = {
        "format": "adeseg_paired_lean_memory_fusion_dice_v1",
        "baseline": baseline,
        "run_arm_mapping": {name: value["arm"] for name, value in runs.items()},
        "provenance": {k: reference[k] for k in keys},
        "per_video": per_video,
        "summary": summary,
        "num_scored_frames": len(per_frame),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap,
        "warning": "Sequences are not necessarily independent patients. Previously inspected PolypGen is exploratory.",
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", "utf8")
    with (output / "per_frame.jsonl").open("w", encoding="utf8") as stream:
        for row in per_frame:
            stream.write(json.dumps(row) + "\n")
    with (output / "per_sequence.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("video_id", "arm", "n_present", "n_empty",
                         "present_dice", "present_iou", "false_alarms"))
        for video, arms_dict in per_video.items():
            for name, values in arms_dict.items():
                writer.writerow((video, name, values["n_present"], values["n_empty"],
                                 values["mean_present_dice"], values["mean_present_iou"],
                                 values["empty_false_alarms"]))
    lines = [
        "# Lean-SAM2 memory-fusion Dice ablation", "",
        "| Arm | Patient/video-macro present Dice | Delta vs baseline | Frame-weighted Dice | Empty false alarms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        lines.append(
            f"| {name} ({runs[name]['arm']}) | {item['sequence_macro_present_dice']:.5f} | "
            f"{item['mean_paired_delta_vs_baseline']:+.5f} | "
            f"{item['frame_weighted_present_dice']:.5f} | "
            f"{item['empty_false_alarms']}/{item['empty_frames']} |"
        )
    lines += [
        "", "All arms use identical backbone weights, split, input manifest, frozen prompts",
        "and exact source-frame IDs. Results score POST-prompt frames only.",
        "Confidence intervals are paired sequence-level bootstrap descriptions, not",
        "proof of independent patient generalization or clinical effectiveness.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", "utf8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True,
                   help="NAME=/path/to/run; specify native and at least one module arm")
    p.add_argument("--baseline", default="native")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261120)
    a = p.parse_args(argv)
    arms = {}
    for raw in a.run:
        name, _, path = raw.partition("=")
        if not path or name in arms:
            p.error("Run argument requires unique NAME=/path")
        arms[name] = Path(path)
    result = score_runs(arms, a.manifest, a.output, baseline=a.baseline,
                        bootstrap=a.bootstrap, seed=a.seed)
    print((a.output / "report.md").read_text("utf8"))


if __name__ == "__main__":
    main()
