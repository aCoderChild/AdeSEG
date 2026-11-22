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


FORMAT = "adeseg_lean_memory_fusion_video_v3"


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
    if reference.get("arm") != "native":
        raise ValueError("The reference run must be native Lean-SAM2")
    if "identity" in runs and runs["identity"].get("arm") != "identity":
        raise ValueError("Identity control receipt must be an identity run")
    keys = ("track", "split", "label_id", "manifest_sha256",
            "initialization_mode", "prompt_manifest_sha256", "backbone_sha256",
            "lean_source_revision", "runtime_signature")
    signature_keys = {
        "adeseg_head", "runner_sha256", "fusion_source_sha256",
        "torch_version", "python_version", "device", "precision",
        "lean_config_sha256", "lvos_config_sha256",
        "sam_config_sha256", "bypass_sha256",
    }
    if (not isinstance(reference.get("runtime_signature"), dict)
            or set(reference["runtime_signature"]) != signature_keys):
        raise ValueError("Missing or incomplete execution provenance")
    if reference.get("split") == "test":
        lock = reference.get("test_lock_sha256")
        if (not isinstance(lock, str) or len(lock) != 64
                or any(x not in "0123456789abcdef" for x in lock)):
            raise ValueError("Held-out TEST scores require a sealed test lock")
        if any(run.get("test_lock_sha256") != lock
               for run in runs.values()):
            raise ValueError("Held-out arms were evaluated under different TEST locks")
    for name, data in runs.items():
        if bool(data.get("dump_tamp_indices", False)) != bool(
                reference.get("dump_tamp_indices", False)):
            raise ValueError(
                "Mixed TAMP token-dump instrumentation across compared arms")
        if any(data.get(k) != reference.get(k) for k in keys):
            raise ValueError(f"Run {name} has unmatched backbone, split, dataset or prompts")
        if set(data.get("sequences", {})) != set(reference.get("sequences", {})):
            raise ValueError(f"Run {name} omitted sequences")
        if name != baseline and data.get("arm") == "native" and reference.get("arm") == "native":
            raise ValueError("Candidate was not run with a module enabled")
    # The inference manifest is IMAGE-ONLY by construction and cannot be
    # byte-identical to the annotation-bearing evaluation manifest. Match
    # source images and exact frame IDs instead; seal evaluation GT separately.
    evaluation_manifest_sha256 = sha(manifest)
    videos = load_manifest(
        manifest, reference["split"],
        require_contiguous=True, require_jpeg=True, safe_output_ids=True,
    )
    if set(videos) != set(reference["sequences"]):
        raise ValueError("Scoring manifest sequences differ from completed receipts")
    per_video, per_frame, summary = {}, [], {}
    # A missing automatic initialization is a measured pipeline failure,
    # never an silently omitted successful video. These GT counts are
    # produced in the scorer ONLY, after predictions have been frozen.
    uncovered_videos = {}
    excluded_pre_prompt_frames = {}
    gt_hasher = hashlib.sha256()
    for video, frames in sorted(videos.items()):
        for frame in frames:
            # Evaluation-only label provenance; never passed to inference.
            gt_hasher.update(video.encode("utf8") + b"\\0")
            gt_hasher.update(str(frame.frame_index).encode("ascii") + b"\\0")
            gt_hasher.update(sha(frame.mask_path).encode("ascii") + b"\\n")
        records = {name: runs[name]["sequences"][video] for name in runs}
        status = reference["sequences"][video].get("status")
        if any(item.get("status") != status for item in records.values()):
            raise ValueError(f"Run status mismatch on {video}")
        if status == "no_initialization":
            if any(item.get("predictions") for item in records.values()):
                raise ValueError(f"Unexpected predictions in unpromptable video {video}")
            truth_flags = [
                bool((load_label_mask(frame.mask_path, [reference["label_id"]]) > 0).any())
                for frame in frames
            ]
            uncovered_videos[video] = {
                "status": "no_initialization",
                "total_frames": len(frames),
                "gt_present_frames_without_prediction": sum(truth_flags),
                "gt_empty_frames_without_prediction": len(frames) - sum(truth_flags),
            }
            continue
        if status != "completed":
            raise ValueError(f"Unexpected run status {status!r} for {video}")
        # Validate entire per-frame temporal-budget history, not only the
        # final resident snapshot. These are storage counters, not full VRAM.
        frame_budget = {}
        for name, record in records.items():
            if bool(reference.get("dump_tamp_indices", False)):
                audit = record.get("tamp_index_dump")
                if (not isinstance(audit, dict)
                        or type(audit.get("records")) is not int
                        or audit["records"] < 0):
                    raise ValueError(f"Missing native TAMP dump receipt: {video}:{name}")
                digest = audit.get("sha256")
                log_path = roots[name] / "predictions" / video / "tamp_indices" / "ours_retained_tokens.jsonl"
                if digest is None:
                    if log_path.exists() or audit["records"] != 0:
                        raise ValueError(f"Unexpected native TAMP log without checksum: {video}:{name}")
                elif (not isinstance(digest, str) or len(digest) != 64
                      or log_path.is_symlink() or not log_path.is_file()
                      or sha(log_path) != digest):
                    raise ValueError(f"Native TAMP token dump checksum mismatch: {video}:{name}")
            resident = record.get("resident_memory")
            resident_keys = {
                "resident_frame_entries", "resident_spatial_tokens",
                "compressed_frame_entries", "protected_frame_entries",
                "insurance_entries", "note",
            }
            if not isinstance(resident, dict) or set(resident) != resident_keys:
                raise ValueError(f"Missing resident memory audit: {video}:{name}")
            if any(type(resident[k]) is not int or resident[k] < 0
                   for k in resident_keys - {"note"}):
                raise ValueError(f"Invalid resident memory audit: {video}:{name}")
            snapshots = record.get("per_frame_resident_memory")
            if not isinstance(snapshots, list):
                raise ValueError(f"Missing frame-level memory audit: {video}:{name}")
            frame_budget[name] = snapshots
        if "identity" in records and records["identity"]["resident_memory"] != (
                records[baseline]["resident_memory"]):
            raise RuntimeError(f"Native/identity retained memory mismatch: {video}")
        prompt_fields = ("prompt_frame_index", "prompt_mask_sha256",
                         "prompt_image_sha256", "prompt_origin",
                         "initialization_mode")
        if any(any(record.get(k) != records[baseline].get(k) for k in prompt_fields)
               for record in records.values()):
            raise ValueError(f"Prompt mismatch on {video}")
        prompt = records[baseline]["prompt_frame_index"]
        frame_ids = [frame.frame_index for frame in frames]
        if prompt not in frame_ids:
            raise ValueError(f"Prompt frame absent from {video}")
        start = frame_ids.index(prompt) + 1
        skipped_truth = [
            bool((load_label_mask(frame.mask_path, [reference["label_id"]]) > 0).any())
            for frame in frames[:start]
        ]
        excluded_pre_prompt_frames[video] = {
            "all_unscored_frames_up_to_prompt": start,
            "gt_present_unscored": sum(skipped_truth),
            "gt_empty_unscored": start - sum(skipped_truth),
            "note": "This post-prompt evaluation deliberately does not score initial/prompt frames",
        }
        target_frames = frames[start:]
        if any(len(item.get("predictions", [])) != len(target_frames)
               for item in records.values()):
            raise ValueError(f"Propagation coverage mismatch on {video}")
        for name, snapshots in frame_budget.items():
            if len(snapshots) != len(target_frames) + 1:
                raise ValueError(f"Incomplete per-frame memory audit: {video}:{name}")
            expected_frames = [frames[start-1].frame_index] + [
                f.frame_index for f in target_frames
            ]
            if any(not isinstance(entry, dict)
                   or entry.get("local_frame_index") != i
                   or entry.get("source_frame_index") != original
                   or not isinstance(entry.get("memory"), dict)
                   or set(entry["memory"]) != resident_keys
                   or any(type(entry["memory"][k]) is not int
                          or entry["memory"][k] < 0
                          for k in resident_keys - {"note"})
                   for i, (entry, original) in
                   enumerate(zip(snapshots, expected_frames))):
                raise ValueError(f"Frame memory audit order/schema invalid: {video}:{name}")
            if snapshots[-1]["memory"] != records[name]["resident_memory"]:
                raise ValueError(f"Final resident-memory audit mismatch: {video}:{name}")
        if "identity" in records and frame_budget["identity"] != frame_budget[baseline]:
            raise RuntimeError(f"Native/identity temporal-memory mismatch: {video}")
        # The source frame providing the initial automatic prompt must also
        # match the labeled evaluation manifest, not only the post-prompt frames.
        prompt_frame = frames[start - 1]
        if sha(prompt_frame.image_path) != records[baseline]["prompt_image_sha256"]:
            raise ValueError(f"Prompt image mismatches labeled cohort: {video}")
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
                digest = record.get("logits_sha256")
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError(
                        f"Missing logit checksum: {video}:{frame.frame_index}")
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
                        != records["identity"]["predictions"][position]["prediction_sha256"]
                        or records[baseline]["predictions"][position]["logits_sha256"]
                        != records["identity"]["predictions"][position]["logits_sha256"]):
                    raise RuntimeError(f"Zero-residual identity diverged at {video}:{frame.frame_index}")
            per_frame.append({
                "video_id": video, "frame_index": frame.frame_index,
                "present": truth_present, "arms": scores,
                "resident_spatial_tokens": {
                    name: frame_budget[name][position+1]["memory"]["resident_spatial_tokens"]
                    for name in runs
                },
                "insurance_entries": {
                    name: frame_budget[name][position+1]["memory"]["insurance_entries"]
                    for name in runs
                },
            })
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
    # Entirely GT-empty cohorts are legitimate negative-control evaluations:
    # report false alarm rates, with positive Dice explicitly undefined.
    # Never convert a GT-empty frame into a perfect Dice=1 surrogate.
    rng = np.random.default_rng(seed)
    all_source_frames = sum(len(v) for v in videos.values())
    not_initialized_present = sum(
        x["gt_present_frames_without_prediction"]
        for x in uncovered_videos.values())
    not_initialized_empty = sum(
        x["gt_empty_frames_without_prediction"]
        for x in uncovered_videos.values())
    preprompt_present = sum(
        x["gt_present_unscored"] for x in excluded_pre_prompt_frames.values())
    preprompt_empty = sum(
        x["gt_empty_unscored"] for x in excluded_pre_prompt_frames.values())
    for name in runs:
        macro = (float(np.mean(
            [per_video[v][name]["mean_present_dice"] for v in eligible]))
            if eligible else None)
        frame_dice = [x["arms"][name]["dice"] for x in per_frame if x["present"]]
        gt_empty = [x["arms"][name]["false_alarm"] for x in per_frame if not x["present"]]
        paired = np.array([
            per_video[v][name]["mean_present_dice"] -
            per_video[v][baseline]["mean_present_dice"] for v in eligible
        ])
        ci = None
        if len(paired):
            sampled = rng.integers(0, len(paired), size=(bootstrap, len(paired)))
            ci = np.quantile(
                paired[sampled].mean(axis=1), (0.025, 0.975)).tolist()
        memory_pairs = [
            (runs[name]["sequences"][v]["resident_memory"],
             runs[baseline]["sequences"][v]["resident_memory"])
            for v in per_video
        ]
        memory_budget_matched = all(
            left["resident_spatial_tokens"] == right["resident_spatial_tokens"]
            and left["insurance_entries"] == right["insurance_entries"]
            for left, right in memory_pairs
        )
        budget_frames = [
            (runs[name]["sequences"][v]["per_frame_resident_memory"][i]["memory"],
             runs[baseline]["sequences"][v]["per_frame_resident_memory"][i]["memory"])
            for v in per_video
            for i in range(len(runs[baseline]["sequences"][v]["per_frame_resident_memory"]))
        ]
        all_frame_budgets_matched = all(
            a["resident_spatial_tokens"] == b["resident_spatial_tokens"]
            and a["insurance_entries"] == b["insurance_entries"]
            for a, b in budget_frames
        )
        write_audits = [runs[name]["sequences"][v] for v in per_video]
        cuda_peaks = [
            int(r["cuda_peak_allocated_bytes"])
            for r in write_audits
            if type(r.get("cuda_peak_allocated_bytes")) is int
            and r["cuda_peak_allocated_bytes"] >= 0
        ]
        propagation_times = [
            float(r["propagation_seconds"]) for r in write_audits
            if isinstance(r.get("propagation_seconds"), (int, float))
            and r["propagation_seconds"] > 0
        ]
        writes_checked = sum(int(x.get("memory_writes_checked", 0)) for x in write_audits)
        writes_modified_after_storage = sum(
            int(x.get("memory_writes_quantized_changed", 0)) for x in write_audits)
        writes_affecting_future = sum(
            int(x.get("effective_memory_writes_affecting_future", 0))
            for x in write_audits)
        anchor_audits = [x.get("anchor_similarity_audit") for x in write_audits]
        valid_anchors = [x for x in anchor_audits
                         if isinstance(x, dict) and x.get("eligible_writes", 0) > 0
                         and x.get("mean_abs_similarity_shift") is not None]
        anchor_count = sum(x["eligible_writes"] for x in valid_anchors)
        anchor_drift = (
            sum(x["eligible_writes"] * x["mean_abs_similarity_shift"]
                for x in valid_anchors) / anchor_count if anchor_count else None)
        summary[name] = {
            "fusion_input_mode": runs[name].get("fusion_input_mode", "full"),
            "fusion_preserve_native_no_object": runs[name].get(
                "fusion_preserve_native_no_object"),
            "native_tamp_index_records": sum(
                int(x.get("tamp_index_dump", {}).get("records", 0))
                for x in write_audits),
            "max_cuda_allocated_mb_observed": (
                max(cuda_peaks) / (1024 * 1024) if cuda_peaks else None),
            "mean_cuda_peak_allocated_mb_observed": (
                float(np.mean(cuda_peaks)) / (1024 * 1024)
                if cuda_peaks else None),
            "mean_single_sequence_propagation_seconds": (
                float(np.mean(propagation_times))
                if propagation_times else None),
            "memory_writes_checked": writes_checked,
            "memory_writes_modified_after_bf16": writes_modified_after_storage,
            "effective_memory_writes_affecting_future": writes_affecting_future,
            "effective_memory_write_fraction": (
                writes_modified_after_storage / writes_checked if writes_checked else None),
            "anchor_similarity_eligible_writes": anchor_count,
            "mean_abs_anchor_cosine_shift_at_write": anchor_drift,
            "all_frame_resident_budgets_matched": all_frame_budgets_matched,
            "resident_budget_frame_match_fraction": (
                sum(a["resident_spatial_tokens"] == b["resident_spatial_tokens"]
                    and a["insurance_entries"] == b["insurance_entries"]
                    for a, b in budget_frames) / len(budget_frames)
                if budget_frames else None
            ),
            "resident_spatial_and_insurance_matched": memory_budget_matched,
            "resident_spatial_token_deltas": {
                v: (runs[name]["sequences"][v]["resident_memory"]["resident_spatial_tokens"] -
                    runs[baseline]["sequences"][v]["resident_memory"]["resident_spatial_tokens"])
                for v in per_video
            },
            "sequence_macro_present_dice": macro,
            "frame_weighted_present_dice": (
                float(np.mean(frame_dice)) if frame_dice else None),
            "mean_paired_delta_vs_baseline": (
                float(paired.mean()) if len(paired) else None),
            "paired_sequence_bootstrap_95_ci": (
                [float(x) for x in ci] if ci is not None else None),
            "present_sequences": len(eligible),
            "source_videos": len(videos),
            "videos_with_initialization": len(per_video),
            "videos_without_initialization": len(uncovered_videos),
            "video_initialization_coverage": len(per_video) / len(videos),
            "gt_present_frames_without_initialization": not_initialized_present,
            "gt_empty_frames_without_initialization": not_initialized_empty,
            "gt_present_frames_unscored_at_or_before_prompt": preprompt_present,
            "gt_empty_frames_unscored_at_or_before_prompt": preprompt_empty,
            "all_source_frames": all_source_frames,
            "post_prompt_scored_frames": len(per_frame),
            "fraction_source_frames_scored_post_prompt":
                len(per_frame) / all_source_frames,
            "conditional_dice_only": True,
            "present_frames": len(frame_dice),
            "empty_frames": len(gt_empty),
            "empty_false_alarms": sum(gt_empty),
            "empty_false_alarm_rate": float(np.mean(gt_empty)) if gt_empty else None,
        }
    report = {
        "format": "adeseg_paired_lean_memory_fusion_dice_v1",
        "baseline": baseline,
        "run_arm_mapping": {name: value["arm"] for name, value in runs.items()},
        "run_input_modes": {
            name: value.get("fusion_input_mode", "full") for name, value in runs.items()
        },
        "initialization_mode": reference.get("initialization_mode", "mask_reprompt"),
        "run_object_presence_gate_modes": {
            name: value.get("fusion_preserve_native_no_object")
            for name, value in runs.items()
        },
        "native_tamp_dump_enabled": bool(reference.get("dump_tamp_indices", False)),
        "test_lock_sha256": reference.get("test_lock_sha256"),
        "unpromptable_video_coverage": uncovered_videos,
        "unscored_frames_through_prompt": excluded_pre_prompt_frames,
        "coverage_warning": (
            "Post-prompt Dice is conditional on successful automatic initialization "
            "and GT-present scored frames. Unpromptable and before/at-prompt "
            "frames are separately counted, NOT imputed as perfect. "
            "Do not compare conditional Dice across different prompt cohorts "
            "without also reporting coverage."),

        "provenance": {**{k: reference[k] for k in keys},
                       "evaluation_manifest_sha256": evaluation_manifest_sha256,
                       "evaluation_label_files_sha256": gt_hasher.hexdigest()},
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
        def _decimal(value, sign=False):
            return (f"{value:+.5f}" if sign else f"{value:.5f}") \
                if value is not None else "N/A (no GT-present frames)"
        lines.append(
            f"| {name} ({runs[name]['arm']}) | "
            f"{_decimal(item['sequence_macro_present_dice'])} | "
            f"{_decimal(item['mean_paired_delta_vs_baseline'], sign=True)} | "
            f"{_decimal(item['frame_weighted_present_dice'])} | "
            f"{item['empty_false_alarms']}/{item['empty_frames']} |"
        )
    lines += [
        "",
        "Automatic initialization / evaluation coverage (all arms share the cohort):",
        f"- Videos initialized: {len(per_video)}/{len(videos)}; "
        f"without initialization: {len(uncovered_videos)}",
        f"- GT-present frames without initialization: {not_initialized_present}; "
        f"GT-empty frames without initialization: {not_initialized_empty}",
        f"- Before/at-prompt frames not scored: "
        f"GT-present={preprompt_present}, GT-empty={preprompt_empty}",
        f"- Post-prompt scored frames: {len(per_frame)}/{all_source_frames} "
        "(conditional metrics; no missing-video imputation)",
        "WARNING: compare detector/FGPI prompt coverage separately; conditional "
        "post-prompt Dice alone cannot measure end-to-end accuracy.",
        "",
        "Resident-memory and insurance entries equal to native (by arm): " +
        ", ".join(f"{name}={summary[name]['resident_spatial_and_insurance_matched']}"
                  for name in summary),
        "Per-frame resident storage and insurance matching is reported in results.json.",
        "Resident count matching is NOT proof of equal runtime peak VRAM or attention bytes.",
        "Uncontrolled one-run CUDA allocated-memory peaks and propagation timing "
        "are separately recorded in results.json; they do NOT establish speedups.",
        "", "Memory-write diagnostics (mode, BF16-effective writes, mean anchor cosine shift):",
        *[
            f"- {name}: {summary[name]['fusion_input_mode']}; "
            f"{summary[name]['memory_writes_modified_after_bf16']}/"
            f"{summary[name]['memory_writes_checked']}; "
            f"future-effective={summary[name]['effective_memory_writes_affecting_future']}; "
            f"{summary[name]['mean_abs_anchor_cosine_shift_at_write']}"
            for name in runs
        ],
        "Anchor cosine drift at WRITE is diagnostic only, NOT measured TAMP top-K change.",
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
