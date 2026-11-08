#!/usr/bin/env python3
"""Run the frozen MemSAM-derived writer conditions on the amended seq2–21 cohort.

This is a descriptive follow-up, not held-out confirmation: its cohort includes
the writer's training and validation sequences as well as seq16–21 from the
parent protocol's test split. It reuses the attempt_05 evaluator and never
selects checkpoints from these results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts import run_memsam_residual_writer as study
from scripts.memory_token_io import digest
from scripts.run_complementary_memory_pilot import activate_author_repo, input_hashes


AMENDMENT_REL = Path(
    "protocols/memsam_residual_memory_writer_v2_full_cohort_amendment_2026-11-08.json"
)
NATIVE_REUSE_AMENDMENT_REL = Path(
    "protocols/memsam_residual_memory_writer_v2_full_cohort_native_reuse_amendment_2026-11-08.json"
)
NATIVE_BASELINE_REL = Path("outputs/experiment/155_region_constrained_tome_polypgen_v1/full")
NATIVE_RESULTS_REL = Path("outputs/experiment/143_official_native_feature_audit_v1/full/native_results.json")
MANIFEST_REL = "data/polypgen_sequence.jsonl"
SEQUENCES = [
    "seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9", "seq10",
    "seq11", "seq12", "seq13", "seq14", "seq15", "seq16", "seq17",
    "seq18", "seq19", "seq20", "seq21",
]
CONDITIONS = ("native_sam2", "mask_neutral_writer", "soft_mask_writer")
FRESH_CONDITIONS = ("mask_neutral_writer", "soft_mask_writer")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed_writer(attempt: Path, arm: str, seed: int, protocol: dict, core_hash: str):
    folder = attempt / "training" / arm
    receipt_path = folder / "completion.json"
    checkpoint_path = folder / "best.pt"
    require(receipt_path.is_file() and checkpoint_path.is_file(),
            f"Writer arm is not complete: {arm}")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("complete") is True and receipt.get("arm") == arm,
            f"Writer completion receipt invalid: {arm}")
    require(receipt.get("clips") == 480 and receipt.get("updates") == 960,
            f"Writer schedule incomplete: {arm}")
    require(digest(checkpoint_path) == receipt.get("best_checkpoint_sha256"),
            f"Writer best-checkpoint hash mismatch: {arm}")
    scores = {int(epoch): float(row["present_dice"])
              for epoch, row in receipt.get("validation", {}).items()}
    require(scores and receipt.get("best_epoch") == max(scores, key=scores.get),
            f"Writer epoch selection does not match validation: {arm}")
    writer = study.writer_init(seed, protocol)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    writer.load_state_dict(state, strict=True)
    writer.eval().requires_grad_(False)
    writer_hash = study.base.state_digest(writer.state_dict())
    selected_validation = receipt.get("validation", {}).get(str(receipt["best_epoch"]), {})
    recorded_summaries = selected_validation.get("summaries", {})
    require(set(recorded_summaries) == set(protocol["data"]["validation_sequences"]),
            f"Selected epoch lacks complete validation summaries: {arm}")
    replay_root = attempt / "validation" / arm / "selected"
    replayed = {}
    for name in protocol["data"]["validation_sequences"]:
        summary_path = replay_root / name / "summary.json"
        require(summary_path.is_file(), f"Selected-checkpoint replay missing: {arm}/{name}")
        replay = json.loads(summary_path.read_text())
        expected = recorded_summaries[name]
        require(replay.get("writer_state_sha256") == writer_hash,
                f"Selected replay used a different writer state: {arm}/{name}")
        require(replay.get("setup_core_sha256") == core_hash,
                f"Selected replay used a different frozen setup: {arm}/{name}")
        for metric in ("present_dice", "present_iou", "present_boundary_f",
                       "gt_empty_false_alarm_count", "target_frames"):
            left, right = replay.get(metric), expected.get(metric)
            require(left == right or (
                isinstance(left, (int, float)) and isinstance(right, (int, float))
                and math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            ), f"Selected replay mismatch for {arm}/{name}/{metric}: {left} != {right}")
        replayed[name] = replay["present_dice"]
    replay_mean = float(sum(replayed.values()) / len(replayed))
    require(math.isclose(replay_mean, receipt["best_validation_present_dice"],
                         rel_tol=0.0, abs_tol=1e-12),
            f"Selected replay mean differs from completion receipt: {arm}")
    return writer, {
        "completion_receipt_sha256": digest(receipt_path),
        "best_checkpoint_sha256": digest(checkpoint_path),
        "best_epoch": receipt["best_epoch"],
        "best_validation_present_dice": receipt["best_validation_present_dice"],
        "validation_present_dice_by_epoch": scores,
        "writer_state_sha256": writer_hash,
        "selected_validation_replay_present_dice": replayed,
        "selected_validation_replay_mean": replay_mean,
    }


def split_descriptives(summaries: dict, sequence_names: list[str], split_map: dict) -> dict:
    result = {}
    metric_names = ("present_dice", "present_iou", "present_boundary_f", "fps")
    for split, names in split_map.items():
        result[split] = {}
        for condition in CONDITIONS:
            rows = [summaries[condition][name] for name in names]
            result[split][condition] = {
                metric: float(sum(row[metric] for row in rows if row[metric] is not None)
                              / sum(row[metric] is not None for row in rows))
                if any(row[metric] is not None for row in rows) else None
                for metric in metric_names
            }
            absent = sum(row["gt_empty_frames"] for row in rows)
            false_alarms = sum(row["gt_empty_false_alarm_count"] for row in rows)
            result[split][condition].update({
                "sequence_count": len(names),
                "false_alarm_count": false_alarms,
                "gt_empty_frames": absent,
                "false_alarm_rate": false_alarms / absent if absent else None,
                "undetected_returns": sum(row["propagation"]["undetected_returns"] for row in rows),
                "return_events": sum(row["propagation"]["return_count"] for row in rows),
            })
    return result


def paired_contrasts(summaries: dict, sequence_names: list[str], seed: int) -> dict:
    pairs = (
        ("mask_neutral_writer", "native_sam2"),
        ("soft_mask_writer", "native_sam2"),
        ("soft_mask_writer", "mask_neutral_writer"),
    )
    contrasts, pvalues = {}, {}
    for candidate, control in pairs:
        delta = {
            name: summaries[candidate][name]["present_dice"]
                  - summaries[control][name]["present_dice"]
            for name in sequence_names
        }
        key = f"{candidate}_minus_{control}"
        contrasts[key] = {
            "mean_dice_gain": float(sum(delta.values()) / len(delta)),
            "bootstrap_95_ci": study.sequence_bootstrap_interval(delta, seed=seed),
            "wins": sum(value > 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "losses": sum(value < 0 for value in delta.values()),
            "per_sequence": delta,
            "leave_one_sequence_out_mean_gain": {
                omitted: float(sum(value for name, value in delta.items() if name != omitted)
                               / (len(delta) - 1))
                for omitted in sequence_names
            },
            "wilcoxon": study.paired_wilcoxon(
                {name: summaries[candidate][name]["present_dice"] for name in sequence_names},
                {name: summaries[control][name]["present_dice"] for name in sequence_names},
            ),
        }
        pvalues[key] = contrasts[key]["wilcoxon"]["p"]
    for name, value in study.holm_adjust(pvalues).items():
        contrasts[name]["holm_adjusted_p"] = value
    return contrasts


def load_native_baseline(drive_root: Path, core: dict, amendment: dict,
                         sequence_names: list[str], current_inputs: dict,
                         videos: dict) -> tuple[dict, dict]:
    reference = amendment["native_baseline"]
    run_dir = drive_root / NATIVE_BASELINE_REL
    native_results_path = drive_root / NATIVE_RESULTS_REL
    setup_path = run_dir / "setup.json"
    protocol_path = run_dir / "protocol.json"
    summary_path = run_dir / "summary.json"
    completion_path = run_dir / "completion.json"
    artifacts = {
        "run155_setup": setup_path,
        "run155_protocol": protocol_path,
        "run155_summary": summary_path,
        "run155_completion": completion_path,
        "run143_native_results": native_results_path,
    }
    expected_hashes = {
        "run155_setup": reference["setup_sha256"],
        "run155_protocol": reference["protocol_sha256"],
        "run155_summary": reference["summary_sha256"],
        "run155_completion": reference["completion_sha256"],
        "run143_native_results": reference["native_results_sha256"],
    }
    for name, path in artifacts.items():
        require(path.is_file(), f"Native baseline artifact missing: {path}")
        require(digest(path) == expected_hashes[name],
                f"Native baseline artifact hash mismatch: {name}")

    native_setup = json.loads(setup_path.read_text())
    native_protocol = json.loads(protocol_path.read_text())
    native_summary = json.loads(summary_path.read_text())
    completion = json.loads(completion_path.read_text())
    native_results = json.loads(native_results_path.read_text())
    model = native_protocol["model"]
    native_scope = native_protocol.get("scope", {})
    require(completion.get("passed") is True,
            "Existing native baseline did not complete successfully")
    require(native_setup.get("native_results_sha256") == digest(native_results_path)
            == reference["native_results_sha256_pinned_by_run155_setup"],
            "Run155 does not pin the supplied Run143 native result file")
    require(native_setup.get("checkpoint_sha256") == core["checkpoint_sha256"]
            == reference["checkpoint_sha256"], "Native checkpoint differs")
    require(native_setup.get("manifest_sha256") == core["manifest_sha256"]
            == reference["manifest_sha256"], "Native dataset manifest differs")
    require(native_setup.get("sam2_revision") == core["sam2_revision"]
            == reference["sam2_revision"], "Native SAM2 source revision differs")
    require(model.get("configuration") == reference["model_config"]
            and native_setup.get("device") == reference["device"]
            and native_setup.get("dtype") == reference["dtype"],
            "Native model configuration or execution dtype differs")
    require(set(sequence_names).issubset(native_scope.get("sequences", []))
            and native_setup.get("conditions", []).count("native") == 1,
            "Native source protocol does not cover the requested native condition")
    require(native_results.get("checkpoint_sha256") == core["checkpoint_sha256"],
            "Native frame evidence is for a different checkpoint")
    require(set(sequence_names).issubset(native_setup.get("sequences", []))
            and set(sequence_names).issubset(completion.get("sequences", []))
            and set(sequence_names).issubset(native_summary.get("per_sequence", {}))
            and set(sequence_names).issubset(native_results.get("sequences", {})),
            "Existing native baseline does not cover the requested cohort")

    normalized, sequence_evidence = {}, {}
    for name in sequence_names:
        row = native_summary["per_sequence"][name]["native"]
        metrics = native_results["sequences"][name]
        source = reference["sequence_evidence"][name]
        receipt_path = drive_root / source["native_receipt_path"]
        frames_path = drive_root / source["native_frame_evidence_path"]
        require(receipt_path.is_file() and frames_path.is_file(),
                f"Native frame receipt missing for {name}")
        require(digest(receipt_path) == source["native_receipt_sha256"],
                f"Native receipt hash mismatch for {name}")
        require(digest(frames_path) == source["native_frame_evidence_sha256"],
                f"Native frame evidence hash mismatch for {name}")
        receipt = json.loads(receipt_path.read_text())
        frames = json.loads(frames_path.read_text())
        prompt = current_inputs[name]["prompt"]
        require(receipt.get("checkpoint_sha256") == core["checkpoint_sha256"]
                and receipt.get("manifest_sha256") == core["manifest_sha256"]
                and receipt.get("model_config") == reference["model_config"]
                and receipt.get("sequence") == name and receipt.get("object_id") == 1,
                f"Native sequence receipt setup mismatch for {name}")
        require(receipt.get("prompt") == prompt
                and receipt.get("num_frames") == len(videos[name])
                and len(frames) == metrics["targets"]
                and len(frames) == len(receipt.get("predictions", []))
                and [item["frame"] for item in frames]
                    == list(range(prompt + 1, len(videos[name]))),
                f"Native frame coverage or prompt differs for {name}")
        prediction_hashes = {item["frame"]: item["sha256"]
                             for item in receipt["predictions"]}
        require(all(prediction_hashes.get(item["frame"]) == item["prediction_sha256"]
                    for item in frames),
                f"Native per-frame prediction hashes differ for {name}")
        positive = [item for item in frames if item["present"]]
        empty = [item for item in frames if not item["present"]]
        require(len(positive) == metrics["present_targets"] == row["present_targets"]
                and len(empty) == metrics["gt_empty_targets"] == row["gt_empty_targets"]
                and metrics["return_count"] == row["return_count"]
                and metrics["undetected_returns"] == row["undetected_returns"],
                f"Native frame counts or propagation summaries differ for {name}")
        recomputed = {
            "present_dice": float(sum(item["scores"]["dice"] for item in positive) / len(positive)),
            "present_iou": float(sum(item["scores"]["iou"] for item in positive) / len(positive)),
            "present_boundary_f": float(sum(item["scores"]["boundary_f"] for item in positive) / len(positive)),
            "gt_empty_false_alarm_count": sum(bool(item["absent_fp"]) for item in empty),
        }
        expected_metrics = {
            "present_dice": metrics["present_dice"],
            "present_iou": metrics["iou"],
            "present_boundary_f": metrics["boundary_f"],
            "gt_empty_false_alarm_count": metrics["gt_empty_false_alarm_count"],
        }
        for key, actual in recomputed.items():
            expected = expected_metrics[key]
            require(actual == expected or (
                isinstance(actual, (int, float)) and isinstance(expected, (int, float))
                and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            ), f"Native frame evidence does not reproduce {name}/{key}")
        for key, value in (("present_dice", row["present_dice"]),
                           ("present_iou", row["present_iou"]),
                           ("present_boundary_f", row["present_boundary_f"]),
                           ("gt_empty_false_alarm_count", row["gt_empty_false_alarms"])):
            actual = recomputed[key]
            require(actual == value or (
                isinstance(actual, (int, float)) and isinstance(value, (int, float))
                and math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12)
            ), f"Run155 native summary differs from verified frame evidence: {name}/{key}")
        require(row.get("complete") is True
                and row.get("parameter_digest") == reference["parameter_digest"],
                f"Native sequence completion or parameter digest mismatch: {name}")

        returns = metrics.get("returns", [])
        detected_delays = [item["delay_sampled_frames"] for item in returns
                           if item.get("detected") and item.get("delay_sampled_frames") is not None]
        wall_seconds = row.get("wall_seconds")
        normalized[name] = {
            "sequence": name, "condition": "native_reused",
            "prompt_frame": prompt, "target_frames": len(frames),
            "present_frames": len(positive), "gt_empty_frames": len(empty),
            "present_dice": recomputed["present_dice"],
            "all_frame_dice": float(sum(item["scores"]["dice"] for item in frames) / len(frames)),
            "present_iou": recomputed["present_iou"],
            "present_boundary_f": recomputed["present_boundary_f"],
            "gt_empty_false_alarm_count": recomputed["gt_empty_false_alarm_count"],
            "gt_empty_false_alarm_rate": (recomputed["gt_empty_false_alarm_count"] / len(empty)
                                           if empty else None),
            "propagation": {
                "return_count": metrics["return_count"],
                "undetected_returns": metrics["undetected_returns"],
                "missed_return_rate": (metrics["undetected_returns"] / metrics["return_count"]
                                       if metrics["return_count"] else None),
                "mean_detected_return_delay_sampled_frames": (
                    float(sum(detected_delays) / len(detected_delays)) if detected_delays else None
                ),
            },
            "elapsed_seconds": wall_seconds,
            "fps": len(frames) / wall_seconds if wall_seconds else None,
            "writer_state_sha256": "native",
            "setup_core_sha256": row.get("setup_sha256"),
            "native_attention_reads": None,
            "source_receipt_sha256": digest(receipt_path),
            "source_frame_evidence_sha256": digest(frames_path),
        }
        sequence_evidence[name] = {
            "native_receipt_sha256": digest(receipt_path),
            "native_frame_evidence_sha256": digest(frames_path),
            "prompt_frame": prompt,
            "target_frames": len(frames),
            "parameter_digest": row["parameter_digest"],
        }

    provenance = {
        "source_run": str(run_dir),
        "native_results_path": str(native_results_path),
        "setup_sha256": digest(setup_path),
        "protocol_sha256": digest(protocol_path),
        "summary_sha256": digest(summary_path),
        "completion_sha256": digest(completion_path),
        "native_results_sha256": digest(native_results_path),
        "native_results_sha256_pinned_by_run155": native_setup["native_results_sha256"],
        "sequence_evidence": sequence_evidence,
        "overlap_evaluator_parity": amendment["native_baseline"]["cross_evaluator_overlap_check"],
        "native_inference_repeated": False,
    }
    return normalized, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    attempt = args.attempt_dir.expanduser().resolve()
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              attempt / "full_cohort_seq2_21_exploratory")
    require(output.is_relative_to(attempt), "Exploratory results must stay under attempt_05")
    drive = study.synced_adeseg_root(output)
    require(output.is_relative_to((drive / "outputs/experiment/memsam_residual_memory_writer_v2").resolve()),
            "Output must be under the synced Google Drive experiment folder")

    amendment_path = ROOT / AMENDMENT_REL
    drive_amendment = output / "protocol.json"
    native_amendment_path = ROOT / NATIVE_REUSE_AMENDMENT_REL
    drive_native_amendment = output / "native_reuse_amendment.json"
    require(amendment_path.is_file() and drive_amendment.is_file()
            and native_amendment_path.is_file() and drive_native_amendment.is_file(),
            "Both versioned amendments and their Drive copies must exist before inference")
    require(amendment_path.read_bytes() == drive_amendment.read_bytes(),
            "Drive amendment differs from the versioned protocol")
    require(native_amendment_path.read_bytes() == drive_native_amendment.read_bytes(),
            "Drive native-reuse amendment differs from the versioned protocol")
    amendment = json.loads(amendment_path.read_text())
    native_amendment = json.loads(native_amendment_path.read_text())
    require(SEQUENCES == amendment["scope"]["sequence_ids"] and len(SEQUENCES) == 19,
            "Sequence list differs from the frozen amendment")
    require(native_amendment["supersedes"]["sha256"] == digest(amendment_path),
            "Native-reuse amendment does not supersede the frozen full-cohort protocol")
    require(SEQUENCES == native_amendment["scope"]["sequence_ids"]
            and native_amendment["fresh_inference_conditions"] == list(FRESH_CONDITIONS),
            "Native-reuse amendment sequence scope or inference arms differ")

    setup_path = attempt / "setup.json"
    setup = json.loads(setup_path.read_text())
    core = setup["core"]
    core_hash = setup["core_sha256"]
    require(canonical_hash(core) == core_hash, "Parent setup core hash mismatch")
    require(digest(attempt / "source_snapshot.zip") == setup["snapshot_sha256"],
            "Parent source snapshot hash mismatch")
    for relative, expected in core["sources"].items():
        require(digest(ROOT / relative) == expected, f"Pinned parent source changed: {relative}")
    protocol = core["protocol"]

    command = setup.get("command", [])
    require("--checkpoint" in command, "Cannot recover the attempt's checkpoint path")
    checkpoint = Path(command[command.index("--checkpoint") + 1]).expanduser().resolve()
    require(digest(checkpoint) == core["checkpoint_sha256"], "SAM2 checkpoint changed")
    videos = (load_manifest(ROOT / MANIFEST_REL, "dev")
              | load_manifest(ROOT / MANIFEST_REL, "test"))
    require(set(SEQUENCES).issubset(videos), "One or more requested sequences are absent")
    current_inputs = input_hashes(videos, SEQUENCES)
    for name in SEQUENCES:
        require(core["inputs"].get(name) == current_inputs[name],
                f"Video or mask inputs changed for {name}")
    gt_presence = {}
    for name in SEQUENCES:
        prompt = current_inputs[name]["prompt"]
        frames = videos[name][prompt + 1:]
        present = sum(bool((load_label_mask(frame.mask_path, [1]) == 1).any())
                      for frame in frames)
        require(present > 0, f"No post-prompt positive GT frame for {name}")
        gt_presence[name] = {"positive_frames": present, "target_frames": len(frames)}
    native_summaries, native_provenance = load_native_baseline(
        drive, core, native_amendment, SEQUENCES, current_inputs, videos
    )

    seed = protocol["training"]["seed"]
    writers = {}
    checkpoint_info = {"native_sam2": {
        "checkpoint_sha256": core["checkpoint_sha256"],
        "parent_setup_core_sha256": core_hash,
        "baseline_reused": True,
        "baseline_native_results_sha256": native_provenance["native_results_sha256"],
    }}
    for arm in ("mask_neutral_writer", "soft_mask_writer"):
        writers[arm], checkpoint_info[arm] = completed_writer(
            attempt, arm, seed, protocol, core_hash
        )

    repo, revision = activate_author_repo()
    require(revision == core["sam2_revision"], "SAM2 repository revision changed")
    split_map = {
        "parent_training_seen": amendment["scope"]["split_membership_from_parent_protocol"]["training"],
        "parent_validation_seen": amendment["scope"]["split_membership_from_parent_protocol"]["validation"],
        "parent_original_test_now_included": amendment["scope"]["split_membership_from_parent_protocol"]["original_test"],
    }
    require(sorted(name for group in split_map.values() for name in group) == sorted(SEQUENCES),
            "Amendment split membership does not partition requested sequences")

    output.mkdir(parents=True, exist_ok=True)
    eval_setup = {
        "amendment_sha256": digest(amendment_path),
        "native_reuse_amendment_sha256": digest(native_amendment_path),
        "attempt_setup_sha256": digest(setup_path),
        "parent_core_sha256": core_hash,
        "evaluation_script_sha256": digest(Path(__file__)),
        "runner_dependency_hashes": {
            relative: digest(ROOT / relative) for relative in (
                "scripts/run_memsam_residual_writer.py",
                "scripts/run_complementary_memory_pilot.py",
                "scripts/run_author_baseline.py",
                "scripts/memory_token_io.py",
                "modeling/complementary_memory/integration.py",
                "modeling/memsam_memory_writer.py",
            )
        },
        "sam2_checkpoint_sha256": core["checkpoint_sha256"],
        "sam2_revision": revision,
        "input_hashes": current_inputs,
        "post_prompt_ground_truth_presence": gt_presence,
        "sequences": SEQUENCES,
        "conditions": CONDITIONS,
        "selected_checkpoints": checkpoint_info,
        "native_baseline_provenance": native_provenance,
        "split_membership": split_map,
        "interpretation": amendment["interpretation_limit"],
    }
    eval_setup_hash = canonical_hash(eval_setup)
    receipt_path = output / "setup.json"
    if args.resume:
        require(receipt_path.is_file(), "Cannot resume without the exploratory setup receipt")
        previous = json.loads(receipt_path.read_text())
        require(previous["eval_setup_hash"] == eval_setup_hash,
                "Resume setup differs from the frozen exploratory evaluation")
    else:
        existing = {path.name for path in output.iterdir()}
        require(existing <= {"protocol.json", "native_reuse_amendment.json", "run_status.json"},
                "Existing exploratory outputs require --resume")
        atomic_json(receipt_path, {"eval_setup": eval_setup, "eval_setup_hash": eval_setup_hash,
                                   "created_at_utc": datetime.now(timezone.utc).isoformat()})

    run_status = output / "run_status.json"
    summaries = {condition: {} for condition in CONDITIONS}
    summaries["native_sam2"] = native_summaries
    atomic_json(output / "native_baseline_reused.json", {
        "source": native_provenance,
        "per_sequence": native_summaries,
        "native_inference_repeated": False,
    })
    for condition in FRESH_CONDITIONS:
        for index, name in enumerate(SEQUENCES, 1):
            atomic_json(run_status, {
                "status": "running", "condition": condition, "sequence": name,
                "sequence_index": index, "sequence_count": len(SEQUENCES),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "eval_setup_hash": eval_setup_hash,
            })
            model = study.init_model(checkpoint, repo)
            writer = writers[condition]
            mode = "generic" if condition == "mask_neutral_writer" else "regional"
            summaries[condition][name] = study.evaluate_sequence(
                model, videos[name], output / condition, name, writer, mode,
                setup_hash=eval_setup_hash,
            )
            del model
            import gc
            gc.collect()
            atomic_json(output / "partial_results.json", {
                "scope": amendment["interpretation_limit"],
                "completed_sequences_by_condition": {
                    arm: sorted(rows) for arm, rows in summaries.items()
                },
                "results": summaries,
            })

    result = study.aggregate(summaries, SEQUENCES, seed)
    result["contrasts"] = paired_contrasts(summaries, SEQUENCES, seed)
    result.update({
        "scope": amendment["interpretation_limit"],
        "sequence_ids": SEQUENCES,
        "sequence_count": len(SEQUENCES),
        "split_stratified_descriptives": split_descriptives(summaries, SEQUENCES, split_map),
        "checkpoint_info": checkpoint_info,
        "native_baseline_provenance": native_provenance,
        "native_inference_repeated": False,
        "eval_setup_hash": eval_setup_hash,
        "no_checkpoint_selection_or_tuning_on_this_cohort": True,
    })
    atomic_json(output / "results.json", result)
    atomic_json(output / "completion.json", {
        "complete": True,
        "results_sha256": digest(output / "results.json"),
        "eval_setup_hash": eval_setup_hash,
        "sequence_count": len(SEQUENCES),
        "conditions": CONDITIONS,
        "fresh_inference_conditions": FRESH_CONDITIONS,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": amendment["interpretation_limit"],
    })
    atomic_json(run_status, {"status": "complete", "sequence_count": len(SEQUENCES),
                             "conditions": CONDITIONS,
                             "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                             "eval_setup_hash": eval_setup_hash})
    print(json.dumps({"status": "complete", "output": str(output),
                      "sequence_count": len(SEQUENCES), "eval_setup_hash": eval_setup_hash}, indent=2))


if __name__ == "__main__":
    main()
