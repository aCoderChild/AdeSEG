#!/usr/bin/env python3
"""Audit spatial region assignments of native memory tokens; no training."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import (
    BOUNDARY_IOU_REVISION,
    EFFICIENT_SAM2_REVISION,
    REGION_ASSIGNMENT,
    REGION_NAMES,
    SAM2_REVISION,
    group_memory_vectors,
    reference_token_regions,
)
from scripts.memory_token_io import (
    digest,
    validate_output,
    extract_memory_features,
    parameter_digest,
    save_feature_overlay,
    prompt_index,
)
from training.tracking import start_wandb, log_results_artifact

AUTHOR_REVISIONS = {
    "boundary_iou": BOUNDARY_IOU_REVISION,
    "sam2": SAM2_REVISION,
    "efficient_sam2": EFFICIENT_SAM2_REVISION,
}


def audit_full_frame(frame, memory, memory_row, predicted, folder, ratios):
    """Same author geometry/grouping as the smoke; save vectors only once per sequence."""
    truth = load_label_mask(frame.mask_path, [1]) == 1
    with Image.open(frame.image_path) as image:
        if image.size != (truth.shape[1], truth.shape[0]):
            raise ValueError("Image and GT coordinates disagree")
    if predicted is not None and predicted.shape != truth.shape:
        raise ValueError("Native prediction and GT coordinates disagree")
    grid = (64, 64)
    if memory is not None and tuple(memory.shape) != (1, 64, *grid):
        raise ValueError("Unexpected registered B+ memory shape")
    if predicted is not None:
        Image.fromarray(predicted.astype(np.uint8) * 255).save(
            folder / f"frame_{frame.frame_index:06d}_predicted_mask.png"
        )
    rows = []
    for ratio in ratios:
        assignments = {}
        sources = [("ground_truth", truth)]
        if predicted is not None:
            sources.append(("predicted", predicted))
        for source, mask in sources:
            assignment = reference_token_regions(mask, grid, ratio)
            labels = torch.from_numpy(assignment["labels"])
            indices = {
                name: torch.where(labels.flatten() == i)[0]
                for i, name in enumerate(REGION_NAMES)
            }
            if memory is not None:
                groups = group_memory_vectors(memory, labels)
                reconstructed = torch.empty_like(memory[0].flatten(1).T)
                for group in groups.values():
                    reconstructed[group["indices"]] = group["vectors"]
                if not torch.equal(reconstructed, memory[0].flatten(1).T):
                    raise RuntimeError("Grouped vectors fail exact reconstruction")
            if not torch.equal(
                torch.cat(list(indices.values())).sort().values,
                torch.arange(labels.numel()),
            ):
                raise RuntimeError("Regions are not disjoint and exhaustive")
            stem = f"frame_{frame.frame_index:06d}_{source}_ratio_{ratio:.2f}"
            torch.save(
                {
                    **{
                        key: torch.from_numpy(value)
                        if isinstance(value, np.ndarray)
                        else value
                        for key, value in assignment.items()
                    },
                    "regions": REGION_NAMES,
                    "indices": indices,
                    "memory_available": memory is not None,
                    "vectors_file": "native_memory.pt" if memory is not None else None,
                    "memory_row": memory_row,
                    "index_order": "row-major y*grid_width+x",
                    "source": source,
                },
                folder / f"{stem}.pt",
            )
            save_feature_overlay(
                frame.image_path,
                F.one_hot(labels, 3).permute(2, 0, 1)[None, [1, 0, 2]].float(),
                folder / f"{stem}.png",
                names=("Foreground interior", "Background", "Boundary-related"),
                title=f"{frame.video_id} frame {frame.frame_index} | {source} | {ratio:.2f}",
                caption="Spatial attribution; no native memory exists for unpromptable/pre-prompt frames."
                if memory is None
                else "Spatial attribution of unchanged native memory; no classifier or intervention.",
            )
            rows.append(
                {
                    "frame": frame.frame_index,
                    "source": source,
                    "ratio": ratio,
                    "width_pixels": assignment["width_pixels"],
                    "memory_available": memory is not None,
                    "counts": {name: len(value) for name, value in indices.items()},
                    "partition_valid": True,
                    "vectors_reconstruct_exactly": True if memory is not None else None,
                }
            )
            assignments[source] = labels
        if predicted is not None:
            confusion = torch.bincount(
                3 * assignments["ground_truth"].flatten()
                + assignments["predicted"].flatten(),
                minlength=9,
            ).reshape(3, 3)
            rows.append(
                {
                    "frame": frame.frame_index,
                    "ratio": ratio,
                    "confusion": confusion.tolist(),
                }
            )
    return rows


def assignment_scores(confusion):
    matrix = torch.as_tensor(confusion).double()
    support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diag()
    union = support + predicted - tp
    return {
        "confusion": matrix.long().tolist(),
        "regions": {
            name: {
                "iou": float(tp[i] / union[i]) if union[i] else None,
                "precision": float(tp[i] / predicted[i]) if predicted[i] else None,
                "recall": float(tp[i] / support[i]) if support[i] else None,
            }
            for i, name in enumerate(REGION_NAMES)
        },
        "mean_class_iou": float((tp[union > 0] / union[union > 0]).mean())
        if (union > 0).any()
        else None,
    }


def cohort_inventory(videos):
    """No sequence silently excluded; unpromptable data receive GT-only audits."""
    if set(videos) != {f"seq{i}" for i in range(1, 24)}:
        raise ValueError(
            "Full validation requires all 23 registered PolypGen sequence IDs"
        )
    result = {}
    for name in sorted(videos, key=lambda value: int(value[3:])):
        frames = videos[name]
        if [frame.frame_index for frame in frames] != list(range(len(frames))):
            raise ValueError("Full audit requires contiguous manifest indices")
        try:
            prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        except ValueError as error:
            if "No eligible labeled prompt" not in str(error):
                raise
            prompt = None
        result[name] = {
            "dataset_frames": len(frames),
            "prompt_frame": prompt,
            "native_memory_frames": len(frames) - prompt if prompt is not None else 0,
            "gt_only_frames": prompt if prompt is not None else len(frames),
        }
    return result


def run_full_validation(args, videos):
    protocol_path = ROOT / "protocols/memory_token_regions.json"
    protocol = json.loads(protocol_path.read_text())
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Unregistered checkpoint")
    cohort = cohort_inventory(videos)
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 12 * 1024**3:
        raise RuntimeError(
            "Full audit requires at least 12 GiB of available Drive-volume space"
        )
    sources = [
        Path(__file__),
        ROOT / "scripts/memory_token_io.py",
        ROOT / "modeling/medsam2.py",
        ROOT / "modeling/region_tokens.py",
        protocol_path,
        ROOT / "external/boundary-iou-api/boundary_iou/utils/boundary_utils.py",
        ROOT / "external/boundary-iou-api/license.txt",
        ROOT / "external/sam2/sam2/sam2_video_predictor.py",
        ROOT / "external/sam2/sam2/modeling/sam2_base.py",
        ROOT / "external/Efficient-SAM2/sam2/sam2_video_predictor_with_bypass_stg.py",
    ]
    config = {
        "scope": "all-frame spatial assignment validation, not classifier or contribution validation",
        "assignment": REGION_ASSIGNMENT,
        "author_revisions": AUTHOR_REVISIONS,
        "cohort": cohort,
        "manifest_sha256": digest(args.manifest),
        "protocol": protocol,
        "checkpoint_sha256": digest(args.checkpoint),
        "device": args.device,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in sources
        },
        "conditioning_memory_audited_separately": True,
        "vectors_saved_once_per_sequence": True,
        "all_frame_source_ratio_overlays_saved": True,
        "no_model_for_unpromptable_sequences": True,
    }
    (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in sources:
        dest = output / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "03 | Memory-token region validation | PolypGen ALL FRAMES",
        "online",
        "https://forge.coreweave.com/api/wandb",
        config,
        output,
        ["memory-regions", "full-assignment-validation"],
    )
    started = time.perf_counter()
    completed, processed = {}, 0

    def progress(phase, sequence=None, frame=None):
        status = {
            "phase": phase,
            "sequence": sequence,
            "frame": frame,
            "audited_dataset_frames": processed,
            "total_dataset_frames": sum(x["dataset_frames"] for x in cohort.values()),
            "completed_sequences": len(completed),
            "total_sequences": 23,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output / "progress.json").write_text(json.dumps(status, indent=2) + "\n")
        run.log(
            {
                "progress/audited_frames": processed,
                "progress/completed_sequences": len(completed),
                "progress/elapsed_seconds": status["elapsed_seconds"],
            }
        )

    try:
        progress("building_predictor")
        predictor = build_video_predictor(
            "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
        ).eval()
        predictor.requires_grad_(False)
        before = parameter_digest(predictor)
        import wandb

        for name, inventory in cohort.items():
            folder = output / name
            folder.mkdir()
            frames = videos[name]
            prompt = inventory["prompt_frame"]
            features, records, predicted_masks = None, [], {}
            rows_by_frame = {}
            if prompt is not None:
                selected = frames[prompt:]
                features, records = extract_memory_features(
                    predictor,
                    selected,
                    videos,
                    args.device,
                    folder,
                    predicted_masks,
                    include_prompt=True,
                    progress=lambda record: progress(
                        "native_capture", name, record["frame"]
                    ),
                )
                repeat, repeat_records = extract_memory_features(
                    predictor,
                    selected,
                    videos,
                    args.device,
                    folder,
                    include_prompt=True,
                    progress=lambda record: progress(
                        "native_replay", name, record["frame"]
                    ),
                )
                if not torch.equal(features, repeat) or records != repeat_records:
                    raise RuntimeError(
                        f"Native replay differs in {name}; full validation blocked"
                    )
                del repeat
                torch.save(
                    {
                        "features": features.to(torch.bfloat16),
                        "records": records,
                        "index_order": "row-major y*grid_width+x",
                        "object_pointers_included": False,
                        "frame_indices": [frame.frame_index for frame in selected],
                    },
                    folder / "native_memory.pt",
                )
                rows_by_frame = {
                    frame.frame_index: i for i, frame in enumerate(selected)
                }
            confusions = {
                str(r): torch.zeros(3, 3, dtype=torch.long)
                for r in protocol["assignment_audit"]["sensitivity_ratios"]
            }
            counts = {
                str(r): {
                    source: {region: 0 for region in REGION_NAMES}
                    for source in ("ground_truth", "predicted")
                }
                for r in protocol["assignment_audit"]["sensitivity_ratios"]
            }
            empty_interiors = {
                str(r): 0 for r in protocol["assignment_audit"]["sensitivity_ratios"]
            }
            with (folder / "frames.jsonl").open("w") as stream:
                frame_manifest = []
                for i, frame in enumerate(frames):
                    frame_manifest.append(
                        {
                            "frame": frame.frame_index,
                            "image": str(frame.image_path),
                            "mask": str(frame.mask_path),
                            "image_sha256": digest(frame.image_path),
                            "mask_sha256": digest(frame.mask_path),
                        }
                    )
                    memory_row = rows_by_frame.get(frame.frame_index)
                    memory = (
                        features[memory_row : memory_row + 1]
                        if memory_row is not None
                        else None
                    )
                    predicted = (
                        predicted_masks[(name, frame.frame_index)].numpy()
                        if memory_row is not None
                        else None
                    )
                    rows = audit_full_frame(
                        frame,
                        memory,
                        memory_row,
                        predicted,
                        folder,
                        protocol["assignment_audit"]["sensitivity_ratios"],
                    )
                    for row in rows:
                        row["memory_stage"] = (
                            "unavailable"
                            if memory is None
                            else "conditioning"
                            if frame.frame_index == prompt
                            else "self_predicted"
                        )
                        if "counts" in row:
                            for region, count in row["counts"].items():
                                counts[str(row["ratio"])][row["source"]][region] += (
                                    count
                                )
                            if (
                                row["source"] == "ground_truth"
                                and row["counts"]["foreground_interior"] == 0
                            ):
                                empty_interiors[str(row["ratio"])] += 1
                        elif frame.frame_index != prompt:
                            confusions[str(row["ratio"])] += torch.tensor(
                                row["confusion"]
                            )
                        stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    processed += 1
                    if i % 10 == 0 or i == len(frames) - 1:
                        progress("region_assignment", name, frame.frame_index)
                        print(
                            f"[full-audit] {name} frame={frame.frame_index} dataset_frames={processed}",
                            flush=True,
                        )
                    if i in {0, len(frames) // 2, len(frames) - 1}:
                        for source in ("ground_truth", "predicted"):
                            image = (
                                folder
                                / f"frame_{frame.frame_index:06d}_{source}_ratio_0.02.png"
                            )
                            if image.exists():
                                run.log({"audit/overlay": wandb.Image(str(image))})
            result = {
                **inventory,
                "completed": True,
                "status": "native_and_reference_audit"
                if prompt is not None
                else "unpromptable_reference_only",
                "native_replay_bit_identical": True if prompt is not None else None,
                "counts": counts,
                "gt_empty_interior_frames": empty_interiors,
                "post_prompt_assignment_scores": {
                    key: assignment_scores(value) for key, value in confusions.items()
                },
            }
            (folder / "frame_manifest.json").write_text(
                json.dumps(frame_manifest, indent=2) + "\n"
            )
            (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            progress("uploading_sequence_artifact", name)
            artifact = wandb.Artifact(
                f"memory-token-regions-{run.id}-{name}", type="experiment-results"
            )
            artifact.add_dir(str(folder))
            run.log_artifact(artifact).wait()
            completed[name] = result
            (output / "sequences.json").write_text(
                json.dumps(completed, indent=2) + "\n"
            )
        if before != parameter_digest(predictor) or any(
            p.grad is not None for p in predictor.parameters()
        ):
            raise RuntimeError("Frozen backbone changed")
        summary = {
            "completed": True,
            "all_dataset_frames_audited": processed
            == sum(x["dataset_frames"] for x in cohort.values()),
            "dataset_sequences": 23,
            "dataset_frames": processed,
            "native_memory_frames": sum(
                x["native_memory_frames"] for x in cohort.values()
            ),
            "gt_only_frames": sum(x["gt_only_frames"] for x in cohort.values()),
            "unpromptable_sequences": [
                name for name, item in cohort.items() if item["prompt_frame"] is None
            ],
            "all_eligible_native_replays_exact": True,
            "backbone_frozen": True,
            "classifier_validated": False,
            "attention_contribution_measured": False,
            "sequences": completed,
        }
        summary["assignment_agreement_by_ratio"] = {}
        for ratio in protocol["assignment_audit"]["sensitivity_ratios"]:
            scores = [
                item["post_prompt_assignment_scores"][str(ratio)]
                for item in completed.values()
                if item["post_prompt_assignment_scores"][str(ratio)]["mean_class_iou"]
                is not None
            ]
            summary["assignment_agreement_by_ratio"][str(ratio)] = {
                "sequences": len(scores),
                "equal_sequence_mean_class_iou": float(
                    np.mean([score["mean_class_iou"] for score in scores])
                )
                if scores
                else None,
                "equal_sequence_region_iou": {
                    region: float(np.mean(values)) if values else None
                    for region in REGION_NAMES
                    for values in [
                        [
                            score["regions"][region]["iou"]
                            for score in scores
                            if score["regions"][region]["iou"] is not None
                        ]
                    ]
                },
                "pooled_token_scores": assignment_scores(
                    sum(
                        (torch.tensor(score["confusion"]) for score in scores),
                        torch.zeros(3, 3, dtype=torch.long),
                    )
                ),
            }
        if not summary["all_dataset_frames_audited"]:
            raise RuntimeError("Dataset coverage incomplete")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report = [
            "# Full PolypGen memory-token assignment validation",
            "",
            "All source frames audited; unpromptable/pre-prompt frames have reference labels only. No classifier or attention intervention.",
            "",
            "| Sequence | Frames | Prompt | Native memory frames | GT-only frames | Post-prompt assignment macro-IoU (0.02) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, item in completed.items():
            score = item["post_prompt_assignment_scores"]["0.02"]["mean_class_iou"]
            report.append(
                f"| {name} | {item['dataset_frames']} | {item['prompt_frame']} | {item['native_memory_frames']} | {item['gt_only_frames']} | "
                + (f"{score:.5f}" if score is not None else "N/A")
                + " |"
            )
        report.extend(
            [
                "",
                "## Equal-sequence post-prompt assignment agreement",
                "",
                "These compare GT/predicted spatial labels, not classifier predictions or segmentation gains. Conditioning frames are excluded. No independent-patient inference is claimed.",
                "",
                "| Boundary ratio | Sequences | Background IoU | Interior IoU | Boundary IoU | Macro-IoU |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for ratio, value in summary["assignment_agreement_by_ratio"].items():
            metrics = [
                value["equal_sequence_region_iou"][region] for region in REGION_NAMES
            ] + [value["equal_sequence_mean_class_iou"]]
            report.append(
                f"| {ratio} | {value['sequences']} | "
                + " | ".join(
                    f"{metric:.5f}" if metric is not None else "N/A"
                    for metric in metrics
                )
                + " |"
            )
        (output / "results.md").write_text("\n".join(report) + "\n")
        run.summary.update(
            {key: value for key, value in summary.items() if key != "sequences"}
        )
        progress("uploading_final_artifact")
        root_files = [
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
            and "wandb" not in path.parts
            and path.relative_to(output).parts[0] not in cohort
        ]
        log_results_artifact(run, output, root_files)
        (output / "delivery.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "wandb_artifact_upload_complete": True,
                    "run_id": run.id,
                },
                indent=2,
            )
            + "\n"
        )
        run.summary["wandb_artifact_upload_complete"] = True
        progress("completed")
    except Exception as error:
        if (output / "summary.json").exists():
            failed = json.loads((output / "summary.json").read_text())
            failed["completed"] = False
            (output / "summary.json").write_text(json.dumps(failed, indent=2) + "\n")
        (output / "error.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "error": str(error),
                    "completed_sequences": list(completed),
                },
                indent=2,
            )
            + "\n"
        )
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq2")
    parser.add_argument("--frame", type=int, action="append")
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Audit every frame of all 23 sequences; no sampling",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    videos = {
        **load_manifest(args.manifest, "dev"),
        **load_manifest(args.manifest, "test"),
    }
    if args.all_sequences:
        if args.frame:
            parser.error("Full validation cannot be combined with frame sampling")
        return run_full_validation(args, videos)
    if not args.frame:
        parser.error("Supply --frame or --all-sequences")
    sequence = videos[args.sequence]
    if (
        len(set(args.frame)) != len(args.frame)
        or min(args.frame) < 0
        or max(args.frame) >= len(sequence)
    ):
        parser.error("Require unique valid frame indices")
    frames = [sequence[i] for i in sorted(args.frame)]
    expected = json.loads((ROOT / "protocols/memory_token_regions.json").read_text())[
        "model"
    ]["checkpoint_sha256"]
    if digest(args.checkpoint) != expected:
        raise ValueError("Unregistered checkpoint")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "sequence": args.sequence,
        "frames": args.frame,
        "device": args.device,
        "boundary_repository": "https://github.com/bowenc0221/boundary-iou-api",
        "boundary_revision": BOUNDARY_IOU_REVISION,
        "primary_ratio": 0.02,
        "sensitivity_ratios": [0.01, 0.02, 0.04],
        "assignment": REGION_ASSIGNMENT,
        "author_revisions": AUTHOR_REVISIONS,
        "regions": REGION_NAMES,
        "gt_used_only_for_prompt_and_external_labels": True,
        "native_modules_changed": False,
        "classifier_training": False,
        "checkpoint_sha256": expected,
        "manifest_sha256": digest(args.manifest),
        "frames_metadata": [
            {
                "frame": f.frame_index,
                "image": str(f.image_path),
                "mask": str(f.mask_path),
                "image_sha256": digest(f.image_path),
                "mask_sha256": digest(f.mask_path),
            }
            for f in frames
        ],
    }
    (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
    for relative in (
        "scripts/audit_memory_token_regions.py",
        "scripts/memory_token_io.py",
        "protocols/memory_token_regions.json",
        "modeling/region_tokens.py",
        "external/boundary-iou-api/boundary_iou/utils/boundary_utils.py",
        "external/boundary-iou-api/license.txt",
    ):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"Memory region assignment audit | {args.sequence}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        config,
        output,
        ["memory-regions", "assignment-audit"],
    )
    try:
        predictor = build_video_predictor(
            "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
        ).eval()
        predictor.requires_grad_(False)
        before = parameter_digest(predictor)
        predicted_masks = {}
        features, records = extract_memory_features(
            predictor, frames, videos, args.device, output, predicted_masks
        )
        repeat, repeat_records = extract_memory_features(
            predictor, frames, videos, args.device, output
        )
        if not torch.equal(features, repeat) or records != repeat_records:
            raise RuntimeError(
                "Native feature/prediction replay differs; assignment audit blocked"
            )
        torch.save(
            {
                "features": features,
                "records": records,
                "predicted_masks": [
                    predicted_masks[(f.video_id, f.frame_index)] for f in frames
                ],
                "feature_space": "stored_maskmem_features",
                "object_pointers_included": False,
            },
            output / "native_memory.pt",
        )
        rows = []
        for i, frame in enumerate(frames):
            truth = load_label_mask(frame.mask_path, [1]) == 1
            predicted = predicted_masks[(frame.video_id, frame.frame_index)].numpy()
            with Image.open(frame.image_path) as image:
                if (
                    image.size != (truth.shape[1], truth.shape[0])
                    or predicted.shape != truth.shape
                ):
                    raise ValueError(
                        "Image, GT and native prediction coordinates disagree"
                    )
            for ratio in config["sensitivity_ratios"]:
                assignments = {}
                for source, mask in (("ground_truth", truth), ("predicted", predicted)):
                    assignment = reference_token_regions(
                        mask, tuple(features.shape[-2:]), ratio
                    )
                    labels = torch.from_numpy(assignment["labels"])
                    groups = group_memory_vectors(features[i : i + 1], labels)
                    reconstructed = torch.empty_like(features[i].flatten(1).T)
                    for group in groups.values():
                        reconstructed[group["indices"]] = group["vectors"]
                    if not torch.equal(reconstructed, features[i].flatten(1).T):
                        raise RuntimeError(
                            "Grouped vectors do not reconstruct native spatial memory"
                        )
                    stem = f"frame_{frame.frame_index:06d}_{source}_ratio_{ratio:.2f}"
                    torch.save(
                        {
                            **{
                                key: torch.from_numpy(value)
                                if isinstance(value, np.ndarray)
                                else value
                                for key, value in assignment.items()
                            },
                            "labels": labels,
                            "groups": groups,
                            "regions": REGION_NAMES,
                            "index_order": "row-major y*grid_width+x",
                            "label_source": source,
                        },
                        output / f"{stem}.pt",
                    )
                    destination = output / f"{stem}.png"
                    save_feature_overlay(
                        frame.image_path,
                        F.one_hot(labels, 3).permute(2, 0, 1)[None, [1, 0, 2]].float(),
                        destination,
                        names=("Foreground interior", "Background", "Boundary-related"),
                        title=f"{source} token regions | ratio {ratio:.2f}",
                        caption="Spatial attribution of unchanged memory vectors; not learned semantic classes.",
                    )
                    import wandb

                    run.log({"audit/overlay": wandb.Image(str(destination))})
                    assignments[source] = labels
                    rows.append(
                        {
                            "frame": frame.frame_index,
                            "source": source,
                            "ratio": ratio,
                            "width_pixels": assignment["width_pixels"],
                            "tokens": labels.numel(),
                            "counts": {
                                name: len(group["indices"])
                                for name, group in groups.items()
                            },
                            "partition_valid": True,
                            "vectors_reconstruct_exactly": True,
                        }
                    )
                confusion = torch.bincount(
                    3 * assignments["ground_truth"].flatten()
                    + assignments["predicted"].flatten(),
                    minlength=9,
                ).reshape(3, 3)
                rows.append(
                    {
                        "frame": frame.frame_index,
                        "ratio": ratio,
                        "gt_rows_predicted_columns": confusion.tolist(),
                        "assignment_agreement": float(
                            confusion.diag().sum() / confusion.sum()
                        ),
                    }
                )
        if before != parameter_digest(predictor) or any(
            p.grad is not None for p in predictor.parameters()
        ):
            raise RuntimeError("Frozen backbone changed")
        summary = {
            "completed": True,
            "frames": len(frames),
            "native_replay_bit_identical": True,
            "backbone_frozen": True,
            "training": False,
            "contribution_measured": False,
            "rows": rows,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report = [
            "# Native memory region assignment audit",
            "",
            "No classifier training or attention intervention; spatial labels are not semantic-purity claims.",
            "",
            "| Frame | Source | Ratio | Width (px) | Background | Foreground interior | Boundary |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            if "counts" in row:
                counts = row["counts"]
                report.append(
                    f"| {row['frame']} | {row['source']} | {row['ratio']:.2f} | {row['width_pixels']} | "
                    + " | ".join(str(counts[name]) for name in REGION_NAMES)
                    + " |"
                )
        (output / "results.md").write_text("\n".join(report) + "\n")
        run.summary.update({k: v for k, v in summary.items() if k != "rows"})
        log_results_artifact(
            run,
            output,
            [
                str(p.relative_to(output))
                for p in output.rglob("*")
                if p.is_file() and "wandb" not in p.parts
            ],
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    except Exception as error:
        (output / "error.json").write_text(
            json.dumps({"completed": False, "error": str(error)}) + "\n"
        )
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
