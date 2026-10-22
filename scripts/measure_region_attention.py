#!/usr/bin/env python3
"""Verify saved overlays and measure native regional attention; no intervention.

Uses existing native predictor/Boundary IoU grouping and Efficient-SAM2 author
attention. Project glue validates alignment, aggregates weights and saves logs.
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.statistics import sequence_bootstrap_interval
from modeling.medsam2 import build_video_predictor
from modeling.region_attention import EFFICIENT_SAM2_REVISION, RegionAttentionObserver
from modeling.region_tokens import REGION_NAMES, reference_token_regions
from scripts.memory_token_io import (
    digest,
    parameter_digest,
    synchronize_storage,
    validate_output,
)
from training.tracking import finalize_tracking, log_results_artifact, start_wandb


def verify_overlays(audit, output):
    """Check every saved region overlay against its labels and original image."""
    summary = json.loads((audit / "summary.json").read_text())
    delivery = json.loads((audit / "delivery.json").read_text())
    if not summary["completed"] or not delivery["wandb_artifact_upload_complete"]:
        raise ValueError("Assignment audit is incomplete")
    if summary["dataset_sequences"] != 23:
        raise ValueError("Expected all 23 sequences")
    rows, gallery = [], []
    palette = np.array([[30, 140, 255], [0, 200, 100], [255, 180, 0]], np.uint8)
    for number in range(1, 24):
        name = f"seq{number}"
        folder = audit / name
        metadata = json.loads((folder / "frame_manifest.json").read_text())
        sequence = json.loads((folder / "summary.json").read_text())
        prompt = sequence["prompt_frame"]
        available = [
            m["frame"] for m in metadata if prompt is not None and m["frame"] > prompt
        ]
        source = "predicted" if available else "ground_truth"
        indices = available or [m["frame"] for m in metadata]
        selected = {indices[0], indices[len(indices) // 2], indices[-1]}
        panels = []
        checked = 0
        for frame in metadata:
            if (
                digest(frame["image"]) != frame["image_sha256"]
                or digest(frame["mask"]) != frame["mask_sha256"]
            ):
                raise ValueError(f"Dataset changed: {name} frame {frame['frame']}")
            with Image.open(frame["image"]) as image:
                original = image.convert("RGB")
                original.thumbnail((640, 480))
            sources = ["ground_truth"]
            if prompt is not None and frame["frame"] >= prompt:
                sources.append("predicted")
            for label_source in sources:
                for ratio in (0.01, 0.02, 0.04):
                    stem = (
                        f"frame_{frame['frame']:06d}_{label_source}_ratio_{ratio:.2f}"
                    )
                    assignment = torch.load(folder / f"{stem}.pt", weights_only=True)
                    labels = assignment["labels"].numpy()
                    if labels.shape != (64, 64) or not np.isin(labels, [0, 1, 2]).all():
                        raise ValueError(f"Invalid labels: {name}/{stem}")
                    if tuple(assignment["regions"]) != REGION_NAMES:
                        raise ValueError("Unexpected region classes")
                    for index, region in enumerate(REGION_NAMES):
                        if not np.array_equal(
                            assignment["indices"][region].numpy(),
                            np.flatnonzero(labels == index),
                        ):
                            raise ValueError(f"Incorrect token indices: {name}/{stem}")
                    with Image.open(folder / f"{stem}.png") as image:
                        image.load()
                        actual = image.convert("RGB")
                    colors = Image.fromarray(palette[labels]).resize(
                        original.size, Image.Resampling.NEAREST
                    )
                    expected = Image.blend(original, colors, 0.45)
                    panel = actual.crop(
                        (original.width, 30, 2 * original.width, 30 + original.height)
                    )
                    if not np.array_equal(np.asarray(panel), np.asarray(expected)):
                        raise ValueError(
                            f"Overlay pixels disagree with token labels: {name}/{stem}"
                        )
                    checked += 1
                    if (
                        label_source == source
                        and ratio == 0.02
                        and frame["frame"] in selected
                    ):
                        actual.thumbnail((640, 300))
                        panels.append(actual.copy())
            # Per-frame progress prevents a silent long verification.
        sheet = Image.new("RGB", (640, sum(p.height for p in panels) + 30), "white")
        ImageDraw.Draw(sheet).text(
            (8, 8), f"{name}: {source}; first/middle/last eligible frame", fill="black"
        )
        y = 30
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(output / f"{name}_region_overlays.png")
        tile = panels[len(panels) // 2].copy()
        tile.thumbnail((340, 180))
        gallery.append((name, source, tile))
        rows.append(
            {
                "sequence": name,
                "frames": len(metadata),
                "overlays_verified": checked,
                "illustration_source": source,
                "illustration_frames": sorted(selected),
            }
        )
        print(f"[overlays] {name}: {checked} label/overlay pairs verified", flush=True)
    atlas = Image.new("RGB", (1360, 6 * 205), "white")
    draw = ImageDraw.Draw(atlas)
    for i, (name, source, tile) in enumerate(gallery):
        x, y = (i % 4) * 340, (i // 4) * 205
        draw.text((x + 4, y + 4), f"{name}: {source}", fill="black")
        atlas.paste(tile, (x, y + 22))
    atlas.save(output / "all_23_sequences_regions.png")
    report = {
        "completed": True,
        "sequences": rows,
        "frames": sum(r["frames"] for r in rows),
        "overlays_verified": sum(r["overlays_verified"] for r in rows),
        "pixel_exact": True,
        "no_new_model_run": True,
    }
    (output / "overlay_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def sequence_report(rows):
    """Average within sequence first; frames are not independent replicates."""
    report = {}
    for layer in sorted({row["layer"] for row in rows}):
        selected = [r for r in rows if r["layer"] == layer]
        regions = {}
        for region in REGION_NAMES:
            values = [r["regions"][region] for r in selected]
            regions[region] = {}
            for metric in (
                "mass",
                "mass_per_token",
                "enrichment",
                "tokens",
                "spatial_token_fraction",
                "spatial_attention_share",
            ):
                valid = [v[metric] for v in values if v[metric] is not None]
                regions[region][metric] = float(np.mean(valid)) if valid else None
            regions[region]["frames_with_region"] = sum(v["tokens"] > 0 for v in values)
            regions[region]["empty_frames"] = sum(v["tokens"] == 0 for v in values)
            regions[region]["empty_frame_fraction"] = sum(
                v["tokens"] == 0 for v in values
            ) / len(values)
            enrichments = [
                v["enrichment"] for v in values if v["enrichment"] is not None
            ]
            regions[region]["median_enrichment"] = (
                float(np.median(enrichments)) if enrichments else None
            )
            fraction = regions[region]["spatial_token_fraction"]
            share = regions[region]["spatial_attention_share"]
            regions[region]["ratio_of_mean_shares"] = (
                share / fraction if fraction and share is not None else None
            )
        report[str(layer)] = {
            "frames": len(selected),
            "regions": regions,
            "pointer_mass": float(np.mean([r["pointer_mass"] for r in selected])),
        }
    return report


def write_results(output, results):
    """Reuse sequence bootstrap statistics; never bootstrap individual frames."""
    valid = [r for r in results if r["status"] == "completed"]
    lines = [
        "# Native regional memory attention",
        "",
        f"Completed promptable sequences: {len(valid)}. Measurement only; no causal segmentation gain is claimed.",
        "All spatial query positions and heads are averaged; pointer mass stays separate.",
        "Enrichment is region spatial-attention share / spatial-token share. Empty regions are undefined, not zero.",
        "Intervals resample whole sequence means (10,000 resamples); patient independence is not established.",
        "",
        "| Layer | Region | Sequences | Mean native mass | Mean enrichment | Enrichment 95% sequence-bootstrap CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    aggregates = []
    for layer in sorted({key for r in valid for key in r["layers"]}):
        for region in REGION_NAMES:
            mass = {
                r["sequence"]: r["layers"][layer]["regions"][region]["mass"]
                for r in valid
                if layer in r["layers"]
            }
            enrichment = {
                r["sequence"]: r["layers"][layer]["regions"][region]["enrichment"]
                for r in valid
                if layer in r["layers"]
                and r["layers"][layer]["regions"][region]["enrichment"] is not None
            }
            mean = float(np.mean(list(enrichment.values()))) if enrichment else None
            ci = (
                sequence_bootstrap_interval(enrichment)
                if len(enrichment) >= 2
                else None
            )
            mean_mass = float(np.mean(list(mass.values())))
            lines.append(
                f"| {layer} | {region} | {len(enrichment)} | {mean_mass:.6f} | {mean if mean is not None else 'N/A'} | {ci if ci else 'N/A'} |"
            )
            aggregates.append(
                {
                    "layer": int(layer),
                    "region": region,
                    "sequences": len(enrichment),
                    "mean_native_mass": mean_mass,
                    "mean_enrichment": mean,
                    "enrichment_ci": ci,
                }
            )
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return aggregates


def load_label_schemes(folder, frame, index):
    """Validate saved assignments against the unchanged Boundary IoU adapter.

    GT supplies external labels only. Neither label source enters the model.
    """
    with Image.open(folder / f"frame_{index:06d}_predicted_mask.png") as image:
        predicted = np.asarray(image) > 0
    truth = load_label_mask(frame.mask_path, [1]) == 1
    result = {}
    for source, mask in (("predicted", predicted), ("ground_truth", truth)):
        for ratio in (0.01, 0.02, 0.04):
            saved = torch.load(
                folder / f"frame_{index:06d}_{source}_ratio_{ratio:.2f}.pt",
                weights_only=True,
            )
            computed = reference_token_regions(mask, (64, 64), ratio)
            if not torch.equal(saved["labels"], torch.from_numpy(computed["labels"])):
                raise RuntimeError(
                    f"Saved region assignment changed: {source}/{ratio}/frame{index}"
                )
            result[f"{source}_{ratio:.2f}"] = saved["labels"]
    return result


def check_primary_agreement(current, previous):
    """The original predicted/0.02 measurement must reproduce, not just logits."""
    for name in REGION_NAMES:
        new, old = current["regions"][name], previous["regions"][name]
        if new["tokens"] != old["tokens"] or not np.isclose(
            new["mass"], old["mass"], atol=1e-6, rtol=1e-5
        ):
            raise RuntimeError(f"Previous primary attention did not reproduce: {name}")
    if not np.isclose(
        current["pointer_mass"], previous["pointer_mass"], atol=1e-6, rtol=1e-5
    ):
        raise RuntimeError("Previous pointer attention did not reproduce")


def write_sensitivity_results(output, results):
    """Descriptive robustness tables, with token support beside enrichment."""
    valid = [r for r in results if r["status"] == "completed"]
    schemes = sorted({scheme for r in valid for scheme in r["sensitivity"]})
    lines = [
        "# Regional attention: boundary-width and label-source sensitivity",
        "",
        f"Completed promptable sequences: {len(valid)}. GT is offline diagnostic labeling only.",
        "All six schemes label the same native attention weights. No attention intervention or segmentation gain is tested.",
        "Mass includes the native pointer denominator; enrichment conditions on spatial keys.",
        "Means weight sequences equally; 95% CIs bootstrap sequence means (10,000 resamples).",
        "Token counts refer to the entire active spatial-memory bank per target, not one memory frame.",
        "Enrichment excludes empty-region frames; mean mass/counts and empty frequencies include them.",
        "Patient independence is not established. Native pointer mass is unchanged across schemes.",
        "",
    ]
    details = [
        "# Sequence-wise sensitivity",
        "",
        "Mean tokens/mass include empty frames. Enrichment is conditional on nonempty regions.",
        "",
    ]
    aggregates = []
    for scheme in schemes:
        lines += [
            f"## {scheme}",
            "",
            "| Layer | Region | Sequences (all / valid enrichment) | Mean tokens | Token fraction | Native mass | Enrichment [95% CI] | Empty-frame fraction |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
        for layer in ("0", "1", "2", "3"):
            for region in REGION_NAMES:
                by_sequence = {
                    r["sequence"]: r["sensitivity"][scheme][layer]["regions"][region]
                    for r in valid
                }
                enrichment = {
                    name: value["enrichment"]
                    for name, value in by_sequence.items()
                    if value["enrichment"] is not None
                }

                def mean(key):
                    return float(
                        np.mean([value[key] for value in by_sequence.values()])
                    )

                value = (
                    float(np.mean(list(enrichment.values()))) if enrichment else None
                )
                ci = (
                    sequence_bootstrap_interval(enrichment)
                    if len(enrichment) >= 2
                    else None
                )
                row = {
                    "scheme": scheme,
                    "layer": int(layer),
                    "region": region,
                    "sequences": len(by_sequence),
                    "valid_enrichment_sequences": len(enrichment),
                    "mean_tokens": mean("tokens"),
                    "mean_spatial_token_fraction": mean("spatial_token_fraction"),
                    "mean_native_mass": mean("mass"),
                    "mean_enrichment": value,
                    "enrichment_ci": ci,
                    "mean_empty_frame_fraction": mean("empty_frame_fraction"),
                    "mean_ratio_of_mean_shares": mean("ratio_of_mean_shares")
                    if all(
                        v["ratio_of_mean_shares"] is not None
                        for v in by_sequence.values()
                    )
                    else None,
                }
                aggregates.append(row)
                formatted = f"{value:.3f}" if value is not None else "N/A"
                if ci:
                    formatted += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
                lines.append(
                    f"| {layer} | {region} | {len(by_sequence)} / {len(enrichment)} | {row['mean_tokens']:.1f} | {row['mean_spatial_token_fraction']:.5f} | {row['mean_native_mass']:.6f} | {formatted} | {row['mean_empty_frame_fraction']:.4f} |"
                )
        lines.append("")
    for result in sorted(valid, key=lambda r: int(r["sequence"][3:])):
        details += [
            f"## {result['sequence']} ({result['attention_frames']} attention frames)",
            "",
            "| Label scheme | Layer | Region | Mean tokens | Token fraction | Native mass | Mean enrichment | Median enrichment | Ratio of mean shares | Empty frames | Pointer mass |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        def fmt(value):
            return "N/A" if value is None else f"{value:.5f}"

        for scheme in schemes:
            for layer in ("0", "1", "2", "3"):
                summary = result["sensitivity"][scheme][layer]
                for region in REGION_NAMES:
                    r = summary["regions"][region]
                    details.append(
                        f"| {scheme} | {layer} | {region} | {r['tokens']:.1f} | {fmt(r['spatial_token_fraction'])} | {fmt(r['mass'])} | {fmt(r['enrichment'])} | {fmt(r['median_enrichment'])} | {fmt(r['ratio_of_mean_shares'])} | {r['empty_frames']}/{summary['frames']} | {fmt(summary['pointer_mass'])} |"
                    )
        details.append("")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    (output / "sequence_results.md").write_text("\n".join(details) + "\n")
    return aggregates


@torch.no_grad()
def measure_sequence(
    predictor,
    name,
    frames,
    audit,
    output,
    device,
    chunk_size,
    run,
    sensitivity=False,
    previous_run=None,
):
    folder = audit / name
    reference = json.loads((folder / "summary.json").read_text())
    metadata = json.loads((folder / "frame_manifest.json").read_text())
    if [(f.frame_index, str(f.image_path), str(f.mask_path)) for f in frames] != [
        (f["frame"], f["image"], f["mask"]) for f in metadata
    ]:
        raise ValueError(f"Manifest differs from verified chronological inputs: {name}")
    prompt = reference["prompt_frame"]
    if prompt is None:
        return {"sequence": name, "status": "unpromptable", "attention_frames": 0}
    native = torch.load(folder / "native_memory.pt", weights_only=True)
    hashes = {r["frame"]: r["prediction_sha256"] for r in native["records"]}
    del native
    labels = {}
    previous = {}
    if previous_run:
        with (previous_run / name / "attention.jsonl").open() as stream:
            previous = {(r["frame"], r["layer"]): r for r in map(json.loads, stream)}

    def label_loader(index):
        if index not in labels:
            if sensitivity:
                labels[index] = load_label_schemes(folder, frames[index], index)
            else:
                labels[index] = torch.load(
                    folder / f"frame_{index:06d}_predicted_ratio_0.02.pt",
                    weights_only=True,
                )["labels"]
        return labels[index]

    target = output / name
    target.mkdir()
    all_rows = []
    seen = set()
    with tempfile.TemporaryDirectory(prefix="frames_", dir=target) as temporary:
        stage_frames(frames, Path(temporary))
        state = predictor.init_state(
            video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False
        )
        try:
            initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
            predictor.add_new_mask(
                state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial)
            )
            with RegionAttentionObserver(
                predictor.memory_attention, label_loader, chunk_size
            ) as observer:
                for index, _, logits in predictor.propagate_in_video(
                    state, start_frame_idx=prompt
                ):
                    synchronize_storage(device)
                    if index in seen:
                        raise RuntimeError("Duplicate native target frame")
                    seen.add(index)
                    actual = hashlib.sha256(
                        logits.detach().float().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    if actual != hashes[index]:
                        raise RuntimeError(
                            f"Native prediction changed: {name} frame {index}; attention results invalid"
                        )
                    if index != prompt:
                        if len(observer.rows) != len(predictor.memory_attention.layers):
                            raise RuntimeError("Missing cross-attention layers")
                        with (target / "attention.jsonl").open("a") as stream:
                            for row in observer.rows:
                                if previous_run:
                                    check_primary_agreement(
                                        row, previous[(index, row["layer"])]
                                    )
                                    row["previous_primary_agrees"] = True
                                row.update(
                                    sequence=name,
                                    frame=index,
                                    baseline_prediction_bit_identical=True,
                                )
                                stream.write(json.dumps(row) + "\n")
                                all_rows.append(row)
                    observer.rows.clear()
                    progress = {
                        "sequence": name,
                        "frame": index,
                        "total_frames": len(frames),
                        "rows": len(all_rows),
                        "state": "running",
                    }
                    (output / "progress.json").write_text(json.dumps(progress) + "\n")
                    run.log(
                        {
                            "progress/frame": index,
                            "progress/sequence": int(name[3:]),
                            "progress/attention_rows": len(all_rows),
                        }
                    )
                    print(
                        f"[attention] {name} frame={index}/{len(frames) - 1} rows={len(all_rows)} native prediction exact",
                        flush=True,
                    )
        finally:
            predictor.reset_state(state)
    if seen != set(range(prompt, len(frames))):
        raise RuntimeError("Incomplete native propagation coverage")
    result = {
        "sequence": name,
        "status": "completed",
        "attention_frames": len(frames) - prompt - 1,
        "native_predictions_bit_identical": True,
        "maximum_reconstruction_error": max(
            (r["reconstruction_max_abs"] for r in all_rows), default=0
        ),
        "layers": sequence_report(all_rows),
        **(
            {
                "sensitivity": {
                    scheme: sequence_report(
                        [
                            {"layer": row["layer"], **row["label_schemes"][scheme]}
                            for row in all_rows
                        ]
                    )
                    for scheme in all_rows[0]["label_schemes"]
                },
                "previous_primary_agrees": True,
            }
            if sensitivity
            else {}
        ),
    }
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    log_results_artifact(
        run, output, [f"{name}/attention.jsonl", f"{name}/summary.json"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument(
        "--sequence",
        action="append",
        help="Default: all 23, seq3 first as a full-sequence correctness gate",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--verify-overlays-only", action="store_true")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Same attention weights, both label sources and all three widths",
    )
    parser.add_argument(
        "--previous-run", type=Path, help="Completed primary attention run to reproduce"
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    if not args.verify_overlays_only and args.checkpoint is None:
        parser.error("Attention measurement requires --checkpoint")
    if args.sensitivity and (not args.previous_run or args.verify_overlays_only):
        parser.error("Sensitivity requires --previous-run and model measurement")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # SDK caches and staging also stay on the explicitly requested Drive path.
    import os

    for key, value in {
        "WANDB_CACHE_DIR": output / "wandb_cache",
        "WANDB_DATA_DIR": output / "wandb_data",
    }.items():
        os.environ[key] = str(value)
        value.mkdir()
    config = {
        **vars(args),
        "efficient_sam2_revision": EFFICIENT_SAM2_REVISION,
        "query_population": "all current-frame spatial queries, averaged over heads",
        "primary_labels": "stored memory-frame predicted-mask assignment at ratio 0.02",
        "metric": "native mass; enrichment conditional on spatial keys",
        "causal_contribution_measured": False,
        "started_at_unix": time.time(),
        "label_schemes": [
            f"{source}_{ratio:.2f}"
            for source in ("predicted", "ground_truth")
            for ratio in (0.01, 0.02, 0.04)
        ]
        if args.sensitivity
        else ["predicted_0.02"],
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
    for relative in (
        "modeling/region_attention.py",
        "modeling/region_tokens.py",
        "scripts/measure_region_attention.py",
        "scripts/memory_token_io.py",
        "protocols/memory_token_regions.json",
        "training/tracking.py",
        "external/Efficient-SAM2/sam2/modeling/sam/transformer.py",
        "external/Efficient-SAM2/LICENSE",
    ):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "05 | Region attention robustness | prediction vs GT; 3 widths"
        if args.sensitivity
        else "04 | Regional attention and overlays | PolypGen",
        "online",
        "https://forge.coreweave.com/api/wandb",
        config,
        output,
        ["native-memory", "regional-attention", "measurement-only"],
    )
    try:
        if args.previous_run:
            prior = json.loads((args.previous_run / "summary.json").read_text())
            if (
                not prior["completed"]
                or sum(r["status"] == "completed" for r in prior["sequences"]) != 21
            ):
                raise ValueError(
                    "Previous all-sequence primary measurements are incomplete"
                )
        overlay = verify_overlays(args.audit_dir, output)
        log_results_artifact(
            run,
            output,
            ["overlay_verification.json", "all_23_sequences_regions.png"]
            + [f"seq{i}_region_overlays.png" for i in range(1, 24)],
        )
        results = []
        if not args.verify_overlays_only:
            protocol = json.loads(
                (ROOT / "protocols/memory_token_regions.json").read_text()
            )
            if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
                raise ValueError("Unregistered checkpoint")
            videos = {
                **load_manifest(args.manifest, "dev"),
                **load_manifest(args.manifest, "test"),
            }
            if set(videos) != {f"seq{i}" for i in range(1, 24)}:
                raise ValueError("Manifest must contain all 23 sequences")
            names = args.sequence or ["seq3"] + [
                f"seq{i}" for i in range(1, 24) if i != 3
            ]
            if len(set(names)) != len(names) or any(n not in videos for n in names):
                raise ValueError("Invalid or duplicate sequence selection")
            predictor = build_video_predictor(
                "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
            ).eval()
            predictor.requires_grad_(False)
            before = parameter_digest(predictor)
            for name in names:
                result = measure_sequence(
                    predictor,
                    name,
                    videos[name],
                    args.audit_dir,
                    output,
                    args.device,
                    args.chunk_size,
                    run,
                    sensitivity=args.sensitivity,
                    previous_run=args.previous_run,
                )
                results.append(result)
                if before != parameter_digest(predictor):
                    raise RuntimeError("Backbone parameters changed")
                (output / "sequences.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                (write_sensitivity_results if args.sensitivity else write_results)(
                    output, results
                )
        summary = {
            "completed": True,
            "overlay_frames": overlay["frames"],
            "overlays_verified": overlay["overlays_verified"],
            "sequences": results,
            "causal_contribution_measured": False,
            "equal_sequence_statistics": (
                write_sensitivity_results if args.sensitivity else write_results
            )(output, results),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finalize_tracking(
            run,
            output,
            {
                "completed": True,
                "overlays_verified": overlay["overlays_verified"],
                "attention_sequences": sum(r["status"] == "completed" for r in results),
            },
            ["setup.json", "summary.json", "results.md"]
            + (["sequence_results.md"] if args.sensitivity else [])
            + [
                str(path.relative_to(output))
                for path in (output / "source").rglob("*")
                if path.is_file()
            ],
        )
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
