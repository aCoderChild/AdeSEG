#!/usr/bin/env python3
"""Gated real-data audit of native-mask region definitions.

Reuse the official CPU runner, complete seq2/seq3 replay and existing author
attention observer. Save all definitions from each frame's own stored logits.
"""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from modeling.region_bias import feature_key  # noqa: E402
from modeling.region_definitions import (  # noqa: E402
    PRIMARY_ASSIGNMENT,
    TRANSFINER_REVISION,
    MATTEFORMER_REVISION,
    region_label_schemes,
)
from modeling.region_tokens import _author, sam2_low_res_mask, reference_token_regions  # noqa: E402
from scripts.calibrate_native_enrichment import NativeEnrichmentObserver  # noqa: E402
from scripts.audit_memory_token_regions import assignment_scores  # noqa: E402
from scripts import run_author_baseline as author  # noqa: E402
from scripts.region_definition_statistics import (  # noqa: E402
    contour_coverage,
    size_bin,
    summarize_frames,
)


class DefinitionObserver(NativeEnrichmentObserver):
    def capture_native(self, output, index, labels):
        if index != self.frame:
            raise RuntimeError("Nonchronological memory capture")
        if index > self.prompt and [r["layer"] for r in self.trace] != list(range(4)):
            raise RuntimeError("Missing native attention layers")
        features = output["maskmem_features"]
        key = feature_key(features.flatten(2).transpose(1, 2))
        if key in self.bank and self.bank[key][0] != index:
            raise RuntimeError("Ambiguous native memory identity")
        self.bank[key] = (index, torch.from_numpy(labels))
        for row in self.observer.rows:
            total = row["pointer_mass"] + sum(
                r["mass"] for r in row["regions"].values()
            )
            if abs(total - 1) > 1e-4:
                raise RuntimeError("Native attention mass does not sum to one")
            self.measurements.append({**row, "frame": index, "mass_sum": total})
        self.observer.rows.clear()
        self.trace.clear()
        self.frame, self.cached = index + 1, None


def square_overlay(frame, labels, destination, title):
    """Use SAM2's exact PIL square resize, then draw the 64x64 token grid."""
    with Image.open(frame.image_path) as source:
        image = source.convert("RGB").resize((1024, 1024))
    palette = np.array([[30, 140, 255], [0, 200, 100], [255, 180, 0]], np.uint8)
    colors = Image.fromarray(palette[labels]).resize(
        (1024, 1024), Image.Resampling.NEAREST
    )
    image = Image.blend(image, colors, 0.42)
    truth = load_label_mask(frame.mask_path, [1]) == 1
    truth = np.asarray(
        Image.fromarray(truth.astype(np.uint8)).resize(
            (1024, 1024), Image.Resampling.NEAREST
        )
    )
    contour = _author()["boundary"](truth, 1 / np.hypot(1024, 1024)).astype(bool)
    pixels = np.asarray(image).copy()
    pixels[contour] = 255
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for position in range(0, 1024, 16):
        draw.line((position, 0, position, 1023), fill=(50, 50, 50), width=1)
        draw.line((0, position, 1023, position), fill=(50, 50, 50), width=1)
    canvas = Image.new("RGB", (1024, 1070), "white")
    canvas.paste(image, (0, 46))
    ImageDraw.Draw(canvas).text(
        (8, 8),
        title + " | blue=BG green=interior orange=boundary; white=GT",
        fill="black",
    )
    canvas.save(destination)


def frame_metrics(frame, low_res, schemes, index, prompt):
    truth = load_label_mask(frame.mask_path, [1]) == 1
    reference_logits = sam2_low_res_mask(truth)
    reference = region_label_schemes(reference_logits)
    predicted = (low_res > 0).numpy()
    record = {
        "frame": index,
        "source_frame_index": frame.frame_index,
        "conditioning": index == prompt,
        "gt_area_fraction": float(truth.mean()),
        "gt_touches_image_edge": bool(
            truth[0].any() or truth[-1].any() or truth[:, 0].any() or truth[:, -1].any()
        ),
        "prediction_touches_image_edge": bool(
            predicted[0].any()
            or predicted[-1].any()
            or predicted[:, 0].any()
            or predicted[:, -1].any()
        ),
        "schemes": {},
    }
    for name, result in schemes.items():
        labels, expected = result["labels"], reference[name]["labels"]
        confusion = np.bincount((3 * expected + labels).ravel(), minlength=9).reshape(
            3, 3
        )
        record["schemes"][name] = {
            "counts": np.bincount(labels.ravel(), minlength=3).tolist(),
            "confusion": confusion.tolist(),
            "agreement": assignment_scores(confusion),
            "self_contour": contour_coverage(predicted, labels),
            "gt_contour": contour_coverage((reference_logits > 0).numpy(), labels),
        }
    return record


def main():
    parser = argparse.ArgumentParser(add_help=False)
    _, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        author.main()
        return
    if (
        remaining[remaining.index("--method") + 1] != "sam2"
        or remaining[remaining.index("--device") + 1] != "cpu"
    ):
        raise ValueError("This audit requires --method sam2 --device cpu")
    original_signature = author.signature

    def signature(*args):
        return {
            **original_signature(*args),
            "primary_assignment": PRIMARY_ASSIGNMENT,
            "transfiner_revision": TRANSFINER_REVISION,
            "matteformer_revision": MATTEFORMER_REVISION,
            "definition_sources": {
                name: author.digest(ROOT / name)
                for name in (
                    "modeling/region_definitions.py",
                    "scripts/audit_region_definitions.py",
                    "scripts/calibrate_native_enrichment.py",
                    "modeling/region_attention.py",
                    "scripts/region_definition_statistics.py",
                    "protocols/region_definition_verification.json",
                )
            },
        }

    author.signature = signature
    original_rollout, original_iterator = author.rollout, author.iter_predictions

    def rollout(
        model,
        method,
        frames,
        prompt,
        folder,
        device,
        run,
        repeat=False,
        capture_logits=False,
    ):
        if repeat:
            return original_rollout(
                model,
                method,
                frames,
                prompt,
                folder,
                device,
                run,
                repeat,
                capture_logits,
            )
        measurements, records = [], []
        encode = model._encode_new_memory
        encode_signature = inspect.signature(encode)
        writes = []

        def writer(*args, **kwargs):
            bound = encode_signature.bind(*args, **kwargs)
            result = encode(*args, **kwargs)
            writes.append(
                {
                    "features_sha256": feature_key(
                        result[0].to(torch.bfloat16).flatten(2).transpose(1, 2)
                    ),
                    "mask_logits": bound.arguments["pred_masks_high_res"]
                    .detach()
                    .float()
                    .cpu()
                    .clone(),
                }
            )
            return result

        def iterator(predictor, state, name, first):
            initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
            observer = DefinitionObserver(model, "zero", frames[0].video_id, prompt)
            with patch.object(model, "_encode_new_memory", writer), observer:
                for index, ids, logits in original_iterator(
                    predictor, state, name, first
                ):
                    outputs = state["output_dict_per_obj"][0]
                    output = outputs["cond_frame_outputs"].get(
                        index, outputs["non_cond_frame_outputs"].get(index)
                    )
                    low = output["pred_masks"][0, 0].detach().float().cpu().clone()
                    _, stored_full = predictor._get_orig_video_res_output(
                        state, output["pred_masks"]
                    )
                    if not torch.equal(logits, stored_full):
                        raise RuntimeError(
                            "Returned prediction differs from its own stored frame mask"
                        )
                    if index == prompt and not torch.equal(
                        low, sam2_low_res_mask(initial)
                    ):
                        raise RuntimeError(
                            "Prompt low-res mask differs from exact SAM2 mask-prompt path"
                        )
                    key = feature_key(
                        output["maskmem_features"].flatten(2).transpose(1, 2)
                    )
                    matched = [w for w in writes if w["features_sha256"] == key]
                    if len(matched) != 1:
                        raise RuntimeError(
                            "Cannot uniquely identify the actual mask-encoder write"
                        )
                    high = F.interpolate(
                        low[None, None],
                        (1024, 1024),
                        mode="bilinear",
                        align_corners=False,
                    )
                    writer_sign_changes = int(
                        ((high > 0) != (matched[0]["mask_logits"] > 0)).sum()
                    )
                    if writer_sign_changes:
                        raise RuntimeError(
                            f"Stored low-res mask/writer input sign disagreement: {writer_sign_changes}"
                        )
                    schemes = region_label_schemes(low)
                    observer.capture_native(
                        output, index, schemes["mixed_cells"]["labels"]
                    )
                    record = frame_metrics(frames[index], low, schemes, index, prompt)
                    record["size_bin"] = size_bin(
                        record["gt_area_fraction"], size_policy["cutpoints"]
                    )
                    full_binary = (logits[0, 0].detach().cpu() > 0).numpy()
                    png = folder / f"native_mask_{index:06d}.png"
                    Image.fromarray(full_binary.astype(np.uint8) * 255).save(png)
                    with Image.open(png) as saved:
                        if not np.array_equal(
                            np.asarray(saved) > 0,
                            (stored_full[0, 0].detach().cpu() > 0).numpy(),
                        ):
                            raise RuntimeError(
                                "Saved predicted PNG differs from its own stored frame"
                            )
                    round_trip = reference_token_regions(full_binary)["labels"]
                    record.update(
                        native_stored_mask_exact=True,
                        saved_png_matches_stored_mask=True,
                        prompt_low_res_exact=True if index == prompt else None,
                        writer_input_sign_changes=writer_sign_changes,
                        v3_round_trip_token_disagreement=float(
                            np.mean(round_trip != schemes["inner_band_0.02"]["labels"])
                        ),
                        native_mask_sha256=author.tensor_hash(low),
                        memory_features_sha256=key,
                    )
                    torch.save(
                        {"native_low_res_logits": low, "schemes": schemes},
                        folder / f"assignments_{index:06d}.pt",
                    )
                    records.append(record)
                    with (folder / "assignment_frames.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    writes.clear()
                    yield index, ids, logits
            measurements.extend(observer.measurements)

        author.iter_predictions = iterator
        try:
            rows, seconds = original_rollout(
                model,
                method,
                frames,
                prompt,
                folder,
                device,
                run,
                repeat,
                capture_logits,
            )
        finally:
            author.iter_predictions = original_iterator
        if len(measurements) != len(rows) * 4:
            raise RuntimeError("Attention pilot misses target/layer reads")
        (folder / "attention.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in measurements)
        )
        postprompt = [r for r in records if not r["conditioning"]]
        positives = [r for r in records if r["gt_area_fraction"] > 0]
        cases = (
            {
                "large": max(positives, key=lambda r: r["gt_area_fraction"]),
                "small": min(positives, key=lambda r: r["gt_area_fraction"]),
            }
            if positives
            else {}
        )
        for kind, field in (("edge_truncated", "gt_touches_image_edge"),):
            candidates = [r for r in records if r[field]]
            if candidates:
                cases[kind] = candidates[0]
        absent = [r for r in records if not r["gt_area_fraction"]]
        if absent:
            cases["absent"] = absent[0]
        for kind, record in cases.items():
            index = record["frame"]
            saved = torch.load(
                folder / f"assignments_{index:06d}.pt", weights_only=False
            )
            for name, scheme in saved["schemes"].items():
                square_overlay(
                    frames[index],
                    scheme["labels"],
                    folder / f"{kind}_{name}.png",
                    f"{frames[0].video_id} frame {index} | {kind} | {name}",
                )
        summary = {
            "primary_assignment": PRIMARY_ASSIGNMENT,
            "frames_including_prompt": len(records),
            "postprompt_targets": len(postprompt),
            "native_mask_checks_passed": True,
            "illustrated_cases": {k: r["frame"] for k, r in cases.items()},
            "attention_reads": len(measurements),
            "gt_edge_contact_frames": sum(r["gt_touches_image_edge"] for r in records),
            "prediction_edge_contact_frames": sum(
                r["prediction_touches_image_edge"] for r in records
            ),
            "v3_round_trip_mean_disagreement": float(
                np.mean([r["v3_round_trip_token_disagreement"] for r in records])
            ),
            "v3_round_trip_max_disagreement": max(
                r["v3_round_trip_token_disagreement"] for r in records
            ),
            "schemes": {},
            "contour_statistics_postprompt": summarize_frames(postprompt),
            "size_statistics_postprompt": {
                group: summarize_frames(
                    [r for r in postprompt if r["size_bin"] == group]
                )
                for group in ("absent", "Q1", "Q2", "Q3", "Q4")
            },
        }
        for name in records[0]["schemes"]:
            selected = [r["schemes"][name] for r in postprompt]
            summary["schemes"][name] = {
                "mean_counts_background_interior_boundary": np.mean(
                    [r["counts"] for r in selected], axis=0
                ).tolist(),
                "empty_interior_frames": sum(r["counts"][1] == 0 for r in selected),
                "empty_boundary_frames": sum(r["counts"][2] == 0 for r in selected),
                "pooled_gt_agreement": assignment_scores(
                    np.sum([r["confusion"] for r in selected], axis=0)
                ),
            }
        (folder / "assignment_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return rows, seconds

    author.rollout = rollout
    original_start, original_finalize = author.start_wandb, author.finalize_tracking
    size_policy = {}
    inventory = []

    def start(project, entity, name, mode, url, config, output, tags):
        config["region_modulation"]["enabled"] = False
        config["region_modulation"]["label_source"] = (
            "own stored native256 prediction; exact mixed-cell primary; read-only"
        )
        manifest = (
            Path(remaining[remaining.index("--manifest") + 1])
            if "--manifest" in remaining
            else ROOT / "data/polypgen_sequence.jsonl"
        )
        videos = {
            **author.load_manifest(manifest, "dev"),
            **author.load_manifest(manifest, "test"),
        }
        sequence_names = (
            ["seq2", "seq3"]
            if "--smoke" in remaining
            else [f"seq{i}" for i in range(1, 24)]
        )
        # Fix size quartiles from ALL source GT masks, before any model outcome.
        positive_areas = []
        for sequence in [f"seq{i}" for i in range(1, 24)]:
            for index, frame in enumerate(videos[sequence]):
                truth = load_label_mask(frame.mask_path, [1]) == 1
                area = float(truth.mean())
                if area:
                    positive_areas.append(area)
                if sequence in sequence_names:
                    inventory.append(
                        {"sequence": sequence, "frame": index, "gt_area_fraction": area}
                    )
        size_policy.update(
            cutpoints=np.quantile(positive_areas, [0.25, 0.5, 0.75]).tolist(),
            population="All positive GT frames in seq1–23, frozen before inference",
            positive_frames=len(positive_areas),
            ties="Lower quartile; absent is separate; never select bins by model outcomes",
        )
        for row in inventory:
            row["size_bin"] = size_bin(
                row["gt_area_fraction"], size_policy["cutpoints"]
            )
        (output / "size_policy.json").write_text(
            json.dumps(size_policy, indent=2) + "\n"
        )
        (output / "frame_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        shutil.copy2(
            ROOT / "protocols/region_definition_verification.json",
            output / "definition_protocol.json",
        )
        sources = [
            "modeling/region_definitions.py",
            "modeling/region_tokens.py",
            "scripts/audit_region_definitions.py",
            "scripts/region_definition_statistics.py",
            "scripts/calibrate_native_enrichment.py",
            "modeling/region_attention.py",
            "external/Mask-Transfiner/detectron2/modeling/roi_heads/mask_head.py",
            "external/MatteFormer/dataloader/data_generator.py",
        ]
        for repo in (
            "Mask-Transfiner",
            "MatteFormer",
            "Efficient-SAM2",
            "boundary-iou-api",
        ):
            for path in (ROOT / "external" / repo).glob("*LICENSE*"):
                sources.append(str(path.relative_to(ROOT)))
        with zipfile.ZipFile(
            output / "definition_sources.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for source in sources:
                archive.write(ROOT / source, source)
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(__file__, output / "definition_audit_source.py")
        return original_start(
            project,
            entity,
            f"15 | Region definitions + contour coverage | {config['scope']}",
            mode,
            url,
            config,
            output,
            tags + ["native-mask-provenance", "definition-verification"],
        )

    def finalize(run, output, summary, names):
        summaries = {
            name: json.loads((output / name / "assignment_summary.json").read_text())
            for name in [
                r["sequence"]
                for r in summary["sequences"]
                if r["status"] == "completed"
            ]
        }
        (output / "assignment_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        lines = [
            "# Native memory region-definition audit",
            "",
            f"Scope: {summary['scope']}; CPU FP32. Primary statistics below exclude the conditioning frame.",
            "Primary: exact mixed cells using author Efficient-SAM2 occupancy on FG and BG.",
            "All assignment inputs are the frame's own native stored logits; not a PNG round trip.",
            "Fresh-model no-observer replay is required in the matching seq2/seq3 gate; the full cohort is not replayed twice.",
            "",
            "| Sequence | Targets | Attention reads | GT edge contacts (including prompt) | Mean/max v3 PNG-roundtrip disagreement |",
            "|---|---:|---:|---:|---|",
        ]
        for name, item in summaries.items():
            lines.append(
                f"| {name} | {item['postprompt_targets']} | {item['attention_reads']} | {item['gt_edge_contact_frames']} | {item['v3_round_trip_mean_disagreement']:.6f} / {item['v3_round_trip_max_disagreement']:.6f} |"
            )
        records = []
        native_keys = set()
        for name in summaries:
            for line in (
                (output / name / "assignment_frames.jsonl").read_text().splitlines()
            ):
                row = {**json.loads(line), "sequence": name}
                records.append(row)
                native_keys.add((name, row["frame"]))
        postprompt = [r for r in records if not r["conditioning"]]
        pooled = summarize_frames(postprompt)
        by_size = {
            group: summarize_frames([r for r in postprompt if r["size_bin"] == group])
            for group in ("absent", "Q1", "Q2", "Q3", "Q4")
        }
        reference_only = []
        manifest = (
            Path(remaining[remaining.index("--manifest") + 1])
            if "--manifest" in remaining
            else ROOT / "data/polypgen_sequence.jsonl"
        )
        videos = {
            **author.load_manifest(manifest, "dev"),
            **author.load_manifest(manifest, "test"),
        }
        for item in inventory:
            if (item["sequence"], item["frame"]) in native_keys:
                continue
            frame = videos[item["sequence"]][item["frame"]]
            low = sam2_low_res_mask(load_label_mask(frame.mask_path, [1]) == 1)
            row = frame_metrics(
                frame, low, region_label_schemes(low), item["frame"], -1
            )
            row.update(
                sequence=item["sequence"],
                size_bin=item["size_bin"],
                role="GT reference only; no prompt/native memory/attention",
            )
            reference_only.append(row)
        coverage = {
            "source_frames": len(inventory),
            "native_memory_frames_including_prompts": len(records),
            "native_postprompt_frames": len(postprompt),
            "reference_only_frames": len(reference_only),
            "reference_only_sequences": sorted({r["sequence"] for r in reference_only}),
            "all_source_frames_accounted_for": len(native_keys) + len(reference_only)
            == len(inventory),
            "gt_edge_contact_native_frames": sum(
                r["gt_touches_image_edge"] for r in records
            ),
            "prediction_edge_contact_native_frames": sum(
                r["prediction_touches_image_edge"] for r in records
            ),
        }
        if not coverage["all_source_frames_accounted_for"]:
            raise RuntimeError("Source-frame coverage mismatch")
        (output / "reference_only_frames.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in reference_only)
        )
        (output / "cohort_statistics.json").write_text(
            json.dumps(
                {
                    "coverage": coverage,
                    "size_policy": size_policy,
                    "postprompt_pooled_descriptive": pooled,
                    "postprompt_by_size": by_size,
                    "reference_only": summarize_frames(reference_only),
                },
                indent=2,
            )
            + "\n"
        )

        def percent(value):
            return "N/A" if value is None else f"{100 * value:.2f}%"

        lines += [
            "",
            "## Contour coverage and region counts",
            "",
            "One-pixel inner contour from unchanged Boundary IoU on native256; boundary-token footprints enlarged by nearest neighbor. Percentages pool contour pixels, not sequences. GT contour misses combine geometry and prediction error; self misses isolate geometry.",
            "",
            "| Definition | BG / FG / boundary tokens (mean) | Self contour missed | Self missed, excluding image border | GT contour missed | Empty FG / boundary frames |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, item in pooled.items():
            lines.append(
                f"| {name} | {' / '.join(f'{v:.1f}' for v in item['mean_token_counts_bg_fg_boundary'])} | {percent(item['self_contour']['including_border']['pooled_missed_fraction'])} | {percent(item['self_contour']['excluding_border']['pooled_missed_fraction'])} | {percent(item['gt_contour']['including_border']['pooled_missed_fraction'])} | {item['empty_interior_frames']} / {item['empty_boundary_frames']} |"
            )
        lines += [
            "",
            "## GT-reference region agreement",
            "",
            "Pooled token IoU; predicted-position labels versus GT-position labels, not feature-classifier accuracy.",
            "",
            "| Definition | BG IoU | FG IoU | Boundary IoU (token labels) |",
            "|---|---:|---:|---:|",
        ]
        for name, item in pooled.items():
            confusion = np.asarray(item["pooled_gt_confusion"])
            diagonal = confusion.diagonal()
            union = confusion.sum(0) + confusion.sum(1) - diagonal
            scores = [float(a / b) if b else None for a, b in zip(diagonal, union)]
            lines.append(f"| {name} | {' | '.join(percent(v) for v in scores)} |")
        lines += [
            "",
            "## Polyp size",
            "",
            f"Positive GT area-fraction quartile cutpoints: {size_policy['cutpoints']}. Frozen from all source frames before inference; ties enter the lower bin.",
            "",
            "| Size bin | Postprompt frames | Mixed-cell self contour missed | Empty FG / boundary frames |",
            "|---|---:|---:|---:|",
        ]
        for group, schemes in by_size.items():
            item = schemes.get("mixed_cells")
            lines.append(
                f"| {group} | {item['frames'] if item else 0} | {percent(item['self_contour']['including_border']['pooled_missed_fraction']) if item else 'N/A'} | {str(item['empty_interior_frames']) + ' / ' + str(item['empty_boundary_frames']) if item else 'N/A'} |"
            )
        lines += [
            "",
            f"Coverage: {coverage['source_frames']} source frames = {len(records)} native memory frames (including prompts) + {len(reference_only)} GT-reference-only frames. Reference-only frames never create memory or attention. Full per-frame counts, contour denominators and size-stratified comparator results are in JSON/JSONL.",
            f"Image-edge contacts among native memory frames: GT {coverage['gt_edge_contact_native_frames']}; predictions {coverage['prediction_edge_contact_native_frames']}. The inner-band comparator retains the author's border definition; the table separately removes the outermost native raster row/column from the contour denominator.",
        ]
        lines += [
            "",
            "Mixed-cell labels identify mixed footprints, not all contours: perfectly grid-aligned edges may have no mixed cells. All vectors remain contextualized.",
            "Mask Transfiner retains its author bilinear sampling/0.01 cutoff; saturated scores are used as categorical boundary-associated cells. MatteFormer uses fixed equal erosion kernels instead of random widths.",
            "Pilot success does not establish full-cohort robustness or a causal segmentation gain. Contiguity-matched exclusions have not been launched.",
        ]
        (output / "assignment_results.md").write_text("\n".join(lines) + "\n")
        return original_finalize(
            run,
            output,
            summary,
            names
            + [
                "definition_audit_source.py",
                "assignment_summary.json",
                "assignment_results.md",
                "size_policy.json",
                "frame_inventory.json",
                "definition_protocol.json",
                "cohort_statistics.json",
                "reference_only_frames.jsonl",
                "definition_sources.zip",
            ],
        )

    author.start_wandb, author.finalize_tracking = start, finalize
    sys.argv = [__file__, *remaining]
    author.main()


if __name__ == "__main__":
    main()
