#!/usr/bin/env python3
"""Audit saved Run155 memory-frame traces without running model inference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "155_region_constrained_tome_polypgen_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run: Path) -> dict:
    summary_path = run / "full/summary.json"
    setup_path = run / "full/setup.json"
    summary = json.loads(summary_path.read_text())
    setup = json.loads(setup_path.read_text())
    sequences = summary["sequences"]
    if len(sequences) != 21 or len(set(sequences)) != 21:
        raise RuntimeError("Expected the 21 unique Run155 evaluation sequences")

    conditions = ("region_constrained_tome", "generic_local_tome")
    results = {}
    issues = []
    targets_total = 0
    layer_calls_total = 0
    memory_span_lengths = []
    input_hashes = {"summary.json": sha256(summary_path), "setup.json": sha256(setup_path)}

    for sequence in sequences:
        condition_rows = {}
        for condition in conditions:
            path = run / f"full/{condition}/{sequence}/frames.json"
            if not path.is_file():
                issues.append(f"missing {condition} trace for {sequence}")
                continue
            input_hashes[str(path.relative_to(run))] = sha256(path)
            condition_rows[condition] = json.loads(path.read_text())
        if len(condition_rows) != len(conditions):
            continue

        regional = condition_rows[conditions[0]]
        generic = condition_rows[conditions[1]]
        if len(regional) != len(generic):
            issues.append(f"{sequence}: regional/generic target counts differ")
            continue
        sequence_targets = 0
        for row_index, (rrow, grow) in enumerate(zip(regional, generic, strict=True)):
            sequence_targets += 1
            targets_total += 1
            r_layers = rrow.get("attention_summary", [])
            g_layers = grow.get("attention_summary", [])
            if len(r_layers) != 4 or len(g_layers) != 4:
                issues.append(f"{sequence}/{row_index}: expected four layer records")
                continue
            layer_calls_total += 8  # four records in each of the two arms

            for condition, layers in ((conditions[0], r_layers), (conditions[1], g_layers)):
                layer_ids = [record.get("layer") for record in layers]
                if layer_ids != [0, 1, 2, 3]:
                    issues.append(f"{sequence}/{row_index}/{condition}: layer IDs {layer_ids}")
                frame_lists = [record.get("memory_frames") for record in layers]
                if any(frames != frame_lists[0] for frames in frame_lists[1:]):
                    issues.append(f"{sequence}/{row_index}/{condition}: layers disagree on memory frames")
                frames = frame_lists[0]
                if not isinstance(frames, list) or not frames or len(frames) > 7:
                    issues.append(f"{sequence}/{row_index}/{condition}: invalid memory-frame list {frames}")
                elif len(set(frames)) != len(frames):
                    issues.append(f"{sequence}/{row_index}/{condition}: duplicate frame IDs")
                elif condition == conditions[0] and layers[0].get("layer") == 0:
                    memory_span_lengths.append(len(frames))
                for record in layers:
                    before = record.get("spatial_keys_before")
                    mass = record.get("spatial_key_mass_sum")
                    pointers = record.get("pointer_keys")
                    all_mass = record.get("key_mass_sum")
                    if before is None or mass is None or abs(float(mass) - float(before)) > 1e-5:
                        issues.append(f"{sequence}/{row_index}/{condition}: spatial mass not conserved")
                    if pointers is None or all_mass is None or before is None:
                        issues.append(f"{sequence}/{row_index}/{condition}: missing pointer/mass fields")
                    elif abs(float(all_mass) - float(before) - float(pointers)) > 1e-5:
                        issues.append(f"{sequence}/{row_index}/{condition}: pointer-inclusive mass mismatch")

            r_frames = [record.get("memory_frames") for record in r_layers]
            g_frames = [record.get("memory_frames") for record in g_layers]
            if r_frames != g_frames:
                issues.append(f"{sequence}/{row_index}: regional/generic memory-frame IDs differ")
            r_after = [record.get("spatial_keys_after") for record in r_layers]
            g_after = [record.get("spatial_keys_after") for record in g_layers]
            if r_after != g_after:
                issues.append(f"{sequence}/{row_index}: regional/generic spatial key budgets differ")
            r_ptr = [record.get("pointer_keys") for record in r_layers]
            g_ptr = [record.get("pointer_keys") for record in g_layers]
            if r_ptr != g_ptr:
                issues.append(f"{sequence}/{row_index}: regional/generic pointer counts differ")

        results[sequence] = {"targets": sequence_targets}

    return {
        "scope": "Static audit of saved Run155 regional/generic attention summaries; no inference",
        "sequence_count": len(results),
        "sequences_expected": sequences,
        "targets_checked": targets_total,
        "attention_layer_records_checked": layer_calls_total,
        "selected_memory_frames_per_target": {
            "min": min(memory_span_lengths) if memory_span_lengths else None,
            "max": max(memory_span_lengths) if memory_span_lengths else None,
            "histogram": {
                str(length): count
                for length, count in sorted(Counter(memory_span_lengths).items())
            },
        },
        "regional_generic_frame_id_lists_equal": not any("frame IDs differ" in x for x in issues),
        "regional_generic_key_budgets_equal": not any("key budgets differ" in x for x in issues),
        "pointer_counts_equal": not any("pointer counts differ" in x for x in issues),
        "all_four_layers_agree_per_target": not any("layers disagree" in x for x in issues),
        "mass_conservation_passed": not any("mass" in x for x in issues),
        "issues": issues,
        "sequence_targets": results,
        "native_trace_limit": (
            "The native frames.json records do not include memory-frame IDs. This audit compares "
            "the regional and generic ToMe traces; native schedule preservation is additionally "
            "supported by the source integration point and SAM2 selector path, not directly by a "
            "native per-target trace comparison."
        ),
        "input_sha256": input_hashes,
        "setup_sha256": sha256(setup_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_RUN / "source_audit_20261109/attention_trace_audit_20261109_v2.json",
    )
    args = parser.parse_args()
    result = audit(args.run_dir)
    result["audit_script_sha256"] = sha256(Path(__file__))
    result["status"] = "pass" if not result["issues"] and result["sequence_count"] == 21 and result["targets_checked"] == 2012 else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "sequences": result["sequence_count"],
        "targets": result["targets_checked"],
        "layer_records": result["attention_layer_records_checked"],
        "memory_frames_min_max": result["selected_memory_frames_per_target"],
        "issues": len(result["issues"]),
        "output": str(args.output),
    }, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
