#!/usr/bin/env python3
"""Compare Run155's saved ToMe frame spans with SAM2's frozen selector rule.

This is a read-only audit. It does not load a model or run inference. Because
Run155's native frame records did not save memory-frame IDs, it independently
reconstructs the expected IDs from the manifest prompt and the pinned SAM2
evaluation selector, then compares them with each ToMe arm and attention layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from datasets.video import load_manifest
from scripts.memory_token_io import prompt_index


DEFAULT_RUN = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "155_region_constrained_tome_polypgen_v1/full"
)
DEFAULT_OUTPUT = DEFAULT_RUN.parent / "source_audit_20261109/native_schedule_audit_20261109.json"
MANIFEST = PROJECT / "data/polypgen_sequence.jsonl"
SAM2_ROOT = PROJECT / "external/sam2"
SAM2_SOURCE = SAM2_ROOT / "sam2/modeling/sam2_base.py"
SAM2_CONFIG = SAM2_ROOT / "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
CONDITIONS = ("region_constrained_tome", "generic_local_tome")
EXPECTED_SEQUENCES = 21
EXPECTED_TARGETS = 2012
EXPECTED_LAYERS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_settings(setup: dict) -> tuple[int, int, dict]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SAM2_ROOT, text=True
    ).strip()
    if revision != setup["sam2_revision"]:
        raise RuntimeError("Mounted SAM2 checkout does not match Run155 setup")

    config_text = SAM2_CONFIG.read_text()
    match = re.search(r"^\s*num_maskmem:\s*(\d+)\s*$", config_text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot read num_maskmem from the pinned SAM2 config")
    num_maskmem = int(match.group(1))

    source_text = SAM2_SOURCE.read_text()
    stride_match = re.search(
        r"^\s*memory_temporal_stride_for_eval\s*=\s*(\d+)\s*,?\s*(?:#.*)?$",
        source_text,
        re.MULTILINE,
    )
    if not stride_match:
        raise RuntimeError("Cannot verify SAM2's default evaluation memory stride")
    stride = int(stride_match.group(1))
    if re.search(r"^\s*memory_temporal_stride_for_eval\s*:", config_text, re.MULTILINE):
        raise RuntimeError("SAM2 config overrides the audited default memory stride")

    return num_maskmem, stride, {
        "sam2_revision": revision,
        "sam2_selector_source_sha256": sha256(SAM2_SOURCE),
        "sam2_config_sha256": sha256(SAM2_CONFIG),
        "num_maskmem": num_maskmem,
        "memory_temporal_stride_for_eval": stride,
        "selection_formula": (
            "SAM2 _prepare_memory_conditioned_features: conditioning frame(s) first; "
            "then t_pos=1..num_maskmem-1 using native evaluation-stride formula. "
            "This protocol has one selected conditioning prompt frame."
        ),
    }


def expected_memory_frames(frame: int, prompt: int, num_maskmem: int, stride: int) -> list[int]:
    """Reproduce SAM2's selected frame IDs in native concatenation order."""
    selected = [prompt]  # sole selected conditioning frame, t_pos=0
    for t_pos in range(1, num_maskmem):
        t_rel = num_maskmem - t_pos
        if t_rel == 1:
            previous = frame - 1
        else:
            previous = ((frame - 2) // stride) * stride - (t_rel - 2) * stride
        # Past frames before the first labeled prompt have no stored memory;
        # the prompt is already represented in the conditioning-frame span.
        if previous <= prompt:
            continue
        selected.append(previous)
    return selected


def audit(run: Path) -> dict:
    summary_path, setup_path = run / "summary.json", run / "setup.json"
    summary, setup = json.loads(summary_path.read_text()), json.loads(setup_path.read_text())
    sequences = summary["sequences"]
    if len(sequences) != EXPECTED_SEQUENCES or len(set(sequences)) != EXPECTED_SEQUENCES:
        raise RuntimeError("Run155 summary does not contain the expected 21 unique sequences")

    videos = {}
    for split in ("dev", "test"):
        videos.update(load_manifest(MANIFEST, split))
    num_maskmem, stride, settings = source_settings(setup)
    source_hashes = {
        "summary.json": sha256(summary_path),
        "setup.json": sha256(setup_path),
        "manifest": sha256(MANIFEST),
        "prompt_selection_helper": sha256(PROJECT / "scripts/memory_token_io.py"),
    }
    issues: list[str] = []
    prompts: dict[str, int] = {}
    counts = {condition: 0 for condition in ("native", *CONDITIONS)}
    layer_calls = 0
    slot_histogram: dict[str, dict[str, int]] = {
        condition: {} for condition in CONDITIONS
    }

    for sequence in sequences:
        if sequence not in videos:
            issues.append(f"{sequence}: missing from dev/test manifest")
            continue
        frames = videos[sequence]
        prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        prompts[sequence] = prompt
        expected_targets = list(range(prompt + 1, len(frames)))

        native_path = run / "native" / sequence / "frames.json"
        if not native_path.is_file():
            issues.append(f"{sequence}: native frame record is missing")
        else:
            source_hashes[str(native_path.relative_to(run))] = sha256(native_path)
            native_rows = json.loads(native_path.read_text())
            counts["native"] += len(native_rows)
            if [row["frame"] for row in native_rows] != expected_targets:
                issues.append(f"{sequence}: native target coverage differs from manifest prompt")

        for condition in CONDITIONS:
            trace_path = run / condition / sequence / "frames.json"
            if not trace_path.is_file():
                issues.append(f"{sequence}/{condition}: trace is missing")
                continue
            source_hashes[str(trace_path.relative_to(run))] = sha256(trace_path)
            rows = json.loads(trace_path.read_text())
            counts[condition] += len(rows)
            if [row["frame"] for row in rows] != expected_targets:
                issues.append(f"{sequence}/{condition}: target coverage differs from manifest prompt")
                continue
            for row in rows:
                expected = expected_memory_frames(row["frame"], prompt, num_maskmem, stride)
                calls = row.get("attention_summary", [])
                if [call.get("layer") for call in calls] != list(range(EXPECTED_LAYERS)):
                    issues.append(f"{sequence}/{row['frame']}/{condition}: unexpected layer records")
                    continue
                layer_frame_lists = [call.get("memory_frames") for call in calls]
                if any(actual != expected for actual in layer_frame_lists):
                    issues.append(
                        f"{sequence}/{row['frame']}/{condition}: saved frame spans differ "
                        f"from reconstructed SAM2 selection (expected {expected}, got {layer_frame_lists})"
                    )
                layer_calls += len(calls)
                count_key = str(len(expected))
                slot_histogram[condition][count_key] = slot_histogram[condition].get(count_key, 0) + 1

    if counts["native"] != EXPECTED_TARGETS:
        issues.append(f"Native targets checked: expected {EXPECTED_TARGETS}, got {counts['native']}")
    for condition in CONDITIONS:
        if counts[condition] != EXPECTED_TARGETS:
            issues.append(f"{condition} targets checked: expected {EXPECTED_TARGETS}, got {counts[condition]}")
    if layer_calls != EXPECTED_TARGETS * len(CONDITIONS) * EXPECTED_LAYERS:
        issues.append(f"Unexpected attention layer-read count: {layer_calls}")

    return {
        "scope": (
            "Offline native memory-schedule reconstruction. No model was loaded and no inference ran. "
            "Native frames.json has no memory-frame IDs; saved ToMe IDs are checked against the "
            "independently reconstructed native SAM2 selector rule."
        ),
        "status": "pass" if not issues else "fail",
        "sequence_count": len(prompts),
        "target_records_checked": counts,
        "attention_layer_calls_checked": layer_calls,
        "prompts": prompts,
        "selector_settings": settings,
        "selected_memory_frame_count_histogram_by_condition": slot_histogram,
        "regional_generic_histograms_equal": slot_histogram[CONDITIONS[0]] == slot_histogram[CONDITIONS[1]],
        "issues": issues,
        "input_sha256": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    result = audit(args.run_dir)
    result["audit_script_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "sequence_count": result["sequence_count"],
        "target_records_checked": result["target_records_checked"],
        "attention_layer_calls_checked": result["attention_layer_calls_checked"],
        "frame_count_histogram": result["selected_memory_frame_count_histogram_by_condition"],
        "issues": len(result["issues"]),
        "output": str(args.output),
    }, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
