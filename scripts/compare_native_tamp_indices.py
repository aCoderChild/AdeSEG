"""Compare native Lean-SAM2 TAMP retained-token indices between paired runs.

This is a mechanistic diagnostic, NOT a segmentation metric. It requires
--dump-tamp-indices for both inference runs, with matched source versions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_lean_memory_fusion_ablation import (
    file_sha256, verify_native_tamp_dump,
)


def read_run(directory: Path):
    directory = directory.resolve(strict=True)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text("utf8"))
    if not receipt.get("dump_tamp_indices", False):
        raise ValueError("Both runs must enable --dump-tamp-indices")
    if receipt.get("format") != "adeseg_lean_memory_fusion_video_v3":
        raise ValueError("Unexpected inference-run receipt")
    return directory, receipt


def load_events(root: Path, video: str, receipt: dict):
    audit = receipt.get("tamp_index_dump")
    if not isinstance(audit, dict):
        raise ValueError("Missing TAMP token audit receipt")
    path = root / "predictions" / video / "tamp_indices" / "ours_retained_tokens.jsonl"
    verified = verify_native_tamp_dump(path)
    if (verified["sha256"] != audit.get("sha256")
            or verified["records"] != audit.get("records")):
        raise ValueError("TAMP dump digest or record count disagrees with receipt")
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text("utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = (row["frame_idx"], row.get("memory_frame_idx"),
                 row.get("object_id"))
        if event in result:
            raise ValueError("Duplicate TAMP selection event")
        result[event] = row
    return result


def compare_runs(reference: Path, candidate: Path, output: Path):
    root_a, a = read_run(Path(reference))
    root_b, b = read_run(Path(candidate))
    if output.exists():
        raise FileExistsError("Refusing to overwrite TAMP comparison output")
    keys = (
        "track", "split", "label_id", "manifest_sha256",
        "prompt_manifest_sha256", "backbone_sha256",
        "lean_source_revision", "runtime_signature",
    )
    if any(a.get(k) != b.get(k) for k in keys):
        raise ValueError("Cannot compare mismatched Lean sources, prompts or runtime")
    if set(a.get("sequences", {})) != set(b.get("sequences", {})):
        raise ValueError("Different video cohorts")
    pairs, missing_a, missing_b = [], [], []
    for video in sorted(a["sequences"]):
        ra = a["sequences"][video]
        rb = b["sequences"][video]
        if ra.get("status") != rb.get("status"):
            raise ValueError("Different video completion status")
        if ra.get("status") == "no_initialization":
            continue
        if ra.get("status") != "completed":
            raise ValueError("Unrecognized sequence status")
        ma = load_events(root_a, video, ra)
        mb = load_events(root_b, video, rb)
        for event in sorted(ma.keys() - mb.keys(), key=str):
            missing_b.append({"video_id": video, "event": event})
        for event in sorted(mb.keys() - ma.keys(), key=str):
            missing_a.append({"video_id": video, "event": event})
        for event in sorted(ma.keys() & mb.keys(), key=str):
            x, y = ma[event], mb[event]
            if x["num_total_tokens"] != y["num_total_tokens"]:
                raise ValueError(
                    "Different candidate token grids at matched TAMP event")
            xset = set(x["retained_token_indices"])
            yset = set(y["retained_token_indices"])
            overlap = len(xset & yset)
            union = len(xset | yset)
            pairs.append({
                "video_id": video,
                "frame_idx": event[0],
                "memory_frame_idx": event[1],
                "object_id": event[2],
                "total_tokens": x["num_total_tokens"],
                "reference_retained": len(xset),
                "candidate_retained": len(yset),
                "same_selection": xset == yset,
                "intersection": overlap,
                "jaccard": overlap / union if union else 1.0,
            })
    report = {
        "format": "adeseg_native_tamp_selection_overlap_v1",
        "reference_arm": a.get("arm"),
        "candidate_arm": b.get("arm"),
        "reference_mode": a.get("fusion_input_mode", "full"),
        "candidate_mode": b.get("fusion_input_mode", "full"),
        "reference_receipt_sha256": file_sha256(root_a / "receipt.json"),
        "candidate_receipt_sha256": file_sha256(root_b / "receipt.json"),
        "matched_events": len(pairs),
        "missing_from_reference": missing_a,
        "missing_from_candidate": missing_b,
        "mean_jaccard": (
            sum(x["jaccard"] for x in pairs) / len(pairs) if pairs else None),
        "identical_selection_fraction": (
            sum(x["same_selection"] for x in pairs) / len(pairs) if pairs else None),
        "per_event": pairs,
        "warning": "Selection overlap does NOT establish correct target retention or improved Dice",
    }
    output.mkdir(parents=True)
    (output / "tamp_selection_overlap.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.reference, args.candidate, args.output)
    print(json.dumps({
        "matched_events": report["matched_events"],
        "mean_jaccard": report["mean_jaccard"],
        "identical_selection_fraction": report["identical_selection_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
