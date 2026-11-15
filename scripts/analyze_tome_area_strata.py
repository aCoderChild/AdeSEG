"""Post-hoc size-stratified audit of completed full-video ToMe trajectories.

This is descriptive analysis only. It uses saved SAM2/ToMe frame metrics and
the exact PolypGen GT masks; it does not rerun inference or treat frames as
independent samples.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest


SEQUENCES = [
    "seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9", "seq10",
    "seq11", "seq12", "seq13", "seq14", "seq15", "seq16", "seq17",
    "seq18", "seq19", "seq20", "seq21", "seq22", "seq23",
]
AREA_BINS = [
    ("<1%", 0.0, 0.01),
    ("1–5%", 0.01, 0.05),
    ("5–15%", 0.05, 0.15),
    ("≥15%", 0.15, 1.01),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bootstrap_equal_sequence_mean(values, *, seed=20261112, draws=20000):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _area_ratio(scores):
    precision = float(scores["precision"])
    recall = float(scores["recall"])
    if recall <= 0:
        return 0.0
    if precision <= 0:
        return None
    return recall / precision


def analyze(run_dir, manifest_path):
    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    setup = json.loads((run_dir / "setup.json").read_text())
    if sha256(manifest_path) != setup["manifest_sha256"]:
        raise RuntimeError("Manifest checksum differs from the completed ToMe run")
    audit_candidates = sorted(run_dir.glob("final_integrity_audit*.json"))
    if len(audit_candidates) != 1:
        raise RuntimeError("Expected exactly one final saved-output integrity audit")
    audit_path = audit_candidates[0]
    integrity_audit = json.loads(audit_path.read_text())
    if integrity_audit.get("status") != "passed" or integrity_audit.get("sequences") != 21:
        raise RuntimeError("Full-run saved-output integrity audit did not pass")
    if not integrity_audit.get("saved_statistics_recomputed_exactly"):
        raise RuntimeError("Full-run audit did not validate saved statistics")
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text())
    videos = load_manifest(manifest_path, "dev")
    test_videos = load_manifest(manifest_path, "test")
    overlap = videos.keys() & test_videos.keys()
    if overlap:
        raise RuntimeError(f"Sequence appears in both manifest splits: {sorted(overlap)}")
    videos.update(test_videos)
    if sorted(results["frame_rows"]["global"]) != sorted(SEQUENCES):
        raise RuntimeError("Saved global ToMe rows do not cover the 21 eligible sequences")
    if sorted(results["frame_rows"]["regional"]) != sorted(SEQUENCES):
        raise RuntimeError("Saved regional ToMe rows do not cover the 21 eligible sequences")

    sequence_rows = {name: [] for name in SEQUENCES}
    gt_masks_used = []
    eligible_target_count = 0
    for sequence in SEQUENCES:
        frames = videos[sequence]
        if [frame.frame_index for frame in frames] != list(range(len(frames))):
            raise RuntimeError(f"Manifest frame indices are not contiguous for {sequence}")
        native_meta = setup["native_reference_receipts"][sequence]
        native_path = Path(native_meta["frames_path"])
        if sha256(native_path) != native_meta["frames_sha256"]:
            raise RuntimeError(f"Native frame receipt checksum failed for {sequence}")
        native_rows = json.loads(native_path.read_text())
        native_by_frame = {int(row["frame"]): row for row in native_rows}
        generic_by_frame = {
            int(row["frame"]): row for row in results["frame_rows"]["global"][sequence]
        }
        regional_by_frame = {
            int(row["frame"]): row for row in results["frame_rows"]["regional"][sequence]
        }
        if native_by_frame.keys() != generic_by_frame.keys() or native_by_frame.keys() != regional_by_frame.keys():
            raise RuntimeError(f"Frame IDs are not aligned across conditions for {sequence}")

        for frame_index in sorted(native_by_frame):
            native = native_by_frame[frame_index]
            generic = generic_by_frame[frame_index]
            regional = regional_by_frame[frame_index]
            if len({native["present"], generic["present"], regional["present"]}) != 1:
                raise RuntimeError(f"GT-presence mismatch at {sequence}/{frame_index}")
            if not native["present"]:
                continue
            if frame_index >= len(frames):
                raise RuntimeError(f"Frame index outside manifest: {sequence}/{frame_index}")
            gt = load_label_mask(frames[frame_index].mask_path, [1]) == 1
            area_fraction = float(gt.mean())
            if area_fraction <= 0:
                raise RuntimeError(f"Saved GT-present label has empty mask: {sequence}/{frame_index}")
            history = int(generic.get("available_noncond_frames", 0))
            regional_history = int(regional.get("available_noncond_frames", history))
            if history != regional_history:
                raise RuntimeError(f"History count differs across ToMe arms at {sequence}/{frame_index}")
            if history <= 6:
                continue
            eligible_target_count += 1
            sequence_rows[sequence].append({
                "frame": frame_index,
                "area_fraction": area_fraction,
                "gt_mask_sha256": sha256(frames[frame_index].mask_path),
                "native_dice": float(native["scores"]["dice"]),
                "generic_dice": float(generic["scores"]["dice"]),
                "regional_dice": float(regional["scores"]["dice"]),
                "native_area_ratio": _area_ratio(native["scores"]),
                "generic_area_ratio": _area_ratio(generic["scores"]),
                "regional_area_ratio": _area_ratio(regional["scores"]),
            })
            gt_masks_used.append({
                "sequence": sequence,
                "frame": frame_index,
                "mask_path": str(frames[frame_index].mask_path),
                "mask_sha256": sha256(frames[frame_index].mask_path),
                "area_fraction": area_fraction,
            })

    output_bins = []
    for bin_name, lower, upper in AREA_BINS:
        seq_deltas = {"generic": {}, "regional": {}}
        pooled = {"native": [], "generic": [], "regional": []}
        area_values = []
        ratio_values = {"native": [], "generic": [], "regional": []}
        frame_count = 0
        for sequence, rows in sequence_rows.items():
            selected = [row for row in rows if lower <= row["area_fraction"] < upper]
            if not selected:
                continue
            frame_count += len(selected)
            area_values.extend(row["area_fraction"] for row in selected)
            for method in pooled:
                pooled[method].extend(row[f"{method}_dice"] for row in selected)
                ratio_values[method].extend(
                    row[f"{method}_area_ratio"] for row in selected
                    if row[f"{method}_area_ratio"] is not None
                )
            seq_deltas["generic"][sequence] = float(np.mean([
                row["generic_dice"] - row["native_dice"] for row in selected
            ]))
            seq_deltas["regional"][sequence] = float(np.mean([
                row["regional_dice"] - row["native_dice"] for row in selected
            ]))

        item = {
            "gt_area_bin": bin_name,
            "frames": frame_count,
            "sequences": len(seq_deltas["regional"]),
            "mean_gt_area_percent": (
                100.0 * float(np.mean(area_values)) if area_values else None
            ),
            "pooled_frame_mean_dice": {
                key: float(np.mean(value)) if value else None
                for key, value in pooled.items()
            },
            "equal_sequence_delta_vs_native": {},
            "sequence_bootstrap_95_ci": {},
            "predicted_to_gt_area_ratio_median": {},
            "predicted_area_direction_percent": {},
        }
        for method in ("generic", "regional"):
            deltas_by_sequence = seq_deltas[method]
            deltas = list(deltas_by_sequence.values())
            item["equal_sequence_delta_vs_native"][method] = (
                float(np.mean(deltas)) if deltas else None
            )
            item["sequence_bootstrap_95_ci"][method] = _bootstrap_equal_sequence_mean(deltas)
            without_seq8 = [
                value for sequence, value in deltas_by_sequence.items()
                if sequence != "seq8"
            ]
            item.setdefault("equal_sequence_delta_excluding_seq8", {})[method] = (
                float(np.mean(without_seq8)) if without_seq8 else None
            )
            item.setdefault("sequence_bootstrap_95_ci_excluding_seq8", {})[method] = (
                _bootstrap_equal_sequence_mean(without_seq8)
            )
            item.setdefault("per_sequence_delta_vs_native", {})[method] = deltas_by_sequence
        for method, ratios in ratio_values.items():
            ratios = np.asarray(ratios, dtype=np.float64)
            item["predicted_to_gt_area_ratio_median"][method] = (
                float(np.median(ratios)) if ratios.size else None
            )
            if ratios.size:
                item["predicted_area_direction_percent"][method] = {
                    "over_gt": 100.0 * float(np.mean(ratios > 1.0)),
                    "under_gt": 100.0 * float(np.mean(ratios < 1.0)),
                    "equal_gt": 100.0 * float(np.mean(ratios == 1.0)),
                }
            else:
                item["predicted_area_direction_percent"][method] = None
        output_bins.append(item)

    seq6 = []
    for row in sequence_rows["seq6"]:
        if row["frame"] in {50, 51, 52, 53, 81, 82}:
            seq6.append({
                **row,
                "gt_area_percent": 100.0 * row["area_fraction"],
                "regional_area_direction": (
                    "over" if row["regional_area_ratio"] > 1
                    else "under" if row["regional_area_ratio"] < 1 else "equal"
                ),
            })

    return {
        "analysis": "post-hoc GT-size strata on full-video saved trajectories",
        "inference_rerun": False,
        "inferential_scope": "descriptive; sequence is the independent unit for the reported bootstrap intervals",
        "run_dir": str(run_dir),
        "manifest_sha256": setup["manifest_sha256"],
        "results_sha256": sha256(results_path),
        "integrity_audit_path": str(audit_path),
        "integrity_audit_sha256": sha256(audit_path),
        "integrity_audit_status": integrity_audit["status"],
        "gt_masks_used": gt_masks_used,
        "eligible_sequences": SEQUENCES,
        "compressed_present_targets_over_six_history_maps": eligible_target_count,
        "area_bin_definition": [
            {"label": name, "min_fraction_inclusive": lo, "max_fraction_exclusive": hi}
            for name, lo, hi in AREA_BINS
        ],
        "area_bins": output_bins,
        "seq6_examples": seq6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "compressed_present_targets_over_six_history_maps": result["compressed_present_targets_over_six_history_maps"],
        "area_bins": result["area_bins"],
        "seq6_examples": result["seq6_examples"],
    }, indent=2))


if __name__ == "__main__":
    main()
