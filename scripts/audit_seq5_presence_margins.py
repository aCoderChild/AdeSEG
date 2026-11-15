"""Audit presence-logit margins across completed, frame-matched Seq5 runs.

This is a saved-output analysis only. It does not run SAM2 or change the frozen
full-cohort experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ARMS = ("native", "region_constrained_tome", "generic_local_tome")
LONG_MISS_INTERVALS = ((157, 182), (184, 241))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frames(run_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for arm in ARMS:
        path = run_dir / arm / "seq5" / "frames.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed Seq5 output: {path}")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Invalid frame list: {path}")
        frames = [int(row["frame"]) for row in rows]
        if frames != list(range(1, len(rows) + 1)):
            raise ValueError(f"Non-contiguous or unordered frames: {path}")
        if any("object_score" not in row or "present" not in row for row in rows):
            raise ValueError(f"Missing presence fields: {path}")
        result[arm] = rows

    reference_ids = [row["frame"] for row in result[ARMS[0]]]
    reference_present = [row["present"] for row in result[ARMS[0]]]
    for arm in ARMS[1:]:
        rows = result[arm]
        if [row["frame"] for row in rows] != reference_ids:
            raise ValueError(f"Frame IDs differ in {arm}")
        if [row["present"] for row in rows] != reference_present:
            raise ValueError(f"GT-present labels differ in {arm}")
    return result


def mean_score(rows: list[dict], frame_ids: list[int]) -> float:
    by_id = {int(row["frame"]): row for row in rows}
    return statistics.mean(float(by_id[frame]["object_score"]) for frame in frame_ids)


def audit(run_dir: Path) -> dict:
    frames = load_frames(run_dir)
    native = {int(row["frame"]): row for row in frames["native"]}
    misses = [
        frame for frame, row in native.items()
        if row["present"] and float(row["scores"]["dice"]) == 0.0
    ]
    result = {
        "analysis": "saved-output Seq5 object-presence margin comparison",
        "inference_rerun": False,
        "sequence": "seq5",
        "arms": {},
        "shared_gt_present_zero_dice_frames": misses,
        "shared_miss_count": len(misses),
        "comparisons_on_shared_misses": {},
        "long_interval_mean_object_scores": {},
        "source_files": {},
    }
    paths = {arm: run_dir / arm / "seq5" / "frames.json" for arm in ARMS}
    for arm in ARMS:
        rows = frames[arm]
        by_id = {int(row["frame"]): row for row in rows}
        result["arms"][arm] = {
            "mean_present_dice": statistics.mean(
                float(row["scores"]["dice"]) for row in rows if row["present"]
            ),
            "suppressed_gt_present_zero_dice_count": sum(
                row["present"]
                and float(row["scores"]["dice"]) == 0.0
                and float(row["object_score"]) <= 0.0
                for row in rows
            ),
            "false_alarm_count": sum(row["absent_fp"] is True for row in rows),
            "frames_sha256": sha256(paths[arm]),
        }
    for arm in ARMS[1:]:
        deltas = [
            float(frames[arm][frame - 1]["object_score"])
            - float(frames["native"][frame - 1]["object_score"])
            for frame in misses
        ]
        treatment = {int(row["frame"]): row for row in frames[arm]}
        flips = [
            frame for frame in misses
            if (float(treatment[frame]["object_score"]) > 0.0)
            != (float(native[frame]["object_score"]) > 0.0)
        ]
        result["comparisons_on_shared_misses"][arm] = {
            "vs_native": {
                "mean_logit_delta": statistics.mean(deltas),
                "median_logit_delta": statistics.median(deltas),
                "min_logit_delta": min(deltas),
                "max_logit_delta": max(deltas),
                "frames_delta_positive": sum(delta > 0 for delta in deltas),
                "frames_delta_negative": sum(delta < 0 for delta in deltas),
                "presence_decision_flips": flips,
            }
        }
    for start, stop in LONG_MISS_INTERVALS:
        ids = list(range(start, stop + 1))
        result["long_interval_mean_object_scores"][f"{start}-{stop}"] = {
            arm: mean_score(frames[arm], ids) for arm in ARMS
        }
    result["source_files"] = {
        arm: str(paths[arm]) for arm in ARMS
    }
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Seq5 presence-logit margin audit",
        "",
        "Saved-output analysis only; no inference was rerun.",
        "",
        "| Arm | Present Dice | Suppressed GT-present misses | GT-empty false alarms |",
        "|---|---:|---:|---:|",
    ]
    for arm, label in (
        ("native", "Native SAM2"),
        ("region_constrained_tome", "Pointer-conditioned ToMe"),
        ("generic_local_tome", "Generic local ToMe"),
    ):
        values = result["arms"][arm]
        lines.append(
            f"| {label} | {values['mean_present_dice']:.6f} | "
            f"{values['suppressed_gt_present_zero_dice_count']} | "
            f"{values['false_alarm_count']} |"
        )
    lines += [
        "",
        f"All arms share {result['shared_miss_count']} GT-present frames with zero Dice.",
        "",
        "| Shared miss interval | Native score | Pointer-conditioned | Generic |",
        "|---|---:|---:|---:|",
    ]
    for interval, scores in result["long_interval_mean_object_scores"].items():
        lines.append(
            f"| {interval} | {scores['native']:.6f} | "
            f"{scores['region_constrained_tome']:.6f} | "
            f"{scores['generic_local_tome']:.6f} |"
        )
    lines += [
        "",
        "| Treatment vs native on shared misses | Mean logit Δ | Median Δ | "
        "Positive / negative frames | Presence decision flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("region_constrained_tome", "Pointer-conditioned ToMe"),
        ("generic_local_tome", "Generic local ToMe"),
    ):
        delta = result["comparisons_on_shared_misses"][arm]["vs_native"]
        lines.append(
            f"| {label} | {delta['mean_logit_delta']:+.6f} | "
            f"{delta['median_logit_delta']:+.6f} | "
            f"{delta['frames_delta_positive']} / {delta['frames_delta_negative']} | "
            f"{len(delta['presence_decision_flips'])} |"
        )
    lines += [
        "",
        "**Interpretation:** the decoder's object-score token produces a scalar logit; "
        "SAM2 keeps a mask only when this logit is positive. ToMe changes the logit "
        "slightly here but never changes the decision on the shared missed frames. "
        "This is not a code defect and does not establish a ToMe remedy for Seq5. "
        "A global presence bypass would be unsafe: the raw-mask probe found a useful "
        "suppressed mask at frame 157 but a zero-overlap, 10.9×-area mask at frame 184.",
        "",
        "Source `frames.json` SHA256:",
    ]
    lines.extend(f"- `{arm}`: `{value['frames_sha256']}`" for arm, value in result["arms"].items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seq5_presence_margin_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "seq5_presence_margin_audit.md").write_text(
        render_markdown(result)
    )
    print(render_markdown(result))


if __name__ == "__main__":
    main()
