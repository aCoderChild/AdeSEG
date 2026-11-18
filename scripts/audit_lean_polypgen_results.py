#!/usr/bin/env python3
"""Independently rescore saved Lean/native masks and write comparison tables."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402
from scripts.run_lean_polypgen_full import SEQUENCES, read  # noqa: E402


def check_mask(path, truth, recorded):
    prediction = np.asarray(Image.open(path)) > 0
    if prediction.shape != truth.shape:
        raise ValueError(f"Mask dimensions differ: {path}")
    intersection = int(np.count_nonzero(prediction & truth))
    denominator = int(prediction.sum() + truth.sum())
    dice = 2 * intersection / denominator if denominator else 1.0
    if not np.isclose(dice, recorded["scores"]["dice"], rtol=0, atol=1e-12):
        raise ValueError(f"Dice does not match saved mask: {path}")
    if bool(truth.any()) != recorded["present"]:
        raise ValueError(f"Presence does not match ground truth: {path}")
    if not truth.any() and bool(prediction.any()) != recorded["absent_fp"]:
        raise ValueError(f"False-alarm record does not match: {path}")
    return dice


def audit(attempt, native):
    comparison = read(attempt / "comparison.json")
    setup = read(attempt / "preflight.json")
    if not read(attempt / "full/delivery.json")["completed"]:
        raise ValueError("Lean delivery incomplete")
    by_sequence = {x["sequence"]: x for x in comparison["sequences"]}
    if set(by_sequence) != set(SEQUENCES):
        raise ValueError("Cohort incomplete")
    checked = 0
    for sequence in SEQUENCES:
        paths = [attempt / "full" / sequence, native / sequence]
        if digest(paths[1] / "frames.jsonl") != setup["native_frames_sha256"][sequence]:
            raise ValueError(f"Native frame receipt changed: {sequence}")
        if digest(paths[1] / "inputs.json") != setup["native_inputs_sha256"][sequence]:
            raise ValueError(f"Native input receipt changed: {sequence}")
        inputs = read(paths[0] / "inputs.json")
        native_inputs = read(paths[1] / "inputs.json")
        identity = lambda records: [(x["frame"], x["image_sha256"], x["mask_sha256"]) for x in records]
        if identity(inputs) != identity(native_inputs):
            raise ValueError(f"Input mismatch: {sequence}")
        rows = [[json.loads(x) for x in (path / "frames.jsonl").read_text().splitlines()] for path in paths]
        if [(x["frame"], x["source_frame_index"]) for x in rows[0]] != [(x["frame"], x["source_frame_index"]) for x in rows[1]]:
            raise ValueError(f"Target mismatch: {sequence}")
        values, alarms = [[], []], [0, 0]
        for pair in zip(*rows):
            index = pair[0]["frame"]
            if inputs[index]["frame"] != pair[0]["source_frame_index"]:
                raise ValueError(f"Source index mismatch: {sequence}/{index}")
            mask_path = Path(inputs[index]["mask"])
            if digest(mask_path) != inputs[index]["mask_sha256"]:
                raise ValueError(f"Ground truth changed: {mask_path}")
            truth = load_label_mask(mask_path, [1]) == 1
            for arm, row in enumerate(pair):
                score = check_mask(paths[arm] / f"frame_{index:06d}.png", truth, row)
                if truth.any():
                    values[arm].append(score)
                else:
                    alarms[arm] += int(row["absent_fp"])
            checked += 1
        ref = by_sequence[sequence]
        if len(rows[0]) != ref["targets"] or len(values[0]) != ref["present"] or len(rows[0]) - len(values[0]) != ref["empty"]:
            raise ValueError(f"Denominator mismatch: {sequence}")
        for arm, key in enumerate(("lean", "native")):
            if not np.isclose(np.mean(values[arm]), ref[f"{key}_dice"], atol=1e-12, rtol=0) or alarms[arm] != ref[f"{key}_false_alarms"]:
                raise ValueError(f"Aggregate mismatch: {sequence}/{key}")
        print(f"Rescored {sequence}: {len(rows[0])} paired masks", flush=True)
    if checked != comparison["target_frames"]:
        raise ValueError("Cohort target count mismatch")
    rows = comparison["sequences"]
    if comparison["empty_frames"] != sum(r["empty"] for r in rows):
        raise ValueError("Cohort empty-frame count mismatch")
    for key in ("lean", "native"):
        if comparison[f"{key}_false_alarms"] != sum(r[f"{key}_false_alarms"] for r in rows):
            raise ValueError(f"Cohort false-alarm mismatch: {key}")
        if not np.isclose(comparison[f"{key}_dice"], np.mean([r[f"{key}_dice"] for r in rows]), atol=1e-12, rtol=0):
            raise ValueError(f"Cohort Dice mismatch: {key}")
    deltas = {r["sequence"]: r["lean_dice"] - r["native_dice"] for r in rows}
    if any(not np.isclose(r["delta"], deltas[r["sequence"]], atol=1e-12, rtol=0) for r in rows):
        raise ValueError("Per-sequence delta mismatch")
    if not np.isclose(comparison["mean_delta"], np.mean(list(deltas.values())), atol=1e-12, rtol=0):
        raise ValueError("Cohort delta mismatch")
    if not np.allclose(comparison["sequence_bootstrap_95_ci"], sequence_bootstrap_interval(deltas), atol=1e-12, rtol=0):
        raise ValueError("Bootstrap interval mismatch")
    return comparison, checked


def report(c):
    lines = ["# Lean-SAM2 vs native SAM2.1 Base+ on PolypGen", "",
             "Exploratory paired comparison; previously inspected sequences. "
             "Initial ground-truth mask, CPU FP32, matching checkpoint and target frames.", "",
             "| Method | Equal-sequence present Dice | False alarms / empty targets |",
             "| --- | ---: | ---: |"]
    for name in ("lean", "native"):
        lines.append(f"| {name} | {c[f'{name}_dice']:.5f} | {c[f'{name}_false_alarms']} / {c['empty_frames']} |")
    lo, hi = c["sequence_bootstrap_95_ci"]
    lines += ["", f"Lean minus native: {c['mean_delta']:+.5f}; sequence-bootstrap 95% interval [{lo:+.5f}, {hi:+.5f}].",
              f"{len(c['sequences'])} sequences, {c['target_frames']} post-prompt targets. Seq1 and seq7 were unpromptable.", "",
              "| Sequence | Present / empty | Lean Dice | Native Dice | Delta | Lean false alarms | Native false alarms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for x in c["sequences"]:
        lines.append(f"| {x['sequence']} | {x['present']} / {x['empty']} | {x['lean_dice']:.5f} | {x['native_dice']:.5f} | {x['delta']:+.5f} | {x['lean_false_alarms']} | {x['native_false_alarms']} |")
    lines += ["", "Full-system comparison; it does not isolate a token mechanism. "
              "This CPU study does not establish the authors' CUDA throughput."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--native-dir", type=Path, required=True)
    args = p.parse_args()
    comparison, checked = audit(args.attempt, args.native_dir)
    (args.attempt / "independent_mask_audit.json").write_text(json.dumps({
        "passed": True, "targets": checked, "saved_masks_scored": checked * 2,
        "comparison_sha256": digest(args.attempt / "comparison.json"),
        "audit_source_sha256": digest(Path(__file__)),
    }, indent=2) + "\n")
    (args.attempt / "comparison.md").write_text(report(comparison))


if __name__ == "__main__":
    main()
