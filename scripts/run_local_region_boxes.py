#!/usr/bin/env python3
"""Bounded development search, frozen held-out test, and honest result summary.

Thin orchestration only: all inference is run_medsam2_region_boxes.py; all
statistics are existing evaluation helpers. Never retries against test Dice.
"""
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_region_token_pilot import write_json
from scripts.run_medsam2_region_boxes import check_development_gate
from evaluation.utility_retrieval import paired_wilcoxon, sequence_bootstrap_interval, holm_adjust
from training.tracking import start_wandb, log_results_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zero-validation", type=Path, required=True)
    parser.add_argument("--nonzero-validation", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root / "region_boxes_local_method_v1"
    output.mkdir(exist_ok=False)
    checkpoint = ROOT / "checkpoints/MedSAM2_latest.pt"
    check_development_gate(args.zero_validation, checkpoint, "mps")
    common = [sys.executable, "-u", str(ROOT / "scripts/run_medsam2_region_boxes.py"),
              "--manifest", str(ROOT / "data/polypgen_sequence.jsonl"),
              "--checkpoint", str(checkpoint),
              "--yolo-checkpoint", str(ROOT / "checkpoints/polypgen_yolov8n.pt"),
              "--gate-dir", str(args.zero_validation), "--device", "mps", "--max-frames", "0"]
    deadline = time.monotonic() + 3600
    while not (args.nonzero_validation / "status.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Nonzero validation did not complete; no development/test launch")
        time.sleep(15)
    if json.loads((args.nonzero_validation / "status.json").read_text()).get("state") != "complete":
        raise RuntimeError("Nonzero validation failed")
    dev = [f"seq{i}" for i in range(1, 16)]
    test = [f"seq{i}" for i in range(16, 24)]

    def execute(name, arguments):
        destination = output / name
        write_json(output / "progress.json", {"stage": name, "state": "running", "time_unix": time.time()})
        with (output / f"{name}.log").open("w") as stream:
            try:
                subprocess.run(common + ["--output-dir", str(destination)] + arguments,
                               cwd=ROOT, check=True, env=os.environ.copy(), stdout=stream,
                               stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                write_json(output / "progress.json", {"stage": name, "state": "failed", "time_unix": time.time()})
                raise
        status = json.loads((destination / "status.json").read_text())
        if status.get("state") != "complete":
            raise RuntimeError(f"Incomplete stage: {name}")
        return json.loads((destination / "summary.json").read_text())

    sequences = [item for seq in dev for item in ("--sequence", seq)]
    up = execute("dev_boundary_up", ["--development", "--weights", "1.2", "1", "1", ".9",
                                    "--scale", ".5"] + sequences)
    down = execute("dev_boundary_down", ["--development", "--weights", ".8", "1", "1", ".9",
                   "--scale", ".5", "--arm", "full", "--detections-dir", str(output / "dev_boundary_up")] + sequences)
    if set(up) != set(dev) or set(down) != set(dev):
        raise RuntimeError("Incomplete development sequence coverage")
    scores = {"up": sum(up[v]["full"] for v in dev) / len(dev),
              "down": sum(down[v]["full"] for v in dev) / len(dev)}
    choice = "up" if scores["up"] >= scores["down"] else "down"
    chosen_dev = up if choice == "up" else down
    selection = {"development_sequences": dev, "development_scores": scores,
                 "weights": [1.2 if choice == "up" else .8, 1., 1., .9], "scale": .5,
                 "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 "criterion": "Higher development sequence-mean Dice of two prospectively fixed region candidates; ties choose up",
                 "warning": "Selection is not evidence that region weighting beats controls",
                 "frozen_unix": time.time()}
    write_json(output / "frozen_selection.json", selection)
    heldout = execute("heldout_once", ["--evaluation", "--selection", str(output / "frozen_selection.json")]
                      + [item for seq in test for item in ("--sequence", seq)])
    if set(heldout) != set(test):
        raise RuntimeError("Incomplete held-out coverage")
    arms = ["framewise", "memory", "reliability_only", "full"]
    report = {"selection": selection, "heldout_sequences": test, "development_sequences": dev,
              "full_registered_study_complete": False, "test_score_directed_retries": False,
              "historical_23_sequence_target": .7700031522346479,
              "warning": "23-sequence aggregate includes development sequences and is descriptive, not a held-out score",
              "heldout": {}, "descriptive_23": {}, "comparisons": {}}
    for arm in arms:
        values = {v: heldout[v][arm] for v in test}
        report["heldout"][arm] = {"mean_dice": sum(values.values()) / len(values), "by_sequence": values}
        development = {v: (chosen_dev[v]["full"] if arm == "full" else up[v][arm]) for v in dev}
        combined = {**development, **values}
        report["descriptive_23"][arm] = sum(combined.values()) / len(combined)
    p_values = {}
    for control in arms[:-1]:
        left = report["heldout"]["full"]["by_sequence"]
        right = report["heldout"][control]["by_sequence"]
        differences = {v: left[v] - right[v] for v in test}
        p_values[control] = paired_wilcoxon(left, right)["p"]
        report["comparisons"][control] = {"mean_dice_difference": sum(differences.values()) / len(test),
                                         "ci95": sequence_bootstrap_interval(differences)}
    report["holm_adjusted_p"] = holm_adjust(p_values)
    write_json(output / "results_summary.json", report)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "MedSAM2 region boxes | frozen comparison",
                      "online", "https://forge.coreweave.com/api/wandb", selection, output, ["region-boxes", "heldout-analysis"])
    run.log({f"heldout/{arm}/dice": report["heldout"][arm]["mean_dice"] for arm in arms})
    log_results_artifact(run, output, ["frozen_selection.json", "results_summary.json"])
    run.finish()
    write_json(output / "progress.json", {"stage": "comparison_complete", "state": "complete",
                                          "time_unix": time.time(), "full_registered_study_complete": False})


if __name__ == "__main__":
    main()
