"""Compare two fresh-process ToMe replays on a fixed real PolypGen clip."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_position_consistent_tome_gate import PROTOCOL, run_condition
from scripts.run_regional_refinement import write


SEQUENCE = "seq2"
TARGET_FRAMES = 2


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_receipt():
    paths = (
        "modeling/position_consistent_tome.py",
        "scripts/run_position_consistent_tome_gate.py",
        "scripts/verify_position_consistent_tome_replay.py",
        "external/ToMe/tome/merge.py",
    )
    return {name: digest(ROOT / name) for name in paths}


def trial(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if digest(args.checkpoint) != protocol["source_and_assets"]["sam2_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the frozen ToMe protocol")
    manifest = ROOT / protocol["source_and_assets"]["manifest"]
    videos = load_manifest(manifest, "dev")
    frames = videos[SEQUENCE]
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    final_target = min(len(frames) - 1, prompt + TARGET_FRAMES)
    inputs = {}
    for frame in frames[prompt:final_target + 1]:
        inputs[f"image:{frame.frame_index}"] = digest(frame.image_path)
        inputs[f"mask:{frame.frame_index}"] = digest(frame.mask_path)
    repo, revision = activate_author_repo()
    if revision != protocol["source_and_assets"]["sam2_revision"]:
        raise RuntimeError("SAM2 checkout differs from the frozen protocol")
    hashes = source_receipt()
    write(output / "setup.json", {
        "protocol_sha256": digest(ROOT / PROTOCOL),
        "manifest_sha256": digest(manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "sam2_revision": revision,
        "tome_revision": protocol["source_and_assets"]["tome_revision"],
        "inputs": inputs,
        "sources": hashes,
        "sequence": SEQUENCE,
        "prompt": prompt,
        "target_frames": list(range(prompt + 1, final_target + 1)),
        "device": "cpu",
        "dtype": "float32",
    })
    rows, run = run_condition(
        SEQUENCE,
        "region_constrained_tome",
        frames,
        prompt,
        {},
        args.checkpoint,
        protocol["source_and_assets"]["sam2_config"],
        repo,
        output,
        target_limit=TARGET_FRAMES,
    )
    if len(rows) != TARGET_FRAMES:
        raise RuntimeError(f"Fixed replay target count differs: {len(rows)}")
    if hashes != source_receipt():
        raise RuntimeError("Source changed during fixed real-clip replay")
    write(output / "replay.json", {
        "setup_sha256": digest(output / "setup.json"),
        "frames": rows,
        "parameter_digest": run["parameter_digest"],
        "parameters_unchanged": run["parameters_unchanged"],
        "attention_layer_calls": run["attention_layer_calls"],
    })
    write(output / "completion.json", {"passed": True})


def orchestrate(args):
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_sha = source_receipt()
    write(output / "protocol.json", json.loads((ROOT / PROTOCOL).read_text()))
    trials = []
    for index in (1, 2):
        trial_dir = output / f"trial_{index:02d}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(trial_dir),
            "--trial",
        ]
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        environment["OMP_NUM_THREADS"] = "2"
        environment["MKL_NUM_THREADS"] = "2"
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        completion = json.loads((trial_dir / "completion.json").read_text())
        replay = json.loads((trial_dir / "replay.json").read_text())
        setup = json.loads((trial_dir / "setup.json").read_text())
        if completion.get("passed") is not True:
            raise RuntimeError(f"Fresh-process replay {index} did not complete")
        rows_bytes = json.dumps(replay["frames"], sort_keys=True, separators=(",", ":")).encode()
        provenance_path = trial_dir / "first_read_cluster_provenance.json"
        trials.append({
            "trial": index,
            "replay_sha256": sha256_bytes(rows_bytes),
            "first_read_cluster_provenance_sha256": digest(provenance_path),
            "setup_sha256": replay["setup_sha256"],
            "prompt": setup["prompt"],
            "targets": [row["frame"] for row in replay["frames"]],
            "attention_layer_calls": replay["attention_layer_calls"],
        })
    if source_sha != source_receipt():
        raise RuntimeError("Source changed during deterministic replay comparison")
    if trials[0]["replay_sha256"] != trials[1]["replay_sha256"]:
        raise RuntimeError("Fresh-process merge assignments/tensors/outputs are not deterministic")
    if trials[0]["first_read_cluster_provenance_sha256"] != trials[1]["first_read_cluster_provenance_sha256"]:
        raise RuntimeError("Fresh-process source-token cluster provenance differs")
    if trials[0]["setup_sha256"] != trials[1]["setup_sha256"] or trials[0]["prompt"] != trials[1]["prompt"]:
        raise RuntimeError("Fresh-process replay inputs or prompt differ")
    expected_targets = list(range(trials[0]["prompt"] + 1, trials[0]["prompt"] + TARGET_FRAMES + 1))
    expected_layer_calls = TARGET_FRAMES * 4
    if any(trial["targets"] != expected_targets for trial in trials) or any(
        trial["attention_layer_calls"] != expected_layer_calls for trial in trials
    ):
        raise RuntimeError(
            "Fixed replay did not cover the manifest-derived target frames and all four layers"
        )
    write(output / "comparison.json", {
        "passed": True,
        "scope": (
            f"Two fresh Python processes, seq2 prompt frame {trials[0]['prompt']} "
            f"plus target frames {expected_targets}, regional ToMe; engineering only."
        ),
        "trials": trials,
        "sources": source_sha,
        "exact_replay_match": True,
    })
    write(output / "completion.json", {
        "passed": True,
        "protocol_sha256": digest(ROOT / PROTOCOL),
        "checkpoint_sha256": digest(args.checkpoint),
        "method_module_sha256": source_sha["modeling/position_consistent_tome.py"],
        "gate_script_sha256": source_sha["scripts/run_position_consistent_tome_gate.py"],
        "tome_source_sha256": source_sha["external/ToMe/tome/merge.py"],
        "comparison_sha256": digest(output / "comparison.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial", action="store_true")
    args = parser.parse_args()
    if args.trial:
        trial(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
