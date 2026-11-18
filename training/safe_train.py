"""Small, fail-closed entrypoint for TRAIN-only capture provenance.

Both public CLI commands use this before and after their model trainer.
A declared source group is not independent proof of patient identity.
"""
import argparse
import json
from pathlib import Path
import sys

from training.split_guard import sha256, validate_train_sources


def run_training(trainer, *, mode, argv=None):
    if mode not in ("med", "lean"):
        raise ValueError("Expected med or lean trainer")
    arguments = list(sys.argv[1:] if argv is None else argv)
    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument("--split-file", type=Path)
    guard.add_argument("--manifest", type=Path)
    guard.add_argument("--output", type=Path)
    args, remaining = guard.parse_known_args(arguments)
    if "-h" in remaining or "--help" in remaining:
        print("Training requires --split-file, source_group and capture_sha256.")
        trainer(["--help"])
        return
    if not args.split_file or not args.manifest or not args.output:
        guard.error("--split-file, --manifest and --output are required")
    file_field = "file" if mode == "med" else "tensor_file"
    expected = validate_train_sources(
        args.manifest, args.split_file, file_field=file_field
    )
    output = args.output.expanduser().resolve()
    if output.exists() or Path(str(output) + ".split.json").exists():
        guard.error("Training checkpoint or source receipt already exists")
    forward = [
        "--manifest", str(args.manifest), "--output", str(args.output),
        *remaining,
    ]
    trainer(forward)
    if not output.is_file():
        raise RuntimeError("Trainer did not create the expected checkpoint")
    observed = validate_train_sources(
        args.manifest, args.split_file, file_field=file_field
    )
    if observed != expected:
        raise RuntimeError(
            "Training inputs/split records changed during the run; checkpoint is unverified"
        )
    sidecar = Path(str(output) + ".split.json")
    receipt = dict(observed)
    receipt["trained_checkpoint_sha256"] = sha256(output)
    receipt["training_method"] = mode
    sidecar.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
