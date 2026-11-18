"""Shared strict split check for public training command wrappers."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .split_guard import sha256, validate_train_sources


@contextmanager
def checked_arguments(*, mode):
    # Keep original training implementations/source hashes untouched.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--split-file", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    args, rest = p.parse_known_args()
    if "--help" in rest or "-h" in rest:
        print("PUBLIC RESEARCH RUN: --split-file PATH is REQUIRED.")
        print("Each training manifest row requires source_group and capture_sha256.")
        print("The frozen split file must have train, dev and test arrays.")
        yield None
        return
    if not args.split_file or not args.manifest or not args.output:
        p.error("--split-file, --manifest and --output are required")
    field = "file" if mode == "med" else "tensor_file"
    receipt = validate_train_sources(args.manifest, args.split_file, file_field=field)
    output = args.output.expanduser().resolve()
    sidecar = Path(str(output) + ".split.json")
    if sidecar.exists():
        p.error("Refusing to overwrite an existing split audit")
    prior = sys.argv
    # Restore the original CLI args except the *new* split-lock flag.
    sys.argv = [prior[0], "--manifest", str(args.manifest),
                "--output", str(args.output), *rest]
    try:
        yield (output, sidecar, receipt)
    finally:
        sys.argv = prior


def save_receipt(info):
    output, sidecar, record = info
    if not output.is_file():
        raise FileNotFoundError("No training checkpoint was created")
    record["trained_checkpoint_sha256"] = sha256(output)
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
