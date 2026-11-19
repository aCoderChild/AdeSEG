"""Prepare a sealed MedSAM2 research contract from an approved DEV/TEST manifest.

This command hashes the real source manifest, original checkpoint and oracle
prompt-frame selections, without loading the large model into VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from adeseg.research.protocol import digest
from adeseg.research.med_video_contract import (
    EVAL_FIXED, MODEL_CONFIG, prompt_records, encode_prompt_records,
    load_med_video_contract,
)
from datasets.video import load_manifest
from modeling.medsam2_latest import verify_medsam2_latest


def prepare(manifest, checkpoint, output, *, split="dev", label_id=1, name="PolypGen"):
    manifest = Path(manifest).resolve(strict=True)
    checkpoint = verify_medsam2_latest(checkpoint)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing protocol directory")
    videos = load_manifest(manifest, split, require_contiguous=True,
                           require_jpeg=True, safe_output_ids=True)
    prompts = encode_prompt_records(prompt_records(videos, label_id))
    output.mkdir(parents=True)
    (output / "prompt_frames.jsonl").write_bytes(prompts)
    evaluation = {
        **EVAL_FIXED, "model_config": MODEL_CONFIG, "label_id": label_id,
        "prompt_frame_manifest_path": "prompt_frames.jsonl",
        "prompt_frame_manifest_sha256": digest(output / "prompt_frames.jsonl"),
    }
    spec = {
        "schema_version": 1, "experiment_id": "med_video_" + split,
        "track": "medsam2_highres",
        "dataset": {"name": name, "manifest": {
            "path": os.path.relpath(manifest, output), "sha256": digest(manifest)}},
        "checkpoint": {"path": os.path.relpath(checkpoint, output),
                       "sha256": digest(checkpoint)},
        "split": split, "initialization": {"source": "oracle_mask", "prompt_budget": 1},
        "evaluation": evaluation,
        "stages": {key: {"command": [], "artifacts": []}
                   for key in ("data", "training", "inference", "evaluation")},
    }
    path = output / "protocol.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    load_med_video_contract(path)  # Fail closed before returning the new protocol.
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--label-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="PolypGen")
    a = parser.parse_args(argv)
    print(prepare(a.manifest, a.checkpoint, a.output, split=a.split,
                  label_id=a.label_id, name=a.dataset_name))


if __name__ == "__main__":
    main()
