#!/usr/bin/env python3
"""Audit labeled-frame initialization eligibility without changing a protocol."""

# ruff: noqa: E402 -- direct CLI execution
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adenoid.io import load_label_mask
from datasets.video import load_manifest


def prompt_index(frames, protocol, label_id=1):
    """Honor the frozen frame-zero protocol or its explicitly approved amendment."""
    candidates = (
        range(len(frames))
        if protocol.get("initialization") == "first_labeled_frame"
        else (0,)
    )
    for index in candidates:
        if (load_label_mask(frames[index].mask_path, [label_id]) == label_id).any():
            return index
    raise ValueError("No eligible labeled prompt under the selected protocol")


def target_indices(frame_count, prompt, count):
    eligible = np.arange(prompt + 7, frame_count)
    if not eligible.size:
        raise ValueError("Too short for a filled-bank pilot after initialization")
    return eligible[
        np.unique(
            np.linspace(0, len(eligible) - 1, min(count, len(eligible))).astype(int)
        )
    ].tolist()


def audit_initialization(videos, label_id=1):
    audit = []
    for video, frames in videos.items():
        first_visible = next(
            (
                i
                for i, frame in enumerate(frames)
                if (load_label_mask(frame.mask_path, [label_id]) == label_id).any()
            ),
            None,
        )
        raw = np.asarray(Image.open(frames[0].mask_path))
        audit.append(
            {
                "video": video,
                "frames": len(frames),
                "first_visible_frame": first_visible,
                "first_frame_raw_min": int(raw.min()),
                "first_frame_raw_max": int(raw.max()),
                "first_frame_mask": str(frames[0].mask_path),
            }
        )
    return {
        "source": "initialization audit; no prompt change or inference",
        "sequences": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_initialization(load_manifest(args.manifest, args.split))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
