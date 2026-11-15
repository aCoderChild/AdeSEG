#!/usr/bin/env python3
"""Stage a single-prompt PolypGen sequence in TCMN's documented input layout.

This only prepares inputs. The released TCMN evaluation checkout lacks its
inference core, test dataset module, and a trained checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, prompt_index, validate_output  # noqa: E402


def stage_sequence(frames, destination):
    """Only the initial visible GT mask is exposed in Annotations."""
    if not frames:
        raise ValueError("Empty sequence")
    video = frames[0].video_id
    if video in ("", ".", "..") or Path(video).name != video:
        raise ValueError("Unsafe video ID")
    if any(frame.video_id != video for frame in frames):
        raise ValueError("Frames belong to different videos")
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    images = destination / "JPEGImages" / video
    annotations = destination / "Annotations" / video
    images.mkdir(parents=True, exist_ok=False)
    annotations.mkdir(parents=True, exist_ok=False)
    mapping = []
    for index, frame in enumerate(frames[prompt:]):
        name = f"{index:06d}"
        if frame.image_path.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"Expected JPEG input: {frame.image_path}")
        (images / f"{name}.jpg").symlink_to(frame.image_path.resolve())
        mapping.append({"staged_frame": name, "source_frame_index": frame.frame_index,
                        "image": str(frame.image_path.resolve())})
    mask = (load_label_mask(frames[prompt].mask_path, [1]) == 1).astype(np.uint8)
    indexed = Image.fromarray(mask, mode="P")
    indexed.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    indexed.save(annotations / "000000.png")
    return {"video_id": video, "source_prompt_frame": frames[prompt].frame_index,
            "prompt_mask_sha256": digest(frames[prompt].mask_path), "frames": mapping}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    videos = load_manifest(args.manifest, args.split)
    if args.sequence not in videos:
        parser.error("Sequence not found in selected split")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mapping = stage_sequence(videos[args.sequence], output)
    (output / "provenance.json").write_text(json.dumps({
        "purpose": "Input staging only; no inference or efficacy claim",
        "manifest_sha256": digest(args.manifest),
        "tcmn_revision": "3463aa4d605cbf0b4eea49d85a6a7964c84867ee",
        **mapping,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
