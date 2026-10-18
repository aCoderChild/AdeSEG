#!/usr/bin/env python3
"""Build a manifest from the public PolypGen Video Sequence Kaggle layout."""

# ruff: noqa: E402 -- direct CLI execution

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import MASK_EXTENSIONS, resolve_mask_path


def numeric_order(paths):
    return sorted(paths, key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)))


def verified_frame_order(images, sequence, mapping=None, allow_reindexed=False):
    """Renumbered mirror filenames cannot establish original temporal order.

    A supplied map binds each mirror file's SHA256 to its independently
    verified original frame index. The running pinned notebook is unaffected.
    """
    if not images:
        raise ValueError("No image frames found")
    reindexed = [image.stem.isdigit() for image in images]
    if any(reindexed) and not all(reindexed):
        raise ValueError("Mixed original and reindexed filenames cannot establish chronology")
    if not any(reindexed):
        return numeric_order(images), "original_numeric_suffix"
    if mapping is None:
        if allow_reindexed:
            return numeric_order(images), "unverified_reindexed_diagnostic_only"
        raise ValueError(
            "Reindexed mirror filenames do not establish chronology. The seq13 "
            "mirror was found to jump original frames 0,100,104,...,12. Supply "
            "--frame-order-map verified against original source frames, or use "
            "--allow-reindexed-order only for explicitly nonchronological diagnostics."
        )
    entries = mapping["sequences"][sequence]
    indices = []
    for image in images:
        entry = entries[image.name]
        if hashlib.sha256(image.read_bytes()).hexdigest() != entry["image_sha256"]:
            raise ValueError(f"Frame-order map content mismatch: {image}")
        index = entry["original_frame_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Original frame indices must be nonnegative integers")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("Frame-order map is not one-to-one")
    return sorted(
        images, key=lambda image: entries[image.name]["original_frame_index"]
    ), "verified_original_frame_map"


def resolve_sequence_root(data_root: Path) -> Path:
    """Resolve PolypGen's positive_cropped directory across Kaggle mount slugs."""
    data_root = data_root.resolve()
    candidates = [data_root / "positive_cropped", data_root]
    candidates.extend(data_root.glob("*/positive_cropped"))
    candidates.extend(data_root.glob("**/positive_cropped"))
    valid = []
    for candidate in candidates:
        if (
            candidate.is_dir()
            and list(candidate.glob("seq*"))
            and candidate not in valid
        ):
            valid.append(candidate)
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected exactly one PolypGen sequence root below {data_root}; found {valid}"
        )
    return valid[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-order-map", type=Path)
    parser.add_argument(
        "--allow-reindexed-order",
        action="store_true",
        help="Diagnostic-only escape hatch; not a chronological video protocol",
    )
    args = parser.parse_args()
    sequence_root = resolve_sequence_root(args.data_root)
    print(f"Resolved PolypGen sequence root: {sequence_root}")
    rows = []
    mapping = (
        json.loads(args.frame_order_map.read_text()) if args.frame_order_map else None
    )
    sequences = sorted(
        sequence_root.glob("seq*"),
        key=lambda path: int(re.search(r"\d+", path.name).group()),
    )
    for sequence in sequences:
        image_dir = next(
            (
                sequence / name
                for name in ("images", f"images_{sequence.name}")
                if (sequence / name).is_dir()
            ),
            None,
        )
        mask_dir = next(
            (
                sequence / name
                for name in ("masks", "mask", "labels", "gt", f"masks_{sequence.name}")
                if (sequence / name).is_dir()
            ),
            None,
        )
        if image_dir is None or mask_dir is None:
            raise FileNotFoundError(
                f"Could not resolve image/mask directories for {sequence}"
            )
        images = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        ordered, ordering_source = verified_frame_order(
            images, sequence.name, mapping, args.allow_reindexed_order
        )
        for index, image in enumerate(ordered):
            mask = resolve_mask_path(mask_dir, image.stem)
            if mask is None:
                candidates = [
                    mask_dir / f"{image.stem}_mask{extension}"
                    for extension in MASK_EXTENSIONS
                ]
                raise FileNotFoundError(f"No mask for {image}; tried {candidates}")
            rows.append(
                {
                    "split": "test",
                    "video_id": sequence.name,
                    "frame_index": index,
                    "image": str(image.resolve()),
                    "mask": str(mask.resolve()),
                    "temporal_order_source": ordering_source,
                    "chronological_assumption_allowed": ordering_source
                    != "unverified_reindexed_diagnostic_only",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(
        f"Wrote {len(rows)} frames from {len(sequences)} PolypGen sequences to {args.output}"
    )


if __name__ == "__main__":
    main()
