#!/usr/bin/env python3
"""Find PolypGen training frames that duplicate a positive-sequence test frame.

The protocol trains on ``data_C1``..``data_C6`` single frames plus
``sequenceData/negativeOnly`` and tests on every ``sequenceData/positive``
sequence. Some single frames are renamed copies of sequence frames, so this
script compares 64-bit difference hashes of the central 80% of each image and
writes the training frames within ``--threshold`` bits of any test frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def difference_hash(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape
    image = image[int(height * 0.1) : int(height * 0.9), int(width * 0.1) : int(width * 0.9)]
    small = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def hashes(paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
    kept, values = [], []
    for path in paths:
        value = difference_hash(path)
        if value is not None:
            kept.append(path)
            values.append(value)
    return np.stack(values), kept


def min_distances(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    distances = np.unpackbits(np.bitwise_xor(queries[:, None], references[None]), axis=2).sum(2)
    return distances.min(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polypgen_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "datasets/splits/polypgen/single_frame_test_overlap.txt",
    )
    args = parser.parse_args()
    root = args.polypgen_root
    test_paths = sorted(root.glob("sequenceData/positive/seq*/images_seq*/*.jpg"))
    train_paths = sorted(
        path for center in range(1, 7) for path in (root / f"data_C{center}" / f"images_C{center}").glob("*.jpg")
    )
    negative_paths = sorted((root / "sequenceData" / "negativeOnly").rglob("*.jpg"))
    if not test_paths or not train_paths:
        sys.exit(f"PolypGen frames not found under {root}")

    test_hashes, test_paths = hashes(test_paths)
    train_hashes, train_paths = hashes(train_paths)
    negative_hashes, negative_paths = hashes(negative_paths)

    train_distance = min_distances(train_hashes, test_hashes)
    excluded = sorted(
        str(path.relative_to(root)) for path, distance in zip(train_paths, train_distance)
        if distance <= args.threshold
    )
    test_distance = min_distances(test_hashes, train_hashes)
    affected = Counter(
        path.parts[-3] for path, distance in zip(test_paths, test_distance) if distance <= args.threshold
    )
    negative_distance = min_distances(negative_hashes, test_hashes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for path in excluded), encoding="utf-8")
    print(json.dumps({
        "threshold_bits": args.threshold,
        "test_frames": len(test_paths),
        "single_frames": len(train_paths),
        "excluded_single_frames": len(excluded),
        "test_frames_with_single_frame_duplicate": dict(affected),
        "negative_frames": len(negative_paths),
        "negative_frames_with_test_duplicate": int((negative_distance <= args.threshold).sum()),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
