"""Convert video JSONL masks to canonical indexed PNG without changing sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_mask(path, *, kind, labels=(0, 1, 2)):
    """Never guess a semantic palette or silently collapse distinct labels."""
    with Image.open(path) as image:
        data = np.asarray(image)
        fmt = image.format
        if data.ndim == 3:
            if fmt != "JPEG" and any(
                not np.array_equal(data[..., 0], data[..., channel])
                for channel in range(1, min(3, data.shape[-1]))
            ):
                raise ValueError("Color-coded mask requires explicit palette mapping")
            if kind == "semantic":
                raise ValueError("Semantic masks require indexed/grayscale class IDs")
            data = np.asarray(image.convert("L"))
        if data.ndim != 2 or not data.size:
            raise ValueError("Expected a nonempty 2D annotation")
        if kind == "binary":
            if fmt == "JPEG":
                return (data > 127).astype(np.uint8)
            ids = set(np.unique(data).tolist())
            if not ids.issubset({0, 1, 255}):
                raise ValueError(f"Ambiguous binary mask IDs: {sorted(ids)}")
            return (data > 0).astype(np.uint8)
        if kind != "semantic":
            raise ValueError("kind must be binary or semantic")
        valid = set(labels)
        if not valid or any(type(v) is not int or not 0 <= v <= 255 for v in valid):
            raise ValueError("Semantic label IDs must be uint8 integers")
        unknown = set(np.unique(data).tolist()) - valid
        if unknown:
            raise ValueError(f"Unexpected semantic IDs: {sorted(unknown)}")
        return data.astype(np.uint8)


def convert_manifest(manifest, output, *, kind, labels=(0, 1, 2)):
    manifest = Path(manifest).resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing conversion directory")
    rows = []
    seen = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"split", "video_id", "frame_index", "image", "mask"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Line {line_number}: invalid video manifest row")
        split, video, frame = row["split"], row["video_id"], row["frame_index"]
        if (split not in ("train", "dev", "test") or not isinstance(video, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", video)
                or type(frame) is not int or frame < 0):
            raise ValueError(f"Line {line_number}: invalid split/video/frame")
        key = (split, video, frame)
        if key in seen:
            raise ValueError(f"Line {line_number}: duplicate video frame")
        seen.add(key)
        image = (manifest.parent / str(row["image"])).resolve(strict=True)
        mask = (manifest.parent / str(row["mask"])).resolve(strict=True)
        if not image.is_file() or not mask.is_file():
            raise ValueError(f"Line {line_number}: missing source image or mask")
        rows.append((row, image, mask, normalized_mask(mask, kind=kind, labels=labels)))
    if not rows:
        raise ValueError("Empty video manifest")
    output.mkdir(parents=True)
    converted_rows = []
    for row, image, source_mask, values in rows:
        rel = Path("masks") / row["split"] / row["video_id"] / f'{row["frame_index"]:06d}.png'
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values).save(destination)
        converted_rows.append({
            **row, "image": os.path.relpath(image, output), "mask": rel.as_posix(),
            "original_mask_sha256": sha256(source_mask),
            "normalized_mask_sha256": sha256(destination),
        })
    target = output / "manifest.jsonl"
    target.write_text("\n".join(json.dumps(r, sort_keys=True) for r in converted_rows)+"\n")
    (output / "conversion.json").write_text(json.dumps({
        "source_manifest_sha256": sha256(manifest),
        "normalized_manifest_sha256": sha256(target),
        "kind": kind, "labels": list(labels), "frames": len(rows)
    }, indent=2)+"\n")
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", required=True, choices=("binary", "semantic"))
    p.add_argument("--labels", default="0,1,2")
    args = p.parse_args(argv)
    try:
        labels = tuple(int(v.strip()) for v in args.labels.split(","))
    except ValueError as error:
        p.error(str(error))
    print(convert_manifest(args.manifest, args.output, kind=args.kind, labels=labels))


if __name__ == "__main__":
    main()
