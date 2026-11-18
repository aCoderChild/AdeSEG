"""Validate JSONL video manifests and prohibit train/dev/test subject leakage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def validate_video_manifest(path):
    path = Path(path).resolve(strict=True)
    records = []
    group_split = {}
    video_split = {}
    frame_keys = set()
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        required = {"patient_id", "video_id", "frame_index", "split", "image", "mask"}
        if not isinstance(row, dict) or required-set(row):
            raise ValueError(f"Line {line_number}: missing required video-manifest fields")
        patient, video, split = row["patient_id"], row["video_id"], row["split"]
        if not patient or not video or split not in ("train", "dev", "test"):
            raise ValueError(f"Line {line_number}: invalid patient/video/split")
        if type(row["frame_index"]) is not int or row["frame_index"] < 0:
            raise ValueError(f"Line {line_number}: invalid frame index")
        key = (video, row["frame_index"])
        if key in frame_keys:
            raise ValueError(f"Duplicate frame {key}")
        frame_keys.add(key)
        for identity, lookup, label in ((patient, group_split, "patient"),
                                        (video, video_split, "video")):
            if identity in lookup and lookup[identity] != split:
                raise ValueError(f"{label} leakage across splits: {identity}")
            lookup[identity] = split
        for field in ("image", "mask"):
            candidate = (path.parent / str(row[field])).resolve(strict=True)
            if not candidate.is_file():
                raise ValueError(f"Line {line_number}: missing {field}")
        records.append(row)
    if not records:
        raise ValueError("Empty dataset manifest")
    return {"frames": len(records), "patients": len(group_split),
            "videos": len(video_split),
            "splits": {s: sum(r["split"] == s for r in records)
                       for s in ("train", "dev", "test")}}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(validate_video_manifest(args.manifest), indent=2))

if __name__ == "__main__":
    main()
