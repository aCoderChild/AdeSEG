"""Audit TRAIN/DEV/TEST provenance before memory-fusion experiments.

Fail closed on video-ID reuse, cross-partition exact image duplication and
contradictory declared source/patient groups. This checks declared source
identities and image bytes; it CANNOT infer actual patient identity or
near-duplicate frames from visually similar images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.video import load_manifest
from scripts.run_lean_memory_fusion_ablation import file_sha256


def audit_partition_integrity(train_manifest, dev_manifest, test_manifest,
                              split_file, group_map, *, output=None):
    paths = {
        "train": Path(train_manifest).resolve(strict=True),
        "dev": Path(dev_manifest).resolve(strict=True),
        "test": Path(test_manifest).resolve(strict=True),
    }
    splits_path = Path(split_file).resolve(strict=True)
    groups_path = Path(group_map).resolve(strict=True)
    videos = {
        split: load_manifest(
            path, split, require_contiguous=True, require_jpeg=True,
            safe_output_ids=True, require_masks=(split == "train"))
        for split, path in paths.items()
    }
    split_membership = json.loads(splits_path.read_text("utf8"))
    video_groups = json.loads(groups_path.read_text("utf8"))
    if not isinstance(split_membership, dict) or set(split_membership) != set(videos):
        raise ValueError("Source-group split file must contain train/dev/test")
    if any(not isinstance(vals, list) or not vals
           or any(not isinstance(g, str) or not g.strip() for g in vals)
           or len(vals) != len(set(vals))
           for vals in split_membership.values()):
        raise ValueError("Each source partition needs unique nonempty groups")
    sets = {k: set(v) for k, v in split_membership.items()}
    if any(sets[a] & sets[b] for a, b in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Declared source/patient groups overlap between splits")
    video_owner = {
        video: split for split, cohort in videos.items() for video in cohort
    }
    if len(video_owner) != sum(len(v) for v in videos.values()):
        raise ValueError("Video ID appears in multiple partitions")
    if not isinstance(video_groups, dict) or set(video_groups) != set(video_owner):
        raise ValueError("Group-map must identify every video in every partition exactly")
    for video, split in video_owner.items():
        group = video_groups[video]
        if not isinstance(group, str) or group not in sets[split]:
            raise ValueError(
                f"Video {video} source group is missing or in wrong split")
    # Same bytes in different source groups can invalidate independence even
    # when filenames, video IDs, and partition declarations look distinct.
    image_owner = {}
    image_count = 0
    source_path_owner = {}
    for split, cohort in videos.items():
        for video, frames in sorted(cohort.items()):
            for frame in frames:
                image_count += 1
                resolved = str(frame.image_path.resolve(strict=True))
                if resolved in source_path_owner and (
                    source_path_owner[resolved][0] != split
                    or source_path_owner[resolved][1] != video_groups[video]
                ):
                    raise ValueError(
                        "Same physical image path appears in different "
                        "split/source groups")
                source_path_owner[resolved] = (split, video_groups[video])
                digest = file_sha256(frame.image_path)
                owner = image_owner.get(digest)
                if owner is not None and (
                    owner["split"] != split
                    or owner["group"] != video_groups[video]
                ):
                    raise ValueError(
                        "Identical image bytes occur in different "
                        "split/source groups")
                image_owner[digest] = {
                    "split": split, "group": video_groups[video],
                }
    report = {
        "format": "adeseg_source_partition_integrity_v1",
        "manifests_sha256": {
            split: file_sha256(path) for split, path in paths.items()},
        "source_group_split_sha256": file_sha256(splits_path),
        "video_group_map_sha256": file_sha256(groups_path),
        "video_count_by_split": {
            split: len(cohort) for split, cohort in videos.items()},
        "frame_count_by_split": {
            split: sum(len(x) for x in cohort.values())
            for split, cohort in videos.items()},
        "image_count_checked": image_count,
        "unique_image_digests": len(image_owner),
        "declared_groups_by_split": {
            k: sorted(v) for k, v in sets.items()},
        "exact_cross_group_image_duplicates": 0,
        "warning": (
            "Only declared video/source IDs and exact image byte duplicates "
            "are checked. Verify true patient/group independence and "
            "near-duplicate video content against source metadata."),
    }
    if output is not None:
        out = Path(output).resolve()
        if out.exists():
            raise FileExistsError("Refusing to overwrite split audit receipt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", "utf8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True,
                   help="TRAIN annotation-bearing manifest (auditor only)")
    p.add_argument("--dev-image-manifest", type=Path, required=True)
    p.add_argument("--test-image-manifest", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--group-map", type=Path, required=True,
                   help="All video IDs mapped to declared source/patient groups")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    result = audit_partition_integrity(
        a.train_manifest, a.dev_image_manifest, a.test_image_manifest,
        a.split_file, a.group_map, output=a.output)
    print(json.dumps({
        "videos": result["video_count_by_split"],
        "images_checked": result["image_count_checked"],
        "audit": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
