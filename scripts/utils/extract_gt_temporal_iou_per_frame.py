#!/usr/bin/env python3
"""Extract per-frame temporal IoU summaries from all-pairs ground-truth CSV."""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


PairKey = Tuple[int, int]


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        (key or "").strip(): (value or "").strip()
        for key, value in row.items()
    }


def read_all_pairs(path: str) -> Dict[str, Dict[str, object]]:
    sequences: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "frames": {},
            "pair_iou": {},
            "neighbors": defaultdict(list),
        }
    )
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw_row in reader:
            row = clean_row(raw_row)
            seq = row["sequence"]
            prev_idx = int(row["previous_index"])
            curr_idx = int(row["current_index"])
            prev_frame = row["previous_frame"]
            curr_frame = row["current_frame"]
            iou = float(row["gt_temporal_iou"])

            seq_data = sequences[seq]
            frames = seq_data["frames"]
            pair_iou = seq_data["pair_iou"]
            neighbors = seq_data["neighbors"]

            frames[prev_idx] = prev_frame
            frames[curr_idx] = curr_frame
            pair_iou[(prev_idx, curr_idx)] = iou
            neighbors[prev_idx].append((curr_idx, curr_frame, iou))
            neighbors[curr_idx].append((prev_idx, prev_frame, iou))

    return sequences


def best_neighbor(neighbors: List[Tuple[int, str, float]], find_max: bool) -> Tuple[str, str, str]:
    if not neighbors:
        return "", "", ""
    if find_max:
        idx, frame, iou = max(neighbors, key=lambda item: (item[2], -item[0]))
    else:
        idx, frame, iou = min(neighbors, key=lambda item: (item[2], item[0]))
    return str(idx), frame, str(iou)


def build_sequence_rows(seq_data: Dict[str, object], max_previous: int) -> List[Dict[str, object]]:
    frames: Dict[int, str] = seq_data["frames"]
    pair_iou: Dict[PairKey, float] = seq_data["pair_iou"]
    neighbors = seq_data["neighbors"]

    rows: List[Dict[str, object]] = []
    for frame_idx in sorted(frames):
        max_idx, max_frame, max_iou = best_neighbor(neighbors[frame_idx], find_max=True)
        min_idx, min_frame, min_iou = best_neighbor(neighbors[frame_idx], find_max=False)
        row: Dict[str, object] = {
            "frame_index": frame_idx,
            "frame": frames[frame_idx],
            "max_iou_frame_index": max_idx,
            "max_iou_frame": max_frame,
            "max_gt_temporal_iou": max_iou,
            "min_iou_frame_index": min_idx,
            "min_iou_frame": min_frame,
            "min_gt_temporal_iou": min_iou,
        }

        previous_indices = list(range(max(0, frame_idx - max_previous), frame_idx))
        previous_indices = previous_indices[-max_previous:]
        previous_indices_most_recent_first = list(reversed(previous_indices))
        for offset in range(1, max_previous + 1):
            if offset <= len(previous_indices_most_recent_first):
                prev_idx = previous_indices_most_recent_first[offset - 1]
                key = (prev_idx, frame_idx)
                row[f"prev_{offset}_frame_index"] = prev_idx
                row[f"prev_{offset}_frame"] = frames.get(prev_idx, "")
                row[f"prev_{offset}_gt_temporal_iou"] = pair_iou.get(key, "")
            else:
                row[f"prev_{offset}_frame_index"] = ""
                row[f"prev_{offset}_frame"] = ""
                row[f"prev_{offset}_gt_temporal_iou"] = ""
        rows.append(row)
    return rows


def fieldnames(max_previous: int) -> List[str]:
    fields = [
        "frame_index",
        "frame",
        "max_iou_frame_index",
        "max_iou_frame",
        "max_gt_temporal_iou",
        "min_iou_frame_index",
        "min_iou_frame",
        "min_gt_temporal_iou",
    ]
    for offset in range(1, max_previous + 1):
        fields.extend(
            [
                f"prev_{offset}_frame_index",
                f"prev_{offset}_frame",
                f"prev_{offset}_gt_temporal_iou",
            ]
        )
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        default="outputs/ground_truth_all_pairs/gt_temporal_iou_pairs.csv",
        help="All-pairs temporal IoU CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/ground_truth_all_pairs_per_frame",
        help="Directory where one CSV per sequence will be written",
    )
    parser.add_argument("--max_previous", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sequences = read_all_pairs(args.input_csv)
    fields = fieldnames(args.max_previous)

    for seq in sorted(sequences, key=lambda value: int(value.replace("seq", ""))):
        rows = build_sequence_rows(sequences[seq], args.max_previous)
        output_path = os.path.join(args.output_dir, f"{seq}.csv")
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
