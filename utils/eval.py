"""Shared evaluation runners for experiment scripts."""

import re

from utils.eval_metrics import evaluate_bbox_sequence, evaluate_mask_sequence


def extract_seq_num(sequence_name):
    """Extract integer sequence number from names such as seq5."""
    seq_match = re.search(r"seq(\d+)", str(sequence_name))
    return int(seq_match.group(1)) if seq_match else None


def run_sequence_evaluation(
    sequence_names=None,
    seq_nums=None,
    data_root="data/polypgen",
    bbox_root="data/bbox",
    output_root="outputs",
    method_name="MedSAM2",
    prompt_source="yolo",
    verbose=True,
):
    """Run bbox and mask evaluation for one or more sequences."""
    if seq_nums is None:
        seq_nums = []
        for sequence_name in sequence_names or []:
            seq_num = extract_seq_num(sequence_name)
            if seq_num is None:
                print(f"Warning: Could not extract sequence number from {sequence_name}")
                continue
            seq_nums.append(seq_num)

    print(f"\n{'='*60}")
    print(f"Running Evaluation on {len(seq_nums)} sequences")
    print(f"{'='*60}")
    for seq_num in seq_nums:
        try:
            if prompt_source == "yolo":
                evaluate_bbox_sequence(
                    seq_num=seq_num,
                    data_root=data_root,
                    bbox_root=bbox_root,
                    output_root=output_root,
                    method_name=method_name,
                    verbose=verbose,
                )
            evaluate_mask_sequence(
                seq_num=seq_num,
                data_root=data_root,
                bbox_root=bbox_root,
                output_root=output_root,
                method_name=method_name,
                save_bbox_overlays=prompt_source == "yolo",
                verbose=verbose,
            )
        except Exception as e:
            print(f"Error evaluating seq{seq_num}: {e}")
    print(f"{'='*60}\n")
