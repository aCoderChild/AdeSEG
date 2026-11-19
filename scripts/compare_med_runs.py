"""Paired native-vs-adapter verification for strict MedSAM2 oracle video runs.

Checks both run receipts, verifies source data/predictions with the evaluator,
and compares identical post-prompt frames. A zero-initialized control should
preserve both binary masks and float32 logits *on the same deterministic setup*.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_med_video import evaluate_run
from adeseg.research.med_video_contract import load_med_video_contract


def compare_runs(manifest, native_root, candidate_root, *, require_logit_identity=False,
                 protocol=None):
    native_root = Path(native_root).resolve(strict=True)
    candidate_root = Path(candidate_root).resolve(strict=True)
    if native_root == candidate_root:
        raise ValueError("Comparison runs must use independent output directories")
    native = json.loads((native_root / "run.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_root / "run.json").read_text(encoding="utf-8"))
    controlled = ("protocol", "research_protocol_sha256",
                  "prompt_manifest_sha256", "evaluation_contract",
                  "split", "manifest_sha256", "backbone_sha256",
                  "model_config", "label_id", "device", "postprocessing")
    mismatched = [field for field in controlled if native.get(field) != candidate.get(field)]
    if mismatched:
        raise ValueError(f"Comparison protocol mismatch: {mismatched}")
    if native.get("adapter_mode") != "native":
        raise ValueError("Reference run must explicitly declare native mode")
    mode = candidate.get("adapter_mode")
    if mode not in ("zero_initialized_control", "trained"):
        raise ValueError("Candidate must explicitly declare zero control or trained adapter")
    if mode == "zero_initialized_control" and candidate.get("adapter_sha256"):
        raise ValueError("Zero control cannot claim a trained adapter")
    if mode == "trained" and not candidate.get("adapter_sha256"):
        raise ValueError("Trained candidate must declare the adapter checkpoint SHA256")
    if require_logit_identity and mode != "zero_initialized_control":
        raise ValueError("Exact logit identity is only expected for zero-initialized control")

    left_summary, left_frames, left_sequences = evaluate_run(
        manifest, native_root, protocol=protocol)
    right_summary, right_frames, right_sequences = evaluate_run(
        manifest, candidate_root, protocol=protocol)
    if native["results"].keys() != candidate["results"].keys():
        raise ValueError("Video populations differ")
    exact_mask, exact_logit = True, True
    compared = 0
    for name in sorted(native["results"]):
        a, b = native["results"][name], candidate["results"][name]
        if a["status"] != b["status"]:
            raise ValueError(f"Promptable status differs: {name}")
        if a["status"] == "unpromptable":
            continue
        for key in ("prompt_source", "prompt_position", "prompt_source_index",
                    "prompt_mask_sha256", "prompt_image_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Prompt provenance differs for {name}: {key}")
        if len(a["predictions"]) != len(b["predictions"]):
            raise ValueError(f"Prediction lengths differ: {name}")
        for p, q in zip(a["predictions"], b["predictions"]):
            for key in ("frame_index", "source_image_sha256", "source_gt_sha256"):
                if p.get(key) != q.get(key):
                    raise ValueError(f"Frame provenance differs for {name}: {key}")
            exact_mask &= p.get("prediction_sha256") == q.get("prediction_sha256")
            x, y = p.get("logit_sha256"), q.get("logit_sha256")
            exact_logit &= bool(x) and bool(y) and x == y
            compared += 1
    if compared == 0:
        raise ValueError("No post-prompt frames available for comparison")
    left_by_frame = {(x["sequence"], x["frame"]): x for x in left_frames}
    right_by_frame = {(x["sequence"], x["frame"]): x for x in right_frames}
    if left_by_frame.keys() != right_by_frame.keys():
        raise ValueError("Evaluation frame populations differ")
    if any(left_by_frame[key]["gt_present"] != right_by_frame[key]["gt_present"]
           for key in left_by_frame):
        raise ValueError("Ground-truth presence changed")
    left_by_seq = {x["sequence"]: x for x in left_sequences}
    right_by_seq = {x["sequence"]: x for x in right_sequences}
    if left_by_seq.keys() != right_by_seq.keys():
        raise ValueError("Evaluation sequence populations differ")
    paired = {}
    for name in sorted(left_by_seq):
        a, b = left_by_seq[name], right_by_seq[name]
        if (a["frames"], a["present_frames"], a["empty_frames"]) != (
                b["frames"], b["present_frames"], b["empty_frames"]):
            raise ValueError(f"Sequence frame eligibility differs: {name}")
        if np.isfinite(a["dice"]) != np.isfinite(b["dice"]):
            raise ValueError(f"Sequence present-mask eligibility differs: {name}")
        paired[name] = {
            "present_frames": a["present_frames"],
            "present_dice_delta": float(b["dice"] - a["dice"])
                                  if np.isfinite(a["dice"]) else None,
            "empty_false_alarm_delta": int(b["empty_false_alarms"] -
                                           a["empty_false_alarms"]),
        }
    if require_logit_identity:
        if any(not (p.get("logit_file") and q.get("logit_file"))
               for key in native["results"] if native["results"][key]["status"] == "completed"
               for p, q in zip(native["results"][key]["predictions"],
                               candidate["results"][key]["predictions"])):
            raise ValueError("Exact logit identity requires independently saved float32 tensors")
    if require_logit_identity and (not exact_mask or not exact_logit):
        raise ValueError("Zero-adapter parity failed: mask or float32 logit hashes differ")
    deltas = [row["present_dice_delta"] for row in paired.values()
              if row["present_dice_delta"] is not None]
    # The unit of resampling is an independently verified sequence/source group,
    # not individual correlated video frames. Here independence is NOT proven.
    if deltas:
        values = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(0)
        sampled = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
        bootstrap_interval = [float(x) for x in np.quantile(sampled, (0.025, 0.975))]
    else:
        bootstrap_interval = None
    return {
        "protocol": native["protocol"],
        "candidate_mode": mode,
        "post_prompt_frames": compared,
        "identical_binary_masks": bool(exact_mask),
        "identical_float32_logits": bool(exact_logit),
        "baseline": left_summary,
        "candidate": right_summary,
        "paired_present_dice_mean_delta": float(np.mean(deltas)) if deltas else None,
        "eligible_paired_present_sequences": len(deltas),
        "exploratory_sequence_bootstrap_95ci": bootstrap_interval,
        "per_sequence": paired,
        "interpretation": (
            "Exact parity is a numerical/implementation check, not proof of clinical efficacy. "
            "Sequence-bootstrap uncertainty assumes independent source groups; "
            "an exploratory short-video comparison cannot establish generalization."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-logit-identity", action="store_true")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Refusing to overwrite existing comparison")
    contract = load_med_video_contract(args.protocol)
    result = compare_runs(contract["manifest"], args.native, args.candidate,
                          require_logit_identity=args.require_logit_identity,
                          protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
