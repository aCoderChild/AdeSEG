"""Compare role assignments with the exact probability map SAM2 encodes in memory."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.soft_role_affinity import mask_logits_to_soft_roles
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write


def roles_from_memory_mask(mask_input, sigmoid_scale, sigmoid_bias):
    """Apply the Run 123 soft mixed-cell partition to 16x16 memory-mask cells."""
    mask_input = torch.as_tensor(mask_input).detach().float().cpu()
    probability = (mask_input - float(sigmoid_bias)) / float(sigmoid_scale)
    while probability.ndim > 2 and probability.shape[0] == 1:
        probability = probability.squeeze(0)
    if probability.shape != (1024, 1024) or not torch.isfinite(probability).all():
        raise ValueError("Expected the finite 1024x1024 mask passed to SAM2 memory encoder")
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError("SAM2 memory-encoder mask probabilities must be in [0, 1]")
    cells = probability.reshape(64, 16, 64, 16).permute(0, 2, 1, 3).reshape(64, 64, 256)
    mean = cells.mean(-1)
    boundary = (4 * cells.var(-1, unbiased=False)).clamp(0, 1)
    roles = torch.stack((
        (1 - mean) * (1 - boundary),
        mean * (1 - boundary),
        boundary,
    ), -1)
    if not torch.allclose(roles.sum(-1), torch.ones_like(mean), atol=1e-6):
        raise RuntimeError("Memory-input soft roles do not sum to one")
    return roles


def compare_roles(low_resolution, memory_mask_input, model):
    low = mask_logits_to_soft_roles(low_resolution).reshape(4096, 3).float().cpu()
    high = roles_from_memory_mask(
        memory_mask_input,
        model.sigmoid_scale_for_mem_enc,
        model.sigmoid_bias_for_mem_enc,
    ).reshape(4096, 3)
    agreement = (low.argmax(-1) == high.argmax(-1)).float().mean()
    return {
        "lowres_token_role_fraction": torch.bincount(low.argmax(-1), minlength=3).float().div(4096).tolist(),
        "memory_input_token_role_fraction": torch.bincount(high.argmax(-1), minlength=3).float().div(4096).tolist(),
        "argmax_role_agreement": float(agreement),
        "mean_absolute_membership_difference": float((low - high).abs().mean()),
        "mean_total_variation_per_token": float(0.5 * (low - high).abs().sum(-1).mean()),
    }


def audit_sequence(model, frames, sequence, temporary_dir, native_reference):
    state, prompt = prepare(model, frames, temporary_dir)
    active_frame = {"index": None}
    encoded_masks = {}
    original_track_step = model.track_step
    original_run_memory_encoder = model._run_memory_encoder

    def track_step(*args, **kwargs):
        active_frame["index"] = int(kwargs["frame_idx"])
        try:
            return original_track_step(*args, **kwargs)
        finally:
            active_frame["index"] = None

    def run_memory_encoder(*args, **kwargs):
        frame_index = kwargs.get("frame_idx", args[1] if len(args) > 1 else None)
        active_frame["index"] = int(frame_index)
        try:
            return original_run_memory_encoder(*args, **kwargs)
        finally:
            active_frame["index"] = None

    def memory_encoder_pre_hook(module, args, kwargs):
        frame_index = active_frame["index"]
        if frame_index is None:
            raise RuntimeError("Could not associate a memory-encoder mask with its video frame")
        mask_probability = kwargs.get("masks", args[1] if len(args) > 1 else None)
        if mask_probability is None:
            raise RuntimeError("SAM2 memory encoder did not receive its mask input")
        mask_probability = mask_probability.detach().float().cpu().clone()
        if frame_index in encoded_masks:
            raise RuntimeError(f"Multiple memory-mask encodings for frame {frame_index}")
        encoded_masks[frame_index] = mask_probability

    model.track_step = track_step
    model._run_memory_encoder = run_memory_encoder
    handle = model.memory_encoder.register_forward_pre_hook(memory_encoder_pre_hook, with_kwargs=True)
    rows = []
    try:
        with torch.inference_mode():
            for frame_index, object_ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if object_ids != [1]:
                    raise RuntimeError("Unexpected object IDs in single-polyp sequence")
                output_dict = state["output_dict_per_obj"][0]
                stored = output_dict["cond_frame_outputs"].get(frame_index)
                if stored is None:
                    stored = output_dict["non_cond_frame_outputs"].get(frame_index)
                if stored is None or stored.get("maskmem_features") is None:
                    raise RuntimeError(f"Frame {frame_index} has no stored memory feature")
                memory_probability = encoded_masks.get(frame_index)
                if memory_probability is None:
                    raise RuntimeError(f"Frame {frame_index} has no captured memory-encoder mask")
                low = stored["pred_masks"]
                row = compare_roles(low, memory_probability, model)
                row.update({
                    "sequence": sequence,
                    "frame": frame_index,
                    "is_prompt": frame_index == prompt,
                    "prediction_sha256": tensor_hash(logits),
                    "mask_input_shape": list(memory_probability.shape),
                    "stored_memory_shape": list(stored["maskmem_features"].shape),
                    "lowres_prediction_shape": list(low.shape),
                })
                if frame_index != prompt:
                    reference_hash = native_reference.get(frame_index)
                    if reference_hash is None:
                        raise RuntimeError(f"Native baseline is missing {sequence}/{frame_index}")
                    row["native_prediction_sha256"] = reference_hash
                    row["native_replay_bit_identical"] = row["prediction_sha256"] == reference_hash
                    if not row["native_replay_bit_identical"]:
                        raise RuntimeError(f"Read-only mask hook changed native prediction {sequence}/{frame_index}")
                else:
                    row["native_prediction_sha256"] = None
                    row["native_replay_bit_identical"] = None
                rows.append(row)
    finally:
        handle.remove()
        model.track_step = original_track_step
        model._run_memory_encoder = original_run_memory_encoder
        model.reset_state(state)
    expected = list(range(prompt, len(frames)))
    if [row["frame"] for row in rows] != expected:
        raise RuntimeError("Alignment audit did not cover every stored frame in order")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--native-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sequence", action="append", default=[])
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if not args.sequence:
        args.sequence = ["seq2", "seq3"]
    videos = load_manifest(args.manifest, "dev")
    if any(sequence not in videos for sequence in args.sequence):
        raise ValueError("Requested sequence is missing from the PolypGen manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo, revision = activate_author_repo()
    model_args = argparse.Namespace(
        method="sam2", device=args.device, model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint=args.checkpoint,
    )
    model = build_author_predictor(model_args, repo)
    model.requires_grad_(False)
    report = {
        "method": "Run 123 soft-region membership computed from saved low-res predicted logits",
        "reference": "same frame's actual mask probability tensor passed into SAM2 MemoryEncoder",
        "partition": "background=(1-mean)*(1-4*variance), foreground=mean*(1-4*variance), boundary=4*variance; 4x4 low-res cells versus corresponding 16x16 1024-grid cells",
        "memory_mask_unscale": {
            "sigmoid_scale_for_mem_enc": model.sigmoid_scale_for_mem_enc,
            "sigmoid_bias_for_mem_enc": model.sigmoid_bias_for_mem_enc,
        },
        "sam2_revision": revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "device": args.device,
        "sequences": {},
        "scope": "Full post-prompt stored-memory trajectories for seq2 and seq3; assignment-alignment audit only, no method intervention.",
    }
    for sequence in args.sequence:
        reference_path = args.native_reference_dir / f"{sequence}_frames.json"
        reference_rows = json.loads(reference_path.read_text())
        reference = {int(row["frame"]): row["prediction_sha256"] for row in reference_rows}
        with tempfile.TemporaryDirectory(prefix=f"{sequence}_", dir=args.output_dir) as temporary:
            rows = audit_sequence(model, videos[sequence], sequence, Path(temporary), reference)
        report["sequences"][sequence] = rows
        write(args.output_dir / f"{sequence}.json", rows)
    write(args.output_dir / "alignment_audit.json", report)
    print(json.dumps({
        sequence: {
            "frames": len(rows),
            "mean_argmax_agreement": sum(row["argmax_role_agreement"] for row in rows) / len(rows),
            "mean_total_variation": sum(row["mean_total_variation_per_token"] for row in rows) / len(rows),
        }
        for sequence, rows in report["sequences"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
