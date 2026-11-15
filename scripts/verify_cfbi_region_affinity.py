"""Real-data engineering gate for CFBI regional matching in SAM2."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_region_affinity import CFBI_REVISION, RegionalAffinityHeads
from modeling.cfbi_region_affinity_training import cfbi_region_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo

PROTOCOL = ROOT / "protocols/cfbi_regional_affinity_polypgen_v2.json"
SOURCES = [
    "protocols/cfbi_regional_affinity_polypgen_v2.json",
    "modeling/cfbi_region_affinity.py",
    "modeling/native_cfbi_region_affinity.py",
    "modeling/cfbi_region_affinity_training.py",
    "scripts/verify_cfbi_region_affinity.py",
    "scripts/run_cfbi_region_affinity_cv.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "modeling/cfbi_region_gate.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(PROTOCOL.read_text())
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    frames = videos["seq5"]
    clip = sample_clip(frames, protocol["training"]["clip_length"],
                       protocol["training"]["seed"])
    args.method = "sam2"
    args.model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    repo, revision = activate_author_repo()
    model = build_author_predictor(args, repo)
    model.requires_grad_(False)
    samples = capture_presence_clip(model, clip, args.output_dir)
    heads = RegionalAffinityHeads()
    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=protocol["training"]["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    zero_loss, zero_calls, zero_deltas = cfbi_region_clip_loss(
        model, heads, samples, "seq5", check_native=True
    )
    zero_loss.backward()
    gradients = []
    for index, head in enumerate(heads.layers):
        grad = head.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f"Missing finite affinity-head gradient in layer {index}")
        gradients.append(float(grad.abs().sum()))
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen SAM2 received gradients")
    optimizer.step()
    heads.eval()
    with torch.no_grad():
        updated_loss, updated_calls, updated_deltas = cfbi_region_clip_loss(
            model, heads, samples, "seq5"
        )
    if len(zero_calls) != 4 * len(samples) or len(updated_calls) != 4 * len(samples):
        raise RuntimeError("Not all four memory-attention layers were observed")
    if any(row["pointer_keys"] <= 0 for row in zero_calls):
        raise RuntimeError("Pointer keys were not present in the observed native read")
    if any(any(present and std <= 1e-6 for present, std in zip(
            row["present_regions"], row["evidence_std"])) for row in zero_calls):
        raise RuntimeError("A present region produced collapsed CFBI matching evidence")
    if any(all(abs(mean - 1.0) <= 1e-6 for mean, present in zip(
            row["evidence_mean"], row["present_regions"]) if present) for row in zero_calls):
        raise RuntimeError("CFBI evidence is saturated at one for every present region")
    if not any(value > 0 for value in updated_deltas):
        raise RuntimeError("Updated regional matcher did not affect propagation")
    if not torch.isfinite(updated_loss):
        raise RuntimeError("Updated clip loss is nonfinite")
    receipt = {
        "passed": True,
        "sequence": "seq5",
        "clip_frames": [frame.frame_index for frame in clip],
        "zero_projection_native_prediction_max_abs_delta": max(zero_deltas),
        "zero_projection_loss": float(zero_loss.detach()),
        "updated_loss": float(updated_loss),
        "updated_prediction_max_abs_delta": max(updated_deltas),
        "layers": 4,
        "pointer_counts": sorted({row["pointer_keys"] for row in zero_calls}),
        "spatial_key_counts": sorted({row["spatial_keys"] for row in zero_calls}),
        "evidence_mean_range": [min(min(row["evidence_mean"]) for row in zero_calls),
                                max(max(row["evidence_mean"]) for row in zero_calls)],
        "evidence_std_min": min(min(row["evidence_std"]) for row in zero_calls),
        "gradient_l1": gradients,
        "cfbi_revision": CFBI_REVISION,
        "sam2_revision": revision,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(PROTOCOL),
        "sources": {path: digest(ROOT / path) for path in SOURCES},
    }
    (args.output_dir / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
