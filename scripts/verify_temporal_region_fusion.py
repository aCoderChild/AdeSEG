"""Gate MatAnyone-informed regional residual fusion on a real PolypGen clip."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.temporal_region_training import temporal_region_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from modeling.native_temporal_region_fusion import NativeTemporalRegionFusion, TemporalRegionHeads

PROTOCOL = "protocols/temporal_region_fusion_polypgen_v1.json"
SOURCES = [
    PROTOCOL,
    "modeling/matanyone_temporal_uncertainty.py",
    "modeling/temporal_region_fusion.py",
    "modeling/native_temporal_region_fusion.py",
    "modeling/temporal_region_training.py",
    "scripts/verify_temporal_region_fusion.py",
    "scripts/capture_presence_clip.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_readout.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/MatAnyone/matanyone/model/big_modules.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq5")
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        frames = sample_clip(load_manifest(manifest, "dev")[args.sequence], 5, args.seed)
        repo, revision = activate_author_repo()
        args.method, args.device = "sam2", "cpu"
        args.model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        native = capture_presence_clip(model, frames, args.output_dir)
        native_hashes = {sample["frame"]: tensor_hash(sample["native_logits"]) for sample in native}
        channels = model.memory_attention.d_model
        if channels != 256:
            raise RuntimeError(f"Unexpected SAM2 attention width: {channels}")
        heads = TemporalRegionHeads(channels=channels).train()
        loss, zero_calls = temporal_region_clip_loss(
            model, heads, native, args.sequence, check_native=True, return_trace=True
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite zero-residual training loss")
        loss.backward()
        projection_gradients = []
        for layer, head in enumerate(heads.layers):
            gradient = head.regional_residual.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise RuntimeError(f"Regional projection gradient failed at layer {layer}")
            projection_gradients.append(float(gradient.abs().sum()))
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen SAM2 received gradients")

        optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["training"]["learning_rate"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated_loss, updated_calls = temporal_region_clip_loss(
            model, heads, native, args.sequence, check_native=False, return_trace=True
        )
        updated_loss.backward()
        uncertainty_gradients = []
        for layer, head in enumerate(heads.layers):
            gradients = [p.grad for p in head.uncertainty.parameters()]
            finite = [g for g in gradients if g is not None]
            magnitude = sum(float(g.abs().sum()) for g in finite)
            if not finite or any(not torch.isfinite(g).all() for g in finite) or magnitude <= 0:
                raise RuntimeError(f"Temporal uncertainty gradient failed at layer {layer}")
            uncertainty_gradients.append(magnitude)

        heads.eval()
        propagation = []
        with tempfile.TemporaryDirectory(prefix="temporal_region_gate_", dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            with NativeTemporalRegionFusion(model, heads, args.sequence) as observer, torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if object_ids != [1]:
                        raise RuntimeError("Unexpected propagated object IDs")
                    if frame == prompt:
                        continue
                    calls = observer.calls[-4:]
                    if [row["layer"] for row in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Four-layer attention trace missing at {frame}")
                    if any(row["spatial_keys"] % 4096 or row["pointer_keys"] < 0 for row in calls):
                        raise RuntimeError(f"Invalid spatial or pointer count at {frame}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite propagated mask at {frame}")
                    propagation.append({
                        "frame": frame,
                        "native_sha256": native_hashes[frame],
                        "regional_sha256": tensor_hash(logits),
                        "changed": tensor_hash(logits) != native_hashes[frame],
                        "layers": calls,
                    })
            model.reset_state(state)

        if len(propagation) != len(native) or not any(row["changed"] for row in propagation):
            raise RuntimeError("Updated regional branch did not alter a complete real propagation")
        report = {
            "passed": True,
            "sequence": args.sequence,
            "frames": [frame.frame_index for frame in frames],
            "seed": args.seed,
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest),
            "sam2_revision": revision,
            "matanyone_revision": protocol["prior_implementation"]["revision"],
            "zero_residual_native_identity": True,
            "zero_residual_loss": float(loss.detach()),
            "first_update_projection_gradient_l1": projection_gradients,
            "second_update_uncertainty_gradient_l1": uncertainty_gradients,
            "zero_read_layer_calls": len(zero_calls),
            "updated_read_layer_calls": len(updated_calls),
            "propagation": propagation,
            "sources": {name: digest(ROOT / name) for name in SOURCES},
            "scope": "one five-frame engineering clip; not an accuracy experiment",
        }
        write(args.output_dir / "completion.json", report)
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
