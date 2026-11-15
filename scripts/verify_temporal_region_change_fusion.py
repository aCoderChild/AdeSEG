"""Run the real-sequence engineering gate for change-supervised regional fusion."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.native_temporal_region_fusion import NativeTemporalRegionFusion
from modeling.temporal_region_fusion_change import (
    ChangeSupervisedRegionHeads,
    temporal_change_loss,
)
from modeling.temporal_region_training import temporal_region_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write

PROTOCOL = ROOT / "protocols/temporal_region_change_fusion_polypgen_v1.json"
SOURCES = [
    "protocols/temporal_region_change_fusion_polypgen_v1.json",
    "scripts/verify_temporal_region_change_fusion.py",
    "modeling/temporal_region_fusion_change.py",
    "modeling/matanyone_temporal_uncertainty.py",
    "modeling/native_temporal_region_fusion.py",
    "modeling/temporal_region_fusion.py",
    "modeling/temporal_region_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_readout.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
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
        protocol = json.loads(PROTOCOL.read_text())
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        frames = sample_clip(load_manifest(manifest, "dev")[args.sequence], 5, args.seed)
        repo, revision = activate_author_repo()
        args.method, args.device = "sam2", "cpu"
        args.model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        samples = capture_presence_clip(model, frames, args.output_dir)
        native_hashes = {sample["frame"]: tensor_hash(sample["native_logits"]) for sample in samples}
        prompt_mask = load_label_mask(frames[0].mask_path, [1]) == 1
        prompt_target = torch.from_numpy(prompt_mask.copy())[None, None].float()
        heads = ChangeSupervisedRegionHeads(model.memory_attention.d_model).train()
        if model.memory_attention.d_model != 256:
            raise RuntimeError("Unexpected SAM2 memory-attention width")

        def clip_loss(check_native):
            for head in heads.layers:
                head.change_logits_history.clear()
            segmentation_loss, calls = temporal_region_clip_loss(
                model, heads, samples, args.sequence,
                check_native=check_native, return_trace=True,
            )
            change_loss = temporal_change_loss(heads, samples, prompt_target)
            return segmentation_loss + protocol["training"]["temporal_change_loss_weight"] * change_loss, change_loss, calls

        loss, change_loss, zero_calls = clip_loss(check_native=True)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite zero-residual clip loss")
        loss.backward()
        first_gradients = {"change": [], "mix": []}
        for layer, head in enumerate(heads.layers):
            uncertainty = [parameter.grad for parameter in head.uncertainty.parameters()]
            uncertainty = [gradient for gradient in uncertainty if gradient is not None]
            if not uncertainty or any(not torch.isfinite(gradient).all() for gradient in uncertainty):
                raise RuntimeError(f"Change-predictor gradient invalid at layer {layer}")
            change_norm = sum(float(gradient.abs().sum()) for gradient in uncertainty)
            mix_gradient = head.mix_logit.grad
            if change_norm <= 0 or mix_gradient is None or not torch.isfinite(mix_gradient) or mix_gradient.abs() == 0:
                raise RuntimeError(f"Change or fusion gradient missing at layer {layer}")
            first_gradients["change"].append(change_norm)
            first_gradients["mix"].append(float(mix_gradient.abs()))
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen SAM2 received gradients")

        optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["training"]["learning_rate"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated_loss, updated_change_loss, updated_calls = clip_loss(check_native=False)
        updated_loss.backward()
        projection_gradients = []
        for layer, head in enumerate(heads.layers):
            gradient = head.regional_residual.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise RuntimeError(f"Regional-message gradient failed at layer {layer}")
            projection_gradients.append(float(gradient.abs().sum()))

        heads.eval()
        propagation = []
        with tempfile.TemporaryDirectory(prefix="region_change_gate_", dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            with NativeTemporalRegionFusion(model, heads, args.sequence) as observer, torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if object_ids != [1]:
                        raise RuntimeError("Unexpected propagated object IDs")
                    if frame == prompt:
                        continue
                    calls = observer.calls[-4:]
                    if [row["layer"] for row in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Four-layer trace missing at frame {frame}")
                    if any(row["spatial_keys"] % 4096 or row["pointer_keys"] < 0 for row in calls):
                        raise RuntimeError(f"Invalid spatial/pointer accounting at frame {frame}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite prediction at frame {frame}")
                    propagation.append({
                        "frame": frame,
                        "native_sha256": native_hashes[frame],
                        "regional_sha256": tensor_hash(logits),
                        "changed": tensor_hash(logits) != native_hashes[frame],
                        "layers": calls,
                    })
            model.reset_state(state)
        if len(propagation) != len(samples) or not any(row["changed"] for row in propagation):
            raise RuntimeError("Updated gate failed complete propagation or changed no predictions")
        write(args.output_dir / "completion.json", {
            "passed": True,
            "sequence": args.sequence,
            "frames": [frame.frame_index for frame in frames],
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest),
            "sam2_revision": revision,
            "matanyone_revision": protocol["prior_implementation"]["revision"],
            "zero_residual_native_identity": True,
            "first_update_change_gradient_l1": first_gradients["change"],
            "first_update_mix_gradient_l1": first_gradients["mix"],
            "second_update_regional_projection_gradient_l1": projection_gradients,
            "first_change_loss": float(change_loss.detach()),
            "second_change_loss": float(updated_change_loss.detach()),
            "zero_read_layer_calls": len(zero_calls),
            "updated_read_layer_calls": len(updated_calls),
            "propagation": propagation,
            "sources": {name: digest(ROOT / name) for name in SOURCES},
            "scope": "one five-frame real-data engineering clip, not an accuracy result",
        })
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
