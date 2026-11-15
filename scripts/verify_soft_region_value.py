"""Gate continuous region-role memory values on a real PolypGen clip."""
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
from modeling.native_region_readout import NativeRegionalReadout
from modeling.paired_readout_training import clip_loss
from modeling.soft_region_roles import predicted_soft_regions
from modeling.soft_region_value_adapter import SoftRegionValueHeads
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = ROOT / "protocols/soft_region_value_polypgen_v1.json"
SOURCES = [
    PROTOCOL.relative_to(ROOT).as_posix(),
    "scripts/verify_soft_region_value.py",
    "scripts/verify_on_demand_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "modeling/native_region_readout.py",
    "modeling/paired_readout_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/soft_region_roles.py",
    "modeling/soft_region_value_adapter.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/RMNet/models/rmnet.py",
    "external/RMNet/LICENSE",
]
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq5")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        protocol = json.loads(PROTOCOL.read_text())
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        frames = sample_clip(load_manifest(manifest, "dev")[args.sequence],
                             protocol["training"]["clip_length"], args.seed)
        repo, revision = activate_author_repo()
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        samples = capture_clip(model, frames, args.output_dir)
        heads = SoftRegionValueHeads().train()

        membership_summary = []
        for sample in samples:
            memberships = predicted_soft_regions(sample["native_logits"][0, 0])
            if memberships.shape != (64, 64, 3):
                raise RuntimeError("Soft memberships do not match the native memory grid")
            if not torch.isfinite(memberships).all() or (memberships < 0).any():
                raise RuntimeError("Invalid soft region membership")
            if not torch.allclose(memberships.sum(-1), torch.ones(64, 64), atol=1e-6):
                raise RuntimeError("Soft memberships do not partition all spatial tokens")
            membership_summary.append({
                "frame": sample["frame"],
                "mean_by_role": memberships.mean((0, 1)).tolist(),
                "fraction_with_mixed_membership": float(
                    ((memberships > 0.01) & (memberships < 0.99)).any(-1).float().mean()
                ),
            })
        if not any(row["fraction_with_mixed_membership"] > 0 for row in membership_summary):
            raise RuntimeError("Gate clip contains no genuinely soft token memberships")

        optimizer = torch.optim.AdamW(
            heads.parameters(), lr=protocol["training"]["learning_rate"],
            weight_decay=protocol["training"]["weight_decay"])
        optimizer.zero_grad(set_to_none=True)
        loss, calls = clip_loss(
            model, heads, samples, "regional_value", args.sequence,
            check_native=True, return_trace=True, region_encoder=predicted_soft_regions)
        if len(calls) != 4 * len(samples) or [row["layer"] for row in calls] != [0, 1, 2, 3] * len(samples):
            raise RuntimeError("Native identity replay did not observe all four layers")
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite native-identity training loss")
        loss.backward()
        gradients = []
        for index, layer in enumerate(heads.layers):
            grad = layer.role_embeddings.grad
            if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
                raise RuntimeError(f"Missing finite nonzero gradient for layer {index}")
            gradients.append(float(grad.abs().sum()))
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen SAM2 received gradients")
        optimizer.step()
        if not all(torch.isfinite(layer.role_embeddings).all()
                   and layer.role_embeddings.count_nonzero() > 0 for layer in heads.layers):
            raise RuntimeError("Optimizer failed to update all role heads")

        heads.eval()
        native = {sample["frame"]: sample["native_logits"] for sample in samples}
        active_rows = []
        with tempfile.TemporaryDirectory(prefix="soft_roles_", dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError("Engineering clip did not prompt on its first frame")
            with NativeRegionalReadout(
                model, heads, "regional_value", args.sequence,
                region_encoder=predicted_soft_regions,
            ) as observer, torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if object_ids != [1]:
                        raise RuntimeError("Unexpected tracked object IDs")
                    if frame == prompt:
                        continue
                    read = observer.observer.trace[-1]
                    layers = observer.calls[-4:]
                    if read["frame"] != frame or [row["layer"] for row in layers] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Incomplete native read trace at clip frame {frame}")
                    if any(row["spatial_keys"] <= 0 or row["pointer_keys"] < 0 for row in layers):
                        raise RuntimeError(f"Invalid token/pointer accounting at clip frame {frame}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite prediction at clip frame {frame}")
                    active_rows.append({
                        "frame": frame,
                        "prediction_sha256": tensor_hash(logits),
                        "differs_from_native": not torch.equal(
                            logits.detach().cpu().float(), native[frame].detach().cpu().float()),
                        "attention_layers": layers,
                    })
            model.reset_state(state)
        if len(active_rows) != len(samples) or not any(r["differs_from_native"] for r in active_rows):
            raise RuntimeError("Updated adapter did not complete a nonzero clip propagation")

        rmnet_revision = __import__("subprocess").check_output(
            ["git", "-C", str(ROOT / "external/RMNet"), "rev-parse", "HEAD"], text=True
        ).strip()
        if rmnet_revision != protocol["prior_work"]["RMNet"]["revision"]:
            raise RuntimeError("RMNet checkout differs from the inspected pinned revision")
        sources = {name: digest(ROOT / name) for name in SOURCES}
        write(args.output_dir / "completion.json", {
            "passed": True,
            "sequence": args.sequence,
            "clip_frames": [frame.frame_index for frame in frames],
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest),
            "sam2_revision": revision,
            "model_config": MODEL_CONFIG,
            "rmnet_revision": rmnet_revision,
            "zero_head_native_prediction_and_memory_identity": True,
            "soft_memberships": membership_summary,
            "gradient_l1_by_layer": gradients,
            "updated_propagation": active_rows,
            "sources": sources,
            "scope": "One real five-frame clip and one optimizer update; engineering gate only.",
        })
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
