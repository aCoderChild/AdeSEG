"""Run a real PolypGen SAM2 identity, gradient and propagation gate."""

import argparse
import json
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.soft_role_affinity import SoftRoleAffinityHeads, mask_logits_to_soft_roles
from modeling.soft_role_attention_training import clip_loss
from modeling.native_region_readout import NativeRegionalReadout
from modeling.region_tokens import _verified_source
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = ROOT / "protocols/soft_role_affinity_polypgen_v1.json"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SOURCES = [
    "protocols/soft_role_affinity_polypgen_v1.json",
    "scripts/verify_soft_role_affinity.py",
    "scripts/verify_on_demand_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "modeling/native_region_readout.py",
    "modeling/soft_role_affinity.py",
    "modeling/soft_role_attention_training.py",
    "modeling/paired_readout_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_mass_attention.py",
    "modeling/region_readout.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/HODOR/hodor/modelling/encoder/soft_masked_attention.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq5")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        protocol = json.loads(PROTOCOL.read_text())
        hodor_revision = protocol["prior_work"]["HODOR_soft_masked_attention"]["revision"]
        _verified_source(
            "HODOR", hodor_revision,
            "hodor/modelling/encoder/soft_masked_attention.py",
        )
        cutie_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "external/Cutie"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if cutie_revision != protocol["prior_work"]["Cutie"]["revision"]:
            raise RuntimeError("Cutie checkout differs from the reviewed revision")
        manifest = ROOT / protocol["data"]["manifest"]
        frames = sample_clip(
            load_manifest(manifest, "dev")[args.sequence],
            protocol["training"]["clip_length"],
            args.seed,
        )
        repo, revision = activate_author_repo()
        build_args = argparse.Namespace(
            method="sam2", device="cpu", model_config=MODEL_CONFIG,
            checkpoint=args.checkpoint,
        )
        model = build_author_predictor(build_args, repo)
        model.requires_grad_(False)
        attention = model.memory_attention.layers[0].cross_attn_image
        heads = SoftRoleAffinityHeads(
            layers=4, heads=attention.num_heads,
            head_dim=attention.q_proj.out_features // attention.num_heads,
            max_bias=protocol["training"]["max_attention_bias"],
        ).train()
        samples = capture_clip(model, frames, args.output_dir)

        role_rows = []
        for sample in samples:
            roles = mask_logits_to_soft_roles(sample["native_logits"][0, 0])
            if roles.shape != (64, 64, 3):
                raise RuntimeError("Stored-mask roles do not match SAM2's memory grid")
            role_rows.append({
                "frame": sample["frame"],
                "mean_membership": roles.mean((0, 1)).tolist(),
                "tokens_with_all_roles_positive": int((roles > 0).all(-1).sum()),
            })
        if not any(row["tokens_with_all_roles_positive"] for row in role_rows):
            raise RuntimeError("Gate clip lacks genuinely soft memory-role assignments")

        optimizer = torch.optim.AdamW(
            heads.parameters(),
            lr=protocol["training"]["learning_rate"],
            weight_decay=protocol["training"]["weight_decay"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss, records, calls = clip_loss(
            model, heads, samples, "soft_role_attention", args.sequence,
            check_native=True,
            role_loss_weight=protocol["training"]["role_loss_weight"],
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite gate training loss")
        loss.backward()
        gradient_records = []
        for layer_index, layer in enumerate(heads.layers):
            scale_grad = layer.role_bias.grad
            role_weight_grad = layer.query_role.weight.grad
            if any(value is None for value in (scale_grad, role_weight_grad)):
                raise RuntimeError(f"Missing gradient in attention layer {layer_index}")
            for name, grad in (("role_bias", scale_grad), ("query_role", role_weight_grad)):
                if not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
                    raise RuntimeError(f"Invalid {name} gradient in layer {layer_index}")
            gradient_records.append({
                "layer": layer_index,
                "role_bias_l1": float(scale_grad.abs().sum()),
                "query_role_l1": float(role_weight_grad.abs().sum()),
            })
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen SAM2 acquired gradients")
        optimizer.step()
        for layer in heads.layers:
            if not torch.isfinite(layer.role_bias).all() or not torch.isfinite(layer.query_role.weight).all():
                raise RuntimeError("Nonfinite adapter update")
            if layer.role_bias.abs().max() > protocol["training"]["max_attention_bias"]:
                raise RuntimeError("Learned role bias exceeded its registered bound")
        if [row["layer"] for row in calls] != [0, 1, 2, 3] * len(samples):
            raise RuntimeError("Training gate did not observe all four layers in order")

        native_by_frame = {sample["frame"]: sample["native_logits"] for sample in samples}
        propagation, changed = [], False
        with tempfile.TemporaryDirectory(prefix="soft_role_gate_", dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            with NativeRegionalReadout(
                model, heads.eval(), "soft_role_attention", args.sequence,
                region_encoder=mask_logits_to_soft_roles,
            ) as observer, torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(
                    state, start_frame_idx=prompt
                ):
                    if object_ids != [1]:
                        raise RuntimeError("Unexpected object ids in gate propagation")
                    if frame == prompt:
                        continue
                    last_calls = observer.calls[-4:]
                    read = observer.observer.trace[-1]
                    if read["frame"] != frame or read["spatial_keys"] <= 0:
                        raise RuntimeError(f"Incomplete region read at frame {frame}")
                    if [row["layer"] for row in last_calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Missing attention layer at frame {frame}")
                    if any(row["spatial_keys"] != read["spatial_keys"] for row in last_calls):
                        raise RuntimeError("Spatial key count changed across layers")
                    if any(row["pointer_keys"] < 0 for row in last_calls):
                        raise RuntimeError("Invalid pointer-key accounting")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Nonfinite propagated mask")
                    reference = native_by_frame.get(frame)
                    if reference is None:
                        raise RuntimeError(f"No native clip reference for frame {frame}")
                    differs = not torch.equal(logits.detach().cpu(), reference.detach().cpu())
                    changed = changed or differs
                    propagation.append({
                        "frame": frame,
                        "prediction_sha256": tensor_hash(logits),
                        "differs_from_native": differs,
                        "spatial_keys": read["spatial_keys"],
                        "pointer_keys": read["pointer_keys"],
                        "layers": [row["layer"] for row in last_calls],
                    })
            model.reset_state(state)
        if len(propagation) != len(samples) or not changed:
            raise RuntimeError("Updated soft-role attention did not alter a completed clip trajectory")

        source_hashes = {name: digest(ROOT / name) for name in SOURCES}
        write(args.output_dir / "completion.json", {
            "passed": True,
            "sequence": args.sequence,
            "frames": [frame.frame_index for frame in frames],
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest),
            "sam2_revision": revision,
            "hodor_revision": hodor_revision,
            "cutie_revision": cutie_revision,
            "model_config": MODEL_CONFIG,
            "native_zero_bias_replay_bit_identical": True,
            "spatial_and_pointer_keys_preserved": True,
            "all_four_layers_observed": True,
            "soft_membership_summary": role_rows,
            "training_loss": float(loss.detach()),
            "training_records": records,
            "gradient_l1_by_layer": gradient_records,
            "updated_propagation": propagation,
            "source_hashes": source_hashes,
            "elapsed_seconds": time.monotonic() - started,
            "scope": "One real five-frame PolypGen clip and one adapter update; engineering gate only.",
        })
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
