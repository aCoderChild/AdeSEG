"""Real-sequence gate for CFBI-region training of SAM2 memory q/k projections."""

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
from modeling.cfbi_region_qk_training import enable_memory_qk_training, native_qk_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_affinity_cv import MODEL_CONFIG
from scripts.run_cfbi_region_pilot import activate_author_repo

PROTOCOL = ROOT / "protocols/cfbi_region_qk_tuning_polypgen_v1.json"
SOURCES = [
    "protocols/cfbi_region_qk_tuning_polypgen_v1.json",
    "scripts/verify_cfbi_region_qk.py",
    "scripts/run_cfbi_region_qk_cv.py",
    "modeling/cfbi_region_qk_training.py",
    "modeling/cfbi_region_affinity.py",
    "modeling/native_cfbi_region_affinity.py",
    "modeling/cfbi_region_affinity_training.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_cfbi_region_affinity_cv.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/CFBI/networks/layers/matching.py",
]


def require_finite_gradients(model, names, *, nonzero):
    parameters = dict(model.named_parameters())
    by_layer = {}
    for name in names:
        grad = parameters[name].grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing or nonfinite q/k gradient: {name}")
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        by_layer[layer] = by_layer.get(layer, 0.0) + float(grad.abs().sum())
    if set(by_layer) != {0, 1, 2, 3}:
        raise RuntimeError("q/k projections in all four layers were not checked")
    if nonzero and any(value <= 0 for value in by_layer.values()):
        raise RuntimeError("At least one memory-attention layer has zero q/k gradient")
    return by_layer


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
    manifest = ROOT / protocol["data"]["manifest"]
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    name = "seq5"
    clip = sample_clip(videos[name], protocol["training"]["clip_length"], protocol["training"]["seed"])
    repo, sam2_revision = activate_author_repo()
    if CFBI_REVISION != protocol["prior_work"]["CFBI"]["revision"]:
        raise RuntimeError("CFBI source revision differs from protocol")
    args.method, args.model_config = "sam2", MODEL_CONFIG

    regional_model = build_author_predictor(args, repo)
    qk_names, qk_parameters = enable_memory_qk_training(regional_model)
    regional_heads = RegionalAffinityHeads(channels=regional_model.memory_attention.d_model)
    regional_optimizer = torch.optim.AdamW(
        [
            {"params": qk_parameters,
             "lr": protocol["training"]["qk_learning_rate"]},
            {"params": list(regional_heads.parameters()),
             "lr": protocol["training"]["regional_projection_learning_rate"]},
        ],
        weight_decay=protocol["training"]["weight_decay"],
    )
    samples = capture_presence_clip(regional_model, clip, args.output_dir)
    regional_optimizer.zero_grad(set_to_none=True)
    zero_loss, zero_calls, zero_deltas = cfbi_region_clip_loss(
        regional_model, regional_heads, samples, name, check_native=True
    )
    zero_loss.backward()
    first_regional_gradients = []
    for layer_index, head in enumerate(regional_heads.layers):
        grad = head.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise RuntimeError(f"Invalid regional projection gradient in layer {layer_index}")
        first_regional_gradients.append(float(grad.abs().sum()))
    first_qk_gradient = require_finite_gradients(regional_model, qk_names, nonzero=False)
    regional_optimizer.step()

    regional_optimizer.zero_grad(set_to_none=True)
    updated_loss, updated_calls, updated_deltas = cfbi_region_clip_loss(
        regional_model, regional_heads, samples, name
    )
    updated_loss.backward()
    updated_qk_gradient = require_finite_gradients(regional_model, qk_names, nonzero=True)
    for layer_index, head in enumerate(regional_heads.layers):
        grad = head.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise RuntimeError(f"Regional gradient disconnected in layer {layer_index}")
    regional_optimizer.step()
    if len(zero_calls) != 4 * len(samples) or len(updated_calls) != 4 * len(samples):
        raise RuntimeError("Regional arm did not traverse all four layers for every clip frame")
    if any(call["spatial_keys"] <= 0 for call in updated_calls):
        raise RuntimeError("Regional arm has an empty spatial key set")
    if any(call["pointer_keys"] < 0 for call in updated_calls):
        raise RuntimeError("Invalid pointer-key accounting")
    if not torch.isfinite(updated_loss):
        raise RuntimeError("Regional arm produced nonfinite post-update loss")

    del regional_model, regional_heads, regional_optimizer, samples

    control_model = build_author_predictor(args, repo)
    control_names, control_parameters = enable_memory_qk_training(control_model)
    if control_names != qk_names:
        raise RuntimeError("Control and regional q/k parameter sets differ")
    control_optimizer = torch.optim.AdamW(
        control_parameters,
        lr=protocol["training"]["qk_learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    control_samples = capture_presence_clip(control_model, clip, args.output_dir)
    control_optimizer.zero_grad(set_to_none=True)
    control_loss = native_qk_clip_loss(control_model, control_samples, check_native=True)
    control_loss.backward()
    control_gradient = require_finite_gradients(control_model, control_names, nonzero=True)
    control_optimizer.step()
    control_optimizer.zero_grad(set_to_none=True)
    control_updated_loss = native_qk_clip_loss(control_model, control_samples)
    if not torch.isfinite(control_updated_loss):
        raise RuntimeError("q/k-only control produced nonfinite post-update loss")

    receipt = {
        "passed": True,
        "sequence": name,
        "clip_frames": [frame.frame_index for frame in clip],
        "regional_zero_projection_prediction_max_abs_delta": max(zero_deltas),
        "regional_zero_projection_loss": float(zero_loss.detach()),
        "regional_updated_loss": float(updated_loss.detach()),
        "regional_updated_prediction_max_abs_delta": max(updated_deltas),
        "regional_layers": 4,
        "regional_clip_calls": len(updated_calls),
        "spatial_key_counts": sorted({call["spatial_keys"] for call in updated_calls}),
        "pointer_counts": sorted({call["pointer_keys"] for call in updated_calls}),
        "regional_projection_gradient_l1": first_regional_gradients,
        "qk_initial_gradient_l1_by_layer": first_qk_gradient,
        "qk_regional_post_update_gradient_l1_by_layer": updated_qk_gradient,
        "qk_control_gradient_l1_by_layer": control_gradient,
        "qk_control_loss": float(control_loss.detach()),
        "qk_control_post_update_loss": float(control_updated_loss.detach()),
        "qk_parameter_names": qk_names,
        "sam2_revision": sam2_revision,
        "cfbi_revision": CFBI_REVISION,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(PROTOCOL),
        "sources": {path: digest(ROOT / path) for path in SOURCES},
    }
    (args.output_dir / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
