"""Real-sequence engineering gate for SimVOS-style regional memory prototypes."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.simvos_region_prototypes import RegionPrototypeHeads, SIMVOS_REVISION
from modeling.simvos_prototype_training import prototype_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo

PROTOCOL = ROOT / "protocols/simvos_region_prototypes_polypgen_v1.json"
SOURCES = [
    "modeling/simvos_region_prototypes.py",
    "modeling/native_simvos_prototype_read.py",
    "modeling/simvos_prototype_training.py",
    "modeling/region_bias.py",
    "scripts/verify_simvos_region_prototypes.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "external/SimVOS/model/models_vit.py",
]


def source_hashes(protocol_path):
    return {
        protocol_path.relative_to(ROOT).as_posix(): digest(protocol_path),
        **{path: digest(ROOT / path) for path in SOURCES},
    }


def gradients(heads):
    records = []
    grad = heads.gains.grad
    if grad is None:
        raise RuntimeError("Missing prototype gain gradient")
    for index, value in enumerate(grad):
        if not torch.isfinite(value) or value.abs() == 0:
            raise RuntimeError(f"Missing prototype gain gradient at layer {index}")
        records.append(float(value.abs()))
    return records


def learner_gradients(heads, active_regions):
    if active_regions != [0, 1, 2]:
        raise RuntimeError(f"Expected all three regions in the real-data gate: {active_regions}")
    records = []
    for region, learner in enumerate(heads.learners):
        total = 0.0
        for name, parameter in learner.named_parameters():
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError(f"Missing/nonfinite TokenLearner gradient {region}/{name}")
            total += float(grad.abs().sum())
        if total <= 0:
            raise RuntimeError(f"TokenLearner {region} has zero gradient")
        records.append(total)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence", action="append", choices=["seq2", "seq3"])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    if ROOT not in args.protocol.parents:
        parser.error("--protocol must be inside the project")
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(args.protocol.read_text())
    manifest = ROOT / protocol["data"]["manifest"]
    sequences = args.sequence or ["seq2", "seq3"]
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    args.method = "sam2"
    args.model_config = protocol["training"]["model_config"]
    repo, sam2_revision = activate_author_repo()
    if SIMVOS_REVISION != protocol["prior_work"]["SimVOS"]["revision"]:
        raise RuntimeError("SimVOS source differs from the frozen protocol")

    model = build_author_predictor(args, repo)
    clips = {}
    samples_by_sequence = {}
    for index, sequence in enumerate(sequences):
        seed = protocol["training"]["seed"] + index
        clip = sample_clip(videos[sequence], protocol["training"]["clip_length"], seed)
        clips[sequence] = [frame.frame_index for frame in clip]
        samples_by_sequence[sequence] = capture_presence_clip(model, clip, args.output_dir)

    heads = RegionPrototypeHeads(prototypes=protocol["method"]["prototype_count"]).to(args.device)
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=protocol["training"]["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    zero = {}
    losses = []
    for sequence in sequences:
        loss, calls, deltas = prototype_clip_loss(
            model, heads, samples_by_sequence[sequence], sequence, "regional", check_native=True
        )
        losses.append(loss)
        zero[sequence] = {"loss": float(loss.detach()), "calls": calls, "max_prediction_delta": max(deltas)}
    optimizer.zero_grad(set_to_none=True)
    torch.stack(losses).mean().backward()
    zero_gain_grad = gradients(heads)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    updated = {}
    losses = []
    for sequence in sequences:
        loss, calls, deltas = prototype_clip_loss(
            model, heads, samples_by_sequence[sequence], sequence, "regional"
        )
        losses.append(loss)
        updated[sequence] = {"loss": float(loss.detach()), "calls": calls, "max_prediction_delta": max(deltas)}
    torch.stack(losses).mean().backward()
    active_regions = sorted({
        region
        for row in updated.values()
        for call in row["calls"]
        for region in call["prototype_regions"]
    })
    prototype_grad = learner_gradients(heads, active_regions)
    updated_gain_grad = gradients(heads)
    optimizer.step()

    if any(len(row["calls"]) != 4 * (len(clips[name]) - 1) for name, row in zero.items()):
        raise RuntimeError("Zero-gain replay missed one or more attention layers")
    if any(len(row["calls"]) != 4 * (len(clips[name]) - 1) for name, row in updated.items()):
        raise RuntimeError("Updated prototype replay missed one or more attention layers")
    if any(call["prototype_keys"] <= 0 for row in updated.values() for call in row["calls"]):
        raise RuntimeError("No learned prototype keys reached attention")
    if any(
        call["spatial_keys"] <= 0 or call["pointer_keys"] < 0
        for row in updated.values() for call in row["calls"]
    ):
        raise RuntimeError("Native spatial/pointer key accounting failed")
    if any(call["gain"] == 0 for row in updated.values() for call in row["calls"]):
        raise RuntimeError("The prototype side-read gain did not update")
    if any(not torch.isfinite(loss) for loss in losses):
        raise RuntimeError("Nonfinite updated gate loss")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen SAM2 unexpectedly received gradients")

    receipt = {
        "passed": True,
        "sequences": sequences,
        "clip_frame_ids": clips,
        "zero_gain_native_replay": zero,
        "post_update": updated,
        "zero_gain_l1_by_layer": zero_gain_grad,
        "token_learner_gradient_l1": prototype_grad,
        "updated_gain_l1_by_layer": updated_gain_grad,
        "native_spatial_and_pointer_counts_preserved": True,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(args.protocol),
        "sam2_revision": sam2_revision,
        "simvos_revision": SIMVOS_REVISION,
        "sources": source_hashes(args.protocol),
    }
    (args.output_dir / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
