"""Run the real-sequence gate for prediction-conditioned memory affinity."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.prediction_conditioned_affinity import (
    PredictionConditionedTrackStep,
    PredictionRoleScales,
)
from modeling.prediction_conditioned_training import clip_loss
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = ROOT / "protocols/prediction_conditioned_region_affinity_polypgen_v2.json"
SOURCES = [
    "protocols/prediction_conditioned_region_affinity_polypgen_v2.json",
    "tests/test_prediction_conditioned_audit.py",
    "modeling/prediction_conditioned_affinity.py",
    "modeling/prediction_conditioned_training.py",
    "modeling/decoder_training_mask.py",
    "scripts/verify_prediction_conditioned_affinity.py",
    "scripts/run_prediction_conditioned_affinity_cv.py",
    "scripts/run_soft_role_affinity_cv.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/verify_on_demand_clip.py",
    "scripts/memory_token_io.py",
    "scripts/false_alarm_statistics.py",
    "scripts/bank_integrity.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "modeling/cfbi_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/region_tokens.py",
    "modeling/region_mass_attention.py",
    "modeling/region_readout.py",
    "modeling/soft_role_affinity.py",
    "modeling/regional_refinement.py",
    "training/tracking.py",
    "external/HODOR/hodor/modelling/encoder/soft_masked_attention.py",
    "external/Cutie/cutie/model/transformer/object_transformer.py",
    "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "external/sam2/sam2/build_sam.py",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/sam2/modeling/memory_attention.py",
    "external/sam2/sam2/modeling/sam2_utils.py",
    "external/sam2/training/loss_fns.py",
    "external/UACANet/lib/modules/layers.py",
    "external/UACANet/lib/modules/attention_module.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq5")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        subprocess.run([
            sys.executable, "-m", "pytest", "-q",
            "tests/test_prediction_conditioned_audit.py",
            "tests/test_soft_role_affinity.py",
            "tests/test_region_definitions.py",
        ], cwd=ROOT, check=True)
        protocol = json.loads(PROTOCOL.read_text())
        data = protocol["data"]
        manifest = ROOT / data["manifest"]
        videos = load_manifest(manifest, "dev")
        frames = sample_clip(
            videos[args.sequence], protocol["engineering_gate"]["clip_length"], args.seed
        )
        prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        if prompt != 0:
            raise RuntimeError("Gate clip must start at the first-frame prompt")

        repo, revision = activate_author_repo()
        if revision != "2b90b9f5ceec907a1c18123530e92e794ad901a4":
            raise RuntimeError("SAM2 repository differs from the frozen protocol")
        cutie_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "external/Cutie"), "rev-parse", "HEAD"], text=True
        ).strip()
        if cutie_revision != protocol["prior_work"]["Cutie"]["revision"]:
            raise RuntimeError("Cutie repository differs from the reviewed revision")

        model = build_author_predictor(SimpleNamespace(
            method="sam2", device="cpu", checkpoint=args.checkpoint,
            model_config=protocol["model"]["config"],
        ), repo)
        model.requires_grad_(False)
        native_parameter_sha = parameter_digest(model)
        samples = capture_clip(model, frames, output)
        with torch.inference_mode():
            for sample in samples:
                native = model.track_step(
                    frame_idx=sample["frame"],
                    is_init_cond_frame=False,
                    current_vision_feats=sample["vision"],
                    current_vision_pos_embeds=sample["positions"],
                    feat_sizes=sample["sizes"],
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=sample["output_dict"],
                    num_frames=sample["num_frames"],
                    track_in_reverse=False,
                    run_mem_encoder=False,
                )
                sample["native_object_score"] = native["object_score_logits"].clone()
                sample["native_obj_ptr"] = native["obj_ptr"].clone()
        arms = {}

        for mode in ("regional", "global"):
            scales = PredictionRoleScales(max_strength=2.0).train()
            optimizer = torch.optim.AdamW(
                scales.parameters(), lr=protocol["training"]["learning_rate"],
                weight_decay=protocol["training"]["weight_decay"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss, records = clip_loss(
                model, scales, samples, mode, check_native=True
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {mode} gate loss")
            loss.backward()
            grad = scales.raw_strength.grad
            if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
                raise RuntimeError(f"Missing or invalid {mode} regional-scale gradient")
            gradient_l1_by_layer = grad.abs().sum(-1)
            if not torch.all(gradient_l1_by_layer > 0):
                raise RuntimeError(f"A {mode} memory-attention layer has no gradient")
            optimizer.step()
            strengths = scales.strengths().detach()
            if not torch.isfinite(strengths).all() or strengths.abs().max() > 2.0:
                raise RuntimeError(f"Invalid or unbounded {mode} update")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen SAM2 weights received gradients")
            if parameter_digest(model) != native_parameter_sha:
                raise RuntimeError("SAM2 parameters changed during the adapter gate")
            for row in records:
                if [call["layer"] for call in row["layers"]] != [0, 1, 2, 3]:
                    raise RuntimeError("A second pass missed a memory-attention layer")
                if any(call["spatial_keys"] <= 0 or call["pointer_keys"] < 0
                       for call in row["layers"]):
                    raise RuntimeError("Invalid spatial or pointer key count")
                if any(call["pointer_bias_max_abs"] != 0.0 for call in row["layers"]):
                    raise RuntimeError("The second read applied a bias to an object pointer")

            with torch.inference_mode():
                post_loss, post_records = clip_loss(model, scales.eval(), samples, mode)
            if not torch.isfinite(post_loss):
                raise RuntimeError(f"Nonfinite post-update {mode} propagation")
            propagation = []
            with tempfile.TemporaryDirectory(prefix=f"{mode}_", dir=output) as temp:
                state, actual_prompt = prepare(model, frames, Path(temp))
                if actual_prompt != 0:
                    raise RuntimeError("Post-update inference prompt differs from gate prompt")
                with PredictionConditionedTrackStep(model, scales.eval(), mode) as tracker, torch.inference_mode():
                    for frame, object_ids, logits in model.propagate_in_video(
                        state, start_frame_idx=actual_prompt
                    ):
                        if object_ids != [1] or not torch.isfinite(logits).all():
                            raise RuntimeError(f"Invalid post-update {mode} prediction at {frame}")
                        layers = tracker.reads[-1]["layers"] if frame != actual_prompt else []
                        if frame != actual_prompt and [row["layer"] for row in layers] != [0, 1, 2, 3]:
                            raise RuntimeError("Post-update propagation missed an attention layer")
                        propagation.append({
                            "frame": int(frame),
                            "prediction_sha256": tensor_hash(logits),
                            "layers": layers,
                        })
                model.reset_state(state)
            if [row["frame"] for row in propagation] != list(range(len(frames))):
                raise RuntimeError(f"Incomplete post-update {mode} propagation")
            arms[mode] = {
                "initial_loss": float(loss.detach()),
                "post_update_loss": float(post_loss),
                "gradient_l1_by_layer": gradient_l1_by_layer.cpu().tolist(),
                "strengths_by_layer_role": strengths.cpu().tolist(),
                "frames": records,
                "post_update_frames": post_records,
                "post_update_propagation": propagation,
            }

        regional_frames = arms["regional"]["post_update_propagation"]
        global_frames = arms["global"]["post_update_propagation"]
        for regional, global_row in zip(regional_frames, global_frames, strict=True):
            if regional["frame"] != global_row["frame"]:
                raise RuntimeError("Post-update regional/global frame schedules differ")
            for regional_layer, global_layer in zip(
                regional["layers"], global_row["layers"], strict=True
            ):
                for key in ("layer", "spatial_keys", "pointer_keys", "pointer_bias_max_abs"):
                    if regional_layer.get(key) != global_layer.get(key):
                        raise RuntimeError(f"Post-update regional/global {key} differs")

        receipt = {
            "passed": True,
            "independent_attention_regressions_passed": True,
            "scope": "One real-data engineering clip and one optimizer update; not an accuracy result.",
            "sequence": args.sequence,
            "sampled_frames": [int(frame.frame_index) for frame in frames],
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest),
            "protocol_sha256": digest(PROTOCOL),
            "sam2_revision": revision,
            "cutie_revision": cutie_revision,
            "native_model_parameter_sha256": native_parameter_sha,
            "source_hashes": {path: digest(ROOT / path) for path in SOURCES},
            "arms": arms,
            "elapsed_seconds": time.monotonic() - start,
        }
        write(output / "completion.json", receipt)
        (output / "results.md").write_text(
            "# Prediction-conditioned region affinity: engineering gate\n\n"
            f"Passed on {args.sequence}, sampled frames {receipt['sampled_frames']}. "
            "The zero-strength two-pass outputs and stored memories matched native SAM2 "
            "bit-for-bit for both regional and global modes. Both modes produced finite "
            "nonzero gradients and finite bounded updates in all four memory-attention layers. "
            "This is an engineering check, not evidence of Dice improvement.\n"
        )
        write(output / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
