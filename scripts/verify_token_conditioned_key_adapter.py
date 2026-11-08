"""Real-frame engineering gate for the token-conditioned SAM2 key adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.decoder_training_mask import decoder_training_mask
from modeling.paired_readout_training import clip_loss
from modeling.token_conditioned_key_adapter import (
    NativeTokenConditionedKeyAdapter,
    TokenConditionedKeyHeads,
)
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = "protocols/token_conditioned_key_adapter_polypgen_v1.json"
CHECKPOINT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/assets/sam2.1_hiera_tiny.pt"
)
DAVIS_ROOT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/datasets/DAVIS"
)
SOURCE_FILES = (
    PROTOCOL,
    "protocols/token_conditioned_key_adapter_engineering_gate_amendment_2026-11-09.json",
    "modeling/token_conditioned_key_adapter.py",
    "modeling/paired_readout_training.py",
    "modeling/decoder_training_mask.py",
    "scripts/verify_token_conditioned_key_adapter.py",
    "tests/test_token_conditioned_key_adapter.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/verify_on_demand_clip.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
)


def parameter_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(str(parameter.dtype).encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    for name, buffer in sorted(model.named_buffers()):
        h.update(name.encode())
        h.update(str(buffer.dtype).encode())
        h.update(str(tuple(buffer.shape)).encode())
        h.update(buffer.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def gradient_summary(heads: TokenConditionedKeyHeads) -> list[dict]:
    rows = []
    for index, layer in enumerate(heads.layers):
        for name, parameter in layer.named_parameters():
            gradient = parameter.grad
            rows.append({
                "layer": index,
                "parameter": name,
                "present": gradient is not None,
                "finite": gradient is None or bool(torch.isfinite(gradient).all()),
                "l1": 0.0 if gradient is None else float(gradient.detach().abs().sum()),
            })
    return rows


def require_layer_order(calls: list[dict], read_count: int) -> None:
    expected = [layer for _ in range(read_count) for layer in range(4)]
    actual = [row["layer"] for row in calls]
    if actual != expected:
        raise RuntimeError(f"Expected four ordered layers per read; got {actual}")
    if any(row["spatial_keys"] % 4096 or row["spatial_frames"] <= 0 for row in calls):
        raise RuntimeError("Spatial keys are not aligned to complete 64x64 frames")
    if any(not row["position_aligned_before_adapter"] or not row["values_unchanged"]
           or not row["pointer_keys_unchanged"] for row in calls):
        raise RuntimeError("Memory position/value/pointer invariant failed")
    if any(
        not torch.isfinite(torch.tensor(row["key_delta_l1"]))
        or row["key_delta_l1"] < 0
        or row["key_delta_nonzero"] < 0
        for row in calls
    ):
        raise RuntimeError("Invalid key-residual trace")


def training_update(model, heads, optimizer, samples, sequence, mode, *, check_native=False):
    optimizer.zero_grad(set_to_none=True)
    loss, calls = clip_loss(
        model,
        heads,
        samples,
        mode="token_key",
        sequence=sequence,
        check_native=check_native,
        return_trace=True,
        attention_context_factory=lambda m, h: NativeTokenConditionedKeyAdapter(m, h, mode),
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite supervised clip loss")
    loss.backward()
    gradients = gradient_summary(heads)
    if any(not row["finite"] for row in gradients):
        raise RuntimeError("Nonfinite adapter gradient")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A frozen SAM2 parameter received a gradient")
    require_layer_order(calls, len(samples))
    optimizer.step()
    return float(loss.detach()), calls, gradients


def clone_optimizer_state(optimizer):
    return copy.deepcopy(optimizer.state_dict())


def require_tree_equal(left, right, path="state"):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"Save/resume replay changed tensor type at {path}")
        if not torch.equal(left, right):
            raise RuntimeError(f"Save/resume replay changed tensor value at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            raise RuntimeError(f"Save/resume replay changed mapping keys at {path}")
        for key in left:
            require_tree_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"Save/resume replay changed sequence structure at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            require_tree_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"Save/resume replay changed value at {path}: {left!r} != {right!r}")


def propagate_signatures(model, frames, output_dir, heads=None, mode="token"):
    """Run one fresh state and retain CPU outputs for exact native parity."""
    records, tensors, calls = [], [], []
    with tempfile.TemporaryDirectory(prefix="propagation_parity_", dir=output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        if prompt != 0:
            raise RuntimeError("Fixed gate clip must use its first frame as prompt")
        observer_context = (
            NativeTokenConditionedKeyAdapter(model, heads, mode)
            if heads is not None else nullcontext(None)
        )
        with observer_context as observer, torch.inference_mode():
            for frame, object_ids, logits in model.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if object_ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs during propagation: {object_ids}")
                if frame == prompt:
                    continue
                bank = state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
                if frame not in bank:
                    raise RuntimeError(f"Missing propagated memory output at frame {frame}")
                low_res = bank[frame]["pred_masks"]
                if not torch.isfinite(logits).all() or not torch.isfinite(low_res).all():
                    raise RuntimeError(f"Nonfinite propagation output at frame {frame}")
                records.append({
                    "frame": int(frame),
                    "prediction_sha256": tensor_hash(logits),
                    "low_res_sha256": tensor_hash(low_res),
                })
                tensors.append((logits.detach().cpu().clone(), low_res.detach().cpu().clone()))
            if observer is not None:
                calls = list(observer.calls)
        model.reset_state(state)
    return records, tensors, calls


def resume_replay_check(model, heads, optimizer, samples, sequence, mode, output_dir):
    """Compare uninterrupted and restored next updates on a frozen real clip."""
    checkpoint_path = output_dir / f"resume_boundary_{sequence}_{mode}.pt"
    torch.save({
        "heads": copy.deepcopy(heads.state_dict()),
        "optimizer": clone_optimizer_state(optimizer),
        "rng_cpu": torch.random.get_rng_state(),
    }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    rng_before = checkpoint["rng_cpu"]

    # The uninterrupted branch starts from the live state.
    torch.random.set_rng_state(rng_before)
    uninterrupted_loss, _, uninterrupted_gradients = training_update(
        model, heads, optimizer, samples, sequence, mode
    )
    uninterrupted_rng_after = torch.random.get_rng_state().clone()
    uninterrupted_weights = copy.deepcopy(heads.state_dict())
    uninterrupted_optimizer = clone_optimizer_state(optimizer)

    # Restore the serialized state into fresh module and optimizer objects.
    resumed_heads = TokenConditionedKeyHeads(channels=64, rank=16)
    resumed_heads.load_state_dict(checkpoint["heads"])
    resumed_optimizer = torch.optim.AdamW(
        resumed_heads.parameters(), lr=1e-4, weight_decay=0.001,
        betas=(0.9, 0.999), eps=1e-8,
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(rng_before)
    resumed_loss, _, resumed_gradients = training_update(
        model, resumed_heads, resumed_optimizer, samples, sequence, mode
    )
    resumed_rng_after = torch.random.get_rng_state().clone()
    if uninterrupted_loss != resumed_loss:
        raise RuntimeError("Save/resume replay changed the next-update loss")
    if [row["l1"] for row in uninterrupted_gradients] != [row["l1"] for row in resumed_gradients]:
        raise RuntimeError("Save/resume replay changed adapter gradients")
    for name, value in uninterrupted_weights.items():
        if not torch.equal(value, resumed_heads.state_dict()[name]):
            raise RuntimeError(f"Save/resume replay changed adapter tensor {name}")
    require_tree_equal(
        uninterrupted_optimizer, resumed_optimizer.state_dict(), "optimizer"
    )
    if not torch.equal(uninterrupted_rng_after, resumed_rng_after):
        raise RuntimeError("Save/resume replay changed the post-update RNG state")
    return {
        "checkpoint_sha256": digest(checkpoint_path),
        "next_update_loss": uninterrupted_loss,
        "exact_loss_replay": True,
        "exact_gradient_replay": True,
        "exact_adapter_state_replay": True,
        "exact_optimizer_state_replay": True,
        "exact_rng_replay": True,
    }


def run_polypgen_gate(sequence, protocol, output_dir, repo, revision):
    settings = protocol["engineering_gate"]["settings"]
    manifest = ROOT / protocol["data"]["manifest"]
    frames = sample_clip(
        load_manifest(manifest, "dev")[sequence],
        settings["polypgen_clip_length"],
        settings["seed"],
    )
    if len(frames) != settings["polypgen_clip_length"]:
        raise RuntimeError(f"Incomplete fixed clip for {sequence}")
    args = SimpleNamespace(
        method="sam2",
        device="cpu",
        checkpoint=CHECKPOINT,
        model_config=protocol["model"]["configuration"],
    )
    model = build_author_predictor(args, repo)
    if model.mem_dim != protocol["model"]["memory_channels"]:
        raise RuntimeError(f"Unexpected SAM2 memory channel count: {model.mem_dim}")
    model.requires_grad_(False)
    original_hash = parameter_digest(model)
    samples = capture_clip(model, frames, output_dir)
    if len(samples) != len(frames) - 1:
        raise RuntimeError(f"Expected every post-prompt frame for {sequence}")

    native_records, native_tensors, native_calls = propagate_signatures(
        model, frames, output_dir
    )
    if len(native_records) != len(samples):
        raise RuntimeError(f"Native propagation missed fixed clip targets for {sequence}")
    if parameter_digest(model) != original_hash:
        raise RuntimeError("Native parity propagation changed a pretrained SAM2 tensor")

    modes = {}
    for mode in ("token", "global"):
        torch.manual_seed(settings["seed"])
        heads = TokenConditionedKeyHeads(channels=model.mem_dim, rank=protocol["method"]["rank"]).train()
        heads.eval()
        zero_records, zero_tensors, zero_calls = propagate_signatures(
            model, frames, output_dir, heads, mode
        )
        if zero_records != native_records or len(zero_tensors) != len(native_tensors):
            raise RuntimeError(f"Zero adapter changed propagated frame receipts for {sequence}/{mode}")
        if any(
            not torch.equal(native_output, zero_output)
            for native_pair, zero_pair in zip(native_tensors, zero_tensors, strict=True)
            for native_output, zero_output in zip(native_pair, zero_pair, strict=True)
        ):
            raise RuntimeError(f"Zero adapter changed full-propagation tensors for {sequence}/{mode}")
        require_layer_order(zero_calls, len(samples))
        if any(row["key_delta_l1"] != 0 or row["key_delta_nonzero"] != 0 for row in zero_calls):
            raise RuntimeError("Zero-initialized adapter produced a nonzero key correction")
        heads.train()
        optimizer = torch.optim.AdamW(
            heads.parameters(), lr=protocol["training"]["optimizer"]["learning_rate"],
            weight_decay=protocol["training"]["optimizer"]["weight_decay"],
            betas=tuple(protocol["training"]["optimizer"]["betas"]),
            eps=protocol["training"]["optimizer"]["eps"],
        )
        initial_loss, initial_calls, initial_grads = training_update(
            model, heads, optimizer, samples, sequence, mode, check_native=True
        )
        first_up = [row["l1"] for row in initial_grads if row["parameter"] == "up.weight"]
        if len(first_up) != 4 or not all(value > 0 for value in first_up):
            raise RuntimeError(f"Zero-initialized output projections did not receive gradients: {first_up}")
        if any(not row["finite"] for row in initial_grads):
            raise RuntimeError("Invalid first-step adapter gradients")
        require_layer_order(initial_calls, len(samples))

        second_loss, second_calls, second_grads = training_update(
            model, heads, optimizer, samples, sequence, mode
        )
        if any(row["parameter"].startswith(("down.", "gate."))
               and row["l1"] <= 0 for row in second_grads):
            raise RuntimeError("Token gate/down-projection did not receive second-step gradients")
        require_layer_order(second_calls, len(samples))

        resume = None
        if sequence == settings["polypgen_sequences"][0] and mode == "token":
            resume = resume_replay_check(
                model, heads, optimizer, samples, sequence, mode, output_dir
            )

        heads.eval()
        predictions = []
        active_calls = []
        with tempfile.TemporaryDirectory(prefix=f"{sequence}_{mode}_", dir=output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError("Fixed gate clip must use its first frame as prompt")
            with NativeTokenConditionedKeyAdapter(model, heads, mode) as observer, torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if object_ids != [1]:
                        raise RuntimeError(f"Unexpected PolypGen object IDs: {object_ids}")
                    if frame == prompt:
                        continue
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite adapted propagation output at {sequence}/{frame}")
                    stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame]
                    predictions.append({
                        "frame": int(frame),
                        "finite": True,
                        "prediction_sha256": tensor_hash(logits),
                        "low_res_sha256": tensor_hash(stored["pred_masks"]),
                    })
                active_calls = list(observer.calls)
            model.reset_state(state)
        require_layer_order(active_calls, len(predictions))
        if len(predictions) != len(samples):
            raise RuntimeError("Adapted propagation missed fixed clip targets")
        if not any(row["key_delta_l1"] > 0 for row in active_calls):
            raise RuntimeError("Trained adapter did not alter any spatial memory key")
        if parameter_digest(model) != original_hash:
            raise RuntimeError("A pretrained SAM2 parameter/buffer changed during the gate")
        modes[mode] = {
            "zero_adapter_loss": initial_loss,
            "second_step_loss": second_loss,
            "first_step_up_projection_gradient_l1": first_up,
            "all_adapter_gradients_finite": True,
            "native_zero_adapter_parity": True,
            "zero_adapter_propagation_receipts": zero_records,
            "zero_adapter_key_trace": zero_calls,
            "post_update_finite_predictions": predictions,
            "post_update_attention_calls": active_calls,
            "save_resume": resume,
        }
    del model
    return {
        "sequence": sequence,
        "source_frames": [int(frame.frame_index) for frame in frames],
        "checkpoint_sha256": digest(CHECKPOINT),
        "manifest_sha256": digest(manifest),
        "sam2_revision": revision,
        "frozen_sam2_parameter_digest": original_hash,
        "modes": modes,
    }


def davis_frames(sequence: str, length: int):
    image_root = DAVIS_ROOT / "JPEGImages" / "480p" / sequence
    mask_root = DAVIS_ROOT / "Annotations" / "480p" / sequence
    images = sorted(image_root.glob("*.jpg"))[:length]
    if len(images) != length:
        raise RuntimeError(f"DAVIS clip {sequence} has fewer than {length} frames")
    masks = [mask_root / f"{image.stem}.png" for image in images]
    if any(not mask.exists() for mask in masks):
        raise RuntimeError(f"DAVIS annotations missing for {sequence} fixed clip")
    return images, masks


def run_davis_multi_object_gate(protocol, output_dir, repo, revision):
    settings = protocol["engineering_gate"]["settings"]
    sequence = settings["davis_multi_object_sequence"]
    images, masks = davis_frames(sequence, settings["davis_clip_length"])
    first_mask = np.asarray(Image.open(masks[0]))
    object_ids = [1, 2]
    if not all(np.any(first_mask == object_id) for object_id in object_ids):
        raise RuntimeError("Frozen DAVIS prompt frame does not contain both object IDs 1 and 2")

    args = SimpleNamespace(
        method="sam2", device="cpu", checkpoint=CHECKPOINT,
        model_config=protocol["model"]["configuration"],
    )
    model = build_author_predictor(args, repo)
    model.requires_grad_(False)
    heads = TokenConditionedKeyHeads(channels=model.mem_dim, rank=protocol["method"]["rank"])
    with torch.no_grad():
        for layer in heads.layers:
            layer.up.weight.normal_(mean=0.0, std=0.01)
    heads.eval()
    rows = []
    with tempfile.TemporaryDirectory(prefix="davis_bike_packing_", dir=output_dir) as temp:
        video_dir = Path(temp) / "video"
        video_dir.mkdir()
        for index, image in enumerate(images):
            shutil.copy2(image, video_dir / f"{index:05d}.jpg")
        state = model.init_state(video_path=str(video_dir), offload_video_to_cpu=True,
                                 offload_state_to_cpu=False)
        for object_id in object_ids:
            mask_array = np.asarray(Image.open(masks[0]))
            model.add_new_mask(
                state,
                frame_idx=0,
                obj_id=object_id,
                mask=torch.from_numpy((mask_array == object_id).copy()),
            )
        if set(state["obj_id_to_idx"]) != set(object_ids):
            raise RuntimeError("SAM2 did not retain both persistent DAVIS object IDs")
        with NativeTokenConditionedKeyAdapter(
            model, heads, "token", inference_state=state
        ) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=0):
                if list(ids) != object_ids:
                    raise RuntimeError(f"Persistent DAVIS object IDs changed at frame {frame}: {ids}")
                if logits.shape[0] != len(object_ids) or not torch.isfinite(logits).all():
                    raise RuntimeError(f"Invalid multi-object DAVIS output at frame {frame}")
                rows.append({
                    "frame": int(frame),
                    "object_ids": list(map(int, ids)),
                    "prediction_sha256": tensor_hash(logits),
                })
        calls = list(observer.calls)
        model.reset_state(state)
    if len(rows) != settings["davis_clip_length"]:
        raise RuntimeError("DAVIS multi-object gate did not cover all five fixed frames")
    if not calls or {row["object_id"] for row in calls} != set(object_ids):
        raise RuntimeError("Attention reads were not associated with both persistent object IDs")
    if any(row["object_id"] not in object_ids for row in calls):
        raise RuntimeError("A memory read was assigned to an unknown DAVIS object")
    for object_id in object_ids:
        object_calls = [row for row in calls if row["object_id"] == object_id]
        if not object_calls or any(row["pointer_keys"] <= 0 for row in object_calls):
            raise RuntimeError(f"Object {object_id} did not use its own pointer-conditioned memory")
        if [row["layer"] for row in object_calls] != [0, 1, 2, 3] * (settings["davis_clip_length"] - 1):
            raise RuntimeError(f"Object {object_id} missed a native attention layer")
    del model
    return {
        "sequence": sequence,
        "frames": [image.stem for image in images],
        "persistent_object_ids": object_ids,
        "predictions": rows,
        "attention_calls": calls,
        "per_object_pointer_isolation": True,
        "checkpoint_sha256": digest(CHECKPOINT),
        "sam2_revision": revision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol_path = ROOT / PROTOCOL
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if args.checkpoint != CHECKPOINT:
            raise RuntimeError("Gate checkpoint must match the frozen protocol path")
        if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("SAM2 checkpoint hash differs from frozen protocol")
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["sam2_revision"]:
            raise RuntimeError("SAM2 repository revision differs from the frozen protocol")
        setup = {
            "protocol_sha256": digest(protocol_path),
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(ROOT / protocol["data"]["manifest"]),
            "sam2_revision": revision,
            "seed": protocol["engineering_gate"]["settings"]["seed"],
            "device": protocol["engineering_gate"]["settings"]["device"],
            "sources": {name: digest(ROOT / name) for name in SOURCE_FILES},
        }
        write(out / "setup.json", setup)
        sequence_results = []
        for sequence in protocol["engineering_gate"]["settings"]["polypgen_sequences"]:
            sequence_results.append(run_polypgen_gate(sequence, protocol, out, repo, revision))
            write(out / "progress.json", {
                "phase": "polypgen_real_frame_gate",
                "completed_sequences": [row["sequence"] for row in sequence_results],
            })
        davis_result = run_davis_multi_object_gate(protocol, out, repo, revision)
        results = {
            "passed": True,
            "scope": "Engineering gate only: seq2/seq3 fixed PolypGen clips plus five-frame DAVIS bike-packing multi-object clip.",
            "polypgen": sequence_results,
            "davis_multi_object": davis_result,
            "full_cohort_training_started": False,
            "efficacy_inference_started": False,
        }
        write(out / "completion.json", results)
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
