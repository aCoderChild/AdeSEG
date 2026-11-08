"""Run a real-frame engineering gate; never run the efficacy cohort here."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest, stage_frames
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.complementary_memory.training import (
    capture_current_features,
    reader_clip_loss,
)
from modeling.cfbi_training import clone_tensors
from modeling.mavos_focal_value_modulation import (
    MAVOSFocalValueResidual,
    MAVOSMemoryValueModulation,
)
from modeling.mavos_value_contrastive import MAVOSValueContrastive
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import (
    build_author_predictor,
    source_identity,
    tensor_hash,
)
from scripts.mavos_value_utils import activate_author_repo, prepare, target_for


PROTOCOL = "protocols/mavos_focal_value_contrastive_polypgen_v1.json"
MANIFEST = "data/polypgen_sequence.jsonl"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def equal_tree(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right)
        )
    return left == right


def make_adapters(model, *, permutation=None):
    adapters = []
    for layer in model.memory_attention.layers:
        attention = layer.cross_attn_image
        grid = int(attention.freqs_cis.shape[0] ** 0.5)
        if grid * grid != attention.freqs_cis.shape[0]:
            raise RuntimeError("SAM2 RoPE grid is not square")
        adapters.append(MAVOSFocalValueResidual(
            attention.kv_in_dim,
            (grid, grid),
            focal_window=3,
            focal_level=2,
            focal_factor=2,
            spatial_permutation=permutation,
        ).to(device=attention.v_proj.weight.device, dtype=attention.v_proj.weight.dtype))
    return torch.nn.ModuleList(adapters)


def rollout(model, frames, output: Path, adapters=None):
    model_hash = parameter_digest(model)
    with tempfile.TemporaryDirectory(prefix="mavos_value_frames_", dir=output) as temporary:
        state, prompt = prepare(model, frames, Path(temporary))
        observer = NativeComplementaryRead(
            model, None, "regional", inference_state=state, trace_only=True
        )
        integration = None if adapters is None else MAVOSMemoryValueModulation(
            model, adapters, record_value_receipts=True
        )
        rows, iterator = [], None
        try:
            with torch.inference_mode():
                with observer:
                    if integration is None:
                        iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                        for frame, ids, logits in iterator:
                            if ids != [1]:
                                raise RuntimeError("Unexpected persistent object ID")
                            if frame == prompt:
                                continue
                            if len(observer.calls) != len(rows) + 1:
                                raise RuntimeError("Missing trace for a propagated memory read")
                            rows.append({
                                "frame": frame,
                                "prediction_sha256": tensor_hash(logits),
                                "read": observer.calls[-1],
                            })
                    else:
                        with integration:
                            iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                            for frame, ids, logits in iterator:
                                if ids != [1]:
                                    raise RuntimeError("Unexpected persistent object ID")
                                if frame == prompt:
                                    continue
                                if len(observer.calls) != len(rows) + 1:
                                    raise RuntimeError("Missing trace for a propagated memory read")
                                rows.append({
                                    "frame": frame,
                                    "prediction_sha256": tensor_hash(logits),
                                    "read": observer.calls[-1],
                                })
            if len(rows) != len(frames) - prompt - 1:
                raise RuntimeError("Incomplete post-prompt frame coverage")
            if len(observer.calls) != len(rows):
                raise RuntimeError("Memory read count differs from propagated targets")
            for call in observer.calls:
                if call["layers"] != [0, 1, 2, 3]:
                    raise RuntimeError("A memory-attention layer was missed")
                if not call["stored_mask_alignment"] or not call["native_positions_verified"]:
                    raise RuntimeError("Native memory or positional alignment failed")
            value_calls = [] if integration is None else list(integration.value_calls)
        finally:
            if iterator is not None:
                iterator.close()
            with torch.inference_mode():
                model.reset_state(state)
        if parameter_digest(model) != model_hash:
            raise RuntimeError("Pretrained SAM2 parameters changed during propagation")
        return {"prompt_frame": prompt, "rows": rows, "reads": observer.calls,
                "value_calls": value_calls, "sam2_parameter_sha256": model_hash}


def train_one_update(model, adapters, optimizer, samples, initial_bank,
                     target_by_frame, contrast_weight):
    optimizer.zero_grad(set_to_none=True)
    observer = NativeComplementaryRead(model, None, "regional", trace_only=True)
    integration = MAVOSMemoryValueModulation(
        model, adapters, record_value_receipts=True,
        capture_adapted_values=True,
    )
    auxiliary = MAVOSValueContrastive(
        integration, observer, target_by_frame, weight=contrast_weight
    )
    with observer, integration:
        loss, _, writes = reader_clip_loss(
            model, samples, clone_tensors(initial_bank), auxiliary=auxiliary
        )
    if not torch.isfinite(loss):
        raise RuntimeError("Real-clip adapter loss is nonfinite")
    loss.backward()
    gradients = []
    for index, adapter in enumerate(adapters):
        grad = adapter.h.weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing/nonfinite output-projection gradient at layer {index}")
        norm = float(grad.detach().abs().sum())
        if norm <= 0:
            raise RuntimeError(f"Zero adapter gradient at layer {index}")
        gradients.append(norm)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen SAM2 acquired a parameter gradient")
    if len(observer.calls) != len(samples) or any(
        row["layers"] != [0, 1, 2, 3] for row in observer.calls
    ):
        raise RuntimeError("Training clip did not traverse all four memory layers")
    expected_calls = 4 * len(samples)
    if len(integration.value_calls) != expected_calls:
        raise RuntimeError("Training adapter did not execute in every memory layer")
    if not all(row["pointer_tail_bit_identical"] for row in integration.value_calls):
        raise RuntimeError("Adapter changed an object-pointer value")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "contrastive_weight": float(contrast_weight),
        "contrastive_records": auxiliary.records,
        "gradient_l1_by_layer": gradients,
        "training_writes": writes,
        "memory_reads": observer.calls,
        "value_calls": integration.value_calls,
    }


def real_gradient_resume_gate(model, frames, output: Path, seed: int, *,
                              permutation=None, contrast_weight=0.1):
    with tempfile.TemporaryDirectory(prefix="mavos_value_train_", dir=output) as temporary:
        state, prompt = prepare(model, frames, Path(temporary))
        with torch.inference_mode():
            model.propagate_in_video_preflight(state)
        with torch.inference_mode(False):
            initial_bank = clone_tensors(state["output_dict_per_obj"][0])
        sample_count = min(2, len(frames) - prompt - 1)
        if sample_count < 1:
            raise RuntimeError("Gradient gate needs a post-prompt frame")
        samples = []
        target_by_frame = {}
        for frame_index, frame_record in enumerate(frames):
            target_by_frame[frame_index] = target_for(frame_record)[0]
        for frame in range(prompt + 1, prompt + 1 + sample_count):
            visual = capture_current_features(model, state, frame)
            samples.append({
                "frame": frame,
                "num_frames": len(frames),
                "vision": visual["vision"],
                "positions": visual["positions"],
                "sizes": visual["sizes"],
                "target": target_by_frame[frame],
                "mask_inputs": None,
                "point_inputs": None,
            })
        model.reset_state(state)

    torch.manual_seed(seed)
    adapters = make_adapters(model, permutation=permutation)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=1e-4, weight_decay=1e-3)
    initial = {
        "adapters": {name: value.detach().clone() for name, value in adapters.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.random.get_rng_state().clone(),
    }
    initial_path = output / "resume_initial.pt"
    torch.save(initial, initial_path)
    model_hash = parameter_digest(model)
    first = train_one_update(
        model, adapters, optimizer, samples, initial_bank,
        target_by_frame, contrast_weight,
    )
    expected = {
        "adapters": {name: value.detach().clone() for name, value in adapters.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "record": first,
    }

    restored = torch.load(initial_path, map_location="cpu", weights_only=True)
    adapters.load_state_dict(restored["adapters"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    torch.random.set_rng_state(restored["rng"])
    second = train_one_update(
        model, adapters, optimizer, samples, initial_bank,
        target_by_frame, contrast_weight,
    )
    replay = {
        "adapters": adapters.state_dict(),
        "optimizer": optimizer.state_dict(),
        "record": second,
    }
    if not equal_tree(expected, replay):
        raise RuntimeError("Real-clip training did not replay exactly after checkpoint restore")
    if parameter_digest(model) != model_hash:
        raise RuntimeError("SAM2 parameter hash changed during real-clip training")
    return {
        "sequence_clip_frames": [sample["frame"] for sample in samples],
        "prompt_frame": prompt,
        "loss": first["loss"],
        "contrastive_weight": float(contrast_weight),
        "contrastive_active_calls": sum(
            row["contrastive_active"] for row in first["contrastive_records"]
        ),
        "contrastive_records": first["contrastive_records"],
        "adapter_final_sha256": parameter_digest(adapters),
        "gradient_l1_by_layer": first["gradient_l1_by_layer"],
        "all_four_layers_trained": True,
        "pointers_preserved": True,
        "resume_replay_exact": True,
        "sam2_parameter_hash_unchanged": True,
        "resume_checkpoint_sha256": digest(initial_path),
    }


def multi_object_rollout(model, frames, prompt, output: Path, object_order, masks, adapters=None):
    model_hash = parameter_digest(model)
    with tempfile.TemporaryDirectory(prefix="mavos_value_multiobj_", dir=output) as temporary:
        stage_frames(frames, Path(temporary))
        state = model.init_state(
            video_path=temporary,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        with torch.inference_mode():
            for object_id in object_order:
                model.add_new_mask(state, frame_idx=prompt, obj_id=object_id, mask=masks[object_id])
        observer = NativeComplementaryRead(
            model, None, "regional", inference_state=state, trace_only=True
        )
        integration = None if adapters is None else MAVOSMemoryValueModulation(
            model, adapters, record_value_receipts=True
        )
        outputs, iterator = {}, None
        try:
            with torch.inference_mode():
                with observer:
                    if integration is None:
                        iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                        for frame, ids, logits in iterator:
                            if frame >= prompt + 3:
                                break
                            if set(ids) != set(object_order) or logits.shape[0] != len(object_order):
                                raise RuntimeError("Persistent multi-object IDs changed")
                            for index, object_id in enumerate(ids):
                                outputs[f"{frame}:{object_id}"] = tensor_hash(logits[index])
                    else:
                        with integration:
                            iterator = model.propagate_in_video(
                                state, start_frame_idx=prompt
                            )
                            for frame, ids, logits in iterator:
                                if frame >= prompt + 3:
                                    break
                                if set(ids) != set(object_order) or logits.shape[0] != len(object_order):
                                    raise RuntimeError("Persistent multi-object IDs changed")
                                for index, object_id in enumerate(ids):
                                    outputs[f"{frame}:{object_id}"] = tensor_hash(logits[index])
        finally:
            if iterator is not None:
                iterator.close()
            with torch.inference_mode():
                model.reset_state(state)
        if parameter_digest(model) != model_hash:
            raise RuntimeError("SAM2 parameters changed in multi-object gate")
        return {"outputs": outputs, "reads": observer.calls,
                "value_calls": [] if integration is None else integration.value_calls}


def multi_object_identity_gate(model, frames, output: Path, *, permutation=None):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    mask = torch.from_numpy(
        (load_label_mask(frames[prompt].mask_path, [1]) == 1).copy()
    )
    coordinates = mask.nonzero()
    if len(coordinates) < 2:
        raise RuntimeError("Prompt mask is too small for the two-instance engineering fixture")
    middle = int(coordinates[:, 1].median())
    left = mask.clone()
    left[:, middle:] = False
    right = mask & ~left
    if not left.any() or not right.any():
        raise RuntimeError("Could not split the real prompt mask into two nonempty fixture IDs")
    masks = {7: left, 42: right}

    native = multi_object_rollout(model, frames, prompt, output, [7, 42], masks)
    zero = make_adapters(model, permutation=permutation)
    adapted = multi_object_rollout(model, frames, prompt, output, [7, 42], masks, zero)
    reversed_order = multi_object_rollout(model, frames, prompt, output, [42, 7], masks, zero)
    if native["outputs"] != adapted["outputs"]:
        raise RuntimeError("Zero-residual adapter changed multi-object native outputs")
    if adapted["outputs"] != reversed_order["outputs"]:
        raise RuntimeError("Prompt insertion order changed persistent-ID outputs")
    if not all(row["pointer_tail_bit_identical"] for row in adapted["value_calls"]):
        raise RuntimeError("Multi-object pointer values changed")
    expected_ids = {7, 42}
    if {row["object_id"] for row in adapted["reads"]} != expected_ids:
        raise RuntimeError("Memory read trace lost a persistent object ID")
    return {
        "passed": True,
        "object_ids": [7, 42],
        "frames_checked": 3,
        "zero_residual_native_identity": True,
        "prompt_order_invariant": True,
        "fixture": "Two disjoint halves of a real PolypGen prompt mask; engineering ID-isolation check only, not natural multi-object efficacy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        manifest = ROOT / MANIFEST
        videos = load_manifest(manifest, "dev")
        sampled = {
            name: sample_clip(videos[name], args.sequence_length, args.seed + index)
            for index, name in enumerate(("seq2", "seq3"))
        }
        repo, revision = activate_author_repo()
        checkpoint_hash = digest(args.checkpoint)
        if checkpoint_hash != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash differs from the frozen candidate protocol")
        model = build_author_predictor(argparse.Namespace(
            method="sam2", device="cpu", checkpoint=args.checkpoint,
            model_config=protocol["model"]["configuration"],
        ), repo)
        model.requires_grad_(False)
        model.eval()
        if revision != protocol["model"]["sam2_revision"]:
            raise RuntimeError("SAM2 repository revision differs from the protocol")
        if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
            raise RuntimeError("SAM2 temporal-memory settings differ from protocol")

        identity = {}
        for name, frames in sampled.items():
            native = rollout(model, frames, output)
            adapters = make_adapters(model)
            control_permutation = torch.randperm(
                4096, generator=torch.Generator().manual_seed(protocol["training"]["seed"])
            )
            control_adapters = make_adapters(model, permutation=control_permutation)
            conditions = {
                "C1_focal_zero": rollout(model, frames, output, adapters),
                "C2_shuffled_zero": rollout(model, frames, output, control_adapters),
            }
            native_hashes = [row["prediction_sha256"] for row in native["rows"]]
            for label, adapted in conditions.items():
                if native_hashes != [row["prediction_sha256"] for row in adapted["rows"]]:
                    raise RuntimeError(f"Zero-residual outputs differ from native SAM2 on {name}/{label}")
                if len(adapted["value_calls"]) != 4 * len(adapted["rows"]):
                    raise RuntimeError(f"Incomplete four-layer value coverage on {name}/{label}")
                if not all(row["pointer_tail_bit_identical"] for row in adapted["value_calls"]):
                    raise RuntimeError(f"Pointer values changed on {name}/{label}")
                if native["reads"] != adapted["reads"]:
                    raise RuntimeError(f"Native memory/position/pointer trace differs on {name}/{label}")
            identity[name] = {
                "frames": [frame.frame_index for frame in frames],
                "prompt_frame": native["prompt_frame"],
                "zero_residual_native_prediction_hashes_match": list(conditions),
                "all_four_layers_verified": True,
                "pointer_values_bit_identical": True,
                "memory_alignment_and_positions_verified": True,
                "native": native,
                **conditions,
            }

        control_permutation = torch.randperm(
            4096, generator=torch.Generator().manual_seed(protocol["training"]["seed"])
        )
        gradient = {
            "adapter_segmentation_only": real_gradient_resume_gate(
                model, sampled["seq2"], output, args.seed, contrast_weight=0.0
            ),
            "adapter_segmentation_plus_contrastive": real_gradient_resume_gate(
                model, sampled["seq2"], output, args.seed, contrast_weight=0.1
            ),
        }
        if gradient["adapter_segmentation_plus_contrastive"]["contrastive_active_calls"] <= 0:
            raise RuntimeError("Real-frame gate did not activate FG/BG contrastive supervision")
        if (gradient["adapter_segmentation_only"]["adapter_final_sha256"]
                == gradient["adapter_segmentation_plus_contrastive"]["adapter_final_sha256"]):
            raise RuntimeError("Contrastive loss did not change the trained adapter state")
        multi_object = {
            "C1_focal": multi_object_identity_gate(model, sampled["seq2"], output),
            "C2_shuffled": multi_object_identity_gate(
                model, sampled["seq2"], output, permutation=control_permutation
            ),
        }
        result = {
            "passed": True,
            "scope": "Real-frame engineering gate only; no efficacy training or PolypGen test inference.",
            "device": "cpu",
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": digest(manifest),
            "sam2_revision": revision,
            "protocol_sha256": digest(ROOT / PROTOCOL),
            "identity": identity,
            "gradient_resume": gradient,
            "multi_object_gate": multi_object,
        }
        write_json(output / "results.json", result)
        write_json(output / "completion.json", {
            "passed": True,
            "results_sha256": digest(output / "results.json"),
            "scope": result["scope"],
        })
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise
    finally:
        if "model" in locals():
            del model
            gc.collect()


if __name__ == "__main__":
    main()
