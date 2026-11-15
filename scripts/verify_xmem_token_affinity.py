"""Real-checkpoint engineering gate for the XMem-inspired SAM2 memory adapter.

This checks parity, key/pointer geometry, frozen base weights, and adapter
gradients on short seq2/seq3 clips. It is not an efficacy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.io import load_label_mask
from datasets.video import assert_disjoint_videos, load_manifest, stage_frames
from modeling.xmem_token_affinity import (
    XMemTokenAffinityRoPEAttention,
    freeze_sam2_train_xmem_affinity,
    install_xmem_token_affinity,
)
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MANIFEST = ROOT / "data/polypgen_sequence.jsonl"
SOURCE_FILES = (
    "protocols/xmem_token_affinity_polypgen_v1.json",
    "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_parameter_diagnostic_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_parameter_diagnostic_results_2026-11-08.json",
    "protocols/xmem_token_affinity_layer0_integration_trace_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_layer0_integration_trace_results_2026-11-08.json",
    "protocols/xmem_token_affinity_full_read_write_trace_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_full_read_write_trace_hook_correction_2026-11-08.json",
    "protocols/xmem_token_affinity_gradient_scaling_trace_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gradient_scaling_trace_results_2026-11-08.json",
    "protocols/xmem_token_affinity_single_scale_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_single_scale_results_2026-11-08.json",
    "protocols/xmem_token_affinity_corrected_source_gate_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_corrected_source_gate_results_2026-11-08.json",
    "protocols/xmem_token_affinity_full_grid_similarity_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gate_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gate_status_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_manifest_split_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gate_verifier_amendment_2026-11-08.json",
    "protocols/iterative_memory_pilot_v1.json",
    "modeling/xmem_token_affinity.py",
    "modeling/xmem_token_affinity_training.py",
    "scripts/run_xmem_token_affinity.py",
    "scripts/verify_xmem_token_affinity.py",
    "tests/test_xmem_token_affinity.py",
    "tests/test_xmem_token_affinity_training.py",
    "tests/test_run_xmem_token_affinity.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "datasets/instance_masks.py",
    "modeling/cfbi_training.py",
    "modeling/complementary_memory/training.py",
    "modeling/complementary_memory/integration.py",
    "modeling/native_memory_layout.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "scripts/verify_on_demand_clip.py",
    "scripts/run_complementary_memory_pilot.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "scripts/bank_integrity.py",
)


def resolve_gate_settings(args, protocol):
    """Apply frozen defaults and reject a gate run with substituted settings."""
    gate = protocol["engineering_gate"]
    expected = gate["settings"]
    requested = {
        "sequences": args.sequences,
        "clip_length": args.length,
        "seed": args.seed,
        "device": args.device,
    }
    defaults = {
        "sequences": protocol["data"]["real_checkpoint_gate_sequences"],
        "clip_length": expected["clip_length"],
        "seed": expected["seed"],
        "device": expected["device"],
    }
    resolved = {}
    for key, default in defaults.items():
        value = default if requested[key] is None else requested[key]
        if value != default:
            raise ValueError(f"Gate setting {key}={value!r} differs from frozen value {default!r}")
        resolved[key] = list(value) if key == "sequences" else value
    if resolved["sequences"] != protocol["data"]["real_checkpoint_gate_sequences"]:
        raise ValueError("Gate sequence order differs from the frozen protocol")
    args.sequences = resolved["sequences"]
    args.length = resolved["clip_length"]
    args.seed = resolved["seed"]
    args.device = resolved["device"]
    return resolved


def seed_everything(seed: int) -> None:
    """Reset Python, NumPy, and PyTorch RNGs for a reproducible arm start."""
    if seed < 0:
        raise ValueError("Experiment seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frame_input_hashes(frames):
    return [{
        "frame_index": int(frame.frame_index),
        "image_sha256": digest(frame.image_path),
        "mask_sha256": digest(frame.mask_path),
    } for frame in frames]


def _activation_receipt(value):
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_hash(value),
    }


def peak_process_rss_bytes() -> int:
    """Return the process high-water RSS in bytes on macOS and Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@contextmanager
def trace_memory_attention_integration(model, layer_indices=(0, 1, 2, 3)):
    """Record selected cross-attention boundaries without changing computation."""
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("Layer trace must select unique attention layer indices")
    rows = []
    rows_by_layer = {index: [] for index in layer_indices}
    handles = []

    for layer_index in layer_indices:
        try:
            attention = model.memory_attention.layers[layer_index].cross_attn_image
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(f"Cannot locate memory cross-attention layer {layer_index}") from exc

        def pre_hook(module, args, kwargs, selected_layer=layer_index):
            q = kwargs.get("q", args[0] if len(args) > 0 else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            if not all(isinstance(value, torch.Tensor) for value in (q, k, v)):
                raise RuntimeError(f"Layer {selected_layer} trace could not read q/k/v")
            utility_q = kwargs.get("utility_q")
            utility_memory = kwargs.get("utility_memory")
            if utility_q is None:
                utility_q = q
            if utility_memory is None:
                utility_memory = v
            pointer_count = kwargs.get(
                "num_k_exclude_rope", args[3] if len(args) > 3 else 0
            )
            row = {
                "layer": selected_layer,
                "call_index": len(rows_by_layer[selected_layer]),
                "inputs": {
                    "q": _activation_receipt(q),
                    "k": _activation_receipt(k),
                    "v": _activation_receipt(v),
                    "effective_affinity_q": _activation_receipt(utility_q),
                    "effective_affinity_memory": _activation_receipt(utility_memory),
                    "pointer_tail_length": int(pointer_count),
                },
                "xmem_strength_before_call": (
                    float(module.xmem_strength.detach().cpu())
                    if hasattr(module, "xmem_strength") else None
                ),
            }
            rows.append(row)
            rows_by_layer[selected_layer].append(row)

        def forward_hook(module, args, kwargs, output, selected_layer=layer_index):
            if not isinstance(output, torch.Tensor) or not rows_by_layer[selected_layer]:
                raise RuntimeError(f"Layer {selected_layer} trace expected one tensor output")
            row = rows_by_layer[selected_layer][-1]
            row["output"] = _activation_receipt(output)
            if not output.requires_grad:
                raise RuntimeError(
                    f"Layer {selected_layer} attention output lacks a training gradient"
                )

            def record_incoming_gradient(gradient, selected_row=row):
                detached = gradient.detach().float()
                selected_row["incoming_output_gradient"] = _activation_receipt(gradient)
                selected_row["incoming_gradient_l1"] = float(detached.abs().sum())
                selected_row["incoming_gradient_max_abs"] = float(detached.abs().max())
                return gradient

            output.register_hook(record_incoming_gradient)

        handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(attention.register_forward_hook(forward_hook, with_kwargs=True))
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def trace_layer0_attention_integration(model):
    """Compatibility helper for the first-layer trace fixture."""
    with trace_memory_attention_integration(model, layer_indices=(0,)) as rows:
        yield rows


def record_track_step_write(rows, frame_index, output):
    """Hash returned prediction/write tensors at the training call site."""
    if not isinstance(output, dict):
        raise RuntimeError("track_step observer expected a tensor dictionary")
    row = {"frame_index": int(frame_index)}
    for key in (
        "pred_masks", "maskmem_features", "maskmem_pos_enc", "obj_ptr",
        "object_score_logits",
    ):
        row[key] = _activation_receipt(output.get(key))
    rows.append(row)


def frozen_parameter_digest(model) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if ".xmem_" in name:
            continue
        value = parameter.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def record_attention_calls(model, *, require_adapter=False):
    rows = []
    handles = []
    for index, layer in enumerate(model.memory_attention.layers):
        attention = layer.cross_attn_image
        if require_adapter and not isinstance(attention, XMemTokenAffinityRoPEAttention):
            raise RuntimeError(f"Memory layer {index} was not adapted")

        def pre_hook(module, args, kwargs, layer_index=index):
            q = kwargs.get("q", args[0] if args else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            pointer_count = int(kwargs.get("num_k_exclude_rope", 0))
            if q is None or k is None or v is None:
                raise RuntimeError("Could not inspect native SAM2 attention arguments")
            if pointer_count > k.shape[1] or k.shape[1] != v.shape[1]:
                raise RuntimeError("Invalid spatial/pointer key geometry")
            rows.append({
                "layer": layer_index,
                "queries": int(q.shape[1]),
                "keys": int(k.shape[1]),
                "spatial_keys": int(k.shape[1] - pointer_count),
                "pointer_keys": pointer_count,
                "query_sha256": tensor_hash(q),
                "positioned_key_sha256": tensor_hash(k),
                "value_sha256": tensor_hash(v),
                "pointer_key_sha256": tensor_hash(k[:, -pointer_count:]) if pointer_count else None,
                "pointer_value_sha256": tensor_hash(v[:, -pointer_count:]) if pointer_count else None,
            })

        handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
    return rows, handles


def validate_post_update_attention_calls(native_calls, updated_calls):
    """Check fixed memory/pointers without forbidding expected later queries."""
    if [row["layer"] for row in native_calls] != [0, 1, 2, 3]:
        raise RuntimeError("Native reference did not traverse all four attention layers")
    if [row["layer"] for row in updated_calls] != [0, 1, 2, 3]:
        raise RuntimeError("Post-update read did not traverse all four attention layers")
    if native_calls[0]["query_sha256"] != updated_calls[0]["query_sha256"]:
        raise RuntimeError("The adapter changed the first-layer current query input")
    preserved = (
        "keys", "spatial_keys", "pointer_keys", "positioned_key_sha256",
        "value_sha256", "pointer_key_sha256", "pointer_value_sha256",
    )
    for native_call, updated_call in zip(native_calls, updated_calls, strict=True):
        if any(native_call[key] != updated_call[key] for key in preserved):
            raise RuntimeError(
                f"Adapter changed memory/pointer structure at layer {updated_call['layer']}"
            )


def run_read(model, sample):
    return model.track_step(
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


def _exact_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys()
                and all(_exact_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _first_exact_difference(left, right, path="root"):
    """Report the first unequal value in nested replay receipts or states."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right):
            return None
        report = {"path": path, "left_shape": list(left.shape),
                  "right_shape": list(right.shape), "left_dtype": str(left.dtype),
                  "right_dtype": str(right.dtype)}
        if left.shape == right.shape and left.numel():
            difference = (left.detach().cpu().float() - right.detach().cpu().float()).abs()
            report["max_abs_difference"] = float(difference.max())
            report["unequal_elements"] = int(difference.ne(0).sum())
        return report
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"path": path, "left_keys": list(left), "right_keys": list(right)}
        for key in left:
            difference = _first_exact_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = _first_exact_difference(a, b, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(left) is not type(right) or left != right:
        report = {"path": path, "left": repr(left), "right": repr(right)}
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            report["absolute_difference"] = abs(float(left) - float(right))
        return report
    return None


def _tree_sha256(value):
    """Hash optimizer state without lossy tensor string formatting."""
    hasher = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                hasher.update(repr(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            hasher.update(repr(item).encode())

    update(value)
    return hasher.hexdigest()


def _validate_predicted_memory_writes(training_result, expected_targets):
    """Confirm the training replay writes each arm's own prediction, never GT."""
    if training_result.get("prompt_index") != 0:
        raise RuntimeError("Training replay did not use the frozen first-frame prompt")
    if training_result.get("post_prompt_targets") != expected_targets:
        raise RuntimeError("Training replay target count differs from the real clip")
    if training_result.get("ground_truth_used_only_for_loss") is not True:
        raise RuntimeError("Training replay does not certify loss-only ground truth")
    writes = [
        write
        for chunk in training_result.get("chunks", [])
        for write in chunk.get("writes", [])
    ]
    if len(writes) != expected_targets:
        raise RuntimeError("Training replay did not write one predicted memory per target")
    if training_result.get("predicted_memory_writes") != expected_targets:
        raise RuntimeError("Training replay summary disagrees with its memory-write records")
    if any(
        row.get("source") != "this_arm_predicted_output"
        or row.get("target_used_for_memory") is not False
        for row in writes
    ):
        raise RuntimeError("A training memory write used an unexpected mask source")
    return True


def verify_training_resume(frames_by_sequence, checkpoint, repo, output):
    """Compare an uninterrupted next real clip with a disk save/resume replay."""
    from scripts import run_xmem_token_affinity as runner

    amendment = json.loads(
        (ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json").read_text()
    )
    rng_amendment_path = ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json"
    autograd_amendment_path = ROOT / "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json"
    reduction_amendment_path = ROOT / "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json"
    config = amendment["training"]
    plan = [
        {"epoch": 1, "sequence": sequence, "draw": 0,
         "indices": list(range(len(frames_by_sequence[sequence]))),
         "source_frames": [int(frame.frame_index) for frame in frames_by_sequence[sequence]]}
        for sequence in ("seq2", "seq3")
    ]
    arm_receipts, state_after_first_clip = {}, {}
    for arm in ("token_affinity", "frame_broadcast"):
        granularity = "token" if arm == "token_affinity" else "frame"
        mode = "regional" if arm == "token_affinity" else "generic"
        model, parameters, _ = runner.make_model(
            checkpoint, repo, granularity, config["seed"]
        )
        initial_adapter = runner.adapter_state(model)
        optimizer = runner.optimizer_for(parameters, config)
        pretrained_hash = frozen_parameter_digest(model)
        first_result = runner.train_clip(
            model, optimizer, frames_by_sequence["seq2"], "seq2", output, config, mode
        )
        first_writes_valid = _validate_predicted_memory_writes(
            first_result, len(frames_by_sequence["seq2"]) - 1
        )
        first_state = runner.adapter_state(model)
        state_after_first_clip[arm] = first_state
        core_hash = hashlib.sha256(f"resume-gate:{arm}".encode()).hexdigest()
        payload = {
            "core_sha256": core_hash,
            "arm": arm,
            "protocol_amendment_sha256": digest(
                ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json"
            ),
            "rng_amendment_sha256": digest(rng_amendment_path),
            "autograd_determinism_amendment_sha256": digest(autograd_amendment_path),
            "reduction_determinism_amendment_sha256": digest(reduction_amendment_path),
            "adapter": first_state,
            "optimizer": optimizer.state_dict(),
            "rng_state": runner.capture_rng_state(),
            "next_clip_index": 1,
            "completed_clip_rows": [{"plan": plan[0], "result": first_result}],
            "validation": {},
            "best_score": -float("inf"),
            "best_epoch": None,
            "best_adapter_sha256": None,
            "pretrained_parameter_sha256": pretrained_hash,
        }
        arm_dir = output / "resume_equivalence" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = arm_dir / "resume.pt"
        receipt_path = arm_dir / "resume_receipt.json"
        runner.atomic_torch_save(checkpoint_path, payload)
        runner.atomic_json(receipt_path, {
            "sha256": digest(checkpoint_path), "next_clip_index": 1,
            "core_sha256": core_hash, "arm": arm,
            "rng_amendment_sha256": digest(rng_amendment_path),
            "autograd_determinism_amendment_sha256": digest(autograd_amendment_path),
            "reduction_determinism_amendment_sha256": digest(reduction_amendment_path),
        })

        # Reference continuation retains the live model, optimizer and RNG.
        expected_result = runner.train_clip(
            model, optimizer, frames_by_sequence["seq3"], "seq3", output, config, mode
        )
        expected_writes_valid = _validate_predicted_memory_writes(
            expected_result, len(frames_by_sequence["seq3"]) - 1
        )
        expected_adapter = runner.adapter_state(model)
        expected_optimizer = optimizer.state_dict()
        expected_rng = runner.capture_rng_state()
        if frozen_parameter_digest(model) != pretrained_hash:
            raise RuntimeError("Resume reference changed pretrained SAM2 weights")
        del model, optimizer

        resumed_model, resumed_parameters, _ = runner.make_model(
            checkpoint, repo, granularity, config["seed"]
        )
        deterministic_initialization = _exact_equal(
            initial_adapter, runner.adapter_state(resumed_model)
        )
        resumed_optimizer = runner.optimizer_for(resumed_parameters, config)
        restored = runner.restore_training_checkpoint(
            checkpoint_path, receipt_path, resumed_model, resumed_optimizer, arm_dir,
            core_hash=core_hash, arm=arm, plan=plan,
            amendment_sha256=digest(
                ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json"
            ),
            rng_amendment_sha256=digest(rng_amendment_path),
            autograd_determinism_amendment_sha256=digest(autograd_amendment_path),
            reduction_determinism_amendment_sha256=digest(reduction_amendment_path),
        )
        if restored["next_clip_index"] != 1:
            raise RuntimeError("Resume cursor did not advance to the second clip")
        actual_result = runner.train_clip(
            resumed_model, resumed_optimizer, frames_by_sequence["seq3"],
            "seq3", output, config, mode
        )
        actual_writes_valid = _validate_predicted_memory_writes(
            actual_result, len(frames_by_sequence["seq3"]) - 1
        )
        actual_adapter = runner.adapter_state(resumed_model)
        actual_optimizer = resumed_optimizer.state_dict()
        actual_rng = runner.capture_rng_state()
        checks = {
            "deterministic_initialization": deterministic_initialization,
            "clip_record": _exact_equal(actual_result, expected_result),
            "adapter_state": _exact_equal(actual_adapter, expected_adapter),
            "optimizer_state": _exact_equal(actual_optimizer, expected_optimizer),
            "rng_state": _exact_equal(actual_rng, expected_rng),
            "pretrained_weights": frozen_parameter_digest(resumed_model) == pretrained_hash,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Actual real-clip save/resume mismatch for {arm}: {checks}")
        arm_receipts[arm] = {
            "checks": checks,
            "own_prediction_memory_writes": (
                first_writes_valid and expected_writes_valid and actual_writes_valid
            ),
            "first_clip_sha256": hashlib.sha256(
                json.dumps(first_result, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "next_clip_sha256": hashlib.sha256(
                json.dumps(actual_result, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "resume_checkpoint_sha256": digest(checkpoint_path),
            "complete_rng_checkpoint_provenance": (
                payload["rng_amendment_sha256"] == digest(rng_amendment_path)
                and json.loads(receipt_path.read_text())["rng_amendment_sha256"]
                == digest(rng_amendment_path)
                and payload["autograd_determinism_amendment_sha256"]
                == digest(autograd_amendment_path)
                and json.loads(receipt_path.read_text())["autograd_determinism_amendment_sha256"]
                == digest(autograd_amendment_path)
                and payload["reduction_determinism_amendment_sha256"]
                == digest(reduction_amendment_path)
                and json.loads(receipt_path.read_text())["reduction_determinism_amendment_sha256"]
                == digest(reduction_amendment_path)
            ),
        }
        del resumed_model, resumed_optimizer

    return {"arms": arm_receipts}, state_after_first_clip


def verify_fresh_start_repeatability(
    clip_specs, checkpoint, repo, output, arms=("token_affinity", "frame_broadcast")
):
    """Replay the first two frozen training clips from two clean arm starts."""
    from scripts import run_xmem_token_affinity as runner

    amendment = json.loads(
        (ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json").read_text()
    )
    config = amendment["training"]
    rng_amendment_hash = digest(
        ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json"
    )
    autograd_amendment_hash = digest(
        ROOT / "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json"
    )
    reduction_amendment_hash = digest(
        ROOT / "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json"
    )
    arm_receipts = {}
    initial_rng_by_arm = {}
    for arm in arms:
        granularity = "token" if arm == "token_affinity" else "frame"
        mode = "regional" if arm == "token_affinity" else "generic"
        repetitions = []
        for repetition in range(2):
            model, parameters, _ = runner.make_model(
                checkpoint, repo, granularity, config["seed"]
            )
            optimizer = runner.optimizer_for(parameters, config)
            initial_adapter = runner.adapter_state(model)
            initial_optimizer = optimizer.state_dict()
            initial_rng = runner.rng_state_digest(runner.capture_rng_state())
            initial_rng_by_arm.setdefault(arm, []).append(initial_rng)
            initial_pretrained = frozen_parameter_digest(model)
            clip_results = []
            clip_summaries = []
            for spec in clip_specs:
                sequence, frames = spec["sequence"], spec["frames"]
                if arm == "token_affinity":
                    track_step_trace = []

                    def observe_write(frame_index, output):
                        record_track_step_write(track_step_trace, frame_index, output)

                    with trace_memory_attention_integration(model) as attention_trace:
                        result = runner.train_clip(
                            model, optimizer, frames, sequence,
                            output, config, mode, include_gradient_hashes=True,
                            write_observer=observe_write,
                        )
                    if not attention_trace or any(
                        "incoming_output_gradient" not in row for row in attention_trace
                    ):
                        raise RuntimeError("Memory attention trace missed an output gradient")
                    expected_reads = len(frames) - 1
                    for layer_index in range(4):
                        layer_calls = [row for row in attention_trace
                                       if row["layer"] == layer_index]
                        if len(layer_calls) != expected_reads:
                            raise RuntimeError(
                                f"Layer {layer_index} trace recorded {len(layer_calls)} reads; "
                                f"expected {expected_reads}"
                            )
                    if len(track_step_trace) != expected_reads:
                        raise RuntimeError(
                            f"Per-frame track_step observer recorded {len(track_step_trace)} writes; "
                            f"expected {expected_reads}"
                        )
                    result["memory_attention_trace"] = attention_trace
                    result["track_step_write_trace"] = track_step_trace
                else:
                    result = runner.train_clip(
                        model, optimizer, frames, sequence,
                        output, config, mode, include_gradient_hashes=True,
                    )
                _validate_predicted_memory_writes(result, len(frames) - 1)
                clip_results.append({"plan": spec["plan"], "result": result})
                clip_summaries.append({
                    "sequence": sequence,
                    "plan": spec["plan"],
                    "losses": [row["loss"] for row in result["chunks"]],
                    "optimizer_updates": result["optimizer_updates"],
                    "read_count": sum(len(row["reads"]) for row in result["chunks"]),
                    "predicted_memory_writes": result["predicted_memory_writes"],
                    "gradient_norms_before_clip": [
                        row["gradient_norm_before_clip"] for row in result["chunks"]
                    ],
                    "scalar_gradients_before_clip": [
                        row.get("scalar_gradients_before_clip") for row in result["chunks"]
                    ],
                    "scalar_gradients_after_clip": [
                        row.get("scalar_gradients_after_clip") for row in result["chunks"]
                    ],
                    "memory_attention_trace": result.get("memory_attention_trace", []),
                    "track_step_write_trace": result.get("track_step_write_trace", []),
                })
            repetitions.append({
                "initial_adapter": initial_adapter,
                "initial_optimizer": initial_optimizer,
                "initial_rng": initial_rng,
                "clip_results": clip_results,
                "clip_summaries": clip_summaries,
                "final_adapter": runner.adapter_state(model),
                "final_optimizer": optimizer.state_dict(),
                "final_rng": runner.rng_state_digest(runner.capture_rng_state()),
                "pretrained_unchanged": frozen_parameter_digest(model) == initial_pretrained,
            })
            runner.atomic_json(output / "fresh_start_repeatability_progress.json", {
                "status": "running",
                "peak_process_rss_bytes": peak_process_rss_bytes(),
                "completed_arms": arm_receipts,
                "current_arm": arm,
                "completed_repetitions": [
                    {
                        "initial_adapter_sha256": runner.state_digest(row["initial_adapter"]),
                        "initial_optimizer_sha256": _tree_sha256(row["initial_optimizer"]),
                        "initial_rng_sha256": row["initial_rng"],
                        "clips": row["clip_summaries"],
                        "final_adapter_sha256": runner.state_digest(row["final_adapter"]),
                        "final_optimizer_sha256": _tree_sha256(row["final_optimizer"]),
                        "final_rng_sha256": row["final_rng"],
                        "pretrained_unchanged": row["pretrained_unchanged"],
                    }
                    for row in repetitions
                ],
            })
            del model, optimizer

        first, second = repetitions
        checks = {
            "initial_adapter": _exact_equal(first["initial_adapter"], second["initial_adapter"]),
            "initial_optimizer": _exact_equal(first["initial_optimizer"], second["initial_optimizer"]),
            "initial_full_rng": first["initial_rng"] == second["initial_rng"],
            "clip_results_and_losses": _exact_equal(first["clip_results"], second["clip_results"]),
            "final_adapter": _exact_equal(first["final_adapter"], second["final_adapter"]),
            "final_optimizer_tensor_state": _exact_equal(
                first["final_optimizer"], second["final_optimizer"]
            ),
            "final_full_rng": first["final_rng"] == second["final_rng"],
            "pretrained_unchanged": first["pretrained_unchanged"] and second["pretrained_unchanged"],
        }
        differences = {
            "first_clip_result": _first_exact_difference(
                first["clip_results"][0], second["clip_results"][0], "clip_results[0]"
            ),
            "second_clip_result": _first_exact_difference(
                first["clip_results"][1], second["clip_results"][1], "clip_results[1]"
            ),
            "final_adapter": _first_exact_difference(
                first["final_adapter"], second["final_adapter"], "adapter"
            ),
            "final_optimizer": _first_exact_difference(
                first["final_optimizer"], second["final_optimizer"], "optimizer"
            ),
        }
        arm_receipts[arm] = {
            "checks": checks,
            "peak_process_rss_bytes": peak_process_rss_bytes(),
            "repetitions": [
                {
                    "initial_adapter_sha256": runner.state_digest(row["initial_adapter"]),
                    "initial_optimizer_sha256": runner._json_sha256(row["initial_optimizer"]),
                    "initial_rng_sha256": row["initial_rng"],
                    "clips": row["clip_summaries"],
                    "final_adapter_sha256": runner.state_digest(row["final_adapter"]),
                    "final_optimizer_sha256": _tree_sha256(row["final_optimizer"]),
                    "final_rng_sha256": row["final_rng"],
                    "pretrained_unchanged": row["pretrained_unchanged"],
                }
                for row in repetitions
            ],
            "first_differences": differences,
            "initial_rng_sha256": first["initial_rng"],
            "final_rng_sha256": first["final_rng"],
            "clip_result_sha256": hashlib.sha256(
                json.dumps(first["clip_results"], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "final_adapter_sha256": runner.state_digest(first["final_adapter"]),
            "final_optimizer_sha256": hashlib.sha256(
                json.dumps(first["final_optimizer"], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "rng_amendment_sha256": rng_amendment_hash,
            "autograd_determinism_amendment_sha256": autograd_amendment_hash,
            "reduction_determinism_amendment_sha256": reduction_amendment_hash,
        }
        runner.atomic_json(output / "fresh_start_repeatability_progress.json", {
            "status": "passed" if all(checks.values()) else "failed",
            "peak_process_rss_bytes": peak_process_rss_bytes(),
            "completed_arms": arm_receipts,
            "matched_arm_initial_rng_state": None,
        })
        if not all(checks.values()):
            loss_summary = [
                [{"sequence": clip["sequence"], "losses": clip["losses"]}
                 for clip in repetition["clip_summaries"]]
                for repetition in repetitions
            ]
            raise RuntimeError(
                f"Fresh-start repeatability failed for {arm}: {checks}; "
                f"first_differences={differences}; "
                f"clip_losses={loss_summary}"
            )
    matched_arm_rng = None
    if set(arms) == {"token_affinity", "frame_broadcast"}:
        matched_arm_rng = (
            initial_rng_by_arm["token_affinity"][0]
            == initial_rng_by_arm["frame_broadcast"][0]
            and initial_rng_by_arm["token_affinity"][1]
            == initial_rng_by_arm["frame_broadcast"][1]
        )
    if matched_arm_rng is False:
        raise RuntimeError("Fresh-start matched arm RNG states differ")
    result = {
        "arms": arm_receipts,
        "matched_arm_initial_rng_state": matched_arm_rng,
        "frozen_clip_prefix": [spec["plan"] for spec in clip_specs],
    }
    runner.atomic_json(output / "fresh_start_repeatability_progress.json", {
        "status": "passed",
        "peak_process_rss_bytes": peak_process_rss_bytes(),
        "completed_arms": arm_receipts,
        "matched_arm_initial_rng_state": matched_arm_rng,
    })
    return result


def multi_object_id_isolation_gate(model, frames, output):
    """Check persistent ID/order isolation on a split-prompt real-image fixture."""
    mask = load_label_mask(frames[0].mask_path, [1]) == 1
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise RuntimeError("Multi-object fixture requires a visible first-frame object")
    middle = int(np.median(coordinates[:, 1]))
    left = mask.copy()
    left[:, middle + 1:] = False
    right = mask & ~left
    masks = {7: left, 42: right}
    if not masks[7].any() or not masks[42].any():
        raise RuntimeError("Split-prompt fixture could not create two nonempty IDs")

    def run(order):
        result = {}
        with tempfile.TemporaryDirectory(prefix="xmem_instances_", dir=output) as folder:
            stage_frames(frames, Path(folder))
            with torch.inference_mode():
                state = model.init_state(
                    video_path=folder, offload_video_to_cpu=True, offload_state_to_cpu=False
                )
                try:
                    for object_id in order:
                        model.add_new_mask(
                            state, frame_idx=0, obj_id=object_id,
                            mask=torch.from_numpy(masks[object_id].copy()),
                        )
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=0):
                        if set(ids) != set(order) or logits.shape[0] != len(order):
                            raise RuntimeError("Persistent object IDs changed during propagation")
                        for index, object_id in enumerate(ids):
                            object_index = state["obj_id_to_idx"][object_id]
                            bank = state["output_dict_per_obj"][object_index]
                            stored = bank["cond_frame_outputs"].get(frame)
                            if stored is None:
                                stored = bank["non_cond_frame_outputs"].get(frame)
                            if stored is None:
                                raise RuntimeError("A persistent ID lacks its own stored memory")
                            result[f"{frame}:{object_id}"] = {
                                "prediction": tensor_hash(logits[index]),
                                "stored_mask": tensor_hash(stored["pred_masks"]),
                                "stored_feature": tensor_hash(stored["maskmem_features"]),
                                "pointer": tensor_hash(stored["obj_ptr"]),
                            }
                finally:
                    model.reset_state(state)
        return result

    forward = run([7, 42])
    reverse = run([42, 7])
    if forward != reverse:
        raise RuntimeError("Multi-object predictions depend on object registration order")
    for object_id in (7, 42):
        isolated = run([object_id])
        expected = {key: value for key, value in forward.items()
                    if key.endswith(f":{object_id}")}
        if isolated != expected:
            raise RuntimeError(f"Memory or predictions leaked across object ID {object_id}")
    return {
        "passed": True,
        "object_ids": [7, 42],
        "frames": len(frames),
        "registration_order_invariant": True,
        "isolated_rollouts_match": True,
        "scope": "Split-prompt engineering fixture on real single-object imagery; not natural multi-object efficacy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--repeatability-only", action="store_true",
        help="Run only the bounded fresh-start repeatability diagnostic on the first two frozen training clips.",
    )
    parser.add_argument(
        "--repeatability-arms", nargs="+", choices=("token_affinity", "frame_broadcast"),
        default=("token_affinity", "frame_broadcast"),
        help="Arms for --repeatability-only; defaults to both matched arms.",
    )
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocols/xmem_token_affinity_polypgen_v1.json").read_text())
    settings = resolve_gate_settings(args, protocol)
    expected_environment = protocol["engineering_gate"]["environment"]
    actual_environment = {"python": platform.python_version(), "torch": str(torch.__version__)}
    if actual_environment != expected_environment:
        raise RuntimeError(
            f"Gate environment {actual_environment} differs from frozen environment "
            f"{expected_environment}"
        )
    if settings["clip_length"] < 2:
        raise ValueError("A clip must contain the prompt and at least one read")
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash differs from the frozen protocol")
    if digest(MANIFEST) != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("PolypGen manifest differs from the frozen protocol")
    if protocol["model"]["sam2_revision"] != SAM2_REVISION:
        raise RuntimeError("Protocol pins a different SAM2 revision")

    split_amendment_path = ROOT / "protocols/xmem_token_affinity_manifest_split_amendment_2026-11-08.json"
    split_amendment = json.loads(split_amendment_path.read_text())
    split_data = split_amendment["data"]
    efficacy_amendment_path = ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json"
    autograd_amendment_path = ROOT / "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json"
    autograd_amendment = json.loads(autograd_amendment_path.read_text())
    reduction_amendment_path = ROOT / "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json"
    reduction_amendment = json.loads(reduction_amendment_path.read_text())
    parameter_diagnostic_path = ROOT / "protocols/xmem_token_affinity_parameter_diagnostic_amendment_2026-11-08.json"
    parameter_diagnostic_amendment = json.loads(parameter_diagnostic_path.read_text())
    parameter_diagnostic_result_path = ROOT / "protocols/xmem_token_affinity_parameter_diagnostic_results_2026-11-08.json"
    layer0_trace_amendment_path = ROOT / "protocols/xmem_token_affinity_layer0_integration_trace_amendment_2026-11-08.json"
    layer0_trace_amendment = json.loads(layer0_trace_amendment_path.read_text())
    layer0_trace_result_path = ROOT / "protocols/xmem_token_affinity_layer0_integration_trace_results_2026-11-08.json"
    full_trace_amendment_path = ROOT / "protocols/xmem_token_affinity_full_read_write_trace_amendment_2026-11-08.json"
    full_trace_amendment = json.loads(full_trace_amendment_path.read_text())
    trace_hook_correction_path = ROOT / "protocols/xmem_token_affinity_full_read_write_trace_hook_correction_2026-11-08.json"
    trace_hook_correction = json.loads(trace_hook_correction_path.read_text())
    gradient_trace_amendment_path = ROOT / "protocols/xmem_token_affinity_gradient_scaling_trace_amendment_2026-11-08.json"
    gradient_trace_amendment = json.loads(gradient_trace_amendment_path.read_text())
    affinity_accumulation_amendment_path = ROOT / "protocols/xmem_token_affinity_single_scale_amendment_2026-11-08.json"
    affinity_accumulation_amendment = json.loads(
        affinity_accumulation_amendment_path.read_text()
    )
    rng_amendment_path = ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json"
    if autograd_amendment["parent_hashes"] != {
        "method_protocol_sha256": digest(ROOT / "protocols/xmem_token_affinity_polypgen_v1.json"),
        "efficacy_amendment_sha256": digest(efficacy_amendment_path),
        "rng_amendment_sha256": digest(rng_amendment_path),
    }:
        raise RuntimeError("Autograd determinism amendment does not match its parent protocols")
    if reduction_amendment["parent_amendment_sha256"] != digest(autograd_amendment_path):
        raise RuntimeError("Reduction determinism amendment does not match its parent amendment")
    if parameter_diagnostic_amendment["parent_amendment_sha256"] != digest(reduction_amendment_path):
        raise RuntimeError("Parameter-diagnostic amendment does not match its parent amendment")
    if layer0_trace_amendment["parent_amendment_sha256"] != digest(parameter_diagnostic_path):
        raise RuntimeError("Layer-0 trace amendment does not match its parent amendment")
    if layer0_trace_amendment["parent_result_sha256"] != digest(parameter_diagnostic_result_path):
        raise RuntimeError("Layer-0 trace amendment does not match its parent diagnostic result")
    if full_trace_amendment["parent_amendment_sha256"] != digest(layer0_trace_amendment_path):
        raise RuntimeError("Full read/write trace amendment does not match its parent amendment")
    if full_trace_amendment["parent_result_sha256"] != digest(layer0_trace_result_path):
        raise RuntimeError("Full read/write trace amendment does not match its parent result")
    if trace_hook_correction["parent_amendment_sha256"] != digest(full_trace_amendment_path):
        raise RuntimeError("Write-trace correction does not match its parent amendment")
    if trace_hook_correction["failed_attempt"]["failure_json_sha256"] != (
        "18ead8366e76202ba5d29f5d468b6a5f65aca4eba0ba13adc32cec1711f34b48"
    ):
        raise RuntimeError("Write-trace correction does not bind the observed instrumentation failure")
    if gradient_trace_amendment["parent_amendment_sha256"] != digest(trace_hook_correction_path):
        raise RuntimeError("Gradient-scaling trace does not match its parent amendment")
    if affinity_accumulation_amendment["parent_amendment_sha256"] != digest(
        gradient_trace_amendment_path
    ) or affinity_accumulation_amendment["parent_result_sha256"] != digest(
        ROOT / "protocols/xmem_token_affinity_gradient_scaling_trace_results_2026-11-08.json"
    ):
        raise RuntimeError("Single-scale correction does not match its frozen evidence")
    corrected_source_gate_path = ROOT / "protocols/xmem_token_affinity_corrected_source_gate_amendment_2026-11-08.json"
    corrected_source_gate = json.loads(corrected_source_gate_path.read_text())
    if corrected_source_gate["parent_amendment_sha256"] != digest(
        affinity_accumulation_amendment_path
    ) or corrected_source_gate["parent_result_sha256"] != digest(
        ROOT / "protocols/xmem_token_affinity_single_scale_results_2026-11-08.json"
    ):
        raise RuntimeError("Corrected-source gate amendment does not match its parent evidence")
    corrected_source_gate_result_path = ROOT / "protocols/xmem_token_affinity_corrected_source_gate_results_2026-11-08.json"
    full_grid_similarity_path = ROOT / "protocols/xmem_token_affinity_full_grid_similarity_amendment_2026-11-08.json"
    full_grid_similarity_amendment = json.loads(full_grid_similarity_path.read_text())
    if full_grid_similarity_amendment["parent_amendment_sha256"] != digest(
        corrected_source_gate_path
    ) or full_grid_similarity_amendment["parent_result_sha256"] != digest(
        corrected_source_gate_result_path
    ):
        raise RuntimeError("Full-grid similarity amendment does not match its parent evidence")
    if split_data["manifest_sha256"] != digest(MANIFEST):
        raise RuntimeError("Manifest split amendment pins a different PolypGen manifest")
    if split_data["method_protocol_sha256"] != digest(
        ROOT / "protocols/xmem_token_affinity_polypgen_v1.json"
    ):
        raise RuntimeError("Manifest split amendment pins a different method protocol")
    if split_data["efficacy_amendment_sha256"] != digest(efficacy_amendment_path):
        raise RuntimeError("Manifest split amendment pins a different efficacy amendment")
    if split_data["role_to_manifest_split"].get("train") != "dev" \
            or split_data["role_to_manifest_split"].get("validation") != "dev" \
            or split_data["role_to_manifest_split"].get("excluded") != "dev" \
            or split_data["role_to_manifest_split"].get("test") != "test":
        raise RuntimeError("Manifest split amendment changed a frozen role assignment")

    torch.set_num_threads(protocol["engineering_gate"]["settings"]["torch_threads"])
    torch.use_deterministic_algorithms(
        protocol["engineering_gate"]["settings"]["deterministic_algorithms"]
    )
    repo, revision = activate_author_repo()
    if revision != SAM2_REVISION:
        raise RuntimeError(f"Unexpected SAM2 revision: {revision}")
    repo_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if repo_status:
        raise RuntimeError("Pinned SAM2 checkout has local changes; refusing the real-data gate")

    videos_by_split = {
        split_name: load_manifest(MANIFEST, split_name)
        for split_name in sorted(set(split_data["role_to_manifest_split"].values()))
    }
    assert_disjoint_videos(*videos_by_split.values())
    for split_name, expected_ids in split_data["expected_sequences_by_manifest_split"].items():
        if set(videos_by_split[split_name]) != set(expected_ids):
            raise RuntimeError(f"Manifest split {split_name!r} differs from its frozen inventory")
    videos = videos_by_split[protocol["data"]["split"]]
    clip_frames = {}
    clip_inputs = {}
    for sequence in settings["sequences"]:
        if sequence not in videos:
            raise ValueError(f"Unknown sequence: {sequence}")
        frames = sample_clip(
            videos[sequence], settings["clip_length"], settings["seed"] + int(sequence[3:])
        )
        if len(frames) != settings["clip_length"]:
            raise RuntimeError(f"Sampler returned the wrong clip length for {sequence}")
        indices = [int(frame.frame_index) for frame in frames]
        expected_indices = protocol["data"]["real_checkpoint_gate_clips"][sequence]
        if indices != expected_indices:
            raise RuntimeError(f"Sampled clip differs from frozen frames for {sequence}: {indices}")
        inputs = frame_input_hashes(frames)
        if inputs != protocol["data"]["real_checkpoint_gate_inputs"][sequence]:
            raise RuntimeError(f"Sampled image/mask files differ from frozen hashes for {sequence}")
        clip_frames[sequence] = frames
        clip_inputs[sequence] = inputs

    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_files = (*SOURCE_FILES, "external/sam2/sam2/modeling/sam/transformer.py",
                    "external/sam2/sam2/modeling/memory_attention.py",
                    "external/sam2/sam2/modeling/sam2_base.py")
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    snapshot_path = output / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            archive.write(ROOT / name, arcname=name)
    setup = {
        "protocol_sha256": digest(ROOT / "protocols/xmem_token_affinity_polypgen_v1.json"),
        "efficacy_amendment_sha256": digest(
            ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json"
        ),
        "rng_amendment_sha256": digest(
            ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json"
        ),
        "autograd_determinism_amendment_sha256": digest(autograd_amendment_path),
        "reduction_determinism_amendment_sha256": digest(reduction_amendment_path),
        "layer0_integration_trace_amendment_sha256": digest(layer0_trace_amendment_path),
        "full_read_write_trace_amendment_sha256": digest(full_trace_amendment_path),
        "full_read_write_trace_hook_correction_sha256": digest(trace_hook_correction_path),
        "gradient_scaling_trace_amendment_sha256": digest(gradient_trace_amendment_path),
        "single_scale_amendment_sha256": digest(affinity_accumulation_amendment_path),
        "corrected_source_gate_amendment_sha256": digest(corrected_source_gate_path),
        "full_grid_similarity_amendment_sha256": digest(full_grid_similarity_path),
        "manifest_split_amendment_sha256": digest(split_amendment_path),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": revision,
        "sam2_worktree_clean": True,
        "settings": settings,
        "adapter_initialization_seeds": {
            sequence: settings["seed"] + int(sequence[3:])
            for sequence in settings["sequences"]
        },
        "environment": actual_environment,
        "clips": clip_inputs,
        "sources": source_hashes,
        "source_snapshot_sha256": digest(snapshot_path),
    }
    write(output / "setup.json", setup)

    try:
        results = {}

        if args.repeatability_only:
            amendment = json.loads(efficacy_amendment_path.read_text())
            training_videos = videos_by_split[
                split_data["role_to_manifest_split"]["train"]
            ]
            from scripts import run_xmem_token_affinity as efficacy_runner

            training_plan = efficacy_runner.clip_plan(training_videos, amendment)
            clip_specs = [{
                "sequence": row["sequence"],
                "frames": [training_videos[row["sequence"]][index] for index in row["indices"]],
                "plan": row,
            } for row in training_plan[:2]]
            fresh_repeatability = verify_fresh_start_repeatability(
                clip_specs, args.checkpoint, repo, output, arms=tuple(args.repeatability_arms)
            )
            if source_hashes != {name: digest(ROOT / name) for name in source_files}:
                raise RuntimeError("A tracked implementation or author source changed during the diagnostic")
            result_path = output / "fresh_start_repeatability_progress.json"
            write(output / "completion.json", {
                "status": "passed",
                "diagnostic_only": True,
                "scope": "Fresh-start repeatability diagnostic only; not an efficacy result and does not authorize training.",
                "checkpoint_sha256": digest(args.checkpoint),
                "setup_sha256": digest(output / "setup.json"),
                "source_snapshot_sha256": digest(snapshot_path),
                "result_sha256": digest(result_path),
                "fresh_start_repeatability": fresh_repeatability,
                "elapsed_seconds": time.monotonic() - started,
            })
            return

        for sequence in settings["sequences"]:
            frames = clip_frames[sequence]
            model = build_author_predictor(SimpleNamespace(
                method="sam2", device=args.device, checkpoint=args.checkpoint,
                model_config=protocol["model"]["config"],
            ), repo)
            model.eval().requires_grad_(False)
            native_parameters = frozen_parameter_digest(model)
            native_calls, native_handles = record_attention_calls(model)
            samples = capture_clip(model, frames, output)
            for handle in native_handles:
                handle.remove()
            if len(samples) != settings["clip_length"] - 1:
                raise RuntimeError(f"Native capture incomplete for {sequence}")
            if len(native_calls) != len(samples) * 4:
                raise RuntimeError("Native propagation did not read all four attention layers per target")
            native_logits = {sample["frame"]: sample["native_logits"].clone() for sample in samples}
            input_hashes = {
                sample["frame"]: fingerprint(sample["output_dict"])["sha256"]
                for sample in samples
            }

            adapter_seed = setup["adapter_initialization_seeds"][sequence]
            seed_everything(adapter_seed)
            install_receipt = install_xmem_token_affinity(
                model, key_dim=64, granularity="token"
            )
            trainable = freeze_sam2_train_xmem_affinity(model)
            model.eval()
            call_rows, handles = record_attention_calls(model, require_adapter=True)
            parity_rows = []
            with torch.inference_mode():
                for sample in samples:
                    calls_before = len(call_rows)
                    output_row = run_read(model, sample)
                    calls = call_rows[calls_before:]
                    if [row["layer"] for row in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Expected all four attention layers on {sequence}")
                    if not torch.equal(output_row["pred_masks"], native_logits[sample["frame"]]):
                        raise RuntimeError(f"Zero-strength SAM2 replay mismatch at {sequence}/{sample['frame']}")
                    if any(row["spatial_keys"] <= 0 for row in calls):
                        raise RuntimeError("A memory read had no spatial keys")
                    native_frame_calls = native_calls[
                        len(parity_rows) * 4:(len(parity_rows) + 1) * 4
                    ]
                    parity_fields = (
                        "layer", "queries", "keys", "spatial_keys", "pointer_keys",
                        "query_sha256", "positioned_key_sha256", "value_sha256",
                        "pointer_key_sha256", "pointer_value_sha256",
                    )
                    for native_call, adapted_call in zip(native_frame_calls, calls, strict=True):
                        if any(native_call[key] != adapted_call[key] for key in parity_fields):
                            raise RuntimeError(
                                f"Native attention inputs/positions/pointers differ at "
                                f"{sequence}/{sample['frame']}/layer{adapted_call['layer']}"
                            )
                    parity_rows.append({
                        "frame": int(sample["frame"]),
                        "bit_identical_to_native": True,
                        "prediction_sha256": tensor_hash(output_row["pred_masks"]),
                        "layers": calls,
                    })

            if frozen_parameter_digest(model) != native_parameters:
                raise RuntimeError("A pretrained SAM2 parameter changed during adapter installation")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Unexpected gradient before adapter training check")
            if any(fingerprint(sample["output_dict"])["sha256"] != input_hashes[sample["frame"]]
                   for sample in samples):
                raise RuntimeError("A replay mutated the saved native memory bank")

            # Check differentiability on actual captured features. Only the
            # adapter is trainable; this single update is an engineering check.
            # Keep the frozen model in evaluation mode. The adapter's forward
            # computes a differentiable affinity whenever gradients are enabled;
            # toggling attention.train() would change SAM2's native dropout path.
            optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
            sample = samples[0]
            reference_calls = parity_rows[0]["layers"]
            call_rows.clear()
            optimizer.zero_grad(set_to_none=True)
            prediction = run_read(model, sample)["pred_masks"]
            first_update_calls = list(call_rows)
            if [row["layer"] for row in first_update_calls] != [0, 1, 2, 3]:
                raise RuntimeError("The first adapter update did not traverse all four layers")
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Nonfinite prediction at the initial adapter strength")
            # The current query must match only at layer 0. Later queries are
            # outputs of earlier adapted layers and are expected to change.
            # Stored spatial keys, values, positions, and pointer tensors must
            # stay fixed at every layer.
            validate_post_update_attention_calls(reference_calls, first_update_calls)
            target = F.interpolate(
                sample["target"], size=prediction.shape[-2:], mode="nearest"
            )
            loss = F.binary_cross_entropy_with_logits(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise RuntimeError("Real-feature adapter loss was nonfinite")
            loss.backward()
            first_grads = [p.grad for p in trainable]
            if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in first_grads):
                raise RuntimeError("No finite nonzero adapter gradient on real features")
            if any(
                p.grad is not None for name, p in model.named_parameters() if ".xmem_" not in name
            ):
                raise RuntimeError("A frozen SAM2 parameter acquired a gradient")
            optimizer.step()
            strengths = [float(layer.cross_attn_image.xmem_strength.detach())
                         for layer in model.memory_attention.layers]
            if any(not math.isfinite(value) or value == 0.0 for value in strengths):
                raise RuntimeError("Each memory layer must reach finite nonzero affinity strength")
            optimizer.zero_grad(set_to_none=True)
            call_rows.clear()
            second_prediction = run_read(model, sample)["pred_masks"]
            second_update_calls = list(call_rows)
            for handle in handles:
                handle.remove()
            if not torch.isfinite(second_prediction).all():
                raise RuntimeError("Nonfinite prediction after the adapter update")
            validate_post_update_attention_calls(reference_calls, second_update_calls)
            F.binary_cross_entropy_with_logits(
                second_prediction.float(), target.float()
            ).backward()
            if any(
                p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0
                for p in trainable
            ):
                raise RuntimeError("An XMem adapter parameter lacks a finite nonzero post-update gradient")
            if frozen_parameter_digest(model) != native_parameters:
                raise RuntimeError("A pretrained SAM2 parameter changed during adapter update")
            if any(fingerprint(sample["output_dict"])["sha256"] != input_hashes[sample["frame"]]
                   for sample in samples):
                raise RuntimeError("A gradient replay mutated the saved native memory bank")

            results[sequence] = {
                "manifest_frames": [int(frame.frame_index) for frame in frames],
                "checkpoint_sha256": digest(args.checkpoint),
                "native_parameter_sha256": native_parameters,
                "adapter_initialization_seed": adapter_seed,
                "adapter_install": install_receipt,
                "zero_strength_parity": parity_rows,
                "four_layer_reads": all(
                    [call["layer"] for call in row["layers"]] == [0, 1, 2, 3]
                    for row in parity_rows
                ),
                "memory_and_pointer_geometry_unchanged": True,
                "adapter_only_finite_nonzero_gradients": True,
                "real_feature_loss": float(loss.detach()),
                "trained_affinity_strengths": strengths,
                "adapter_parameters_with_gradients": len(trainable),
                "native_bank_inputs_unchanged": True,
                "pretrained_parameters_unchanged": True,
                "scope": "Short real-data engineering check only; no efficacy result.",
            }
            del model

        resume_result, adapters_after_clip = verify_training_resume(
            clip_frames, args.checkpoint, repo, output
        )
        from scripts import run_xmem_token_affinity as efficacy_runner
        amendment = json.loads(
            (ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json").read_text()
        )
        training_videos = videos_by_split[
            split_data["role_to_manifest_split"]["train"]
        ]
        training_plan = efficacy_runner.clip_plan(training_videos, amendment)
        clip_specs = []
        for row in training_plan[:2]:
            clip_specs.append({
                "sequence": row["sequence"],
                "frames": [training_videos[row["sequence"]][index] for index in row["indices"]],
                "plan": row,
            })
        fresh_repeatability = verify_fresh_start_repeatability(
            clip_specs, args.checkpoint, repo, output
        )
        multi_object = {}
        for arm in ("token_affinity", "frame_broadcast"):
            granularity = "token" if arm == "token_affinity" else "frame"
            seed_everything(amendment["training"]["seed"])
            multi_model = build_author_predictor(SimpleNamespace(
                method="sam2", device=args.device, checkpoint=args.checkpoint,
                model_config=protocol["model"]["config"],
            ), repo)
            install_xmem_token_affinity(multi_model, key_dim=64, granularity=granularity)
            efficacy_runner.load_adapter_state(multi_model, adapters_after_clip[arm])
            for parameter in multi_model.parameters():
                parameter.requires_grad_(False)
            multi_model.eval()
            multi_object[arm] = multi_object_id_isolation_gate(
                multi_model, clip_frames["seq2"], output
            )
            del multi_model

        if source_hashes != {name: digest(ROOT / name) for name in source_files}:
            raise RuntimeError("A tracked implementation or author source changed during the gate")
        if clip_inputs != {
            sequence: frame_input_hashes(clip_frames[sequence]) for sequence in settings["sequences"]
        }:
            raise RuntimeError("A sampled image or mask changed during the gate")
        required_checks = json.loads(
            (ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json").read_text()
        )["gate_completion_schema"]["required_boolean_checks"]
        rng_required_checks = json.loads(
            (ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json").read_text()
        )["required_engineering_gate_checks"]
        resume_checks = [
            arm_result["checks"] for arm_result in resume_result["arms"].values()
        ]
        checks = {
            "native_zero_strength_parity": all(
                row["bit_identical_to_native"]
                for sequence_result in results.values()
                for row in sequence_result["zero_strength_parity"]
            ),
            "four_layer_reads": all(
                sequence_result["four_layer_reads"] for sequence_result in results.values()
            ),
            "memory_and_pointer_geometry_unchanged": all(
                sequence_result["memory_and_pointer_geometry_unchanged"]
                for sequence_result in results.values()
            ),
            "pretrained_sam2_weights_unchanged": all(
                sequence_result["pretrained_parameters_unchanged"]
                for sequence_result in results.values()
            ) and all(check["pretrained_weights"] for check in resume_checks),
            "adapter_only_finite_gradients": all(
                sequence_result["adapter_only_finite_nonzero_gradients"]
                for sequence_result in results.values()
            ),
            "native_bank_integrity": all(
                sequence_result["native_bank_inputs_unchanged"]
                for sequence_result in results.values()
            ),
            "deterministic_initialization": all(
                check["deterministic_initialization"] for check in resume_checks
            ),
            "own_prediction_memory_writes": all(
                arm_result["own_prediction_memory_writes"]
                for arm_result in resume_result["arms"].values()
            ),
            "adapter_optimizer_rng_resume_equivalence": all(
                all(check[name] for name in (
                    "clip_record", "adapter_state", "optimizer_state", "rng_state"
                ))
                for check in resume_checks
            ),
            "persistent_multi_object_id_isolation": all(
                arm_result["passed"]
                and arm_result["registration_order_invariant"]
                and arm_result["isolated_rollouts_match"]
                for arm_result in multi_object.values()
            ),
            "fresh_start_repeatability": all(
                all(arm_result["checks"].values())
                for arm_result in fresh_repeatability["arms"].values()
            ),
            "matched_arm_initial_rng_state": fresh_repeatability[
                "matched_arm_initial_rng_state"
            ],
            "complete_rng_checkpoint_provenance": all(
                arm_result["complete_rng_checkpoint_provenance"]
                for arm_result in resume_result["arms"].values()
            ),
        }
        missing_or_failed = [
            name for name in (*required_checks, *rng_required_checks)
            if checks.get(name) is not True
        ]
        if missing_or_failed:
            raise RuntimeError(f"Required engineering checks failed: {missing_or_failed}")
        results_payload = {
            "sequences": results,
            "save_resume_equivalence": resume_result,
            "fresh_start_repeatability": fresh_repeatability,
            "persistent_multi_object_id_isolation": multi_object,
            "scope": "Short engineering gate only; no efficacy result.",
        }
        write(output / "results.json", results_payload)
        results_sha256 = digest(output / "results.json")
        receipt = {
            "passed": True,
            "protocol_complete": True,
            "status": "complete",
            "checks": checks,
            "sequences": results,
            "save_resume_equivalence": resume_result,
            "fresh_start_repeatability": fresh_repeatability,
            "persistent_multi_object_id_isolation": multi_object,
            "sam2_revision": revision,
            "setup_sha256": digest(output / "setup.json"),
            "protocol_sha256": setup["protocol_sha256"],
            "efficacy_amendment_sha256": setup["efficacy_amendment_sha256"],
            "rng_amendment_sha256": setup["rng_amendment_sha256"],
            "autograd_determinism_amendment_sha256": setup["autograd_determinism_amendment_sha256"],
            "reduction_determinism_amendment_sha256": setup["reduction_determinism_amendment_sha256"],
            "manifest_split_amendment_sha256": setup["manifest_split_amendment_sha256"],
            "manifest_sha256": setup["manifest_sha256"],
            "source_snapshot_sha256": digest(snapshot_path),
            "results_sha256": results_sha256,
            "implementation_sha256": source_hashes["modeling/xmem_token_affinity.py"],
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Engineering gate only. No segmentation improvement is claimed.",
        }
        write(output / "completion.json", receipt)
        (output / "results.md").write_text(
            "# XMem token-affinity engineering gate\n\n"
            "All frozen engineering checks passed on short real SAM2/PolypGen clips, "
            "including full-state save-resume, repeated fresh-start prefixes, and split-prompt "
            "persistent-ID isolation for token and frame-broadcast arms. This does not "
            "measure efficacy or Dice improvement.\n"
        )
    except BaseException as error:
        write(output / "failure.json", {
            "status": "stopped",
            "exception_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "scope": "Engineering gate only; no efficacy inference was started.",
            "failed_at_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
