"""Real-frame smoke gate for the frozen cohort runner's exact train path.

This is an engineering check only. It exercises reader_clip_loss with the
token-conditioned key hooks, checks zero-adapter memory-write parity, both
trainable modes, and exact next-update save/resume replay on a short seq2 clip.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features, clone_tensors
from scripts.mavos_value_utils import activate_author_repo, atomic_json, prepare, target_for
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT,
    PROTOCOL,
    _capture_rng,
    _make_heads,
    _model_digest,
    _restore_rng,
    _source_hashes,
    _train_chunk,
    _tree_digest,
)


def _assert_equal(left, right, path="value"):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"Tensor type changed at {path}")
        if not torch.equal(left, right):
            raise RuntimeError(f"Tensor differs at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            raise RuntimeError(f"Mapping differs at {path}")
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"Sequence structure differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_equal(a, b, f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"Value differs at {path}: {left!r} != {right!r}")


def _samples(model, state, frames, indices):
    rows = []
    for index in indices:
        visual = capture_current_features(model, state, index)
        rows.append({
            "frame": index,
            "num_frames": len(frames),
            "vision": visual["vision"],
            "positions": visual["positions"],
            "sizes": visual["sizes"],
            "target": target_for(frames[index])[0],
            "mask_inputs": None,
            "point_inputs": None,
        })
    return rows


def _adamw(heads, protocol):
    cfg = protocol["training"]["optimizer"]
    return torch.optim.AdamW(
        heads.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
    )


def run(output_dir: Path, checkpoint: Path) -> dict:
    output_dir = validate_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(protocol["training"]["seed"]))
    manifest = ROOT / protocol["data"]["manifest"]
    sequences = load_manifest(manifest, "dev")
    frames = sample_clip(sequences["seq2"], 6, int(protocol["engineering_gate"]["settings"]["seed"]))
    repo_root, revision = activate_author_repo()
    if revision != protocol["model"]["sam2_revision"]:
        raise RuntimeError("SAM2 source revision differs from the frozen protocol")
    model = build_author_predictor(argparse.Namespace(
        method="sam2", device="cpu", checkpoint=checkpoint,
        model_config=protocol["model"]["configuration"],
    ), repo_root)
    model.requires_grad_(False)
    model.eval()
    original_hash = _model_digest(model)
    fixed_frames = [frame.frame_index for frame in frames]
    results = {"sequence": "seq2", "sampled_manifest_frames": fixed_frames, "modes": {}}

    # Reference ordinary native autoregressive writes for every sampled target.
    native_bank = {}
    with tempfile.TemporaryDirectory(prefix="native_reference_", dir=output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        if prompt != 0:
            raise RuntimeError("Runner smoke clip did not begin with the prompt frame")
        iterator = None
        try:
            with torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                for frame_index, object_ids, _ in iterator:
                    if object_ids != [1]:
                        raise RuntimeError(f"Unexpected native object IDs {object_ids}")
                    if frame_index == prompt:
                        continue
                    output = state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame_index]
                    selected = {
                        key: value for key, value in output.items()
                        if key in {"maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits", "maskmem_features"}
                    }
                    native_bank[frame_index] = clone_tensors(selected)
        finally:
            if iterator is not None:
                iterator.close()
            with torch.inference_mode():
                model.reset_state(state)
    if len(native_bank) != len(frames) - 1:
        raise RuntimeError("Native reference propagation missed sampled targets")

    # Re-run the exact full-run training path for token and pooled-control modes.
    token_heads = token_optimizer = token_bank = token_samples = None
    for arm, mode in (("token", "token"), ("global", "global")):
        with tempfile.TemporaryDirectory(prefix=f"{arm}_train_", dir=output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError("Runner smoke prompt index changed")
            with torch.inference_mode():
                model.propagate_in_video_preflight(state)
            bank = clone_tensors(state["output_dict_per_obj"][0])
            model._run_state = state
            heads = _make_heads(model, protocol)
            optimizer = _adamw(heads, protocol)
            chunk = _samples(model, state, frames, range(1, 5))
            loss, next_bank, writes, diagnostics = _train_chunk(
                model, heads, optimizer, chunk, bank, "seq2", mode,
                float(protocol["training"]["optimizer"]["gradient_clip_norm"]),
            )
            if len(writes) != 4 or diagnostics["gradient_l1_by_layer"] == [0.0] * 4:
                raise RuntimeError(f"No valid training update was produced in {mode} mode")
            for frame_index in range(1, 5):
                candidate = next_bank["non_cond_frame_outputs"][frame_index]
                for key in ("maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits", "maskmem_features"):
                    _assert_equal(candidate[key], native_bank[frame_index][key], f"{mode}.{frame_index}.{key}")
            results["modes"][arm] = {
                "mode": mode,
                "first_chunk_loss": loss,
                "first_chunk_zero_adapter_native_memory_write_parity": True,
                "gradient_l1_by_layer": diagnostics["gradient_l1_by_layer"],
                "four_layer_trace_count": len(diagnostics["memory_read_trace"]),
                "frame_writes": writes,
            }
            if mode == "token":
                token_heads = heads
                token_optimizer = optimizer
                token_bank = next_bank
                token_samples = _samples(model, state, frames, [5])
                with torch.inference_mode():
                    model.reset_state(state)
                model._run_state = None
                continue
            with torch.inference_mode():
                model.reset_state(state)
            model._run_state = None

    if token_heads is None:
        raise RuntimeError("Missing token-mode state for resume replay")
    # Compare an uninterrupted next update with a checkpoint-restored update.
    resume_snapshot = {
        "heads": copy.deepcopy(token_heads.state_dict()),
        "optimizer": copy.deepcopy(token_optimizer.state_dict()),
        "bank": clone_tensors(token_bank),
        "bank_sha256": _tree_digest(token_bank),
        "rng": _capture_rng(),
    }
    if _tree_digest(resume_snapshot["bank"]) != resume_snapshot["bank_sha256"]:
        raise RuntimeError("Runner smoke bank snapshot was not exact")
    uninterrupted_loss, uninterrupted_bank, _, uninterrupted_diag = _train_chunk(
        model, token_heads, token_optimizer, token_samples, token_bank,
        "seq2", "token", float(protocol["training"]["optimizer"]["gradient_clip_norm"]),
    )
    uninterrupted_heads = copy.deepcopy(token_heads.state_dict())
    uninterrupted_optimizer = copy.deepcopy(token_optimizer.state_dict())
    uninterrupted_rng = _capture_rng()
    uninterrupted_bank_hash = _tree_digest(uninterrupted_bank)

    resumed_heads = _make_heads(model, protocol)
    resumed_heads.load_state_dict(resume_snapshot["heads"], strict=True)
    resumed_optimizer = _adamw(resumed_heads, protocol)
    resumed_optimizer.load_state_dict(resume_snapshot["optimizer"])
    _restore_rng(resume_snapshot["rng"])
    resumed_loss, resumed_bank, _, resumed_diag = _train_chunk(
        model, resumed_heads, resumed_optimizer, token_samples, resume_snapshot["bank"],
        "seq2", "token", float(protocol["training"]["optimizer"]["gradient_clip_norm"]),
    )
    _assert_equal(uninterrupted_heads, resumed_heads.state_dict(), "adapter")
    _assert_equal(uninterrupted_optimizer, resumed_optimizer.state_dict(), "optimizer")
    _assert_equal(uninterrupted_rng, _capture_rng(), "rng")
    if (uninterrupted_loss != resumed_loss
            or uninterrupted_bank_hash != _tree_digest(resumed_bank)
            or uninterrupted_diag["gradient_l1_by_layer"] != resumed_diag["gradient_l1_by_layer"]):
        raise RuntimeError("Runner save/resume next-update replay differed")
    results["resume_replay"] = {
        "exact_loss": True,
        "exact_adapter_state": True,
        "exact_optimizer_state": True,
        "exact_rng_state": True,
        "exact_next_memory_bank": True,
        "next_update_loss": uninterrupted_loss,
    }
    if _model_digest(model) != original_hash:
        raise RuntimeError("SAM2 parameters/buffers changed in the runner smoke check")
    results.update({
        "passed": True,
        "scope": "Engineering only; one six-frame sampled seq2 clip, both adapter modes, no cohort training/effectiveness claim.",
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(manifest),
        "sam2_revision": revision,
        "sam2_parameter_buffer_sha256": original_hash,
        "runner_source_hashes": _source_hashes(),
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = validate_output(args.output_dir)
    try:
        results = run(output_dir, args.checkpoint)
        results_path = output_dir / "results.json"
        atomic_json(results_path, results)
        atomic_json(output_dir / "completion.json", {
            "passed": True,
            "results_sha256": digest(results_path),
            "protocol_sha256": digest(PROTOCOL),
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(ROOT / json.loads(PROTOCOL.read_text())["data"]["manifest"]),
            "scope": results["scope"],
        })
    except BaseException as error:
        atomic_json(output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
