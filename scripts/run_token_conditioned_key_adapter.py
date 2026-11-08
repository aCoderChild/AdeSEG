"""Train and evaluate the frozen token-conditioned SAM2 memory-key pilot.

The runner reuses the project SAM2 autoregressive training/evaluation helpers.
It trains only the new memory-key adapters, stores all artifacts on Drive, and
does not open the test split unless the frozen validation screen passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.complementary_memory.training import (
    capture_current_features,
    clone_tensors,
    detach_tensors,
    reader_clip_loss,
)
from modeling.token_conditioned_key_adapter import (
    NativeTokenConditionedKeyAdapter,
    TokenConditionedKeyHeads,
)
from scripts.false_alarm_statistics import propagation_summary
from scripts.mavos_value_utils import (
    activate_author_repo,
    atomic_json,
    atomic_torch_save,
    prepare,
    state_digest,
    target_for,
)
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor


PROTOCOL_REL = Path("protocols/token_conditioned_key_adapter_polypgen_v1.json")
PROTOCOL = ROOT / PROTOCOL_REL
MANIFEST = ROOT / "data/polypgen_sequence.jsonl"
CHECKPOINT_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/assets/sam2.1_hiera_tiny.pt"
)
GATE_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/token_conditioned_key_adapter_polypgen_v1/"
    "attempt_02_engineering_gate_2026-11-09"
)
OUTPUT_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
    "My Drive/AdeSEG/outputs/experiment/token_conditioned_key_adapter_polypgen_v1/full"
)

SOURCES = (
    "protocols/token_conditioned_key_adapter_polypgen_v1.json",
    "protocols/token_conditioned_key_adapter_engineering_gate_amendment_2026-11-09.json",
    "protocols/token_conditioned_key_adapter_runner_amendment_2026-11-09.json",
    "scripts/run_token_conditioned_key_adapter.py",
    "scripts/verify_token_conditioned_key_adapter.py",
    "scripts/verify_token_conditioned_key_adapter_runner.py",
    "tests/test_run_token_conditioned_key_adapter.py",
    "scripts/memory_token_io.py",
    "scripts/mavos_value_utils.py",
    "scripts/run_author_baseline.py",
    "scripts/false_alarm_statistics.py",
    "modeling/token_conditioned_key_adapter.py",
    "modeling/complementary_memory/training.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "modeling/region_tokens.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "datasets/io.py",
    "datasets/video.py",
    "data/polypgen_sequence.jsonl",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/sam2/modeling/memory_attention.py",
    "external/sam2/training/loss_fns.py",
    "external/RobustPVOS/sam2/modeling/moga.py",
    "external/RobustPVOS/LICENSE",
)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    )
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in SOURCES}


def _guard_sources(expected: dict[str, str]) -> None:
    for name, checksum in expected.items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError(f"Frozen source changed during the run: {name}")


def _status(out: Path, phase: str, **fields) -> None:
    atomic_json(out / "run_status.json", {
        "phase": phase,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    })
    print(json.dumps({"phase": phase, **fields}, default=str), flush=True)


def _tree_digest(value) -> str:
    """Hash a nested tensor bank without relying on dictionary insertion order."""
    hasher = hashlib.sha256()

    def visit(item, path: str) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(path.encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            hasher.update(f"{path}:{item!r}".encode())

    visit(value, "root")
    return hasher.hexdigest()


def _capture_rng() -> dict:
    np_state = np.random.get_state()
    return {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()),
                  int(np_state[2]), int(np_state[3]), float(np_state[4])),
    }


def _restore_rng(state: dict) -> None:
    torch.random.set_rng_state(state["torch_cpu"])
    random.setstate(tuple(state["python"]))
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32),
                         int(position), int(has_gauss), float(cached)))


def _make_heads(model: torch.nn.Module, protocol: dict) -> TokenConditionedKeyHeads:
    torch.manual_seed(int(protocol["training"]["seed"]))
    heads = TokenConditionedKeyHeads(
        layers=int(protocol["model"]["attention_layers"]),
        channels=int(protocol["model"]["memory_channels"]),
        rank=int(protocol["method"]["rank"]),
    ).to(device="cpu", dtype=torch.float32)
    if len(heads.layers) != len(model.memory_attention.layers):
        raise RuntimeError("Adapter layer count differs from native SAM2")
    return heads


def _chunk_frames(protocol: dict) -> int:
    # The committed protocol records the four-frame schedule in prose rather
    # than a numeric field. Fail closed if that frozen statement changes.
    schedule = protocol["training"]["schedule"].lower()
    if "four-frame chunking" not in schedule:
        raise RuntimeError("The frozen training schedule no longer specifies four-frame chunks")
    return 4


def _check_training_trace(calls: list[dict], n_frames: int) -> None:
    if len(calls) != 4 * n_frames:
        raise RuntimeError("Training clip did not trace all four memory-attention layers")
    expected = [layer for _ in range(n_frames) for layer in range(4)]
    if [row["layer"] for row in calls] != expected:
        raise RuntimeError("Training trace has a missing or reordered attention layer")
    for row in calls:
        if row["spatial_keys"] <= 0 or row["spatial_keys"] % 4096:
            raise RuntimeError("Training read contains incomplete 64x64 spatial frames")
        if not all(row[key] for key in (
            "position_aligned_before_adapter", "values_unchanged", "pointer_keys_unchanged"
        )):
            raise RuntimeError("Training read violated a native key/value/pointer invariant")
        if not math.isfinite(row["key_delta_l1"]):
            raise RuntimeError("Nonfinite key residual in training trace")


def _train_chunk(
    model: torch.nn.Module,
    heads: TokenConditionedKeyHeads,
    optimizer: torch.optim.Optimizer,
    samples: list[dict],
    bank: dict,
    sequence: str,
    mode: str,
    gradient_clip_norm: float,
) -> tuple[float, dict, list[dict], dict]:
    optimizer.zero_grad(set_to_none=True)
    integration = NativeTokenConditionedKeyAdapter(model, heads, mode)
    with integration:
        loss, next_bank, writes = reader_clip_loss(model, samples, bank)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Nonfinite loss in {sequence}/{samples[0]['frame']}")
    loss.backward()
    gradients = []
    for layer_index, layer in enumerate(heads.layers):
        params = list(layer.parameters())
        if any(parameter.grad is None for parameter in params):
            raise RuntimeError(f"Missing adapter gradient in {mode} layer {layer_index}")
        if any(not torch.isfinite(parameter.grad).all() for parameter in params):
            raise RuntimeError(f"Nonfinite adapter gradient in {mode} layer {layer_index}")
        grad_l1 = float(sum(parameter.grad.detach().abs().sum() for parameter in params))
        if grad_l1 <= 0:
            raise RuntimeError(f"Zero aggregate adapter gradient in {mode} layer {layer_index}")
        gradients.append(grad_l1)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A frozen SAM2 parameter received a gradient")
    _check_training_trace(integration.calls, len(samples))
    torch.nn.utils.clip_grad_norm_(heads.parameters(), gradient_clip_norm)
    optimizer.step()
    for parameter in heads.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Nonfinite trained adapter parameter")
    return float(loss.detach()), detach_tensors(next_bank), writes, {
        "gradient_l1_by_layer": gradients,
        "memory_read_trace": integration.calls,
    }


def _train_sequence(
    model: torch.nn.Module,
    heads: TokenConditionedKeyHeads,
    optimizer: torch.optim.Optimizer,
    frames: list,
    sequence: str,
    bank: dict,
    first_frame: int,
    mode: str,
    out: Path,
    source_hashes: dict[str, str],
    checkpoint_callback,
    prior_rows: list[dict],
    protocol: dict,
    gradient_clip_norm: float,
) -> tuple[dict, list[dict]]:
    chunk_size = _chunk_frames(protocol)
    index = first_frame
    rows = list(prior_rows)
    while index < len(frames):
        stop = min(index + chunk_size, len(frames))
        samples = []
        for frame_index in range(index, stop):
            visual = capture_current_features(model, model._run_state, frame_index)
            target = target_for(frames[frame_index])[0]
            samples.append({
                "frame": frame_index,
                "num_frames": len(frames),
                "vision": visual["vision"],
                "positions": visual["positions"],
                "sizes": visual["sizes"],
                "target": target,
                "mask_inputs": None,
                "point_inputs": None,
            })
        loss, bank, writes, diagnostics = _train_chunk(
            model, heads, optimizer, samples, bank, sequence, mode, gradient_clip_norm
        )
        row = {
            "sequence": sequence,
            "frame_start": index,
            "frame_stop_exclusive": stop,
            "loss": loss,
            "gradient_l1_by_layer": diagnostics["gradient_l1_by_layer"],
            "memory_read_trace": diagnostics["memory_read_trace"],
            "memory_writes": writes,
            "ground_truth_used_only_for_post_forward_loss": True,
            "mode": mode,
        }
        rows.append(row)
        index = stop
        _guard_sources(source_hashes)
        checkpoint_callback(index, bank, rows)
        atomic_json(out / "progress.json", {
            "phase": "training",
            "mode": mode,
            "sequence": sequence,
            "next_frame": index,
            "sequence_frames": len(frames),
            "updates_in_sequence": len(rows),
        })
    return bank, rows


def _sequence_source_hashes(frames: list) -> list[dict]:
    return [{
        "frame": index,
        "image_sha256": digest(frame.image_path),
        "mask_sha256": digest(frame.mask_path),
    } for index, frame in enumerate(frames)]


def _summary_from_rows(sequence: str, prompt: int, rows: list[dict], elapsed: float,
                       adapter_sha: str, model_sha: str, setup_hash: str,
                       input_hashes: list[dict]) -> dict:
    present = [row for row in rows if row["present"]]
    empty = [row for row in rows if not row["present"]]
    return {
        "sequence": sequence,
        "prompt_frame": prompt,
        "target_frames": len(rows),
        "present_frames": len(present),
        "gt_empty_frames": len(empty),
        "present_dice": float(np.mean([row["scores"]["dice"] for row in present])) if present else None,
        "present_iou": float(np.mean([row["scores"]["iou"] for row in present])) if present else None,
        "present_boundary_f": float(np.mean([row["scores"]["boundary_f"] for row in present])) if present else None,
        "gt_empty_false_alarm_count": sum(bool(row["absent_fp"]) for row in empty),
        "gt_empty_false_alarm_rate": float(np.mean([bool(row["absent_fp"]) for row in empty])) if empty else None,
        "propagation": propagation_summary(rows),
        "elapsed_seconds": float(elapsed),
        "fps": len(rows) / elapsed if elapsed else None,
        "adapter_sha256": adapter_sha,
        "sam2_sha256": model_sha,
        "setup_core_sha256": setup_hash,
        "input_frames": input_hashes,
    }


def _validate_cached_evaluation(directory: Path, frames: list, prompt: int,
                                adapter_sha: str, model_sha: str, setup_hash: str) -> dict:
    summary_path = directory / "summary.json"
    rows_path = directory / "frames.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        raise RuntimeError(f"Incomplete cached evaluation; preserve it and inspect: {directory}")
    summary = json.loads(summary_path.read_text())
    if (summary.get("adapter_sha256") != adapter_sha
            or summary.get("sam2_sha256") != model_sha
            or summary.get("setup_core_sha256") != setup_hash
            or summary.get("prompt_frame") != prompt):
        raise RuntimeError(f"Cached evaluation provenance mismatch: {directory}")
    if summary.get("input_frames") != _sequence_source_hashes(frames):
        raise RuntimeError(f"Cached evaluation source frame/mask hashes changed: {directory}")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    expected_frames = list(range(prompt + 1, len(frames)))
    if [row.get("frame") for row in rows] != expected_frames:
        raise RuntimeError(f"Cached evaluation frame coverage mismatch: {directory}")
    for row in rows:
        frame_index = row["frame"]
        source = frames[frame_index]
        if (row.get("image_sha256") != digest(source.image_path)
                or row.get("ground_truth_sha256") != digest(source.mask_path)):
            raise RuntimeError(f"Cached source input changed at {directory}/{frame_index}")
        mask_path = directory / "masks" / f"{frame_index:06d}.png"
        if not mask_path.is_file() or digest(mask_path) != row.get("mask_sha256"):
            raise RuntimeError(f"Cached prediction mask integrity failed: {mask_path}")
        with Image.open(mask_path) as im:
            prediction = np.asarray(im.convert("L")) > 0
        truth = load_label_mask(source.mask_path, [1]) == 1
        if bool(truth.any()) != bool(row.get("present")):
            raise RuntimeError(f"Cached presence label differs from source mask: {mask_path}")
        score = calculate_scores(prediction, truth)
        expected_boundary = float(db_eval_boundary(prediction, truth))
        if (not np.isclose(score["dice"], row["scores"]["dice"], atol=1e-12, rtol=0)
                or not np.isclose(score["iou"], row["scores"]["iou"], atol=1e-12, rtol=0)
                or not np.isclose(expected_boundary, row["scores"]["boundary_f"], atol=1e-12, rtol=0)):
            raise RuntimeError(f"Cached metrics do not match saved masks: {mask_path}")
        expected_absent_fp = bool(prediction.any()) if not truth.any() else None
        if row.get("absent_fp") != expected_absent_fp:
            raise RuntimeError(f"Cached false-alarm label differs from source mask: {mask_path}")
    elapsed = summary.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"Cached evaluation has an invalid elapsed time: {directory}")
    recomputed = _summary_from_rows(
        summary["sequence"], prompt, rows, float(elapsed), adapter_sha, model_sha,
        setup_hash, _sequence_source_hashes(frames),
    )
    for key, expected in recomputed.items():
        actual = summary.get(key)
        if isinstance(expected, float):
            if actual is None or not np.isclose(actual, expected, atol=1e-12, rtol=0):
                raise RuntimeError(f"Cached evaluation summary differs for {key}: {directory}")
        elif actual != expected:
            raise RuntimeError(f"Cached evaluation summary differs for {key}: {directory}")
    return summary


def _evaluate_video(model, frames: list, heads, mode: str | None, sequence: str,
                    out: Path, setup_hash: str,
                    source_hashes: dict[str, str]) -> dict:
    _guard_sources(source_hashes)
    before_hash = _model_digest(model)
    directory = out / "evaluation" / sequence
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    if directory.exists():
        summary = _validate_cached_evaluation(
            directory, frames, prompt,
            "native" if heads is None else state_digest(heads.state_dict()),
            before_hash, setup_hash,
        )
        _guard_sources(source_hashes)
        return summary
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{sequence}.", dir=directory.parent) as temp_name:
        staging = Path(temp_name)
        masks_dir = staging / "masks"
        masks_dir.mkdir()
        rows, calls = [], []
        with tempfile.TemporaryDirectory(prefix=f"stage_{sequence}_", dir=staging) as video_temp:
            state, actual_prompt = prepare(model, frames, Path(video_temp))
            if actual_prompt != prompt:
                raise RuntimeError("Prepared prompt index differs from the frozen manifest prompt")
            context = (NativeTokenConditionedKeyAdapter(
                model, heads, mode or "token", inference_state=state
            ) if heads is not None else _Null())
            iterator = None
            started = time.monotonic()
            try:
                with context as observer, torch.inference_mode():
                    iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                    for frame_index, object_ids, logits in iterator:
                        if object_ids != [1]:
                            raise RuntimeError(f"Unexpected object IDs in {sequence}: {object_ids}")
                        if frame_index == prompt:
                            continue
                        if not torch.isfinite(logits).all():
                            raise RuntimeError(f"Nonfinite logits in {sequence}/{frame_index}")
                        source = frames[frame_index]
                        prediction = logits[0, 0].detach().float().cpu().numpy() > 0
                        truth = load_label_mask(source.mask_path, [1]) == 1
                        scores = calculate_scores(prediction, truth)
                        scores["boundary_f"] = float(db_eval_boundary(prediction, truth))
                        mask_path = masks_dir / f"{frame_index:06d}.png"
                        Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
                        rows.append({
                            "frame": int(frame_index),
                            "present": bool(truth.any()),
                            "scores": scores,
                            "absent_fp": bool(prediction.any()) if not truth.any() else None,
                            "prediction_sha256": hashlib.sha256(
                                str(tuple(logits.shape)).encode()
                                + logits.detach().float().cpu().contiguous().numpy().tobytes()
                            ).hexdigest(),
                            "mask_sha256": digest(mask_path),
                            "image_sha256": digest(source.image_path),
                            "ground_truth_sha256": digest(source.mask_path),
                        })
                if heads is not None:
                    calls = list(observer.calls)
                    if len(calls) != 4 * len(rows):
                        raise RuntimeError("Inference did not trace four attention layers per target frame")
                    if [row["layer"] for row in calls] != [i for _ in rows for i in range(4)]:
                        raise RuntimeError("Inference attention layer order/coverage is invalid")
                    if any(row["frame"] is None for row in calls):
                        raise RuntimeError("Inference trace is missing the active frame index")
                    if any(row["object_id"] != 1 for row in calls):
                        raise RuntimeError("Inference trace mixed or lost persistent object identity")
                    for row in calls:
                        if row["spatial_keys"] <= 0 or row["spatial_keys"] % 4096:
                            raise RuntimeError("Inference trace contains malformed spatial frame blocks")
                        if not row["position_aligned_before_adapter"] or not row["values_unchanged"] or not row["pointer_keys_unchanged"]:
                            raise RuntimeError("Inference violated a native key/value/pointer invariant")
            finally:
                if iterator is not None:
                    iterator.close()
                with torch.inference_mode():
                    model.reset_state(state)
            if len(rows) != len(frames) - prompt - 1:
                raise RuntimeError(f"Incomplete post-prompt coverage in {sequence}")
            elapsed = time.monotonic() - started
        if _model_digest(model) != before_hash:
            raise RuntimeError("SAM2 parameters/buffers changed during evaluation")
        _guard_sources(source_hashes)
        adapter_sha = "native" if heads is None else state_digest(heads.state_dict())
        summary = _summary_from_rows(
            sequence, prompt, rows, elapsed, adapter_sha, before_hash, setup_hash,
            _sequence_source_hashes(frames),
        )
        _atomic_jsonl(staging / "frames.jsonl", rows)
        if calls:
            _atomic_jsonl(staging / "memory_trace.jsonl", calls)
        atomic_json(staging / "summary.json", summary)
        os.replace(staging, directory)
    return summary


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _summarize(groups: dict, names: list[str], out: Path, protocol: dict,
               validation: bool) -> dict:
    arms = ("native", "token_conditioned", "global_gate")
    metrics = ("present_dice", "present_iou", "present_boundary_f", "gt_empty_false_alarm_rate")
    means = {
        arm: {
            metric: (float(np.mean([groups[arm][name][metric] for name in names
                                    if groups[arm][name][metric] is not None]))
                     if any(groups[arm][name][metric] is not None for name in names) else None)
            for metric in metrics
        }
        for arm in arms
    }
    contrasts = {}
    for learned, control in (("token_conditioned", "native"), ("token_conditioned", "global_gate"),
                            ("global_gate", "native")):
        comparable = [name for name in names
                      if groups[learned][name]["present_dice"] is not None
                      and groups[control][name]["present_dice"] is not None]
        if not comparable:
            raise RuntimeError(f"No comparable present-frame Dice values for {learned} vs {control}")
        sequence_delta = {
            name: groups[learned][name]["present_dice"] - groups[control][name]["present_dice"]
            for name in comparable
        }
        contrasts[f"{learned}_minus_{control}"] = {
            "mean": float(np.mean(list(sequence_delta.values()), dtype=np.float64)),
            "sequence_delta": sequence_delta,
            "wins": sum(value > 0 for value in sequence_delta.values()),
            "ties": sum(value == 0 for value in sequence_delta.values()),
            "losses": sum(value < 0 for value in sequence_delta.values()),
            "bootstrap_95_ci": sequence_bootstrap_interval(
                sequence_delta, seed=int(protocol["training"]["seed"])
            ),
            "wilcoxon": paired_wilcoxon(
                {name: groups[learned][name]["present_dice"] for name in comparable},
                {name: groups[control][name]["present_dice"] for name in comparable},
            ),
            "leave_one_sequence_out_mean_delta": {
                omitted: (float(np.mean([sequence_delta[name] for name in comparable if name != omitted],
                                        dtype=np.float64))
                          if len(comparable) > 1 and omitted in comparable else None)
                for omitted in comparable
            },
        }
    result = {
        "conditions": groups,
        "equal_sequence_means": means,
        "contrasts": contrasts,
        "holm_adjusted_p": holm_adjust({key: value["wilcoxon"]["p"] for key, value in contrasts.items()}),
    }
    if validation:
        if names != list(protocol["data"]["validation_sequences"]):
            raise RuntimeError("Validation release screen must use the frozen validation sequence list")
        if len(names) != 3:
            raise RuntimeError("Frozen release threshold is defined for exactly three validation sequences")
        rule = protocol["evaluation"]["release_rule"]
        if "at least 0.01 equal-sequence present Dice" not in rule or "at least 2/3 sequences" not in rule or "no more than 0.02 absolute" not in rule:
            raise RuntimeError("Human-readable release rule no longer matches the frozen executable thresholds")
        token_native = contrasts["token_conditioned_minus_native"]
        token_global = contrasts["token_conditioned_minus_global_gate"]
        token_fa = means["token_conditioned"]["gt_empty_false_alarm_rate"]
        native_fa = means["native"]["gt_empty_false_alarm_rate"]
        false_alarm_increase = (
            None if token_fa is None or native_fa is None else token_fa - native_fa
        )
        passed = (
            false_alarm_increase is not None
            and token_native["mean"] >= 0.01
            and token_global["mean"] >= 0.01
            and token_native["wins"] >= 2
            and token_global["wins"] >= 2
            and false_alarm_increase <= 0.02
        )
        result["validation_release_screen"] = {
            "passed": bool(passed),
            "false_alarm_increase_vs_native": false_alarm_increase,
            "thresholds": {
                "min_mean_dice_gain_vs_native_and_global": 0.01,
                "min_sequence_wins_vs_each": 2,
                "validation_sequences": len(names),
                "max_gt_empty_false_alarm_increase": 0.02,
            },
            "interpretation": "Exploratory release screen only; not evidence of statistical significance or independent confirmation.",
        }
    atomic_json(out / ("validation_results.json" if validation else "test_results.json"), result)
    _write_report(out / ("validation_results.md" if validation else "test_results.md"),
                  result, names, validation)
    return result


def _write_report(path: Path, result: dict, names: list[str], validation: bool) -> None:
    means = result["equal_sequence_means"]
    lines = [
        "# Token-conditioned SAM2 memory-key adapter",
        "",
        "Exploratory PolypGen sequence-level evaluation; the cohort has informed prior project analyses, and sequence/patient independence is unverified.",
        "",
        "| Condition | Present Dice | Present IoU | Boundary F | GT-empty false-alarm rate |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"native": "Native SAM2", "token_conditioned": "Token-conditioned key adapter", "global_gate": "Capacity-matched global-gate adapter"}
    for arm, label in labels.items():
        values = [means[arm][key] for key in (
            "present_dice", "present_iou", "present_boundary_f", "gt_empty_false_alarm_rate"
        )]
        lines.append("| " + label + " | " + " | ".join(
            f"{value:.6f}" if value is not None else "N/A" for value in values
        ) + " |")
    lines += ["", "| Sequence | Native Dice | Token-conditioned Dice | Global-gate Dice |",
              "|---|---:|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(
            f"{result['conditions'][arm][name]['present_dice']:.6f}" for arm in labels
        ) + " |")
    for contrast, row in result["contrasts"].items():
        ci = row["bootstrap_95_ci"]
        lines.append(
            f"\n{contrast}: {row['mean']:+.6f} Dice, paired sequence-bootstrap 95% CI "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}], wins/ties/losses "
            f"{row['wins']}/{row['ties']}/{row['losses']}, Wilcoxon p="
            f"{row['wilcoxon']['p']:.6g}, Holm-adjusted p="
            f"{result['holm_adjusted_p'][contrast]:.6g}."
        )
        loso = row["leave_one_sequence_out_mean_delta"]
        if loso:
            lines.append("Leave-one-sequence-out mean Dice delta: " + ", ".join(
                f"omit {name}: {value:+.6f}" if value is not None else f"omit {name}: N/A"
                for name, value in loso.items()
            ) + ".")
    if validation:
        screen = result["validation_release_screen"]
        lines.append(f"\nFrozen validation release screen: {'PASS' if screen['passed'] else 'FAIL'}. ")
        lines.append("Test inference remains closed unless this screen passes.")
        lines.append("This small validation screen is exploratory and cannot establish independent generalization.")
    else:
        lines.append("\nTest results are exploratory because this PolypGen cohort informed prior work; they are not independent confirmation.")
    path.write_text("\n".join(lines) + "\n")


def _load_existing_setup(out: Path, core_hash: str, resume: bool) -> None:
    setup_path = out / "setup.json"
    if setup_path.exists():
        saved = json.loads(setup_path.read_text())
        if saved.get("core_sha256") != core_hash:
            raise RuntimeError("Existing output setup does not match protocol/source/data identity")
        if not resume:
            raise FileExistsError("Output already exists; use --resume to continue it")
    elif resume:
        raise FileNotFoundError("Cannot resume without a frozen setup.json")


def _train_arm(model, videos, arm: str, mode: str, protocol: dict, out: Path,
               core_hash: str, source_hashes: dict[str, str], original_model_hash: str,
               resume: bool, paired_initial_hash: str | None) -> tuple[TokenConditionedKeyHeads, str]:
    arm_dir = out / "training" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    completed_path = arm_dir / "completion.json"
    weights_path = arm_dir / "adapters.pt"
    if completed_path.exists() and weights_path.exists():
        complete = json.loads(completed_path.read_text())
        if complete.get("core_sha256") != core_hash or digest(weights_path) != complete.get("adapter_file_sha256"):
            raise RuntimeError(f"Completed arm provenance/checkpoint mismatch: {arm}")
        heads = _make_heads(model, protocol)
        heads.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        initial_hash = complete["initial_adapter_state_sha256"]
        if paired_initial_hash is not None and initial_hash != paired_initial_hash:
            raise RuntimeError("Token and global control did not share initialization")
        for sequence in protocol["data"]["train_sequences"]:
            receipt = json.loads((arm_dir / f"{sequence}.json").read_text())
            if receipt.get("source_hashes") != _sequence_source_hashes(videos[sequence]):
                raise RuntimeError(f"Completed training inputs changed for {arm}/{sequence}")
        return heads.eval(), initial_hash

    heads = _make_heads(model, protocol)
    initial_hash = state_digest(heads.state_dict())
    if paired_initial_hash is not None and initial_hash != paired_initial_hash:
        raise RuntimeError("Token and global control initializations differ")
    optimizer_cfg = protocol["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
    )
    checkpoint_path = arm_dir / "resume.pt"
    completed_sequences: list[str] = []
    next_sequence_index = 0
    saved = None
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("core_sha256") != core_hash or saved.get("arm") != arm:
            raise RuntimeError("Training resume checkpoint identity mismatch")
        if _tree_digest(saved["bank"]) != saved.get("bank_sha256"):
            raise RuntimeError("Training resume bank failed its digest")
        heads.load_state_dict(saved["heads"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        completed_sequences = list(saved["completed_sequences"])
        next_sequence_index = int(saved["next_sequence_index"])
    elif not resume and checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists; use --resume: {checkpoint_path}")

    train_sequences = protocol["data"]["train_sequences"]
    if completed_sequences != train_sequences[:next_sequence_index]:
        raise RuntimeError("Resume checkpoint completed-sequence cursor is inconsistent")
    for sequence in completed_sequences:
        receipt = json.loads((arm_dir / f"{sequence}.json").read_text())
        if receipt.get("source_hashes") != _sequence_source_hashes(videos[sequence]):
            raise RuntimeError(f"Previously trained inputs changed for {arm}/{sequence}")
    for sequence_index in range(next_sequence_index, len(train_sequences)):
        sequence = train_sequences[sequence_index]
        frames = videos[sequence]
        with tempfile.TemporaryDirectory(prefix=f"train_{sequence}_", dir=arm_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            current_input_hashes = _sequence_source_hashes(frames)
            if saved is not None and saved.get("sequence") == sequence:
                if saved.get("sequence_source_hashes") != current_input_hashes:
                    raise RuntimeError(f"Inputs changed while resuming {arm}/{sequence}")
            with torch.inference_mode():
                model.propagate_in_video_preflight(state)
            bank = clone_tensors(state["output_dict_per_obj"][0])
            model._run_state = state
            start_frame = prompt + 1
            if saved is not None and saved.get("sequence") == sequence:
                bank = saved["bank"]
                start_frame = int(saved["next_frame"])
            chunk_dir = arm_dir / "chunks" / sequence
            chunk_dir.mkdir(parents=True, exist_ok=True)
            prior_rows = []
            for receipt_path in sorted(chunk_dir.glob("frame_*.json")):
                row = json.loads(receipt_path.read_text())
                if row["frame_stop_exclusive"] <= start_frame:
                    prior_rows.append(row)
            expected_frame = prompt + 1
            for row in prior_rows:
                if row["frame_start"] != expected_frame:
                    raise RuntimeError("Saved training chunks do not form a contiguous prefix")
                expected_frame = row["frame_stop_exclusive"]
            if expected_frame != start_frame:
                raise RuntimeError("Saved training receipts do not match the resume frame cursor")
            before_hash = _model_digest(model)

            def checkpoint_callback(next_frame: int, current_bank: dict, rows: list[dict]) -> None:
                last = rows[-1]
                atomic_json(chunk_dir / f"frame_{last['frame_start']:06d}.json", last)
                atomic_torch_save(checkpoint_path, {
                    "core_sha256": core_hash,
                    "arm": arm,
                    "sequence": sequence,
                    "sequence_source_hashes": current_input_hashes,
                    "next_sequence_index": sequence_index,
                    "next_frame": next_frame,
                    "heads": heads.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "bank": current_bank,
                    "bank_sha256": _tree_digest(current_bank),
                    "rng": _capture_rng(),
                    "completed_sequences": completed_sequences,
                })
                _guard_sources(source_hashes)

            try:
                bank, rows = _train_sequence(
                    model, heads, optimizer, frames, sequence, bank, start_frame,
                    mode, out, source_hashes, checkpoint_callback, prior_rows, protocol,
                    float(optimizer_cfg["gradient_clip_norm"]),
                )
            finally:
                with torch.inference_mode():
                    model.reset_state(state)
                model._run_state = None
            if _model_digest(model) != before_hash:
                raise RuntimeError("Frozen SAM2 parameters/buffers changed during training")
            expected_targets = len(frames) - prompt - 1
            if sum(row["frame_stop_exclusive"] - row["frame_start"] for row in rows) != expected_targets:
                raise RuntimeError(f"Training target coverage mismatch in {sequence}")
            atomic_json(arm_dir / f"{sequence}.json", {
                "sequence": sequence,
                "prompt_frame": prompt,
                "post_prompt_frames": expected_targets,
                "update_count": len(rows),
                "updates": rows,
                "source_hashes": _sequence_source_hashes(frames),
            })
            completed_sequences.append(sequence)
            atomic_torch_save(checkpoint_path, {
                "core_sha256": core_hash,
                "arm": arm,
                "sequence": None,
                "sequence_source_hashes": None,
                "next_sequence_index": sequence_index + 1,
                "next_frame": None,
                "heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "bank": {},
                "bank_sha256": _tree_digest({}),
                "rng": _capture_rng(),
                "completed_sequences": completed_sequences,
            })
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            _status(out, "training", arm=arm, mode=mode,
                    completed_sequences=len(completed_sequences),
                    expected_sequences=len(train_sequences), last_sequence=sequence)

    if completed_sequences != train_sequences:
        raise RuntimeError("Training did not cover all frozen sequences")
    chunk_size = _chunk_frames(protocol)
    expected_updates = sum(
        math.ceil((len(videos[name]) - prompt_index(videos[name], {"initialization": "first_labeled_frame"}) - 1)
                  / chunk_size)
        for name in train_sequences
    )
    update_count = sum(json.loads((arm_dir / f"{name}.json").read_text())["update_count"]
                       for name in train_sequences)
    if update_count != expected_updates:
        raise RuntimeError(f"Expected {expected_updates} manifest-derived updates; got {update_count}")
    if _model_digest(model) != original_model_hash:
        raise RuntimeError("Frozen SAM2 digest changed across the full training arm")
    atomic_torch_save(weights_path, heads.state_dict())
    atomic_json(completed_path, {
        "complete": True,
        "arm": arm,
        "mode": mode,
        "sequences": completed_sequences,
        "optimizer_updates": update_count,
        "initial_adapter_state_sha256": initial_hash,
        "adapter_file_sha256": digest(weights_path),
        "adapter_state_sha256": state_digest(heads.state_dict()),
        "sam2_sha256": original_model_hash,
        "core_sha256": core_hash,
    })
    heads.eval()
    del optimizer
    gc.collect()
    return heads, initial_hash


def _release_gate(gate_dir: Path, checkpoint: Path,
                   current_sources: dict[str, str]) -> dict:
    completion_path = gate_dir / "completion.json"
    setup_path = gate_dir / "setup.json"
    if not completion_path.is_file() or not setup_path.is_file():
        raise RuntimeError("Missing completed real-frame engineering-gate receipt")
    completion = json.loads(completion_path.read_text())
    setup = json.loads(setup_path.read_text())
    if (not completion.get("passed")
            or completion.get("full_cohort_training_started")
            or completion.get("efficacy_inference_started")):
        raise RuntimeError("Engineering gate is failed or incorrectly includes efficacy work")
    if (setup.get("protocol_sha256") != digest(PROTOCOL)
            or setup.get("checkpoint_sha256") != digest(checkpoint)
            or setup.get("manifest_sha256") != digest(MANIFEST)
            or setup.get("sam2_revision") != json.loads(PROTOCOL.read_text())["model"]["sam2_revision"]):
        raise RuntimeError("Engineering-gate protocol/checkpoint/manifest differs from this run")
    gate_sources = setup.get("sources", {})
    shared_sources = set(gate_sources) & set(current_sources)
    required_sources = {
        "protocols/token_conditioned_key_adapter_polypgen_v1.json",
        "protocols/token_conditioned_key_adapter_engineering_gate_amendment_2026-11-09.json",
        "modeling/token_conditioned_key_adapter.py",
        "modeling/decoder_training_mask.py",
        "scripts/verify_token_conditioned_key_adapter.py",
        "scripts/run_author_baseline.py",
        "datasets/video.py",
    }
    if not required_sources.issubset(shared_sources):
        raise RuntimeError("Engineering gate lacks hashes for required current implementation sources")
    mismatched = [name for name in shared_sources if gate_sources[name] != current_sources[name]]
    if mismatched:
        raise RuntimeError(f"Engineering-gate source hashes differ: {sorted(mismatched)}")
    identities = [*completion.get("polypgen", []), completion.get("davis_multi_object", {})]
    for identity in identities:
        if (identity.get("checkpoint_sha256") != digest(checkpoint)
                or identity.get("sam2_revision") != setup.get("sam2_revision")):
            raise RuntimeError("Embedded engineering-gate receipt identity is inconsistent")
    if any(row.get("manifest_sha256") != digest(MANIFEST)
           for row in completion.get("polypgen", [])):
        raise RuntimeError("Embedded PolypGen gate receipt manifest identity is inconsistent")
    return {
        "completion": completion,
        "setup": setup,
        "results_sha256": digest(completion_path),
        "setup_sha256": digest(setup_path),
    }


def _runner_training_smoke(smoke_dir: Path, checkpoint: Path,
                           source_hashes: dict[str, str]) -> dict:
    completion_path = smoke_dir / "completion.json"
    results_path = smoke_dir / "results.json"
    if not completion_path.is_file() or not results_path.is_file():
        raise RuntimeError("Missing runner-specific real-frame training smoke receipt")
    completion = json.loads(completion_path.read_text())
    results = json.loads(results_path.read_text())
    if not completion.get("passed") or completion.get("results_sha256") != digest(results_path):
        raise RuntimeError("Runner-specific real-frame training smoke failed or has invalid hash")
    expected_manifest_sha = digest(MANIFEST)
    if (completion.get("protocol_sha256") != digest(PROTOCOL)
            or completion.get("checkpoint_sha256") != digest(checkpoint)
            or completion.get("manifest_sha256") != expected_manifest_sha
            or results.get("runner_source_hashes") != source_hashes):
        raise RuntimeError("Runner smoke protocol/checkpoint/data/source identity differs")
    return {"completion": completion, "results_sha256": digest(results_path)}


def _model_digest(model: torch.nn.Module) -> str:
    """Hash SAM2 parameters and buffers, matching the real-frame gate."""
    hasher = hashlib.sha256()
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in sorted(tensors):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--gate", type=Path, default=GATE_DEFAULT)
    parser.add_argument("--runner-smoke", type=Path, default=Path(
        "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/"
        "My Drive/AdeSEG/outputs/experiment/token_conditioned_key_adapter_polypgen_v1/"
        "attempt_03_runner_training_smoke_2026-11-09"
    ))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    os.environ["TMPDIR"] = str(out)
    model = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(PROTOCOL.read_text())
        if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("SAM2 checkpoint hash differs from the frozen protocol")
        source_hashes = _source_hashes()
        gate = _release_gate(args.gate, args.checkpoint, source_hashes)
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["sam2_revision"]:
            raise RuntimeError("SAM2 repository revision differs from the frozen protocol")
        if not os.popen(f"git -C {ROOT} status --porcelain -- {PROTOCOL_REL}").read().strip() == "":
            raise RuntimeError("Frozen efficacy protocol has uncommitted changes")
        protocol_commit = os.popen(
            f"git -C {ROOT} log -1 --format=%H -- {PROTOCOL_REL}"
        ).read().strip()
        if not protocol_commit:
            raise RuntimeError("Efficacy protocol must be committed before training")
        # Keep test image/mask records unopened until the frozen validation
        # release screen passes. The split names are already fixed in protocol.
        videos = load_manifest(MANIFEST, "dev")
        train = list(protocol["data"]["train_sequences"])
        validation = list(protocol["data"]["validation_sequences"])
        test = list(protocol["data"]["test_sequences"])
        if (set(train + validation) - set(videos)
                or set(train) & set(validation)
                or set(test) & set(train + validation)):
            raise RuntimeError("Frozen train/validation splits overlap or contain absent sequences")
        post_prompt_counts = {
            name: len(videos[name]) - prompt_index(
                videos[name], {"initialization": "first_labeled_frame"}
            ) - 1
            for name in train + validation
        }
        training_targets = sum(post_prompt_counts[name] for name in train)
        chunk_size = _chunk_frames(protocol)
        training_updates = sum(math.ceil(post_prompt_counts[name] / chunk_size) for name in train)
        if training_targets <= 0 or training_updates <= 0:
            raise RuntimeError("Frozen manifest has no post-prompt training targets")
        runner_smoke = _runner_training_smoke(args.runner_smoke, args.checkpoint, source_hashes)
        core = {
            "protocol_sha256": digest(PROTOCOL),
            "protocol_commit": protocol_commit,
            "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(MANIFEST),
            "sam2_revision": revision,
            "sources": source_hashes,
            "engineering_gate_results_sha256": gate["results_sha256"],
            "runner_training_smoke_results_sha256": runner_smoke["results_sha256"],
            "train_sequences": train,
            "validation_sequences": validation,
            "test_sequences": test,
            "post_prompt_frames": post_prompt_counts,
            "training_updates_per_arm": training_updates,
            "seed": int(protocol["training"]["seed"]),
            "device": protocol["model"]["device"],
        }
        core_hash = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        _load_existing_setup(out, core_hash, args.resume)
        if not (out / "setup.json").exists():
            setup = {
                "core": core,
                "core_sha256": core_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv,
                "python": sys.version,
                "torch": torch.__version__,
            }
            atomic_json(out / "setup.json", setup)
            (out / "protocol.json").write_bytes(PROTOCOL.read_bytes())
            atomic_json(out / "source_hashes.json", source_hashes)
            (out / "engineering_gate_receipt.json").write_text(
                json.dumps(gate, indent=2, sort_keys=True) + "\n"
            )
            (out / "runner_training_smoke_receipt.json").write_text(
                json.dumps(runner_smoke, indent=2, sort_keys=True) + "\n"
            )
            with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for relative in SOURCES:
                    archive.write(ROOT / relative, relative)
        model = build_author_predictor(argparse.Namespace(
            method="sam2", device="cpu", checkpoint=args.checkpoint,
            model_config=protocol["model"]["configuration"],
        ), repo)
        model.requires_grad_(False)
        model.eval()
        if (model.num_maskmem != 7
                or model.memory_temporal_stride_for_eval != 1
                or model.mem_dim != int(protocol["model"]["memory_channels"])):
            raise RuntimeError("Native SAM2 memory settings differ from the frozen protocol")
        original_model_hash = _model_digest(model)
        run_core_hash = core_hash

        groups = {arm: {} for arm in ("native", "token_conditioned", "global_gate")}
        # Native validation is freshly run once under this attempt. It is not
        # borrowed from another pilot whose per-frame input receipts differ.
        for sequence in validation:
            _status(out, "validation_native", sequence=sequence)
            groups["native"][sequence] = _evaluate_video(
                model, videos[sequence], None, None, sequence, out / "native_validation",
                run_core_hash, source_hashes,
            )

        paired_initial_hash = None
        for arm, mode in (("token_conditioned", "token"), ("global_gate", "global")):
            _status(out, "training_start", arm=arm, mode=mode)
            heads, initial_hash = _train_arm(
                model, videos, arm, mode, protocol, out, core_hash,
                source_hashes, original_model_hash, args.resume, paired_initial_hash,
            )
            if paired_initial_hash is None:
                paired_initial_hash = initial_hash
            for sequence in validation:
                _status(out, "validation_adapter", arm=arm, sequence=sequence)
                groups[arm][sequence] = _evaluate_video(
                    model, videos[sequence], heads, mode, sequence,
                    out / f"validation_{arm}", run_core_hash, source_hashes,
                )
            del heads
            gc.collect()
        if _model_digest(model) != original_model_hash:
            raise RuntimeError("SAM2 parameter/buffer digest changed during training/validation")
        validation_result = _summarize(groups, validation, out, protocol, validation=True)
        if not validation_result["validation_release_screen"]["passed"]:
            atomic_json(out / "completion.json", {
                "complete": True,
                "test_run": False,
                "reason": "predeclared validation release screen failed",
                "validation_results_sha256": digest(out / "validation_results.json"),
                "interpretation": "No test inference was performed.",
            })
            _status(out, "complete_validation_screen_failed")
            return

        # Test access is conditional on the frozen validation-only rule.
        test_videos = load_manifest(MANIFEST, "test")
        if set(test_videos) != set(test):
            raise RuntimeError("Loaded test split differs from the frozen protocol sequence list")
        videos.update(test_videos)
        test_groups = {arm: {} for arm in ("native", "token_conditioned", "global_gate")}
        for arm, mode in (("native", None), ("token_conditioned", "token"), ("global_gate", "global")):
            heads = None
            if arm != "native":
                heads = _make_heads(model, protocol)
                heads.load_state_dict(torch.load(
                    out / "training" / arm / "adapters.pt", map_location="cpu", weights_only=True
                ), strict=True)
                heads.eval()
            for sequence in test:
                _status(out, "test_evaluation", arm=arm, sequence=sequence)
                test_groups[arm][sequence] = _evaluate_video(
                    model, videos[sequence], heads, mode, sequence,
                    out / f"test_{arm}", run_core_hash, source_hashes,
                )
            del heads
            gc.collect()
        _summarize(test_groups, test, out, protocol, validation=False)
        atomic_json(out / "completion.json", {
            "complete": True,
            "test_run": True,
            "validation_results_sha256": digest(out / "validation_results.json"),
            "test_results_sha256": digest(out / "test_results.json"),
        })
        _status(out, "complete", validation_release_passed=True)
    except BaseException as error:
        atomic_json(out / "failure.json", {"error": repr(error), "time": datetime.now(timezone.utc).isoformat()})
        _status(out, "failed", error=repr(error))
        raise
    finally:
        if model is not None:
            del model
            gc.collect()


if __name__ == "__main__":
    _main()
