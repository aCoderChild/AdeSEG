"""Train and evaluate the frozen-SAM2 XMem-inspired affinity pilot.

This is an exploratory PolypGen runner. It refuses to start training unless a
protocol-complete real-frame gate receipt is supplied. A short seq2/seq3 gate
is necessary but not sufficient for efficacy training.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import assert_disjoint_videos, load_manifest
from evaluation.statistics import (
    holm_adjust,
    paired_wilcoxon,
    sequence_bootstrap_interval,
)
from modeling.cfbi_training import clone_tensors
from modeling.complementary_memory.training import (
    capture_current_features,
    detach_tensors,
)
from modeling.xmem_token_affinity import (
    freeze_sam2_train_xmem_affinity,
    install_xmem_token_affinity,
)
from modeling.xmem_token_affinity_training import (
    xmem_clip_loss,
    xmem_native_read_audit,
)
from scripts import run_complementary_memory_pilot as base
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.verify_xmem_token_affinity import (
    frozen_parameter_digest,
    seed_everything,
)

METHOD_PROTOCOL = ROOT / "protocols/xmem_token_affinity_polypgen_v1.json"
EFFICACY_AMENDMENT = ROOT / "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json"
RNG_AMENDMENT = ROOT / "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json"
AUTOGRAD_AMENDMENT = ROOT / "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json"
REDUCTION_AMENDMENT = ROOT / "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json"
PARAMETER_DIAGNOSTIC_AMENDMENT = ROOT / "protocols/xmem_token_affinity_parameter_diagnostic_amendment_2026-11-08.json"
GRADIENT_TRACE_AMENDMENT = ROOT / "protocols/xmem_token_affinity_gradient_scaling_trace_amendment_2026-11-08.json"
GRADIENT_TRACE_RESULT = ROOT / "protocols/xmem_token_affinity_gradient_scaling_trace_results_2026-11-08.json"
AFFINITY_ACCUMULATION_AMENDMENT = ROOT / "protocols/xmem_token_affinity_single_scale_amendment_2026-11-08.json"
AFFINITY_ACCUMULATION_RESULT = ROOT / "protocols/xmem_token_affinity_single_scale_results_2026-11-08.json"
CORRECTED_SOURCE_GATE_AMENDMENT = ROOT / "protocols/xmem_token_affinity_corrected_source_gate_amendment_2026-11-08.json"
CORRECTED_SOURCE_GATE_RESULT = ROOT / "protocols/xmem_token_affinity_corrected_source_gate_results_2026-11-08.json"
FULL_GRID_SIMILARITY_AMENDMENT = ROOT / "protocols/xmem_token_affinity_full_grid_similarity_amendment_2026-11-08.json"
MANIFEST_SPLIT_AMENDMENT = ROOT / "protocols/xmem_token_affinity_manifest_split_amendment_2026-11-08.json"
MANIFEST = ROOT / "data/polypgen_sequence.jsonl"
SOURCE_PATHS = (
    "protocols/xmem_token_affinity_polypgen_v1.json",
    "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_rng_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_autograd_determinism_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_reduction_determinism_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_parameter_diagnostic_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gradient_scaling_trace_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gradient_scaling_trace_results_2026-11-08.json",
    "protocols/xmem_token_affinity_single_scale_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_single_scale_results_2026-11-08.json",
    "protocols/xmem_token_affinity_corrected_source_gate_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_corrected_source_gate_results_2026-11-08.json",
    "protocols/xmem_token_affinity_full_grid_similarity_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_manifest_split_amendment_2026-11-08.json",
    "protocols/xmem_token_affinity_gate_verifier_amendment_2026-11-08.json",
    "protocols/iterative_memory_pilot_v1.json",
    "scripts/run_xmem_token_affinity.py",
    "scripts/run_complementary_memory_pilot.py",
    "scripts/run_author_baseline.py",
    "scripts/verify_xmem_token_affinity.py",
    "modeling/xmem_token_affinity.py",
    "modeling/xmem_token_affinity_training.py",
    "modeling/complementary_memory/training.py",
    "modeling/complementary_memory/integration.py",
    "modeling/native_memory_layout.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "scripts/memory_token_io.py",
    "external/sam2/sam2/modeling/memory_attention.py",
    "external/sam2/sam2/modeling/sam/transformer.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "tests/test_xmem_token_affinity.py",
    "tests/test_xmem_token_affinity_training.py",
)


def atomic_json(path: Path, value) -> None:
    base.atomic_json(path, value)


def atomic_torch_save(path: Path, value) -> None:
    base.atomic_torch_save(path, value)


def write_failure_receipt(
    out, *, phase, arm, error, core_hash, gate_receipt, last_progress,
    traceback_text, failed_at=None,
):
    """Persist an auditable stop record before propagating a run exception."""
    failed_at = time.time() if failed_at is None else failed_at
    failure = {
        "status": "stopped",
        "phase": phase,
        "arm": arm,
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": traceback_text,
        "core_sha256": core_hash,
        "gate_receipt": gate_receipt,
        "last_successful_progress": last_progress,
        "failed_at_unix": failed_at,
    }
    atomic_json(out / "failure.json", failure)
    atomic_json(out / "progress.json", {
        "phase": phase,
        "status": "stopped",
        "arm": arm,
        "last_successful_progress": last_progress,
        "failure_file": "failure.json",
        "updated_at": failed_at,
    })
    return failure


def state_digest(state) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        # Flatten first: XMem has scalar per-layer strengths, and viewing a
        # zero-dimensional float directly as bytes raises on current PyTorch.
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def gradient_tensor_hashes(named_parameters) -> dict[str, str]:
    """Hash each trainable gradient separately for actionable replay diagnostics."""
    receipts = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for trainable adapter parameter {name}")
        receipts[name] = state_digest(OrderedDict(((name, parameter.grad.detach()),)))
    return receipts


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def rng_state_digest(state) -> str:
    """Hash Python, NumPy, and CPU PyTorch RNG state using stable encodings."""
    hasher = hashlib.sha256()
    tensor = state["torch_cpu"].detach().cpu().contiguous()
    hasher.update(b"torch_cpu")
    hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    hasher.update(json.dumps(state["python"], sort_keys=True, separators=(",", ":")).encode())
    numpy_state = state["numpy"]
    hasher.update(numpy_state["algorithm"].encode())
    keys = numpy_state["keys"].detach().cpu().contiguous()
    hasher.update(keys.numpy().tobytes())
    scalar_state = {key: value for key, value in numpy_state.items()
                    if key not in ("algorithm", "keys")}
    hasher.update(json.dumps(scalar_state, sort_keys=True, separators=(",", ":")).encode())
    return hasher.hexdigest()


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "torch_cpu": torch.get_rng_state(),
        "python": {
            "version": py_state[0],
            "state": list(py_state[1]),
            "gauss": py_state[2],
        },
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    py = state["python"]
    random.setstate((py["version"], tuple(py["state"]), py["gauss"]))
    n = state["numpy"]
    random_state = (
        n["algorithm"], n["keys"].cpu().numpy().astype(np.uint32, copy=True),
        n["position"], n["has_gauss"], n["cached_gaussian"],
    )
    np.random.set_state(random_state)


def adapter_state(model) -> OrderedDict:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items()
        if ".xmem_" in name
    )


def load_adapter_state(model, saved: dict) -> None:
    expected = adapter_state(model)
    if set(expected) != set(saved):
        raise RuntimeError("Checkpoint XMem parameter names differ from the installed adapter")
    full = model.state_dict()
    with torch.no_grad():
        for name, value in saved.items():
            if full[name].shape != value.shape:
                raise RuntimeError(f"Checkpoint XMem tensor shape changed: {name}")
            full[name].copy_(value.to(device=full[name].device, dtype=full[name].dtype))


def make_model(checkpoint, repo, granularity, seed):
    seed_everything(seed)
    model = base.init_predictor(checkpoint, repo)
    seed_everything(seed)
    receipt = install_xmem_token_affinity(model, key_dim=64, granularity=granularity)
    parameters = freeze_sam2_train_xmem_affinity(model)
    if len(model.memory_attention.layers) != 4 or not model.num_maskmem:
        raise RuntimeError("The pinned SAM2 memory-attention layout changed")
    return model, parameters, receipt


def clip_plan(videos, amendment):
    data, train = amendment["data"], amendment["training"]
    rows = []
    for epoch in range(1, train["epochs"] + 1):
        epoch_rows = []
        for sequence in data["train"]:
            for draw in range(train["clips_per_train_sequence_per_epoch"]):
                sampler_seed = train["seed"] + epoch * 100000 + int(sequence[3:]) * 100 + draw
                selected = sample_clip(videos[sequence], train["clip_length"], sampler_seed)
                if not (load_label_mask(selected[0].mask_path, [1]) == 1).any():
                    raise RuntimeError(
                        f"Author sampler returned a clip without a visible prompt: {sequence}"
                    )
                positions = {frame.frame_index: i for i, frame in enumerate(videos[sequence])}
                indices = [positions[frame.frame_index] for frame in selected]
                if len(indices) != train["clip_length"] or indices != list(
                    range(indices[0], indices[0] + train["clip_length"])
                ):
                    raise RuntimeError(f"Sampler produced a noncontiguous clip: {sequence}/{indices}")
                epoch_rows.append({
                    "epoch": epoch,
                    "sequence": sequence,
                    "draw": draw,
                    "sampler_seed": sampler_seed,
                    "indices": indices,
                    "source_frames": [int(frame.frame_index) for frame in selected],
                })
        random.Random(train["seed"] + epoch).shuffle(epoch_rows)
        rows.extend(epoch_rows)
    return rows


def validate_roles(amendment, videos, videos_by_split, split_amendment):
    data = amendment["data"]
    roles = [data["train"], data["validation"], data["test"], data["excluded"]]
    if any(len(role) != len(set(role)) for role in roles):
        raise RuntimeError("A sequence is duplicated within a split role")
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            if set(left) & set(right):
                raise RuntimeError("Sequence leakage across frozen roles")
    present = set().union(*(set(role) for role in roles))
    if present != set(videos):
        raise RuntimeError(f"Split roles differ from manifest sequences: {sorted(present ^ set(videos))}")
    source_path = ROOT / data["sequence_roles_source"]
    if not source_path.is_file():
        raise RuntimeError(f"Frozen sequence-role source is missing: {source_path}")
    source = json.loads(source_path.read_text())["data"]
    for role in ("train", "validation", "test", "excluded"):
        if data[role] != source[role]:
            raise RuntimeError(f"Amendment {role} role differs from its cited frozen source")

    split_data = split_amendment["data"]
    role_to_split = split_data["role_to_manifest_split"]
    if set(role_to_split) != {"train", "validation", "test", "excluded"}:
        raise RuntimeError("Manifest split amendment must map all four frozen roles")
    expected_by_split = {}
    for role, sequence_ids in ((role, data[role]) for role in role_to_split):
        expected_by_split.setdefault(role_to_split[role], set()).update(sequence_ids)
    if set(videos_by_split) != set(expected_by_split):
        raise RuntimeError("Loaded manifest splits differ from the frozen split amendment")
    for split_name, expected_ids in expected_by_split.items():
        actual_ids = set(videos_by_split[split_name])
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"Manifest split {split_name!r} does not match frozen sequence roles: "
                f"{sorted(actual_ids ^ expected_ids)}"
            )
    frozen_inventory = split_data["expected_sequences_by_manifest_split"]
    if {name: set(ids) for name, ids in frozen_inventory.items()} != expected_by_split:
        raise RuntimeError("Manifest split inventory disagrees with the frozen role mapping")
    for role, sequence_ids in ((role, data[role]) for role in role_to_split):
        if not set(sequence_ids) <= set(videos_by_split[role_to_split[role]]):
            raise RuntimeError(f"Frozen role {role!r} was assigned to the wrong manifest split")


def load_role_videos():
    """Load every frozen role from its actual manifest split without relabeling."""
    amendment = json.loads(EFFICACY_AMENDMENT.read_text())
    split_amendment = json.loads(MANIFEST_SPLIT_AMENDMENT.read_text())
    split_data = split_amendment["data"]
    expected_hashes = {
        "manifest_sha256": (MANIFEST, split_data["manifest_sha256"]),
        "method_protocol_sha256": (METHOD_PROTOCOL, split_data["method_protocol_sha256"]),
        "efficacy_amendment_sha256": (EFFICACY_AMENDMENT, split_data["efficacy_amendment_sha256"]),
        "sequence_roles_source_sha256": (
            ROOT / split_data["sequence_roles_source"],
            split_data["sequence_roles_source_sha256"],
        ),
    }
    for label, (path, expected) in expected_hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f"Manifest split amendment {label} does not match its pinned file")
    if split_data["manifest"] != str(MANIFEST.relative_to(ROOT)):
        raise RuntimeError("Manifest split amendment points to a different manifest")
    if split_data["method_protocol"] != str(METHOD_PROTOCOL.relative_to(ROOT)):
        raise RuntimeError("Manifest split amendment points to a different method protocol")
    if split_data["efficacy_amendment"] != str(EFFICACY_AMENDMENT.relative_to(ROOT)):
        raise RuntimeError("Manifest split amendment points to a different efficacy amendment")
    if amendment["data"]["manifest_sha256"] != split_data["manifest_sha256"]:
        raise RuntimeError("Method and split amendments pin different manifests")

    splits = sorted(set(split_data["role_to_manifest_split"].values()))
    videos_by_split = {name: load_manifest(MANIFEST, name) for name in splits}
    assert_disjoint_videos(*videos_by_split.values())
    videos = {}
    for split_name in splits:
        videos.update(videos_by_split[split_name])
    validate_roles(amendment, videos, videos_by_split, split_amendment)
    return videos, videos_by_split, split_amendment


def create_setup(args, videos, videos_by_split, split_amendment):
    method = json.loads(METHOD_PROTOCOL.read_text())
    amendment = json.loads(EFFICACY_AMENDMENT.read_text())
    if digest(MANIFEST) != amendment["data"]["manifest_sha256"]:
        raise RuntimeError("PolypGen manifest hash differs from the frozen amendment")
    if amendment["data"]["manifest_sha256"] != method["data"]["manifest_sha256"]:
        raise RuntimeError("Method protocol and efficacy amendment pin different manifests")
    if digest(args.checkpoint) != method["model"]["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint hash differs from the frozen method protocol")
    if digest(EFFICACY_AMENDMENT) != args.amendment_sha256:
        raise RuntimeError("Efficacy amendment hash differs from the requested frozen receipt")
    rng_amendment = json.loads(RNG_AMENDMENT.read_text())
    if rng_amendment["parent_efficacy_amendment_sha256"] != digest(EFFICACY_AMENDMENT):
        raise RuntimeError("RNG amendment does not apply to the frozen efficacy amendment")
    autograd_amendment = json.loads(AUTOGRAD_AMENDMENT.read_text())
    if autograd_amendment["parent_hashes"] != {
        "method_protocol_sha256": digest(METHOD_PROTOCOL),
        "efficacy_amendment_sha256": digest(EFFICACY_AMENDMENT),
        "rng_amendment_sha256": digest(RNG_AMENDMENT),
    }:
        raise RuntimeError("Autograd determinism amendment does not match its parent protocols")
    reduction_amendment = json.loads(REDUCTION_AMENDMENT.read_text())
    if reduction_amendment["parent_amendment_sha256"] != digest(AUTOGRAD_AMENDMENT):
        raise RuntimeError("Reduction determinism amendment does not match its parent amendment")
    diagnostic_amendment = json.loads(PARAMETER_DIAGNOSTIC_AMENDMENT.read_text())
    if diagnostic_amendment["parent_amendment_sha256"] != digest(REDUCTION_AMENDMENT):
        raise RuntimeError("Parameter-diagnostic amendment does not match its parent amendment")
    affinity_accumulation_amendment = json.loads(AFFINITY_ACCUMULATION_AMENDMENT.read_text())
    if affinity_accumulation_amendment["parent_amendment_sha256"] != digest(GRADIENT_TRACE_AMENDMENT):
        raise RuntimeError("Affinity-accumulation amendment does not match its parent amendment")
    if affinity_accumulation_amendment["parent_result_sha256"] != digest(
        GRADIENT_TRACE_RESULT
    ):
        raise RuntimeError("Affinity-accumulation amendment does not match its parent result")
    corrected_source_gate = json.loads(CORRECTED_SOURCE_GATE_AMENDMENT.read_text())
    if corrected_source_gate["parent_amendment_sha256"] != digest(
        AFFINITY_ACCUMULATION_AMENDMENT
    ) or corrected_source_gate["parent_result_sha256"] != digest(
        AFFINITY_ACCUMULATION_RESULT
    ):
        raise RuntimeError("Corrected-source gate amendment does not match its parent evidence")
    full_grid_amendment = json.loads(FULL_GRID_SIMILARITY_AMENDMENT.read_text())
    if full_grid_amendment["parent_amendment_sha256"] != digest(
        CORRECTED_SOURCE_GATE_AMENDMENT
    ) or full_grid_amendment["parent_result_sha256"] != digest(
        CORRECTED_SOURCE_GATE_RESULT
    ):
        raise RuntimeError("Full-grid similarity amendment does not match its parent evidence")
    validate_roles(amendment, videos, videos_by_split, split_amendment)
    repo, revision = source_identity("sam2")
    if revision != method["model"]["sam2_revision"]:
        raise RuntimeError("SAM2 revision differs from the frozen protocol")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Pinned SAM2 repository has local changes")
    missing = [rel for rel in SOURCE_PATHS if not (ROOT / rel).is_file()]
    if missing:
        raise RuntimeError(f"Experiment source inventory is incomplete: {missing}")
    plan = clip_plan(videos, amendment)
    expected_clips = (
        amendment["training"]["epochs"]
        * len(amendment["data"]["train"])
        * amendment["training"]["clips_per_train_sequence_per_epoch"]
    )
    if len(plan) != expected_clips:
        raise RuntimeError(f"Frozen training plan has {len(plan)} clips, expected {expected_clips}")
    clips_per_epoch_expected = expected_clips // amendment["training"]["epochs"]
    if any(
        sum(row["epoch"] == epoch for row in plan) != clips_per_epoch_expected
        for epoch in range(1, amendment["training"]["epochs"] + 1)
    ):
        raise RuntimeError("Frozen training plan has an incomplete epoch")
    per_sequence_per_epoch = amendment["training"]["clips_per_train_sequence_per_epoch"]
    if any(
        sum(row["epoch"] == epoch and row["sequence"] == sequence for row in plan)
        != per_sequence_per_epoch
        for epoch in range(1, amendment["training"]["epochs"] + 1)
        for sequence in amendment["data"]["train"]
    ):
        raise RuntimeError("Frozen plan does not match the per-sequence clip budget")
    clip_updates = (
        amendment["training"]["clip_length"] - 1
        + amendment["training"]["truncated_bptt_frames"] - 1
    ) // amendment["training"]["truncated_bptt_frames"]
    expected_updates = expected_clips * clip_updates
    if expected_updates != amendment["training"]["updates_per_arm"]:
        raise RuntimeError(
            f"Frozen update total is inconsistent: planned {expected_updates}, "
            f"protocol says {amendment['training']['updates_per_arm']}"
        )
    inputs = base.input_hashes(
        videos, amendment["data"]["train"] + amendment["data"]["validation"] + amendment["data"]["test"]
    )
    source_hashes = {rel: digest(ROOT / rel) for rel in SOURCE_PATHS}
    seed = amendment["training"]["seed"]
    initial_rng = {}
    token_model, token_parameters, _ = make_model(args.checkpoint, repo, "token", seed)
    token_init = state_digest(adapter_state(token_model))
    token_optimizer_state = optimizer_for(token_parameters, amendment["training"]).state_dict()
    initial_rng["token"] = rng_state_digest(capture_rng_state())
    del token_model
    frame_model, frame_parameters, _ = make_model(args.checkpoint, repo, "frame", seed)
    frame_init = state_digest(adapter_state(frame_model))
    frame_optimizer_state = optimizer_for(frame_parameters, amendment["training"]).state_dict()
    initial_rng["frame"] = rng_state_digest(capture_rng_state())
    del frame_model
    if token_init != frame_init:
        raise RuntimeError("Token and frame-control adapter initialization is not matched")
    if initial_rng["token"] != initial_rng["frame"]:
        raise RuntimeError("Token and frame-control initial RNG states are not matched")
    if not _tree_equal(token_optimizer_state, frame_optimizer_state):
        raise RuntimeError("Token and frame-control initial optimizer states are not matched")
    seed_everything(seed)
    core = {
        "method_protocol_sha256": digest(METHOD_PROTOCOL),
        "efficacy_amendment_sha256": digest(EFFICACY_AMENDMENT),
        "rng_amendment_sha256": digest(RNG_AMENDMENT),
        "autograd_determinism_amendment_sha256": digest(AUTOGRAD_AMENDMENT),
        "reduction_determinism_amendment_sha256": digest(REDUCTION_AMENDMENT),
        "manifest_split_amendment_sha256": digest(MANIFEST_SPLIT_AMENDMENT),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": revision,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "source_hashes": source_hashes,
        "input_receipts": inputs,
        "training_plan": plan,
        "training_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "matched_initial_adapter_sha256": token_init,
        "matched_initial_optimizer_sha256": _json_sha256(token_optimizer_state),
        "matched_initial_rng_sha256": initial_rng["token"],
        "split_roles": {key: amendment["data"][key] for key in
                        ("train", "validation", "test", "excluded")},
        "manifest_splits": {
            split_name: sorted(split_videos)
            for split_name, split_videos in videos_by_split.items()
        },
        "role_to_manifest_split": split_amendment["data"]["role_to_manifest_split"],
    }
    core_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return method, amendment, core, core_hash, repo


def save_source_snapshot(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in SOURCE_PATHS:
            archive.write(ROOT / relative, relative)
    return digest(path)


def optimizer_for(parameters, config):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    return optimizer


def restore_training_checkpoint(
    checkpoint, receipt_path, model, optimizer, output, *, core_hash, arm, plan,
    amendment_sha256, rng_amendment_sha256, autograd_determinism_amendment_sha256,
    reduction_determinism_amendment_sha256,
):
    """Load one clip-boundary checkpoint and validate its complete cursor."""
    receipt = json.loads(Path(receipt_path).read_text())
    if digest(checkpoint) != receipt.get("sha256"):
        raise RuntimeError("XMem resume checkpoint hash differs from its receipt")
    if receipt.get("core_sha256") != core_hash or receipt.get("arm") != arm:
        raise RuntimeError("XMem resume receipt provenance mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("core_sha256") != core_hash or saved.get("arm") != arm
            or saved.get("protocol_amendment_sha256") != amendment_sha256
            or saved.get("rng_amendment_sha256") != rng_amendment_sha256
            or receipt.get("rng_amendment_sha256") != rng_amendment_sha256
            or saved.get("autograd_determinism_amendment_sha256")
            != autograd_determinism_amendment_sha256
            or receipt.get("autograd_determinism_amendment_sha256")
            != autograd_determinism_amendment_sha256
            or saved.get("reduction_determinism_amendment_sha256")
            != reduction_determinism_amendment_sha256
            or receipt.get("reduction_determinism_amendment_sha256")
            != reduction_determinism_amendment_sha256):
        raise RuntimeError("XMem resume checkpoint provenance mismatch")
    completed_rows = saved["completed_clip_rows"]
    start_index = saved["next_clip_index"]
    if (start_index != len(completed_rows)
            or receipt.get("next_clip_index") != start_index
            or not 0 <= start_index <= len(plan)):
        raise RuntimeError("Resume cursor differs from completed clip records")
    for index, row in enumerate(completed_rows):
        if row.get("plan") != plan[index]:
            raise RuntimeError("Resume completed clip order differs from frozen plan")
    load_adapter_state(model, saved["adapter"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng_state(saved["rng_state"])
    best_epoch = saved["best_epoch"]
    best_hash = saved["best_adapter_sha256"]
    best_path = Path(output) / "best_adapter.pt"
    if best_epoch is None:
        if best_hash is not None or best_path.exists():
            raise RuntimeError("Unvalidated best adapter exists in the resume state")
    elif not best_path.is_file() or digest(best_path) != best_hash:
        raise RuntimeError("Validation-selected adapter hash differs from resume state")
    return saved


def make_samples(model, state, frames, indices):
    samples = []
    for index in indices:
        visual = capture_current_features(model, state, index)
        target, _ = base.target_for(frames[index])
        samples.append({
            "frame": index,
            "num_frames": len(frames),
            "vision": visual["vision"],
            "positions": visual["positions"],
            "sizes": visual["sizes"],
            "target": target,
            "mask_inputs": None,
            "point_inputs": None,
        })
    return samples


def train_clip(
    model, optimizer, frames, sequence, output, config, mode, *,
    include_gradient_hashes=False, write_observer=None
):
    """Train one sampled clip using autoregressive predicted memory writes."""
    with tempfile.TemporaryDirectory(prefix=f"xmem_train_{sequence}_", dir=output) as temp:
        state, prompt = base.prepare(model, frames, Path(temp))
        if prompt != 0:
            raise RuntimeError(
                f"Frozen clip plan requires a visible first-frame prompt: {sequence}/{prompt}"
            )
        with torch.inference_mode():
            model.propagate_in_video_preflight(state)
        with torch.inference_mode(False):
            bank = clone_tensors(state["output_dict_per_obj"][0])
        end = len(frames)
        chunks = []
        start = prompt + 1
        while start < end:
            stop = min(start + config["truncated_bptt_frames"], end)
            samples = make_samples(model, state, frames, list(range(start, stop)))
            optimizer.zero_grad(set_to_none=True)
            with xmem_native_read_audit(model, mode) as audit:
                loss, bank, writes = xmem_clip_loss(
                    model, samples, bank, step_observer=write_observer
                )
            expected_layers = [layer for _ in samples for layer in range(4)]
            actual_layers = [layer for row in audit.calls for layer in row["layers"]]
            if actual_layers != expected_layers:
                raise RuntimeError(f"Not all four memory layers were read in {sequence}")
            if len(audit.calls) != len(samples) or any(
                not row["stored_mask_alignment"] or not row["native_positions_verified"]
                for row in audit.calls
            ):
                raise RuntimeError("Autoregressive read failed bank or position alignment")
            if any(row["target_used_for_memory"] for row in writes):
                raise RuntimeError("Ground truth entered an autoregressive memory write")
            # Avoid CPU autograd and intra-op reduction scheduling changing
            # gradients by a few ULPs across clean process starts. Forward
            # numerics and optimizer semantics remain unchanged.
            previous_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with torch.autograd.set_multithreading_enabled(False):
                    loss.backward()
                trainable = [
                    (name, parameter) for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                gradients = [parameter.grad for _, parameter in trainable]
                if not gradients or any(
                    gradient is None or not torch.isfinite(gradient).all()
                    for gradient in gradients
                ):
                    raise RuntimeError("Adapter gradients are missing or nonfinite")
                if not any(float(gradient.abs().sum()) > 0 for gradient in gradients):
                    raise RuntimeError("All adapter gradients are zero")
                if any(parameter.grad is not None for name, parameter in model.named_parameters()
                       if ".xmem_" not in name):
                    raise RuntimeError("A pretrained SAM2 parameter acquired a gradient")
                gradient_l1_before_clip = float(
                    sum(gradient.detach().abs().sum() for gradient in gradients)
                )
                scalar_gradients_before_clip = {
                    name: float(parameter.grad.detach())
                    for name, parameter in trainable
                    if parameter.ndim == 0 and name.endswith("xmem_strength")
                }
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config["gradient_clip_norm"], error_if_nonfinite=True,
                )
                scalar_gradients_after_clip = {
                    name: float(parameter.grad.detach())
                    for name, parameter in trainable
                    if parameter.ndim == 0 and name.endswith("xmem_strength")
                }
            finally:
                torch.set_num_threads(previous_num_threads)
            optimizer.step()
            chunk = {
                "frames": [sample["frame"] for sample in samples],
                "gradient_tensors_sha256": gradient_tensor_hashes(trainable)
                if include_gradient_hashes else None,
                "loss": float(loss.detach()),
                "gradient_l1_before_clip": gradient_l1_before_clip,
                "gradient_norm_before_clip": float(grad_norm),
                "reads": audit.calls,
                "writes": writes,
            }
            if not include_gradient_hashes:
                chunk.pop("gradient_tensors_sha256")
            else:
                chunk["scalar_gradients_before_clip"] = scalar_gradients_before_clip
                chunk["scalar_gradients_after_clip"] = scalar_gradients_after_clip
            chunks.append(chunk)
            bank = detach_tensors(bank)
            start = stop
        with torch.inference_mode():
            model.reset_state(state)
    target_count = len(frames) - prompt - 1
    if sum(len(chunk["frames"]) for chunk in chunks) != target_count:
        raise RuntimeError(f"Incomplete post-prompt clip training: {sequence}")
    return {
        "sequence": sequence,
        "prompt_index": prompt,
        "post_prompt_targets": target_count,
        "optimizer_updates": len(chunks),
        "chunks": chunks,
        "predicted_memory_writes": sum(len(chunk["writes"]) for chunk in chunks),
        "ground_truth_used_only_for_loss": True,
    }


def _resume_probe_adapter(model):
    # The actual runner checkpoints at complete clips; clip sampling is frozen
    # in the plan, so no runtime sampler state can drift across a resume.
    return adapter_state(model)


def train_arm(args, arm, videos, out, core, core_hash, amendment, repo, resume):
    config = amendment["training"]
    output = out / "training" / arm
    output.mkdir(parents=True, exist_ok=True)
    model, parameters, install_receipt = make_model(
        args.checkpoint, repo, "frame" if arm == "frame_broadcast" else "token", config["seed"]
    )
    pretrained_hash = frozen_parameter_digest(model)
    optimizer = optimizer_for(parameters, config)
    if rng_state_digest(capture_rng_state()) != core["matched_initial_rng_sha256"]:
        raise RuntimeError("Fresh arm RNG state differs from the frozen matched setup")
    if _json_sha256(optimizer.state_dict()) != core["matched_initial_optimizer_sha256"]:
        raise RuntimeError("Fresh arm optimizer state differs from the frozen matched setup")
    checkpoint = output / "resume.pt"
    completed_rows, validation = [], {}
    best_score, best_epoch = -float("inf"), None
    start_index = 0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Use --resume to continue {checkpoint}")
        saved = restore_training_checkpoint(
            checkpoint, output / "resume_receipt.json", model, optimizer, output,
            core_hash=core_hash, arm=arm, plan=core["training_plan"],
            amendment_sha256=digest(EFFICACY_AMENDMENT),
            rng_amendment_sha256=digest(RNG_AMENDMENT),
            autograd_determinism_amendment_sha256=digest(AUTOGRAD_AMENDMENT),
            reduction_determinism_amendment_sha256=digest(REDUCTION_AMENDMENT),
        )
        completed_rows = saved["completed_clip_rows"]
        validation = saved["validation"]
        best_score, best_epoch = saved["best_score"], saved["best_epoch"]
        start_index = saved["next_clip_index"]
    elif any(output.iterdir()):
        raise FileExistsError(f"Nonempty training directory without a valid resume checkpoint: {output}")

    initial_plan = core["training_plan"]
    clips_per_epoch = (
        len(amendment["data"]["train"])
        * config["clips_per_train_sequence_per_epoch"]
    )
    for index in range(start_index, len(initial_plan)):
        row = initial_plan[index]
        sequence = row["sequence"]
        frames = [videos[sequence][position] for position in row["indices"]]
        clip_result = train_clip(model, optimizer, frames, sequence, out, config,
                                 "generic" if arm == "frame_broadcast" else "regional")
        expected_updates = (
            config["clip_length"] - 1 + config["truncated_bptt_frames"] - 1
        ) // config["truncated_bptt_frames"]
        if clip_result["optimizer_updates"] != expected_updates:
            raise RuntimeError(
                f"Clip update count differs from frozen schedule: "
                f"{sequence}/{clip_result['optimizer_updates']}"
            )
        completed_rows.append({"plan": row, "result": clip_result})
        epoch_complete = (index + 1) % clips_per_epoch == 0
        if epoch_complete and row["epoch"] in amendment["validation_and_selection"]["epochs"]:
            if frozen_parameter_digest(model) != pretrained_hash:
                raise RuntimeError("Training modified pretrained SAM2 parameters")
            selected = evaluate_adapter(
                args, arm, videos, amendment["data"]["validation"], out,
                core_hash, repo, adapter_state(model), f"validation/epoch_{row['epoch']:02d}",
            )
            score = float(np.mean([item["present_dice"] for item in selected.values()]))
            validation[str(row["epoch"])] = {
                "equal_sequence_present_dice": score,
                "per_sequence": selected,
                "adapter_sha256": state_digest(adapter_state(model)),
            }
            if score > best_score:
                best_score, best_epoch = score, row["epoch"]
                atomic_torch_save(output / "best_adapter.pt", adapter_state(model))
        payload = {
            "core_sha256": core_hash,
            "arm": arm,
            "protocol_amendment_sha256": digest(EFFICACY_AMENDMENT),
            "rng_amendment_sha256": digest(RNG_AMENDMENT),
            "autograd_determinism_amendment_sha256": digest(AUTOGRAD_AMENDMENT),
            "reduction_determinism_amendment_sha256": digest(REDUCTION_AMENDMENT),
            "adapter": _resume_probe_adapter(model),
            "optimizer": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "next_clip_index": index + 1,
            "completed_clip_rows": completed_rows,
            "validation": validation,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_adapter_sha256": digest(output / "best_adapter.pt")
            if best_epoch is not None else None,
            "pretrained_parameter_sha256": pretrained_hash,
        }
        atomic_torch_save(checkpoint, payload)
        atomic_json(output / "resume_receipt.json", {
            "sha256": digest(checkpoint), "next_clip_index": index + 1,
            "core_sha256": core_hash, "arm": arm,
            "rng_amendment_sha256": digest(RNG_AMENDMENT),
            "autograd_determinism_amendment_sha256": digest(AUTOGRAD_AMENDMENT),
            "reduction_determinism_amendment_sha256": digest(REDUCTION_AMENDMENT),
        })
        atomic_json(out / "progress.json", {
            "phase": "training", "arm": arm, "epoch": row["epoch"],
            "clip": index + 1, "expected_clips": len(initial_plan),
            "sequence": sequence, "updated_at": time.time(),
        })

    if best_epoch is None or not (output / "best_adapter.pt").is_file():
        raise RuntimeError("No validation-selected XMem adapter checkpoint exists")
    if frozen_parameter_digest(model) != pretrained_hash:
        raise RuntimeError("Training changed pretrained SAM2 parameters")
    rows_path = output / "training_records.json"
    atomic_json(rows_path, completed_rows)
    receipt = {
        "complete": True,
        "arm": arm,
        "core_sha256": core_hash,
        "protocol_amendment_sha256": digest(EFFICACY_AMENDMENT),
        "install_receipt": install_receipt,
        "clips": len(completed_rows),
        "updates": sum(row["result"]["optimizer_updates"] for row in completed_rows),
        "predicted_memory_writes": sum(row["result"]["predicted_memory_writes"] for row in completed_rows),
        "validation": validation,
        "best_epoch": best_epoch,
        "best_validation_dice": best_score,
        "best_adapter_sha256": digest(output / "best_adapter.pt"),
        "pretrained_parameter_sha256": pretrained_hash,
        "training_records_sha256": digest(rows_path),
    }
    atomic_json(output / "completion.json", receipt)
    del model, optimizer
    gc.collect()
    return torch.load(output / "best_adapter.pt", map_location="cpu", weights_only=True), receipt


class _AuditOnlyReader(torch.nn.Module):
    """No-op post-attention reader used to reuse the project's eval harness."""

    summaries = 0

    def __init__(self, state):
        super().__init__()
        self._keys = []
        for index, (name, value) in enumerate(state.items()):
            buffer_name = f"adapter_{index:03d}"
            self.register_buffer(buffer_name, value.detach().cpu().clone())
            self._keys.append((name, buffer_name))

    def forward(self, native, query_positions, memory, positions, probabilities, mode):
        return native


def evaluate_adapter(args, arm, videos, names, out, core_hash, repo, state, phase):
    results = {}
    granularity = "frame" if arm == "frame_broadcast" else "token"
    for name in names:
        folder = out / "evaluation" / arm / phase
        folder.mkdir(parents=True, exist_ok=True)
        model = base.init_predictor(args.checkpoint, repo)
        install_xmem_token_affinity(model, key_dim=64, granularity=granularity)
        load_adapter_state(model, state)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        reader = _AuditOnlyReader(state).eval()
        mode = "generic" if arm == "frame_broadcast" else "regional"
        summary = base.evaluate_video(
            model, videos[name], reader, mode, folder, name,
            int(json.loads(EFFICACY_AMENDMENT.read_text())["training"]["seed"]),
            hashlib.sha256(f"{core_hash}:{arm}:{state_digest(state)}".encode()).hexdigest(),
        )
        summary["condition"] = arm
        summary["adapter_state_sha256"] = state_digest(state)
        summary["model_parameter_sha256_including_adapter"] = summary.pop(
            "original_sam2_parameter_sha256"
        )
        summary["frozen_sam2_parameter_sha256"] = frozen_parameter_digest(model)
        # The reused evaluator verifies read geometry via a no-op post-read hook;
        # this label identifies the per-layer XMem condition being evaluated.
        atomic_json(folder / name / "summary.json", summary)
        results[name] = summary
        del model
        gc.collect()
    return results


def evaluate_native(args, videos, names, out, core_hash, repo):
    results = {}
    for name in names:
        folder = out / "evaluation" / "native" / "test"
        folder.mkdir(parents=True, exist_ok=True)
        model = base.init_predictor(args.checkpoint, repo)
        summary = base.evaluate_video(
            model, videos[name], reader=None, mode="generic", directory=folder,
            name=name,
            seed=int(json.loads(EFFICACY_AMENDMENT.read_text())["training"]["seed"]),
            setup_hash=hashlib.sha256(f"{core_hash}:native".encode()).hexdigest(),
        )
        results[name] = summary
        del model
        gc.collect()
    return results


def summarize_results(per_arm, amendment):
    names = amendment["data"]["test"]
    means = {}
    for arm, rows in per_arm.items():
        if set(rows) != set(names):
            raise RuntimeError(f"Test coverage incomplete for {arm}")
        if any(rows[name]["present_dice"] is None for name in names):
            raise RuntimeError(
                f"Primary equal-sequence present Dice is undefined for at least one {arm} sequence"
            )
        means[arm] = {
            key: float(np.mean([rows[name][key] for name in names]))
            for key in ("present_dice", "present_iou", "present_boundary_f", "elapsed_seconds")
        }
        means[arm]["false_alarms"] = sum(rows[name]["gt_empty_false_alarm_count"] for name in names)
        means[arm]["empty_frames"] = sum(rows[name]["gt_empty_frames"] for name in names)
        means[arm]["undetected_returns"] = sum(
            rows[name]["propagation"]["undetected_returns"] for name in names
        )
        means[arm]["return_events"] = sum(
            rows[name]["propagation"]["return_count"] for name in names
        )
        means[arm]["fps"] = float(np.mean([rows[name]["fps"] for name in names if rows[name]["fps"]]))
    contrasts, raw_p = {}, {}
    for control in ("native", "frame_broadcast"):
        delta = {
            name: per_arm["token_affinity"][name]["present_dice"]
            - per_arm[control][name]["present_dice"]
            for name in names
        }
        contrasts[control] = {
            "equal_sequence_mean_dice_gain": float(np.mean(list(delta.values()))),
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(
                delta, seed=int(amendment["training"]["seed"])
            ),
            "wins": sum(value > 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "losses": sum(value < 0 for value in delta.values()),
            "per_sequence": delta,
        }
        raw_p[control] = paired_wilcoxon(
            {name: per_arm["token_affinity"][name]["present_dice"] for name in names},
            {name: per_arm[control][name]["present_dice"] for name in names},
        )["p"]
    false_alarm_rate = {}
    for arm in per_arm:
        total_empty = means[arm]["empty_frames"]
        if total_empty == 0:
            raise RuntimeError(f"False-alarm rate has no GT-empty denominator for {arm}")
        false_alarm_rate[arm] = means[arm]["false_alarms"] / total_empty
    pass_rule = amendment["stopping_rule"]["continue_only_if"]
    rule_pass = all(
        contrasts[control]["equal_sequence_mean_dice_gain"] >= 0.01
        and contrasts[control]["sequence_bootstrap_95_ci"][0] > 0
        for control in ("native", "frame_broadcast")
    )
    max_fa_increase = float(
        amendment["stopping_rule"]["max_absolute_false_alarm_rate_increase"]
    )
    for control in ("native", "frame_broadcast"):
        if false_alarm_rate["token_affinity"] > false_alarm_rate[control] + max_fa_increase:
            rule_pass = False
        if (means["token_affinity"]["undetected_returns"]
                > means[control]["undetected_returns"]):
            rule_pass = False
    return {
        "scope": "Exploratory PolypGen test on previously examined sequences; one frozen seed.",
        "per_sequence": per_arm,
        "means": means,
        "paired_contrasts": contrasts,
        "holm_adjusted_p": holm_adjust(raw_p),
        "false_alarm_rates": false_alarm_rate,
        "frozen_continuation_rule_text": pass_rule,
        "frozen_continuation_rule_passed": bool(rule_pass),
    }


def verify_gate_receipt(path, expected_core):
    path = Path(path)
    completion_path = path / "completion.json"
    setup_path = path / "setup.json"
    if not completion_path.is_file() or not setup_path.is_file():
        raise RuntimeError("Engineering gate must contain setup.json and completion.json")
    completion = json.loads(completion_path.read_text())
    setup = json.loads(setup_path.read_text())
    if completion.get("status") != "complete" or completion.get("protocol_complete") is not True:
        raise RuntimeError("XMem engineering gate is partial; efficacy training is blocked")
    if completion.get("setup_sha256") != digest(setup_path):
        raise RuntimeError("Engineering gate setup hash differs from completion")
    for key in (
        "protocol_sha256",
        "efficacy_amendment_sha256",
        "rng_amendment_sha256",
        "autograd_determinism_amendment_sha256",
        "reduction_determinism_amendment_sha256",
        "manifest_split_amendment_sha256",
        "manifest_sha256",
        "sam2_revision",
    ):
        if completion.get(key) != setup.get(key):
            raise RuntimeError(f"Engineering gate completion {key} differs from setup")
    expected_fields = {
        "checkpoint_sha256": "checkpoint_sha256",
        "protocol_sha256": "method_protocol_sha256",
        "efficacy_amendment_sha256": "efficacy_amendment_sha256",
        "rng_amendment_sha256": "rng_amendment_sha256",
        "autograd_determinism_amendment_sha256": "autograd_determinism_amendment_sha256",
        "reduction_determinism_amendment_sha256": "reduction_determinism_amendment_sha256",
        "manifest_split_amendment_sha256": "manifest_split_amendment_sha256",
        "manifest_sha256": "manifest_sha256",
        "sam2_revision": "sam2_revision",
    }
    for gate_key, core_key in expected_fields.items():
        if setup.get(gate_key) != expected_core.get(core_key):
            raise RuntimeError(f"Engineering gate {gate_key} differs from the planned run")
    environment = setup.get("environment", {})
    for gate_key, core_key in (("python", "python"), ("torch", "torch")):
        if environment.get(gate_key) != expected_core.get(core_key):
            raise RuntimeError(f"Engineering gate {gate_key} environment differs")
    current_sources = expected_core.get("source_hashes", {})
    gate_sources = setup.get("sources", {})
    if not isinstance(current_sources, dict) or not current_sources:
        raise RuntimeError("Planned run lacks a current source inventory")
    mismatched_sources = [
        name for name, current_hash in current_sources.items()
        if gate_sources.get(name) != current_hash
    ]
    if mismatched_sources:
        raise RuntimeError(
            "Engineering gate source inventory differs from the planned run: "
            f"{mismatched_sources}"
        )
    schema = json.loads(EFFICACY_AMENDMENT.read_text())["gate_completion_schema"]
    checks = completion.get("checks", {})
    rng_schema = json.loads(RNG_AMENDMENT.read_text())
    required_checks = (*schema["required_boolean_checks"],
                       *rng_schema["required_engineering_gate_checks"])
    missing = [name for name in required_checks if checks.get(name) is not True]
    if missing:
        raise RuntimeError(f"Engineering gate omits or fails required checks: {missing}")
    snapshot_path = path / "source_snapshot.zip"
    if (not snapshot_path.is_file()
            or setup.get("source_snapshot_sha256") != digest(snapshot_path)
            or completion.get("source_snapshot_sha256") != digest(snapshot_path)):
        raise RuntimeError("Engineering gate source snapshot is missing or has a mismatched hash")
    sources = setup.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Engineering gate lacks a source hash inventory")
    with zipfile.ZipFile(snapshot_path) as archive:
        archived = set(archive.namelist())
        if archived != set(sources):
            raise RuntimeError("Engineering gate snapshot entries differ from the source inventory")
        for name, expected_hash in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise RuntimeError(f"Engineering gate snapshot source hash mismatch: {name}")
    results_path = path / "results.json"
    if (not results_path.is_file()
            or completion.get("results_sha256") != digest(results_path)):
        raise RuntimeError("Engineering gate result record is missing or hash-mismatched")
    return {
        "completion_sha256": digest(completion_path),
        "setup_sha256": digest(setup_path),
        "source_snapshot_sha256": digest(snapshot_path),
        "results_sha256": digest(results_path),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "train"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    out = validate_output(args.output_dir)
    if out.parent.name != "xmem_token_affinity_polypgen_v1":
        raise RuntimeError("Output must be inside Drive outputs/experiment/xmem_token_affinity_polypgen_v1")
    if args.stage == "train" and args.gate_dir is None:
        raise RuntimeError("Training requires a protocol-complete --gate-dir")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    videos, videos_by_split, split_amendment = load_role_videos()
    method, amendment, core, core_hash, repo = create_setup(
        args, videos, videos_by_split, split_amendment
    )
    gate_receipt = None
    if args.stage == "train":
        gate_receipt = verify_gate_receipt(
            args.gate_dir, core
        )
    if args.stage == "plan":
        print(json.dumps({
            "status": "plan_verified_no_training",
            "train_sequences": amendment["data"]["train"],
            "validation_sequences": amendment["data"]["validation"],
            "test_sequences": amendment["data"]["test"],
            "clips": len(core["training_plan"]),
            "plan_sha256": core["training_plan_sha256"],
            "matched_adapter_initialization_sha256": core["matched_initial_adapter_sha256"],
            "amendment_sha256": digest(EFFICACY_AMENDMENT),
            "note": "Split roles are sequence-disjoint but exploratory; no efficacy run was started.",
        }, indent=2))
        return

    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty; use --resume: {out}")
    out.mkdir(parents=True, exist_ok=args.resume)
    snapshot = out / "source_snapshot.zip"
    if args.resume:
        setup_saved = json.loads((out / "setup.json").read_text())
        if setup_saved["core_sha256"] != core_hash or digest(snapshot) != setup_saved["source_snapshot_sha256"]:
            raise RuntimeError("Resume setup or source snapshot differs")
    else:
        snapshot_hash = save_source_snapshot(snapshot)
        atomic_json(out / "setup.json", {
            "core": core,
            "core_sha256": core_hash,
            "source_snapshot_sha256": snapshot_hash,
            "protocol_complete_gate": gate_receipt,
            "command": sys.argv,
            "scope": "Exploratory PolypGen training; no claim of independent confirmation.",
        })

    started = time.monotonic()
    trained, receipts = {}, {}
    phase, active_arm = "training", None
    try:
        for arm in ("token_affinity", "frame_broadcast"):
            active_arm = arm
            state, receipts[arm] = train_arm(
                args, arm, videos, out, core, core_hash, amendment, repo, args.resume
            )
            trained[arm] = state
        # Test data are evaluated only after both training and all validation-based
        # checkpoint selection has finished.
        phase, active_arm = "evaluation", None
        test_sequences = amendment["data"]["test"]
        per_arm = {}
        for arm in ("token_affinity", "frame_broadcast"):
            active_arm = arm
            per_arm[arm] = evaluate_adapter(
                args, arm, videos, test_sequences, out, core_hash, repo,
                trained[arm], "test",
            )
        active_arm = "native"
        per_arm["native"] = evaluate_native(
            args, videos, test_sequences, out, core_hash, repo
        )
        results = summarize_results(per_arm, amendment)
        results["training_receipts"] = receipts
        results["elapsed_seconds"] = time.monotonic() - started
        atomic_json(out / "results.json", results)
        atomic_json(out / "completion.json", {
            "complete": True,
            "core_sha256": core_hash,
            "results_sha256": digest(out / "results.json"),
            "training_receipts_sha256": {
                arm: digest(out / "training" / arm / "completion.json")
                for arm in receipts
            },
            "continuation_rule_passed": results["frozen_continuation_rule_passed"],
            "elapsed_seconds": results["elapsed_seconds"],
        })
        atomic_json(out / "progress.json", {
            "phase": "complete", "status": "complete", "updated_at": time.time()
        })
        print(json.dumps({"status": "complete", "output": str(out),
                          "continuation_rule_passed": results["frozen_continuation_rule_passed"]}, indent=2))
    except BaseException as error:
        progress_path = out / "progress.json"
        try:
            previous_progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous_progress = None
        write_failure_receipt(
            out,
            phase=phase,
            arm=active_arm,
            error=error,
            core_hash=core_hash,
            gate_receipt=gate_receipt,
            last_progress=previous_progress,
            traceback_text=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
