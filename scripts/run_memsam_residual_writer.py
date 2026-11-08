"""Run the frozen SAM2.1 soft-mask memory-writer study after its hold clears.

The executable refuses all real-frame inference until Run174 has a verified
completion receipt with a failed frozen continuation rule.  It then runs an
engineering gate before any writer training and releases the pilot-test split
only after the frozen validation rule passes.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.complementary_memory.training import capture_current_features, detach_tensors, reader_clip_loss
from modeling.cfbi_training import clone_tensors
from modeling.memsam_memory_writer import NativeMemoryWriter, SoftMaskResidualWriter
from scripts import run_complementary_memory_pilot as base
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import source_identity, tensor_hash
from scripts.verify_complementary_memory_pilot import stored_output

PROTOCOL_REL = "protocols/memsam_residual_memory_writer_v2.json"
MANIFEST_REL = "data/polypgen_sequence.jsonl"
RUN174_REL = Path("174_iterative_memory_reader_v1/attempt_01")
ARMS = ("mask_neutral_writer", "soft_mask_writer")
GATE_SEQUENCES = ("seq2", "seq3")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def status(out, phase, **values):
    record = {"phase": phase, "pid": os.getpid(),
              "updated_at": datetime.now(timezone.utc).isoformat(), **values}
    atomic_json(out / "run_status.json", record)
    print(json.dumps(record, sort_keys=True), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def synced_adeseg_root(output_directory):
    requested = Path(output_directory).expanduser().resolve()
    matches = [path.resolve() for path in
               (Path.home() / "Library/CloudStorage").glob("GoogleDrive-*/My Drive/AdeSEG")
               if path.is_dir()]
    matches = [path for path in matches
               if requested.is_relative_to((path / "outputs/experiment").resolve())]
    if len(matches) != 1:
        raise RuntimeError(
            "Output must be beneath exactly one synced AdeSEG/outputs/experiment folder; "
            f"found {len(matches)}"
        )
    return matches[0]


def current_command(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def run174_hold(run_dir):
    """Fail closed unless Run174 completed and failed its frozen criterion."""
    run_dir = Path(run_dir)
    completion_path = run_dir / "completion.json"
    failure_path = run_dir / "failure.json"
    status_path = run_dir / "run_status.json"
    if not completion_path.is_file():
        live = None
        if status_path.is_file():
            record = json.loads(status_path.read_text())
            command = current_command(record.get("pid", -1))
            if "scripts/run_iterative_memory_pilot.py" in command and str(run_dir) in command:
                live = {"pid": record.get("pid"), "phase": record.get("phase"),
                        "epoch": record.get("epoch"), "clip": record.get("clip")}
        return {"released": False, "reason": "Run174 has no completion receipt",
                "run174_live_process": live,
                "run174_failure_receipt_exists": failure_path.exists()}

    completion = json.loads(completion_path.read_text())
    if status_path.is_file():
        record = json.loads(status_path.read_text())
        command = current_command(record.get("pid", -1))
        if "scripts/run_iterative_memory_pilot.py" in command and str(run_dir) in command:
            return {"released": False, "reason": "Run174 process remains active after writing status",
                    "run174_live_process": {"pid": record.get("pid"), "phase": record.get("phase")}}
    results_path = run_dir / "results.json"
    if (not completion.get("complete") or not results_path.is_file()
            or digest(results_path) != completion.get("results_sha256")):
        return {"released": False, "reason": "Run174 completion/results integrity check failed"}
    if completion.get("continuation_passed") is True:
        return {"released": False,
                "reason": "Run174 passed its continuation rule; prospective alternative remains on hold"}
    if completion.get("continuation_passed") is not False:
        return {"released": False, "reason": "Run174 continuation status is not a frozen boolean"}
    return {"released": True, "reason": "Run174 completed and failed its frozen continuation rule",
            "completion_sha256": digest(completion_path),
            "results_sha256": digest(results_path)}


def training_plan(videos, protocol):
    """Rebuild Run174's deterministic author-sampled schedule exactly."""
    data, config = protocol["data"], protocol["training"]
    plan = []
    for epoch in range(1, config["epochs"] + 1):
        epoch_rows = []
        for name in data["train_sequences"]:
            for draw in range(config["clips_per_sequence_per_epoch"]):
                seed = protocol["training"]["seed"] + epoch * 100000 + int(name[3:]) * 100 + draw
                selected = sample_clip(videos[name], config["clip_length"], seed)
                positions = {frame.frame_index: i for i, frame in enumerate(videos[name])}
                indices = [positions[frame.frame_index] for frame in selected]
                require(indices == list(range(indices[0], indices[0] + config["clip_length"])),
                        "Author sampler changed contiguous clip geometry")
                epoch_rows.append({"epoch": epoch, "sequence": name, "draw": draw,
                                   "sampler_seed": seed, "indices": indices})
        random.Random(protocol["training"]["seed"] + epoch).shuffle(epoch_rows)
        plan.extend(epoch_rows)
    return plan


def source_inventory():
    return [
        PROTOCOL_REL, "protocols/memsam_residual_writer_engineering_amendment_2026-11-08.json",
        MANIFEST_REL, "scripts/run_memsam_residual_writer.py",
        "modeling/memsam_memory_writer.py", "tests/test_memsam_memory_writer.py",
        "tests/test_run_memsam_residual_writer.py",
        "scripts/run_complementary_memory_pilot.py", "scripts/run_iterative_memory_pilot.py",
        "scripts/verify_complementary_memory_pilot.py", "scripts/run_author_baseline.py",
        "scripts/memory_token_io.py", "scripts/bank_integrity.py",
        "scripts/false_alarm_statistics.py", "modeling/complementary_memory/integration.py",
        "modeling/complementary_memory/training.py", "modeling/cfbi_training.py",
        "modeling/decoder_training_mask.py", "datasets/sam2_clip_sampler.py",
        "datasets/video.py", "datasets/io.py", "evaluation/segmentation.py",
        "evaluation/statistics.py", "evaluation/davis2017/metrics.py",
        "external/sam2/sam2/sam2_video_predictor.py",
        "external/sam2/sam2/modeling/sam2_base.py",
        "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        "external/MemSAM/models/segment_anything_memsam/modeling/mem_modules.py",
    ]


def verify_source_inventory(paths, expected, root=ROOT):
    """Verify every runner dependency against the committed hash manifest."""
    paths = tuple(paths)
    if set(paths) != set(expected):
        missing = sorted(set(paths) - set(expected))
        extra = sorted(set(expected) - set(paths))
        raise RuntimeError(
            f"Frozen source inventory differs from protocol; missing={missing}, extra={extra}"
        )
    sources = {}
    for relative in paths:
        actual = digest(root / relative)
        require(actual == expected[relative], f"Frozen study source changed: {relative}")
        sources[relative] = actual
    return sources


def guard(core, videos, sequence=None):
    for relative, expected in core["sources"].items():
        require(digest(ROOT / relative) == expected, f"Frozen study source changed: {relative}")
    if sequence is not None:
        current = base.input_hashes(videos, [sequence])[sequence]
        require(current == core["inputs"][sequence], f"Video input changed: {sequence}")


def setup(args, videos, out, drive, release):
    protocol_path = ROOT / PROTOCOL_REL
    protocol = json.loads(protocol_path.read_text())
    head = base.git_text("rev-parse", "HEAD")
    committed_protocol = subprocess.check_output(
        ["git", "show", f"{head}:{PROTOCOL_REL}"], cwd=ROOT
    )
    require(committed_protocol == protocol_path.read_bytes(),
            "Protocol must be committed unchanged before the real-frame gate")
    require(digest(ROOT / MANIFEST_REL) == protocol["model"]["manifest_sha256"],
            "Manifest hash differs from the frozen protocol")
    require(digest(args.checkpoint) == protocol["model"]["checkpoint_sha256"],
            "Checkpoint hash differs from the frozen protocol")
    for split in ("train_sequences", "validation_sequences", "test_sequences"):
        require(set(protocol["data"][split]).issubset(videos), f"Missing sequences in {split}")
    train = protocol["data"]["train_sequences"]
    validation = protocol["data"]["validation_sequences"]
    test = protocol["data"]["test_sequences"]
    require(not (set(train) & set(validation) or set(train) & set(test)
                 or set(validation) & set(test)), "Split leakage")

    repo, revision = base.activate_author_repo()
    require(revision == protocol["model"]["sam2_revision"], "SAM2 revision changed")
    upstream = ROOT / "external/MemSAM"
    memsam_revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    memsam_status = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    require(memsam_revision == protocol["model"]["memsam_revision"] and not memsam_status,
            "MemSAM repository is not the pinned clean checkout")

    paths = source_inventory()
    source_paths = [relative for relative in paths if relative != PROTOCOL_REL]
    sources = verify_source_inventory(
        source_paths, protocol["implementation"]["source_sha256"]
    )
    sources[PROTOCOL_REL] = digest(protocol_path)
    input_hashes = base.input_hashes(videos, train + validation + test)
    plan = training_plan(videos, protocol)
    run174_setup = json.loads((drive / "outputs/experiment" / RUN174_REL / "setup.json").read_text())
    require(plan == run174_setup["core"]["plan"],
            "Candidate clip plan is not byte-for-byte equivalent to Run174's frozen plan")
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    core = {
        "protocol": protocol, "protocol_sha256": digest(protocol_path), "sources": sources,
        "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(ROOT / MANIFEST_REL),
        "inputs": input_hashes, "sam2_revision": revision,
        "memsam_revision": memsam_revision, "plan": plan, "plan_sha256": plan_hash,
        "run174_release": release, "torch": torch.__version__,
    }
    core_hash = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    setup_path = out / "setup.json"
    if args.resume:
        saved = json.loads(setup_path.read_text())
        require(saved["core_sha256"] == core_hash, "Resume source/data/protocol mismatch")
        require(digest(out / "source_snapshot.zip") == saved["snapshot_sha256"],
                "Source snapshot changed")
    else:
        require(not setup_path.exists() and not any(out.iterdir()),
                "Nonempty attempt directory requires --resume")
        with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for relative in paths:
                archive.write(ROOT / relative, relative)
        base.atomic_json(setup_path, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": head,
            "core": core, "core_sha256": core_hash,
            "snapshot_sha256": digest(out / "source_snapshot.zip"),
            "command": sys.argv, "scope": "Exploratory PolypGen pilot; frozen SAM2 weights",
        })
    return protocol, core, core_hash, repo


def init_model(checkpoint, repo):
    model = base.init_predictor(checkpoint, repo)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def writer_init(seed, protocol):
    torch.manual_seed(seed)
    model = SoftMaskResidualWriter(
        channels=protocol["writer"]["channels"],
        mid_channels=protocol["writer"]["mid_channels"],
        kernel=protocol["writer"]["kernel"],
    )
    return model


def writer_optimizer(writer, protocol):
    config = protocol["training"]["optimizer"]
    require(config["name"] == "AdamW", "Unexpected frozen writer optimizer")
    return torch.optim.AdamW(
        writer.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"], betas=tuple(config["betas"]),
        eps=config["eps"],
    )


def expected_writer_state(mode, writer, expected_frames, records):
    if mode is None:
        require(not records, "Native baseline unexpectedly has writer records")
        return
    frame_ids = [row["frame"] for row in records]
    require(len(records) == len(set(frame_ids)), "Memory writer wrote a frame more than once")
    require(set(frame_ids) == set(expected_frames),
            "Memory writer did not record each frame exactly once")
    for row in records:
        require(row["mode"] == mode, "Writer mode differs from requested condition")
        require(row["memory_mask_scale"] != 0, "Invalid memory-mask scale receipt")
        require(row["feature_shape"][0] == row["mask_shape"][0],
                "Memory feature/mask object batch differs")
        if mode == "regional":
            require(len(row["recovered_probability_sha256"]) == 64,
                    "Missing recovered soft-mask probability hash")
        if not torch.isfinite(torch.tensor([row["memory_mask_scale"], row["memory_mask_bias"]])).all():
            raise RuntimeError("Invalid writer transform values")


def rollout_gate(checkpoint, repo, frames, output, writer=None, mode="regional"):
    model = init_model(checkpoint, repo)
    model_hash = parameter_digest(model)
    writes = []
    writer_context = (NativeMemoryWriter(model, writer, mode) if writer is not None
                      else nullcontext(None))
    with tempfile.TemporaryDirectory(prefix="memwriter_gate_", dir=output) as temporary:
        with writer_context as write_observer:
            state, prompt = base.prepare(model, frames, Path(temporary))
            observer = NativeComplementaryRead(
                model, None, "regional", inference_state=state, trace_only=True
            )
            rows, calls = [], []
            iterator = None
            try:
                with torch.inference_mode():
                    with observer:
                        iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                        for frame, ids, logits in iterator:
                            if ids != [1] or logits.shape[:2] != (1, 1):
                                raise RuntimeError("Unexpected object identity or output shape")
                            stored = stored_output(state["output_dict_per_obj"][0], frame)
                            row = {
                                "frame": frame,
                                "prediction_sha256": tensor_hash(logits),
                                "stored_sha256": fingerprint(stored)["sha256"],
                                "pointer_sha256": tensor_hash(stored["obj_ptr"]),
                                "position_sha256": fingerprint(stored["maskmem_pos_enc"])["sha256"],
                            }
                            if frame != prompt:
                                row["read"] = observer.calls[-1]
                                calls.append(observer.calls[-1])
                            rows.append(row)
                if write_observer is not None:
                    writes = list(write_observer.records)
            finally:
                if iterator is not None:
                    iterator.close()
                with torch.inference_mode():
                    model.reset_state(state)
    require(parameter_digest(model) == model_hash, "SAM2 parameters changed during gate rollout")
    expected_writer_state(mode if writer is not None else None, writer,
                          range(prompt, len(frames)), writes)
    del model
    gc.collect()
    return {"prompt_frame": prompt, "rows": rows, "reader_calls": calls,
            "memory_writer_calls": writes, "model_parameter_sha256": model_hash}


def compare_identity(native, adapted, name):
    require(native["prompt_frame"] == adapted["prompt_frame"], f"Prompt differs: {name}")
    left, right = native["rows"], adapted["rows"]
    require(len(left) == len(right), f"Frame coverage differs: {name}")
    for a, b in zip(left, right):
        for key in ("frame", "prediction_sha256", "stored_sha256", "pointer_sha256", "position_sha256"):
            require(a[key] == b[key], f"Zero-writer identity differs: {name}/{a['frame']}/{key}")
    native_reads, adapted_reads = native["reader_calls"], adapted["reader_calls"]
    require(len(native_reads) == len(adapted_reads), f"Read count differs: {name}")
    for a, b in zip(native_reads, adapted_reads):
        for key in ("frame", "object_id", "memory_frames", "spatial_keys", "pointer_keys",
                    "pointer_input_sha256", "pointer_position_sha256", "layers",
                    "stored_mask_alignment", "native_positions_verified"):
            require(a.get(key) == b.get(key), f"Memory trace differs: {name}/{a.get('frame')}/{key}")
        require(b["layers"] == [0, 1, 2, 3], "Not all native attention layers were observed")
        require(b["stored_mask_alignment"] and b["native_positions_verified"],
                "Stored mask or native positional encoding verification failed")
    for row in adapted["memory_writer_calls"]:
        require(row["input_feature_sha256"] == row["adapted_feature_sha256"],
                f"Zero-initialized writer changed a memory feature: {name}/{row['frame']}")
        require(row["memory_mask_scale"] == 20.0 and row["memory_mask_bias"] == -10.0,
                "Unexpected SAM2.1 memory-mask affine transform")
    require(len(adapted["memory_writer_calls"]) == len(adapted["rows"]),
            f"Memory write/frame coverage differs: {name}")
    return {"sequence": name, "frames_including_prompt": len(right),
            "post_prompt_reads": len(adapted_reads), "native_logits_memory_positions_pointers_bit_identical": True,
            "all_four_attention_layers_verified": True}


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def multi_object_rollout(checkpoint, repo, frames, output, order, masks, writer=None):
    model = init_model(checkpoint, repo)
    model_hash = parameter_digest(model)
    writer_context = NativeMemoryWriter(model, writer, "regional") if writer is not None else nullcontext(None)
    result, read_ids, writer_records = {}, set(), []
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    clip = frames[prompt:prompt + 3]
    with tempfile.TemporaryDirectory(prefix="writer_multiobj_", dir=output) as temporary:
        stage_frames(clip, Path(temporary))
        with writer_context as write_observer:
            with torch.inference_mode():
                state = model.init_state(video_path=temporary, offload_video_to_cpu=True,
                                         offload_state_to_cpu=False)
                try:
                    for object_id in order:
                        model.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=masks[object_id])
                except BaseException:
                    model.reset_state(state)
                    raise
            observer = NativeComplementaryRead(
                model, None, "regional", inference_state=state, trace_only=True
            )
            iterator = None
            try:
                with torch.inference_mode():
                    with observer:
                        iterator = model.propagate_in_video(state, start_frame_idx=0)
                        for frame, ids, logits in iterator:
                            if frame >= 3:
                                break
                            require(set(ids) == set(order) and logits.shape[0] == len(order),
                                    "Persistent object IDs changed in multi-object propagation")
                            for index, object_id in enumerate(ids):
                                object_index = state["obj_id_to_idx"][object_id]
                                bank = state["output_dict_per_obj"][object_index]
                                stored = stored_output(bank, frame)
                                result[f"{frame}:{object_id}"] = {
                                    "prediction_sha256": tensor_hash(logits[index]),
                                    "stored_sha256": fingerprint(stored)["sha256"],
                                    "pointer_sha256": tensor_hash(stored["obj_ptr"]),
                                }
                    read_ids = {row["object_id"] for row in observer.calls}
                if write_observer is not None:
                    writer_records = list(write_observer.records)
            finally:
                if iterator is not None:
                    iterator.close()
                with torch.inference_mode():
                    model.reset_state(state)
    require(parameter_digest(model) == model_hash, "SAM2 parameters changed in multi-object gate")
    del model
    gc.collect()
    return {"outputs": result, "read_object_ids": sorted(read_ids),
            "memory_writer_calls": writer_records, "model_parameter_sha256": model_hash}


def multi_object_gate(args, repo, frames, output, seed):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    mask = torch.from_numpy((load_label_mask(frames[prompt].mask_path, [1]) == 1).copy())
    coordinates = mask.nonzero()
    require(len(coordinates) > 1, "seq2 prompt mask too small for two-object fixture")
    middle = int(coordinates[:, 1].median())
    left = mask.clone(); left[:, middle:] = False
    right = mask & ~left
    require(left.any() and right.any(), "Cannot split seq2 prompt into two objects")
    masks = {7: left, 42: right}
    native = multi_object_rollout(args.checkpoint, repo, frames, output, [7, 42], masks)
    zero = writer_init(seed, json.loads((ROOT / PROTOCOL_REL).read_text()).get("writer", {}))
    adapted = multi_object_rollout(args.checkpoint, repo, frames, output, [7, 42], masks, zero)
    reversed_order = multi_object_rollout(args.checkpoint, repo, frames, output, [42, 7], masks, zero)
    singletons = {
        str(object_id): multi_object_rollout(args.checkpoint, repo, frames, output,
                                              [object_id], masks, zero)
        for object_id in (7, 42)
    }
    require(native["outputs"] == adapted["outputs"], "Zero writer changed multi-object outputs")
    require(adapted["outputs"] == reversed_order["outputs"],
            "Object prompt order changed zero-writer outputs")
    require(adapted["read_object_ids"] == [7, 42], "Attention trace lost an object ID")
    for object_id, single in ((7, singletons["7"]), (42, singletons["42"])):
        expected = {key: value for key, value in adapted["outputs"].items()
                    if key.endswith(f":{object_id}")}
        actual = {key: value for key, value in single["outputs"].items()
                  if key.endswith(f":{object_id}")}
        require(expected == actual, f"Cross-object influence detected for ID {object_id}")
    return {"passed": True, "object_ids": [7, 42], "frames": 3,
            "zero_writer_identity": True, "order_invariant": True,
            "single_object_isolation": True,
            "scope": "Synthetic two-instance split-prompt fixture on real seq2 frames; not natural multi-object efficacy"}


def train_one_clip(model, writer, mode, optimizer, frames, name, output, seed, training):
    model_hash = parameter_digest(model)
    writes, calls, updates = [], [], []
    writer_context = NativeMemoryWriter(model, writer, mode)
    with tempfile.TemporaryDirectory(prefix=f"writer_train_{name}_", dir=output) as temporary:
        with writer_context as write_observer:
            state, prompt = base.prepare(model, frames, Path(temporary))
            with torch.inference_mode():
                model.propagate_in_video_preflight(state)
            with torch.inference_mode(False):
                bank = clone_tensors(state["output_dict_per_obj"][0])
            total = len(frames) - prompt - 1
            require(total == training["clip_length"] - 1,
                    f"Expected {training['clip_length'] - 1} post-prompt frames in {name}, got {total}")
            start = prompt + 1
            update_number = 0
            while start < len(frames):
                stop = min(start + training["truncated_bptt_frames"], len(frames))
                samples = []
                for current in range(start, stop):
                    visual = capture_current_features(model, state, current)
                    target, _ = base.target_for(frames[current])
                    samples.append({
                        "frame": current, "num_frames": len(frames),
                        "vision": visual["vision"], "positions": visual["positions"],
                        "sizes": visual["sizes"], "target": target,
                        "mask_inputs": None, "point_inputs": None,
                    })
                optimizer.zero_grad(set_to_none=True)
                observer = NativeComplementaryRead(
                    model, None, "regional", inference_state=state, trace_only=True
                )
                with observer:
                    loss, bank, frame_writes = reader_clip_loss(model, samples, bank)
                loss.backward()
                gradients = [p.grad for p in writer.parameters() if p.grad is not None]
                require(bool(gradients) and all(torch.isfinite(g).all() for g in gradients),
                        f"Writer gradients absent/nonfinite in {name}")
                gradient_l1 = float(sum(g.detach().abs().sum() for g in gradients))
                require(gradient_l1 > 0, f"Writer gradient is zero in {name}")
                require(not any(p.grad is not None for p in model.parameters()),
                        "Frozen SAM2 received parameter gradients")
                torch.nn.utils.clip_grad_norm_(
                    writer.parameters(), training["optimizer"]["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                bank = detach_tensors(bank)
                update_number += 1
                calls.extend(observer.calls)
                updates.append({"update": update_number, "frames": [start, stop - 1],
                                "loss": float(loss.detach()), "gradient_l1": gradient_l1,
                                "reads": observer.calls, "writes": frame_writes})
                start = stop
            writes = list(write_observer.records)
            with torch.inference_mode():
                model.reset_state(state)
    expected_updates = math.ceil(total / training["truncated_bptt_frames"])
    require(update_number == expected_updates and len(calls) == total,
            "Sampled training clip coverage mismatch")
    require(parameter_digest(model) == model_hash, "SAM2 parameter hash changed during writer training")
    require(len(writes) == len(frames), "Memory writer did not write prompt + all sampled frames")
    return {"sequence": name, "prompt_frame": prompt, "post_prompt_frames": total,
            "optimizer_updates": update_number, "finite_nonzero_writer_gradients": True,
            "original_sam2_parameters_unchanged": True, "memory_writer_calls": writes,
            "attention_reads": calls, "chunks": updates,
            "post_prompt_ground_truth_memory_writes": False}


def engineering_gate(args, videos, out, protocol, repo):
    gate_dir = out / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    completion = gate_dir / "completion.json"
    if completion.exists():
        receipt = json.loads(completion.read_text())
        require(receipt.get("passed") and digest(gate_dir / "results.json") == receipt["results_sha256"],
                "Existing writer gate receipt is invalid")
        return json.loads((gate_dir / "results.json").read_text())

    identity_rows = []
    for name in GATE_SEQUENCES:
        status(out, "native_identity_gate", sequence=name)
        native = rollout_gate(args.checkpoint, repo, videos[name], gate_dir)
        base.atomic_json(gate_dir / f"{name}_native.json", native)
        zero = writer_init(protocol["training"]["seed"], protocol)
        zero.eval()
        adapted = rollout_gate(args.checkpoint, repo, videos[name], gate_dir, zero, "regional")
        identity_rows.append(compare_identity(native, adapted, name))
        base.atomic_json(gate_dir / f"{name}_zero_writer.json", adapted)
        del zero, native, adapted
        gc.collect()

    # A deterministic real clip checks added-module gradients and checkpoint replay.
    plan = training_plan(videos, protocol)
    clip_spec = plan[0]
    frames = [videos[clip_spec["sequence"]][index] for index in clip_spec["indices"]]
    status(out, "real_clip_gradient_resume_gate", sequence=clip_spec["sequence"])
    model = init_model(args.checkpoint, repo)
    original_hash = parameter_digest(model)
    seed = protocol["training"]["seed"]
    writer_a = writer_init(seed, protocol).train()
    optimizer_a = writer_optimizer(writer_a, protocol)
    initial = {
        "writer": {k: v.detach().clone() for k, v in writer_a.state_dict().items()},
        "optimizer": optimizer_a.state_dict(), "rng": torch.random.get_rng_state().clone(),
    }
    initial_path = gate_dir / "resume_initial.pt"
    base.atomic_torch_save(initial_path, initial)
    first = train_one_clip(model, writer_a, "regional", optimizer_a, frames,
                           clip_spec["sequence"], gate_dir, seed, protocol["training"])
    expected_writer = {k: v.detach().clone() for k, v in writer_a.state_dict().items()}
    expected_optimizer = optimizer_a.state_dict()
    restored = torch.load(initial_path, map_location="cpu", weights_only=True)
    writer_b = writer_init(seed, protocol).train()
    writer_b.load_state_dict(restored["writer"], strict=True)
    optimizer_b = writer_optimizer(writer_b, protocol)
    optimizer_b.load_state_dict(restored["optimizer"])
    torch.random.set_rng_state(restored["rng"])
    second = train_one_clip(model, writer_b, "regional", optimizer_b, frames,
                            clip_spec["sequence"], gate_dir, seed, protocol["training"])
    require(tree_equal(expected_writer, writer_b.state_dict()), "Real-clip writer resume differs")
    require(tree_equal(expected_optimizer, optimizer_b.state_dict()), "Real-clip optimizer resume differs")
    require(parameter_digest(model) == original_hash, "SAM2 changed in real-clip gate")
    training_gate = {"clip_plan": clip_spec, "writer_gradients": first["finite_nonzero_writer_gradients"],
                     "sam2_hash_unchanged": True, "real_clip_save_resume_exact": True,
                     "writer_receipts": len(first["memory_writer_calls"]),
                     "attention_reads": len(first["attention_reads"]),
                     "training_replay_receipts_equal": first == second}
    require(training_gate["training_replay_receipts_equal"], "Real-clip training receipts did not replay")
    del model, writer_a, writer_b, optimizer_a, optimizer_b
    gc.collect()

    status(out, "multi_object_id_gate", sequence="seq2")
    multi_object = multi_object_gate(args, repo, videos["seq2"], gate_dir,
                                     protocol["training"]["seed"])

    results = {
        "passed": True, "native_identity": identity_rows, "real_clip_training": training_gate,
        "multi_object_identity": multi_object,
        "scope": "Engineering gate only. No efficacy result or test-set inference.",
    }
    base.atomic_json(gate_dir / "results.json", results)
    base.atomic_json(completion, {"passed": True, "results_sha256": digest(gate_dir / "results.json")})
    return results


def evaluate_sequence(model, frames, output, name, writer=None, mode=None, setup_hash=None):
    model_hash = parameter_digest(model)
    final = output / name
    writer_state_hash = "native" if writer is None else base.state_digest(writer.state_dict())
    if (final / "summary.json").is_file():
        cached = json.loads((final / "summary.json").read_text())
        require(cached.get("writer_state_sha256") == writer_state_hash
                and cached.get("setup_core_sha256") == setup_hash,
                f"Cached writer evaluation differs: {final}")
        return cached
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial.", dir=output))
    masks_dir = temporary / "masks"
    masks_dir.mkdir()
    writer_context = NativeMemoryWriter(model, writer, mode) if writer is not None else nullcontext(None)
    try:
        with tempfile.TemporaryDirectory(prefix=f"frames_{name}_", dir=temporary) as frame_temp:
            with writer_context as write_observer:
                state, prompt = base.prepare(model, frames, Path(frame_temp))
                observer = NativeComplementaryRead(
                    model, None, "regional", inference_state=state, trace_only=True
                )
                rows, reads = [], []
                iterator = None
                start = time.monotonic()
                try:
                    with torch.inference_mode():
                        with observer:
                            iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                            for frame, ids, logits in iterator:
                                if ids != [1] or logits.ndim != 4 or logits.shape[:2] != (1, 1):
                                    raise RuntimeError("Unexpected object ID or prediction shape")
                                if frame == prompt:
                                    continue
                                if frame != prompt + len(rows) + 1:
                                    raise RuntimeError("Nonchronological or missing prediction frame")
                                prediction = logits[0, 0].detach().cpu().numpy() > 0
                                truth = load_label_mask(frames[frame].mask_path, [1]) == 1
                                if prediction.shape != truth.shape:
                                    raise RuntimeError("Prediction/GT geometry mismatch")
                                scores = calculate_scores(prediction, truth)
                                row = {
                                    "frame": frame, "source_frame_index": frames[frame].frame_index,
                                    "prediction_sha256": tensor_hash(logits),
                                    "prediction_nonempty": bool(prediction.any()),
                                    "present": bool(truth.any()),
                                    "scores": {**scores, "boundary_f": float(db_eval_boundary(prediction, truth))},
                                    "absent_fp": bool(prediction.any()) if not truth.any() else None,
                                }
                                row["read"] = observer.calls[-1]
                                reads.append(observer.calls[-1])
                                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                                    masks_dir / f"{frame:06d}.png"
                                )
                                row["mask_sha256"] = digest(masks_dir / f"{frame:06d}.png")
                                rows.append(row)
                    writer_records = [] if write_observer is None else list(write_observer.records)
                finally:
                    if iterator is not None:
                        iterator.close()
                        with torch.inference_mode():
                            model.reset_state(state)
        expected = len(frames) - prompt - 1
        require(len(rows) == expected and len(reads) == expected,
                f"Incomplete evaluation coverage: {name}")
        if writer is not None:
            require(len(writer_records) == len(frames) - prompt,
                    f"Incomplete writer coverage: {name}")
        require(parameter_digest(model) == model_hash, "SAM2 parameters changed during evaluation")
        present = [row for row in rows if row["present"]]
        elapsed = time.monotonic() - start
        summary = {
            "sequence": name, "condition": "native" if writer is None else mode,
            "prompt_frame": prompt, "target_frames": len(rows),
            "present_frames": len(present), "gt_empty_frames": len(rows) - len(present),
            "present_dice": float(np.mean([row["scores"]["dice"] for row in present])) if present else None,
            "all_frame_dice": float(np.mean([row["scores"]["dice"] for row in rows])) if rows else None,
            "present_iou": float(np.mean([row["scores"]["iou"] for row in present])) if present else None,
            "present_boundary_f": float(np.mean([row["scores"]["boundary_f"] for row in present])) if present else None,
            "gt_empty_false_alarm_count": sum(bool(row["absent_fp"]) for row in rows if not row["present"]),
            "gt_empty_false_alarm_rate": float(np.mean([bool(row["absent_fp"]) for row in rows if not row["present"]])) if any(not row["present"] for row in rows) else None,
            "propagation": propagation_summary(rows), "elapsed_seconds": elapsed,
            "fps": len(rows) / elapsed if elapsed else None,
            "writer_state_sha256": writer_state_hash,
            "setup_core_sha256": setup_hash,
            "original_sam2_parameter_sha256": model_hash,
            "native_attention_reads": reads,
        }
        if writer_records:
            atomic_json(temporary / "memory_writer_calls.json", writer_records)
        base.state_to_json_rows(temporary / "frames.jsonl", rows)
        atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, final)
        return summary
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def aggregate(summaries, sequence_names, seed):
    means = {}
    for arm, rows in summaries.items():
        means[arm] = {
            key: float(np.mean([rows[name][key] for name in sequence_names if rows[name][key] is not None]))
            for key in ("present_dice", "present_iou", "present_boundary_f", "fps")
        }
        empty = sum(rows[name]["gt_empty_frames"] for name in sequence_names)
        false = sum(rows[name]["gt_empty_false_alarm_count"] for name in sequence_names)
        means[arm]["false_alarm_count"] = false
        means[arm]["gt_empty_frames"] = empty
        means[arm]["false_alarm_rate"] = false / empty if empty else None
        means[arm]["undetected_returns"] = sum(
            rows[name]["propagation"]["undetected_returns"] for name in sequence_names
        )
        means[arm]["return_events"] = sum(
            rows[name]["propagation"]["return_count"] for name in sequence_names
        )
        means[arm]["missed_return_rate"] = (
            means[arm]["undetected_returns"] / means[arm]["return_events"]
            if means[arm]["return_events"] else None
        )
    contrasts, pvalues = {}, {}
    for candidate, control in (("soft_mask_writer", "native_sam2"),
                               ("soft_mask_writer", "mask_neutral_writer")):
        delta = {name: summaries[candidate][name]["present_dice"] - summaries[control][name]["present_dice"]
                 for name in sequence_names}
        key = f"{candidate}_minus_{control}"
        contrasts[key] = {
            "mean_dice_gain": float(np.mean(list(delta.values()))),
            "bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=seed),
            "wins": sum(value > 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "losses": sum(value < 0 for value in delta.values()),
            "per_sequence": delta,
            "leave_one_sequence_out_mean_gain": {
                omitted: float(np.mean([value for name, value in delta.items() if name != omitted]))
                for omitted in sequence_names
            },
            "wilcoxon": paired_wilcoxon(
                {name: summaries[candidate][name]["present_dice"] for name in sequence_names},
                {name: summaries[control][name]["present_dice"] for name in sequence_names},
            ),
        }
        pvalues[key] = contrasts[key]["wilcoxon"]["p"]
    adjusted = holm_adjust(pvalues)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value
    return {"per_sequence": summaries, "means": means, "contrasts": contrasts}


def train_arm(args, protocol, core, core_hash, repo, videos, out, arm, resume):
    folder = out / "training" / arm
    folder.mkdir(parents=True, exist_ok=True)
    model = init_model(args.checkpoint, repo)
    original_hash = parameter_digest(model)
    seed = protocol["training"]["seed"]
    writer = writer_init(seed, protocol).train()
    optimizer = writer_optimizer(writer, protocol)
    init_hash = base.state_digest(writer.state_dict())
    checkpoint_path = folder / "resume.pt"
    clip_hashes, validation = [], {}
    completed, best_score, best_epoch = 0, -math.inf, None
    if checkpoint_path.exists():
        require(resume, f"Checkpoint exists; resume explicitly: {checkpoint_path}")
        receipt = json.loads((folder / "resume_receipt.json").read_text())
        require(digest(checkpoint_path) == receipt["sha256"], "Resume checkpoint digest mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(saved["core_sha256"] == core_hash and saved["arm"] == arm,
                "Resume checkpoint setup/arm mismatch")
        require(saved["writer_initial_sha256"] == init_hash, "Writer initialization changed")
        writer.load_state_dict(saved["writer"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["torch_rng_state"])
        completed, validation = saved["completed_clips"], saved["validation"]
        clip_hashes = saved["clip_hashes"]
        best_score, best_epoch = saved["best_score"], saved["best_epoch"]
        require(len(clip_hashes) == completed == receipt["completed_clips"],
                "Resume clip coverage differs")
        for index, sha in enumerate(clip_hashes):
            require(digest(folder / "clips" / f"{index:04d}.json") == sha,
                    f"Completed clip receipt changed: {index}")
    else:
        require(not any(folder.iterdir()), "Nonempty training directory without resume checkpoint")

    plan = core["plan"]
    config = protocol["training"]
    per_epoch = len(protocol["data"]["train_sequences"]) * config["clips_per_sequence_per_epoch"]
    for index in range(completed, len(plan)):
        item = plan[index]
        name = item["sequence"]
        status(out, "training", arm=arm, epoch=item["epoch"], clip=index + 1,
               total_clips=len(plan), sequence=name)
        frames = [videos[name][frame_index] for frame_index in item["indices"]]
        record = train_one_clip(model, writer, "generic" if arm == "mask_neutral_writer" else "regional",
                                optimizer, frames, name, out, seed, config)
        require(record["post_prompt_frames"] == protocol["training"]["clip_length"] - 1
                and record["optimizer_updates"] == math.ceil(
                    (protocol["training"]["clip_length"] - 1)
                    / protocol["training"]["truncated_bptt_frames"]),
                "Training clip geometry differs from frozen plan")
        payload = {"plan": item, "record": record}
        clip_path = folder / "clips" / f"{index:04d}.json"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        if clip_path.exists():
            require(json.loads(clip_path.read_text()) == payload,
                    f"Replay after an interrupted checkpoint differs: clip {index}")
        else:
            atomic_json(clip_path, payload)
        clip_hashes.append(digest(clip_path))
        completed = index + 1
        epoch_done = completed % per_epoch == 0
        if epoch_done and item["epoch"] in config["validation_epochs"]:
            require(parameter_digest(model) == original_hash, "SAM2 parameters changed before validation")
            epoch = item["epoch"]
            eval_model = init_model(args.checkpoint, repo)
            val_rows = {}
            for val_name in protocol["data"]["validation_sequences"]:
                mode = "generic" if arm == "mask_neutral_writer" else "regional"
                val_rows[val_name] = evaluate_sequence(
                    eval_model, videos[val_name], out / "validation" / arm / f"epoch_{epoch:02d}",
                    val_name, writer.eval(), mode, core_hash
                )
                writer.train()
            del eval_model; gc.collect()
            score = float(np.mean([val_rows[seq]["present_dice"] for seq in val_rows]))
            validation[str(epoch)] = {"present_dice": score, "summaries": val_rows}
            if score > best_score:
                best_score, best_epoch = score, epoch
                atomic_torch_save(folder / "best.pt", writer.state_dict())

        # Save at every complete sampled clip so an interruption replays at most one clip.
        atomic_torch_save(checkpoint_path, {
            "core_sha256": core_hash, "arm": arm, "writer": writer.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng_state": torch.random.get_rng_state(),
            "completed_clips": completed, "clip_hashes": clip_hashes,
            "validation": validation, "best_score": best_score, "best_epoch": best_epoch,
            "writer_initial_sha256": init_hash,
        })
        atomic_json(folder / "resume_receipt.json", {
            "sha256": digest(checkpoint_path), "completed_clips": completed,
        })

    require(completed == 480 and best_epoch is not None, "Training did not complete frozen schedule")
    require(parameter_digest(model) == original_hash, "SAM2 parameters changed during training")
    writer.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True), strict=True)
    atomic_json(folder / "completion.json", {
        "complete": True, "arm": arm, "clips": completed, "updates": completed * 2,
        "validation": validation, "best_epoch": best_epoch, "best_validation_present_dice": best_score,
        "best_checkpoint_sha256": digest(folder / "best.pt"),
        "original_sam2_parameter_sha256": original_hash,
    })
    del model; gc.collect()
    return writer.eval()


def validation_release(protocol, summaries):
    names = protocol["data"]["validation_sequences"]
    regional = summaries["soft_mask_writer"]
    neutral = summaries["mask_neutral_writer"]
    native = summaries["native_sam2"]
    checks = {}
    for control in ("native_sam2", "mask_neutral_writer"):
        wins = sum(regional[name]["present_dice"] > summaries[control][name]["present_dice"] for name in names)
        gain = float(np.mean([regional[name]["present_dice"] - summaries[control][name]["present_dice"] for name in names]))
        checks[control] = {"mean_gain": gain, "wins": wins,
                           "pass": gain >= 0.005 and wins >= 2}
    empty = sum(native[name]["gt_empty_frames"] for name in names)
    native_false = sum(native[name]["gt_empty_false_alarm_count"] for name in names)
    regional_false = sum(regional[name]["gt_empty_false_alarm_count"] for name in names)
    native_rate = native_false / empty if empty else 0.0
    regional_rate = regional_false / empty if empty else 0.0
    checks["false_alarm_safety"] = {"native_rate": native_rate, "regional_rate": regional_rate,
                                    "pass": regional_rate <= native_rate + 0.02}
    return {"checks": checks, "passed": all(row["pass"] for row in checks.values())}


def final_report(summaries, protocol):
    names = protocol["data"]["test_sequences"]
    result = aggregate(summaries, names, protocol["training"]["seed"])
    regional, native, neutral = (summaries[key] for key in (
        "soft_mask_writer", "native_sam2", "mask_neutral_writer"))
    c_native = result["contrasts"]["soft_mask_writer_minus_native_sam2"]
    c_neutral = result["contrasts"]["soft_mask_writer_minus_mask_neutral_writer"]
    checks = {
        "gain_vs_native": c_native["mean_dice_gain"] >= 0.010 and c_native["bootstrap_95_ci"][0] > 0 and c_native["wins"] >= 5,
        "gain_vs_matched_control": c_neutral["mean_dice_gain"] >= 0.005 and c_neutral["bootstrap_95_ci"][0] > 0 and c_neutral["wins"] >= 5,
        "false_alarm_safety": result["means"]["soft_mask_writer"]["false_alarm_rate"] <= result["means"]["native_sam2"]["false_alarm_rate"] + 0.02,
        "missed_return_safety": result["means"]["soft_mask_writer"]["missed_return_rate"] <= result["means"]["native_sam2"]["missed_return_rate"] + 0.05,
    }
    result.update({"scope": "Exploratory full-propagation PolypGen pilot; not external confirmation",
                   "continuation_checks": checks, "continuation_passed": all(checks.values()),
                   "outlier_sensitivity": "Per-sequence leave-one-out mean Dice contrasts included."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--check-hold", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    out = validate_output(args.output_dir)
    drive = synced_adeseg_root(out)
    require(out.is_relative_to((drive / "outputs/experiment/memsam_residual_memory_writer_v2").resolve()),
            "Output must be under the frozen v2 experiment folder on Google Drive")
    release = run174_hold(drive / "outputs/experiment" / RUN174_REL)
    if args.check_hold:
        print(json.dumps(release, indent=2))
        return
    require(release["released"], "Hold active: " + release["reason"])
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    else:
        require(out.is_dir(), "Resume attempt directory does not exist")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    try:
        videos = load_manifest(ROOT / MANIFEST_REL, "dev") | load_manifest(ROOT / MANIFEST_REL, "test")
        protocol, core, core_hash, repo = setup(args, videos, out, drive, release)
        gate = engineering_gate(args, videos, out, protocol, repo)
        if args.gate_only:
            status(out, "engineering_gate_complete_no_efficacy_run")
            return

        # Native validation is the identical starting model and prompt protocol.
        baseline_model = init_model(args.checkpoint, repo)
        native_validation = {}
        for name in protocol["data"]["validation_sequences"]:
            status(out, "native_validation", sequence=name)
            native_validation[name] = evaluate_sequence(
                baseline_model, videos[name], out / "validation/native", name,
                setup_hash=core_hash
            )
        del baseline_model; gc.collect()
        summaries = {"native_sam2": native_validation}
        trained = {}
        for arm in ARMS:
            trained[arm] = train_arm(args, protocol, core, core_hash, repo, videos, out, arm, args.resume)
            summaries[arm] = {}
            eval_model = init_model(args.checkpoint, repo)
            mode = "generic" if arm == "mask_neutral_writer" else "regional"
            for name in protocol["data"]["validation_sequences"]:
                summaries[arm][name] = evaluate_sequence(
                    eval_model, videos[name], out / "validation" / arm / "selected", name,
                    trained[arm], mode, core_hash
                )
            del eval_model; gc.collect()
        release_validation = validation_release(protocol, summaries)
        atomic_json(out / "validation_release.json", release_validation)
        if not release_validation["passed"]:
            atomic_json(out / "completion.json", {
                "complete": True, "scope": "Validation failed; pilot-test split was not opened",
                "core_sha256": core_hash, "validation_release_passed": False,
                "validation_release_sha256": digest(out / "validation_release.json"),
                "elapsed_seconds": time.monotonic() - started,
            })
            status(out, "complete_validation_failed_test_not_run")
            return

        # Only after all trained arms and the frozen validation release pass.
        test_summaries = {}
        for arm in ("native_sam2", *ARMS):
            test_summaries[arm] = {}
            eval_model = init_model(args.checkpoint, repo)
            mode = None if arm == "native_sam2" else (
                "generic" if arm == "mask_neutral_writer" else "regional"
            )
            writer = None if arm == "native_sam2" else trained[arm]
            for name in protocol["data"]["test_sequences"]:
                status(out, "pilot_test", arm=arm, sequence=name)
                test_summaries[arm][name] = evaluate_sequence(
                    eval_model, videos[name], out / "evaluation" / arm, name, writer, mode, core_hash
                )
            del eval_model; gc.collect()
        results = final_report(test_summaries, protocol)
        atomic_json(out / "results.json", results)
        atomic_json(out / "completion.json", {
            "complete": True, "core_sha256": core_hash,
            "results_sha256": digest(out / "results.json"),
            "validation_release_passed": True,
            "continuation_passed": results["continuation_passed"],
            "elapsed_seconds": time.monotonic() - started,
        })
        status(out, "complete", continuation_passed=results["continuation_passed"])
    except BaseException as error:
        atomic_json(out / "failure.json", {"error": repr(error), "time_utc": datetime.now(timezone.utc).isoformat()})
        status(out, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
