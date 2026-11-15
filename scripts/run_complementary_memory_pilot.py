"""Run the frozen, full-video complementary-memory PolypGen pilot."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
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
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import (
    holm_adjust,
    paired_wilcoxon,
    sequence_bootstrap_interval,
)
from modeling.complementary_memory import ComplementaryMemoryReader, NativeComplementaryRead
from modeling.complementary_memory.training import (
    capture_current_features,
    detach_tensors,
    reader_clip_loss,
)
from modeling.cfbi_training import clone_tensors
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity, tensor_hash

PROTOCOL_REL = "protocols/complementary_memory_pilot_v1.json"
MANIFEST_REL = "data/polypgen_sequence.jsonl"
TRAIN_SEQUENCES = [
    "seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9",
    "seq10", "seq11", "seq12", "seq13", "seq14", "seq15",
]
TEST_SEQUENCES = [f"seq{i}" for i in range(16, 24)]
SEEDS = (20261106, 20261107, 20261108)
MODES = ("generic", "regional")
SOURCE_REL = (
    PROTOCOL_REL,
    MANIFEST_REL,
    "scripts/run_complementary_memory_pilot.py",
    "scripts/verify_complementary_memory_pilot.py",
    "scripts/run_author_baseline.py",
    "scripts/memory_token_io.py",
    "scripts/bank_integrity.py",
    "scripts/false_alarm_statistics.py",
    "training/tracking.py",
    "modeling/complementary_memory/__init__.py",
    "modeling/complementary_memory/reader.py",
    "modeling/complementary_memory/integration.py",
    "modeling/complementary_memory/training.py",
    "modeling/native_memory_layout.py",
    "modeling/cutie_memory.py",
    "modeling/cutie_region_queries.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_bias.py",
    "modeling/region_definitions.py",
    "modeling/regional_refinement.py",
    "modeling/region_tokens.py",
    "evaluation/segmentation.py",
    "evaluation/statistics.py",
    "evaluation/davis2017/metrics.py",
    "datasets/io.py",
    "datasets/video.py",
    "tests/test_complementary_memory_training.py",
    "tests/test_complementary_memory.py",
    "tests/test_prediction_conditioned_audit.py",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/sam2/modeling/sam2_utils.py",
    "external/sam2/sam2/modeling/memory_attention.py",
    "external/sam2/training/loss_fns.py",
    "external/sam2/sam2/utils/transforms.py",
    "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "external/Cutie/cutie/model/transformer/object_summarizer.py",
    "external/Cutie/cutie/model/transformer/transformer_layers.py",
    "external/Cutie/cutie/model/transformer/object_transformer.py",
    "external/Cutie/cutie/model/transformer/positional_encoding.py",
    "external/Cutie/cutie/model/channel_attn.py",
    "external/Cutie/cutie/model/group_modules.py",
    "external/Cutie/cutie/utils/tensor_utils.py",
)


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def state_digest(state):
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        hasher.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def git_text(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_hashes():
    paths = [ROOT / rel for rel in SOURCE_REL]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Pilot source inventory is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def input_hashes(videos, sequences):
    records = {}
    for name in sequences:
        frames = videos[name]
        prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        entries = []
        for index in range(prompt, len(frames)):
            frame = frames[index]
            entries.append({
                "local_frame": index,
                "source_frame": frame.frame_index,
                "image_sha256": digest(frame.image_path),
                "mask_sha256": digest(frame.mask_path),
            })
        records[name] = {"prompt": prompt, "frames": entries}
    return records


def make_setup(args, videos):
    protocol_path = ROOT / PROTOCOL_REL
    protocol = json.loads(protocol_path.read_text())
    if digest(ROOT / MANIFEST_REL) != protocol["data"]["manifest_sha256_at_freeze"]:
        raise RuntimeError("PolypGen manifest hash differs from the frozen protocol")
    if TRAIN_SEQUENCES != protocol["data"]["training_sequences"]:
        raise RuntimeError("Training sequence order differs from the frozen protocol")
    if TEST_SEQUENCES != protocol["data"]["evaluation_sequences"]:
        raise RuntimeError("Evaluation sequence list differs from the frozen protocol")
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash does not match the frozen protocol")
    sources = source_hashes()
    repo, revision = activate_author_repo()
    if revision != protocol["model"]["revision"]:
        raise RuntimeError("SAM2 repository revision differs from protocol")
    cutie = ROOT / "external/Cutie"
    cutie_revision = subprocess.check_output(
        ["git", "-C", str(cutie), "rev-parse", "HEAD"], text=True
    ).strip()
    if cutie_revision != protocol["candidate"]["upstream_components"]["revision"]:
        raise RuntimeError("Cutie repository revision differs from protocol")
    input_records = input_hashes(videos, TRAIN_SEQUENCES + TEST_SEQUENCES)
    post_prompt_counts = {
        name: len(videos[name]) - input_records[name]["prompt"] - 1
        for name in TRAIN_SEQUENCES + TEST_SEQUENCES
    }
    train_frames = sum(post_prompt_counts[name] for name in TRAIN_SEQUENCES)
    test_frames = sum(post_prompt_counts[name] for name in TEST_SEQUENCES)
    if train_frames != protocol["data"]["training_post_prompt_frames"]:
        raise RuntimeError(f"Training target count changed: expected 1640, got {train_frames}")
    if test_frames != protocol["data"]["evaluation_post_prompt_frames"]:
        raise RuntimeError(f"Evaluation target count changed: expected 372, got {test_frames}")
    expected_updates = sum(math.ceil(post_prompt_counts[name] / 4) for name in TRAIN_SEQUENCES)
    if expected_updates != protocol["data"]["training_updates_per_arm_per_seed"]:
        raise RuntimeError(f"Training update count changed: expected 415, got {expected_updates}")
    input_sha = hashlib.sha256(
        json.dumps(input_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.manual_seed(SEEDS[0])
    init_reader = ComplementaryMemoryReader()
    if sum(parameter.numel() for parameter in init_reader.parameters()) != 560644:
        raise RuntimeError("Complementary reader parameter count differs from frozen protocol")
    reader_init_hash = state_digest(init_reader.state_dict())
    core = {
        "protocol_sha256": digest(protocol_path),
        "protocol_commit": git_text("log", "-1", "--format=%H", "--", PROTOCOL_REL),
        "repository_head": git_text("rev-parse", "HEAD"),
        "sources": sources,
        "sam2_revision": revision,
        "cutie_revision": cutie_revision,
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_hash},
        "manifest_sha256": digest(ROOT / MANIFEST_REL),
        "input_receipt_sha256": input_sha,
        "input_receipts": input_records,
        "training_sequences": TRAIN_SEQUENCES,
        "evaluation_sequences": TEST_SEQUENCES,
        "post_prompt_frames_by_sequence": post_prompt_counts,
        "training_post_prompt_frames": train_frames,
        "evaluation_post_prompt_frames": test_frames,
        "training_updates_per_arm": expected_updates,
        "seeds": list(SEEDS),
        "reader_initial_state_sha256_seed_20261106": reader_init_hash,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "device": "cpu",
    }
    core_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return protocol, core, core_hash, repo


def save_source_snapshot(out):
    path = out / "source_snapshot.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in SOURCE_REL:
            archive.write(ROOT / rel, rel)
    return digest(path)


def init_predictor(checkpoint, repo):
    args = argparse.Namespace(
        method="sam2",
        device="cpu",
        checkpoint=Path(checkpoint),
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    model = build_author_predictor(args, repo)
    if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
        raise RuntimeError("Native SAM2 memory settings differ from protocol")
    if model.fill_hole_area != 0:
        raise RuntimeError("Unexpected native mask postprocessing")
    return model


def activate_author_repo():
    repo, revision = source_identity("sam2")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo, revision


def prepare(model, frames, directory):
    """Reuse the project's standard first-GT-mask SAM2 prompt setup."""
    stage_frames(frames, directory)
    state = model.init_state(
        video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=False
    )
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    mask = torch.from_numpy(load_label_mask(frames[prompt].mask_path, [1]) == 1)
    model.add_new_mask(state, frame_idx=prompt, obj_id=1, mask=mask)
    return state, prompt


def target_for(frame):
    gt = load_label_mask(frame.mask_path, [1]) == 1
    target = torch.from_numpy(gt.astype(np.float32))[None, None]
    target = F.interpolate(target, size=(256, 256), mode="nearest")
    return target, gt


def train_video(model, reader, mode, optimizer, frames, name, output, progress, seed, arm,
                max_post_frames=None, auxiliary_factory=None):
    """Train one chronological video while carrying the arm's own memory bank."""
    with tempfile.TemporaryDirectory(prefix=f"train_{name}_", dir=output) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        with torch.inference_mode():
            model.propagate_in_video_preflight(state)
        with torch.inference_mode(False):
            bank = clone_tensors(state["output_dict_per_obj"][0])
        end_frame = len(frames) if max_post_frames is None else min(
            len(frames), prompt + 1 + max_post_frames
        )
        total = end_frame - prompt - 1
        if total <= 0:
            raise RuntimeError(f"No post-prompt training frames in {name}")
        frame_index = prompt + 1
        updates, frame_losses, reader_calls = [], [], []
        update_index = 0
        while frame_index < end_frame:
            stop = min(frame_index + 4, end_frame)
            samples = []
            for current in range(frame_index, stop):
                visual = capture_current_features(model, state, current)
                target, _ = target_for(frames[current])
                samples.append({
                    "frame": current,
                    "num_frames": len(frames),
                    "vision": visual["vision"],
                    "positions": visual["positions"],
                    "sizes": visual["sizes"],
                    "target": target,
                    "mask_inputs": None,
                    "point_inputs": None,
                })

            optimizer.zero_grad(set_to_none=True)
            auxiliary_records = []
            with NativeComplementaryRead(model, reader, mode) as observer:
                if auxiliary_factory is None:
                    loss, bank, writes = reader_clip_loss(model, samples, bank)
                else:
                    with auxiliary_factory(reader, observer) as auxiliary:
                        loss, bank, writes = reader_clip_loss(model, samples, bank, auxiliary)
                        auxiliary_records = auxiliary.records
            loss.backward()
            gradients = [p.grad for p in reader.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
                raise RuntimeError(f"Missing or nonfinite reader gradient in {name}")
            if not any(float(grad.abs().sum()) > 0 for grad in gradients):
                raise RuntimeError(f"Zero reader gradient in {name}")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen SAM2 received parameter gradients")
            gradient_groups = {}
            for group, module in (
                ("output_projection", reader.output_projection),
                ("summary_input_projection", reader.summarizer.input_proj),
                ("feature_pool", reader.summarizer.feature_pred),
                ("pool_weights", reader.summarizer.weights_pred),
                ("position_projection", reader.position_projection),
                ("reader_attention", reader.reader),
            ):
                group_grads = [p.grad for p in module.parameters() if p.grad is not None]
                gradient_groups[group] = float(sum(g.detach().abs().sum() for g in group_grads)) if group_grads else 0.0
            if len(observer.calls) != len(samples) or any(
                row["layers"] != [0, 1, 2, 3] for row in observer.calls
            ):
                raise RuntimeError("Training clip omitted a native attention layer/read")
            optimizer.step()
            bank = detach_tensors(bank)
            update_index += 1
            frame_losses.extend([float(loss.detach())] * len(samples))
            updates.append({
                "update": update_index,
                "frames": [frame_index, stop - 1],
                "loss": float(loss.detach()),
                "gradient_l1": float(sum(g.detach().abs().sum() for g in gradients)),
                "gradient_groups_l1": gradient_groups,
                "reads": observer.calls,
                "writes": writes,
                **({"auxiliary": auxiliary_records} if auxiliary_factory is not None else {}),
            })
            reader_calls.extend(observer.calls)
            frame_index = stop
            atomic_json(progress, {
                "phase": "training",
                "seed": seed,
                "arm": arm,
                "sequence": name,
                "completed_frames": frame_index - prompt - 1,
                "expected_frames": total,
                "completed_updates": update_index,
                **({"updated_at": datetime.now(timezone.utc).isoformat(),
                    "loss": float(loss.detach()), "auxiliary": auxiliary_records}
                   if auxiliary_factory is not None else {}),
            })
        with torch.inference_mode():
            model.reset_state(state)
    expected_updates = math.ceil(total / 4)
    if update_index != expected_updates or len(reader_calls) != total:
        raise RuntimeError(f"Incomplete autoregressive training coverage in {name}")
    return {
        "sequence": name,
        "prompt_frame": prompt,
        "post_prompt_frames": total,
        "optimizer_updates": update_index,
        "mean_chunk_loss": float(np.mean(frame_losses)),
        "reader_calls": len(reader_calls),
        "own_prediction_writes": total,
        "ground_truth_used_only_for_loss": True,
        "chunks": updates,
        "truncated_for_engineering_gate": max_post_frames is not None,
    }


def state_to_json_rows(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_arm(out, seed, mode, videos, checkpoint, repo, core_hash, resume):
    folder = out / "training" / f"seed_{seed}" / mode
    folder.mkdir(parents=True, exist_ok=True)
    resume_path = folder / "resume.pt"
    progress = out / "progress.json"
    model = init_predictor(checkpoint, repo)
    original_hash = parameter_digest(model)
    torch.manual_seed(seed)
    reader = ComplementaryMemoryReader().train()
    init_hash = state_digest(reader.state_dict())
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    sequence_rows = []
    start_index = 0
    if resume_path.exists():
        if not resume:
            raise FileExistsError(f"Training checkpoint exists; resume explicitly: {resume_path}")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["setup_core_sha256"] != core_hash or saved["seed"] != seed or saved["mode"] != mode:
            raise RuntimeError("Resume checkpoint provenance does not match this attempt")
        if saved["reader_initial_sha256"] != init_hash:
            raise RuntimeError("Resume reader initialization changed")
        reader.load_state_dict(saved["reader_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        sequence_rows = saved["completed_sequence_rows"]
        start_index = len(sequence_rows)
        if saved.get("next_sequence_index") != start_index:
            raise RuntimeError("Resume next-sequence index disagrees with its completed rows")
        receipt_path = folder / "completed_sequences.json"
        if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != sequence_rows:
            raise RuntimeError("Resume completed-sequence receipt is missing or differs")
        torch.random.set_rng_state(saved["torch_rng_state"])
        for index, row in enumerate(sequence_rows):
            expected_frames = len(videos[TRAIN_SEQUENCES[index]]) - prompt_index(
                videos[TRAIN_SEQUENCES[index]], {"initialization": "first_labeled_frame"}
            ) - 1
            if (row["sequence"] != TRAIN_SEQUENCES[index]
                    or row["post_prompt_frames"] != expected_frames
                    or row["optimizer_updates"] != math.ceil(expected_frames / 4)
                    or row["reader_calls"] != expected_frames
                    or row["own_prediction_writes"] != expected_frames):
                raise RuntimeError("Completed-sequence receipt is out of order or incomplete")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Nonempty arm directory without resumable receipt: {folder}")

    for index in range(start_index, len(TRAIN_SEQUENCES)):
        name = TRAIN_SEQUENCES[index]
        row = train_video(model, reader, mode, optimizer, videos[name], name, out,
                          progress, seed, mode)
        sequence_rows.append(row)
        atomic_json(folder / "completed_sequences.json", sequence_rows)
        atomic_torch_save(resume_path, {
            "setup_core_sha256": core_hash,
            "seed": seed,
            "mode": mode,
            "reader_initial_sha256": init_hash,
            "next_sequence_index": len(sequence_rows),
            "completed_sequence_rows": sequence_rows,
            "reader_state": reader.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.random.get_rng_state(),
        })

    if len(sequence_rows) != len(TRAIN_SEQUENCES):
        raise RuntimeError("Training completion receipt is missing sequences")
    if (sum(row["post_prompt_frames"] for row in sequence_rows) != 1640
            or sum(row["optimizer_updates"] for row in sequence_rows) != 415):
        raise RuntimeError("Full training frame/update totals differ from the frozen protocol")
    if parameter_digest(model) != original_hash:
        raise RuntimeError("Original SAM2 weights changed during training")
    weights_path = folder / "reader_final.pt"
    atomic_torch_save(weights_path, reader.state_dict())
    receipt = {
        "seed": seed,
        "mode": mode,
        "setup_core_sha256": core_hash,
        "reader_initial_sha256": init_hash,
        "reader_final_sha256": digest(weights_path),
        "reader_final_state_sha256": state_digest(reader.state_dict()),
        "original_sam2_parameter_sha256": original_hash,
        "completed_sequences": sequence_rows,
        "completed_frames": sum(row["post_prompt_frames"] for row in sequence_rows),
        "optimizer_updates": sum(row["optimizer_updates"] for row in sequence_rows),
        "complete": True,
    }
    atomic_json(folder / "training_receipt.json", receipt)
    del model
    gc.collect()
    return weights_path, receipt


def validate_training_receipt(receipt_path, weights_path, setup_core, seed, mode):
    if not receipt_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"Missing completed training artifact for {seed}/{mode}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "seed": seed,
        "mode": mode,
        "setup_core_sha256": hashlib.sha256(
            json.dumps(setup_core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "complete": True,
        "completed_frames": 1640,
        "optimizer_updates": 415,
        "reader_final_sha256": digest(weights_path),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Completed training receipt does not match frozen run: {receipt_path}")
    expected_rows = setup_core["post_prompt_frames_by_sequence"]
    rows = receipt.get("completed_sequences", [])
    if [row.get("sequence") for row in rows] != TRAIN_SEQUENCES:
        raise RuntimeError(f"Training sequence receipt is incomplete: {receipt_path}")
    for row in rows:
        count = expected_rows[row["sequence"]]
        if (row.get("post_prompt_frames") != count
                or row.get("reader_calls") != count
                or row.get("own_prediction_writes") != count
                or row.get("optimizer_updates") != math.ceil(count / 4)
                or row.get("ground_truth_used_only_for_loss") is not True):
            raise RuntimeError(f"Training sequence receipt mismatch: {receipt_path}/{row.get('sequence')}")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if state_digest(weights) != receipt.get("reader_final_state_sha256"):
        raise RuntimeError(f"Reader checkpoint differs from its training receipt: {weights_path}")
    if sum(value.numel() for value in weights.values()) != 560644:
        raise RuntimeError("Reader parameter count changed after training")
    return receipt


def evaluate_video(model, frames, reader, mode, directory, name, seed, setup_hash):
    model_hash = parameter_digest(model)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / name
    if (final / "summary.json").is_file():
        cached = json.loads((final / "summary.json").read_text())
        expected_reader = "native" if reader is None else state_digest(reader.state_dict())
        if cached.get("setup_core_sha256") == setup_hash and cached.get("reader_state_sha256") == expected_reader:
            validate_eval_artifacts(final, cached, setup_hash, expected_reader,
                                    len(frames) - cached["prompt_frame"] - 1)
            return cached
        raise RuntimeError(f"Existing evaluation output has incompatible provenance: {final}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial.", dir=directory))
    masks_dir = temporary / "masks"
    masks_dir.mkdir()
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"frames_{name}_", dir=temporary) as temp_frames:
            state, prompt = prepare(model, frames, Path(temp_frames))
            observer = NativeComplementaryRead(model, reader, mode, inference_state=state) if reader is not None else None
            start = time.monotonic()
            with torch.inference_mode():
                context = observer if observer is not None else _NullContext()
                with context:
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1] or logits.ndim != 4 or logits.shape[:2] != (1, 1):
                            raise RuntimeError("Unexpected object IDs or output rank")
                        if frame == prompt:
                            continue
                        if frame != prompt + len(rows) + 1:
                            raise RuntimeError("Missing or nonchronological target frame")
                        prediction = logits[0, 0].detach().cpu().numpy() > 0
                        truth = load_label_mask(frames[frame].mask_path, [1]) == 1
                        if prediction.shape != truth.shape:
                            raise RuntimeError("Prediction and GT output geometry differ")
                        score = calculate_scores(prediction, truth)
                        row = {
                            "frame": frame,
                            "source_frame_index": frames[frame].frame_index,
                            "prediction_sha256": tensor_hash(logits),
                            "present": bool(truth.any()),
                            "scores": {
                                **score,
                                "boundary_f": float(db_eval_boundary(prediction, truth)),
                            },
                            "absent_fp": bool(prediction.any()) if not truth.any() else None,
                        }
                        if observer is not None:
                            if len(observer.calls) != len(rows) + 1:
                                raise RuntimeError("Missing complementary reader trace")
                            row["read"] = observer.calls[-1]
                        mask_path = masks_dir / f"{frame:06d}.png"
                        Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
                        row["mask_sha256"] = digest(mask_path)
                        rows.append(row)
            elapsed = time.monotonic() - start
            with torch.inference_mode():
                model.reset_state(state)
        expected = len(frames) - prompt - 1
        if expected <= 0:
            raise RuntimeError(f"Evaluation sequence {name} has no post-prompt frames")
        if len(rows) != expected or (observer is not None and len(observer.calls) != expected):
            raise RuntimeError(f"Incomplete evaluation coverage for {name}")
        if parameter_digest(model) != model_hash:
            raise RuntimeError("Evaluation changed original SAM2 parameters")
        present = [row for row in rows if row["present"]]
        summary = {
            "sequence": name,
            "condition": "native" if reader is None else mode,
            "seed": seed,
            "prompt_frame": prompt,
            "target_frames": len(rows),
            "present_frames": len(present),
            "gt_empty_frames": len(rows) - len(present),
            "present_dice": float(np.mean([row["scores"]["dice"] for row in present])) if present else None,
            "all_frame_dice": float(np.mean([row["scores"]["dice"] for row in rows])) if rows else None,
            "present_iou": float(np.mean([row["scores"]["iou"] for row in present])) if present else None,
            "present_boundary_f": float(np.mean([row["scores"]["boundary_f"] for row in present])) if present else None,
            "gt_empty_false_alarm_count": sum(bool(row["absent_fp"]) for row in rows if not row["present"]),
            "gt_empty_false_alarm_rate": float(np.mean([bool(row["absent_fp"]) for row in rows if not row["present"]])) if any(not row["present"] for row in rows) else None,
            "propagation": propagation_summary(rows),
            "elapsed_seconds": elapsed,
            "fps": len(rows) / elapsed if elapsed else None,
            "reader_state_sha256": "native" if reader is None else state_digest(reader.state_dict()),
            "original_sam2_parameter_sha256": model_hash,
            "setup_core_sha256": setup_hash,
        }
        state_to_json_rows(temporary / "frames.jsonl", rows)
        atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, final)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ensure_eval(reader_path, mode, seed, videos, out, checkpoint, repo, setup_hash):
    if reader_path is None:
        arm_name = "native"
        reader = None
        reader_state_hash = "native"
    else:
        reader_state = torch.load(reader_path, map_location="cpu", weights_only=True)
        reader = ComplementaryMemoryReader()
        reader.load_state_dict(reader_state, strict=True)
        reader.eval()
        arm_name = mode
        reader_state_hash = state_digest(reader.state_dict())
    arm_dir = out / "evaluation" / ("native" if reader is None else f"seed_{seed}/{arm_name}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in TEST_SEQUENCES:
        target_dir = arm_dir / name
        if (target_dir / "summary.json").is_file():
            cached = json.loads((target_dir / "summary.json").read_text())
            if cached.get("setup_core_sha256") != setup_hash or cached.get("reader_state_sha256") != reader_state_hash:
                raise RuntimeError(f"Incompatible existing evaluation receipt {target_dir}")
            expected_targets = len(videos[name]) - cached["prompt_frame"] - 1
            validate_eval_artifacts(target_dir, cached, setup_hash, reader_state_hash,
                                    expected_targets)
            summaries[name] = cached
            continue
        model = init_predictor(checkpoint, repo)
        before = parameter_digest(model)
        summary = evaluate_video(model, videos[name], reader, mode, arm_dir, name, seed, setup_hash)
        if parameter_digest(model) != before:
            raise RuntimeError("Evaluation changed original SAM2 parameters")
        summaries[name] = summary
        del model
        gc.collect()
        atomic_json(out / "progress.json", {
            "phase": "evaluation",
            "seed": seed,
            "arm": arm_name,
            "sequence": name,
            "completed_sequences": len(summaries),
            "expected_sequences": len(TEST_SEQUENCES),
        })
    return summaries


def validate_eval_artifacts(directory, summary, setup_hash, reader_hash, expected_targets):
    frames_path = directory / "frames.jsonl"
    masks_dir = directory / "masks"
    if (summary.get("setup_core_sha256") != setup_hash
            or summary.get("reader_state_sha256") != reader_hash
            or summary.get("target_frames") != expected_targets
            or not frames_path.is_file() or not masks_dir.is_dir()):
        raise RuntimeError(f"Evaluation receipt is incomplete or incompatible: {directory}")
    rows = [json.loads(line) for line in frames_path.read_text().splitlines() if line]
    if len(rows) != expected_targets:
        raise RuntimeError(f"Evaluation frame receipt count differs: {directory}")
    first = summary["prompt_frame"] + 1
    for offset, row in enumerate(rows):
        frame = first + offset
        mask_path = masks_dir / f"{frame:06d}.png"
        if (row.get("frame") != frame or not mask_path.is_file()
                or row.get("mask_sha256") != digest(mask_path)):
            raise RuntimeError(f"Evaluation mask/receipt mismatch: {directory}/{frame}")
    if len(list(masks_dir.glob("*.png"))) != expected_targets:
        raise RuntimeError(f"Evaluation mask count differs: {directory}")


def aggregate(summaries):
    arms = sorted(summaries)
    if set(arms) != {"native", "generic", "regional"}:
        raise RuntimeError(f"Expected native and both matched reader arms, got {arms}")
    per_sequence = {}
    for name in TEST_SEQUENCES:
        if any(name not in summaries[arm] for arm in arms):
            raise RuntimeError(f"Missing evaluation sequence in an arm: {name}")
        for arm in arms:
            receipt = summaries[arm][name]
            if (receipt.get("target_frames", 0) <= 0
                    or receipt.get("present_frames", 0) <= 0
                    or receipt.get("present_dice") is None
                    or receipt.get("gt_empty_false_alarm_rate") is None):
                raise RuntimeError(f"Missing primary/false-alarm denominator for {arm}/{name}")
        parameter_hashes = {
            summaries[arm][name].get("original_sam2_parameter_sha256") for arm in arms
        }
        if len(parameter_hashes) != 1 or None in parameter_hashes:
            raise RuntimeError(f"Original SAM2 weights differ across arms for {name}")
        per_sequence[name] = {arm: summaries[arm][name] for arm in arms}
    result = {"per_sequence": per_sequence, "equal_sequence": {}}
    for arm in arms:
        dice = [per_sequence[name][arm]["present_dice"] for name in TEST_SEQUENCES]
        valid = [value for value in dice if value is not None]
        result["equal_sequence"][arm] = float(np.mean(valid)) if valid else None
    contrasts = {}
    pvalues = {}
    for control in ("native", "generic"):
        left = {name: per_sequence[name]["regional"]["present_dice"] for name in TEST_SEQUENCES}
        right = {name: per_sequence[name][control]["present_dice"] for name in TEST_SEQUENCES}
        delta = {name: left[name] - right[name] for name in TEST_SEQUENCES if left[name] is not None and right[name] is not None}
        test = paired_wilcoxon(left, right)
        pvalues[control] = test["p"]
        fa_deltas = []
        for name in TEST_SEQUENCES:
            region_fa = per_sequence[name]["regional"]["gt_empty_false_alarm_rate"]
            control_fa = per_sequence[name][control]["gt_empty_false_alarm_rate"]
            if region_fa is not None and control_fa is not None:
                fa_deltas.append(region_fa - control_fa)
        contrasts[f"regional_minus_{control}"] = {
            "n_sequences": len(delta),
            "mean_present_dice_difference": float(np.mean(list(delta.values()))) if delta else None,
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=20261106) if delta else None,
            "wins": sum(value > 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "losses": sum(value < 0 for value in delta.values()),
            "wilcoxon_p": test["p"],
            "mean_equal_sequence_false_alarm_difference": float(np.mean(fa_deltas)) if fa_deltas else None,
            "max_sequence_false_alarm_difference": float(max(fa_deltas)) if fa_deltas else None,
        }
    adjusted = holm_adjust(pvalues)
    for control in ("native", "generic"):
        contrasts[f"regional_minus_{control}"]["holm_adjusted_p"] = adjusted[control]
    result["primary_contrasts"] = contrasts
    return result


def seed_screen(summary):
    native = summary["primary_contrasts"]["regional_minus_native"]
    generic = summary["primary_contrasts"]["regional_minus_generic"]
    checks = {
        "dice_margin_vs_native": native["mean_present_dice_difference"] >= 0.020,
        "dice_margin_vs_generic": generic["mean_present_dice_difference"] >= 0.010,
        "ci_lower_vs_native_positive": native["sequence_bootstrap_95_ci"][0] > 0,
        "ci_lower_vs_generic_positive": generic["sequence_bootstrap_95_ci"][0] > 0,
        "false_alarm_limit_vs_native": (native["mean_equal_sequence_false_alarm_difference"] is not None and native["mean_equal_sequence_false_alarm_difference"] <= 0.05),
        "false_alarm_limit_vs_generic": (generic["mean_equal_sequence_false_alarm_difference"] is not None and generic["mean_equal_sequence_false_alarm_difference"] <= 0.05),
    }
    return {"passed": all(checks.values()), "checks": checks}


def all_seed_summary(seed_summaries):
    per_seed_per_sequence = {
        seed: summary["per_sequence"] for seed, summary in seed_summaries.items()
    }
    averaged = {"per_sequence": {}, "equal_sequence": {}}
    for name in TEST_SEQUENCES:
        averaged["per_sequence"][name] = {}
        for arm in ("native", "generic", "regional"):
            values = []
            for seed in seed_summaries:
                source_arm = "native" if arm == "native" else arm
                if source_arm == "native":
                    values.append(per_seed_per_sequence[seed][name]["native"])
                else:
                    values.append(per_seed_per_sequence[seed][name][source_arm])
            # Native is repeated in each seed summary only as a reference; all copies must agree.
            if arm == "native":
                native_hashes = {
                    value.get("original_sam2_parameter_sha256")
                    for value in values
                }
                native_dice = {value.get("present_dice") for value in values}
                if len(native_hashes) != 1 or len(native_dice) != 1:
                    raise RuntimeError(f"Native reference changed across seeds for {name}")
                averaged["per_sequence"][name][arm] = values[0]
            else:
                metrics = dict(values[0])
                for key in ("present_dice", "all_frame_dice", "present_iou", "present_boundary_f", "gt_empty_false_alarm_rate"):
                    items = [row[key] for row in values if row.get(key) is not None]
                    metrics[key] = float(np.mean(items)) if items else None
                counts = [row["gt_empty_false_alarm_count"] for row in values]
                denominators = {row["gt_empty_frames"] for row in values}
                if len(denominators) != 1:
                    raise RuntimeError(f"False-alarm denominator changed across seeds for {name}")
                metrics["gt_empty_false_alarm_count"] = float(np.mean(counts))
                metrics["gt_empty_false_alarm_counts_by_seed"] = {
                    str(seed): per_seed_per_sequence[seed][name][arm]["gt_empty_false_alarm_count"]
                    for seed in seed_summaries
                }
                averaged["per_sequence"][name][arm] = metrics
    for arm in ("native", "generic", "regional"):
        values = [averaged["per_sequence"][name][arm]["present_dice"] for name in TEST_SEQUENCES]
        averaged["equal_sequence"][arm] = float(np.mean([v for v in values if v is not None]))
    contrasts = {}
    pvalues = {}
    for control in ("native", "generic"):
        candidate = {name: averaged["per_sequence"][name]["regional"]["present_dice"] for name in TEST_SEQUENCES}
        baseline = {name: averaged["per_sequence"][name][control]["present_dice"] for name in TEST_SEQUENCES}
        delta = {name: candidate[name] - baseline[name] for name in TEST_SEQUENCES}
        test = paired_wilcoxon(candidate, baseline)
        pvalues[control] = test["p"]
        fa_deltas = [
            averaged["per_sequence"][name]["regional"]["gt_empty_false_alarm_rate"]
            - averaged["per_sequence"][name][control]["gt_empty_false_alarm_rate"]
            for name in TEST_SEQUENCES
            if averaged["per_sequence"][name]["regional"]["gt_empty_false_alarm_rate"] is not None
            and averaged["per_sequence"][name][control]["gt_empty_false_alarm_rate"] is not None
        ]
        seed_means = {}
        for seed, rows in per_seed_per_sequence.items():
            differences = [
                rows[name]["regional"]["present_dice"] - rows[name][control]["present_dice"]
                for name in TEST_SEQUENCES
            ]
            seed_means[str(seed)] = float(np.mean(differences))
        contrasts[f"regional_minus_{control}"] = {
            "mean_present_dice_difference": float(np.mean(list(delta.values()))),
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=20261106),
            "wins": sum(value > 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "losses": sum(value < 0 for value in delta.values()),
            "wilcoxon_p": test["p"],
            "mean_equal_sequence_false_alarm_difference": float(np.mean(fa_deltas)) if fa_deltas else None,
            "max_sequence_false_alarm_difference": float(max(fa_deltas)) if fa_deltas else None,
            "seed_mean_differences": seed_means,
        }
    adjusted = holm_adjust(pvalues)
    for control in ("native", "generic"):
        contrasts[f"regional_minus_{control}"]["holm_adjusted_p"] = adjusted[control]
    averaged["primary_contrasts"] = contrasts
    return averaged


def final_decision(summary):
    native = summary["primary_contrasts"]["regional_minus_native"]
    generic = summary["primary_contrasts"]["regional_minus_generic"]
    conditions = {
        "dice_margin_vs_native": native["mean_present_dice_difference"] >= 0.020,
        "dice_margin_vs_generic": generic["mean_present_dice_difference"] >= 0.010,
        "ci_lower_vs_native_positive": native["sequence_bootstrap_95_ci"][0] > 0,
        "ci_lower_vs_generic_positive": generic["sequence_bootstrap_95_ci"][0] > 0,
        "holm_p_vs_native_below_0_05": native["holm_adjusted_p"] < 0.05,
        "holm_p_vs_generic_below_0_05": generic["holm_adjusted_p"] < 0.05,
        "false_alarm_limit_vs_native": (native["mean_equal_sequence_false_alarm_difference"] is not None and native["mean_equal_sequence_false_alarm_difference"] <= 0.05),
        "false_alarm_limit_vs_generic": (generic["mean_equal_sequence_false_alarm_difference"] is not None and generic["mean_equal_sequence_false_alarm_difference"] <= 0.05),
        "positive_seed_mean_vs_native_in_at_least_two": sum(v > 0 for v in native["seed_mean_differences"].values()) >= 2,
        "positive_seed_mean_vs_generic_in_at_least_two": sum(v > 0 for v in generic["seed_mean_differences"].values()) >= 2,
    }
    return {"status": "positive_by_frozen_rule" if all(conditions.values()) else "not_positive_by_frozen_rule",
            "conditions": conditions, "passed": all(conditions.values())}


def write_results_md(out, summary, decision):
    lines = [
        "# Complementary regional memory reader: PolypGen pilot",
        "",
        "Exploratory result on previously examined PolypGen sequences. This is not independent confirmation.",
        "",
        "| Sequence | Native SAM2 | Generic reader | Regional reader | Regional − native | Regional − generic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in TEST_SEQUENCES:
        row = summary["per_sequence"][name]
        values = [row[arm]["present_dice"] for arm in ("native", "generic", "regional")]
        lines.append(f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[2]-values[0]:+.4f} | {values[2]-values[1]:+.4f} |")
    lines.extend(["", "| Aggregate | Native SAM2 | Generic reader | Regional reader |", "|---|---:|---:|---:|"])
    lines.append("| Equal-sequence present Dice | " + " | ".join(f"{summary['equal_sequence'][arm]:.4f}" for arm in ("native", "generic", "regional")) + " |")
    for name, row in summary["primary_contrasts"].items():
        lines.append(f"| {name} | delta {row['mean_present_dice_difference']:+.4f} | CI [{row['sequence_bootstrap_95_ci'][0]:+.4f}, {row['sequence_bootstrap_95_ci'][1]:+.4f}] | Holm p {row['holm_adjusted_p']:.4g} |")
    lines.extend(["", f"**Frozen-rule classification:** `{decision['status']}`.", "", "Per-sequence values are primary. Seed-level and secondary metrics are in `summary.json`; frame records and predicted masks are in `evaluation/`. A failure to meet the frozen rule is not repaired by tuning on these sequences.", ""])
    (out / "results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    out = validate_output(args.output_dir)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = ROOT / MANIFEST_REL
    videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
    protocol, core, core_hash, repo = make_setup(args, videos)
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError(f"Cannot resume missing attempt: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)

    setup_path = out / "setup.json"
    if args.resume:
        if not setup_path.is_file():
            raise RuntimeError("Resume has no setup receipt")
        saved_setup = json.loads(setup_path.read_text())
        if saved_setup.get("setup_core_sha256") != core_hash or saved_setup.get("setup_core") != core:
            raise RuntimeError("Attempt setup differs from current code/data/protocol")
        with (out / "attempt_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "resume", "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
    else:
        setup = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "output_dir": str(out),
            "setup_core_sha256": core_hash,
            "setup_core": core,
            "source_snapshot_sha256": save_source_snapshot(out),
        }
        atomic_json(setup_path, setup)
        atomic_json(out / "progress.json", {"phase": "initialized", "completed_units": 0})
    initial_parameter_hash = None
    seed_summaries = {}
    try:
        # The native reference is scored once and reused for each paired seed.
        native_results = ensure_eval(None, "native", SEEDS[0], videos,
                                     out, args.checkpoint, repo, core_hash)
        native_hashes = {
            native_results[name].get("original_sam2_parameter_sha256")
            for name in TEST_SEQUENCES
        }
        if len(native_hashes) != 1 or None in native_hashes:
            raise RuntimeError("Native evaluation used inconsistent SAM2 parameter states")
        initial_parameter_hash = next(iter(native_hashes))

        for seed_index, seed in enumerate(SEEDS):
            arm_results = {"native": native_results}
            for mode in MODES:
                reader_path = out / "training" / f"seed_{seed}" / mode / "reader_final.pt"
                receipt_path = reader_path.parent / "training_receipt.json"
                if reader_path.exists() and not receipt_path.exists():
                    raise RuntimeError(f"Final weights exist without a completed receipt: {reader_path}")
                if not reader_path.exists():
                    reader_path, _ = train_arm(
                        out, seed, mode, videos, args.checkpoint, repo, core_hash, args.resume
                    )
                receipt = validate_training_receipt(
                    receipt_path, reader_path, core, seed, mode
                )
                if receipt["original_sam2_parameter_sha256"] != initial_parameter_hash:
                    raise RuntimeError("Training and evaluation used different original SAM2 weights")
                reader_state = torch.load(reader_path, map_location="cpu", weights_only=True)
                reader = ComplementaryMemoryReader()
                reader.load_state_dict(reader_state, strict=True)
                reader.eval()
                arm_results[mode] = ensure_eval(reader_path, mode, seed, videos, out,
                                                args.checkpoint, repo, core_hash)
            summary = aggregate(arm_results)
            seed_summaries[str(seed)] = summary
            atomic_json(out / f"seed_{seed}_summary.json", summary)
            atomic_json(out / "progress.json", {
                "phase": "seed_complete",
                "seed": seed,
                "seed_index": seed_index + 1,
                "expected_seeds_max": len(SEEDS),
            })
            if seed_index == 0:
                screen = seed_screen(summary)
                atomic_json(out / "seed_1_continuation_screen.json", screen)
                if not screen["passed"]:
                    completion = {
                        "status": "stopped_after_seed_1_screen",
                        "seed": seed,
                        "screen": screen,
                        "efficacy_training_completed": True,
                        "additional_seeds_run": [],
                        "interpretation": "Candidate did not meet the prospectively frozen continuation rule; no result-driven tuning was performed.",
                    }
                    atomic_json(out / "completion.json", completion)
                    write_results_md(out, summary, {"status": "seed_1_screen_not_passed"})
                    atomic_json(out / "progress.json", {"phase": "complete", **completion})
                    return

        final = all_seed_summary(seed_summaries)
        decision = final_decision(final)
        final["per_seed"] = seed_summaries
        final["decision"] = decision
        atomic_json(out / "summary.json", final)
        write_results_md(out, final, decision)
        atomic_json(out / "completion.json", {
            "status": decision["status"],
            "decision": decision,
            "seeds_completed": list(seed_summaries),
            "sequences": TEST_SEQUENCES,
            "target_frames": sum(len(videos[name]) - prompt_index(videos[name], {"initialization": "first_labeled_frame"}) - 1 for name in TEST_SEQUENCES),
            "original_sam2_parameter_sha256": initial_parameter_hash,
            "setup_core_sha256": core_hash,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(out / "progress.json", {"phase": "complete", "status": decision["status"]})
    except BaseException as error:
        atomic_json(out / "failure.json", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "phase": json.loads((out / "progress.json").read_text()).get("phase") if (out / "progress.json").exists() else "unknown",
        })
        raise


if __name__ == "__main__":
    main()
