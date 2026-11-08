"""Matched native / value-adapter / value-adapter+contrastive PolypGen pilot.

The implementation reuses the project's autoregressive SAM2 training path,
official SAM2 predictor, frozen MAVOS focal residual, and pinned SupContrast
loss. Artifacts are written to the caller-provided Drive output directory.
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
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval, holm_adjust
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.complementary_memory.training import (
    capture_current_features, clone_tensors, detach_tensors, reader_clip_loss,
)
from modeling.mavos_focal_value_modulation import MAVOSFocalValueResidual, MAVOSMemoryValueModulation
from modeling.mavos_value_contrastive import MAVOSValueContrastive
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.mavos_value_utils import (
    activate_author_repo, prepare, target_for, atomic_json, atomic_torch_save, state_digest,
)

PROTOCOL = ROOT / "protocols/mavos_focal_value_contrastive_polypgen_v1.json"
MANIFEST = ROOT / "data/polypgen_sequence.jsonl"
TRAIN = ["seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9", "seq10", "seq11", "seq12"]
VALIDATION = ["seq13", "seq14", "seq15"]
TEST = [f"seq{i}" for i in range(16, 24)]
ARMS = {"segmentation_only": 0.0, "segmentation_plus_contrastive": 0.1}
SOURCES = [
    "protocols/mavos_focal_value_contrastive_polypgen_v1.json",
    "protocols/mavos_focal_value_modulation_polypgen_v1.json",
    "scripts/run_mavos_value_contrastive.py",
    "scripts/verify_mavos_focal_value_modulation.py",
    "scripts/run_author_baseline.py",
    "scripts/mavos_value_utils.py",
    "scripts/memory_token_io.py",
    "scripts/false_alarm_statistics.py",
    "modeling/mavos_focal_value_modulation.py",
    "modeling/mavos_value_contrastive.py",
    "modeling/complementary_memory/contrastive.py",
    "modeling/complementary_memory/integration.py",
    "modeling/complementary_memory/training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "datasets/io.py",
    "datasets/video.py",
    "data/polypgen_sequence.jsonl",
    "tests/test_mavos_focal_value_modulation.py",
    "external/SupContrast/losses.py",
    "external/SupContrast/LICENSE",
    "external/MAVOS/networks/layers/externel_utils.py",
    "external/MAVOS/LICENSE",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/sam2/modeling/memory_attention.py",
]


def status(out, phase, **fields):
    atomic_json(out / "run_status.json", {
        "phase": phase, "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(), **fields,
    })
    print(json.dumps({"phase": phase, **fields}, default=str), flush=True)


def atomic_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))
    os.replace(temporary, path)


def source_receipt():
    return {name: digest(ROOT / name) for name in SOURCES}


def guard_sources(expected):
    for name, sha in expected.items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Source changed during run: {name}")


def make_adapters(model):
    result = []
    for layer in model.memory_attention.layers:
        attention = layer.cross_attn_image
        side = math.isqrt(attention.freqs_cis.shape[0])
        if side * side != attention.freqs_cis.shape[0]:
            raise RuntimeError("Native SAM2 positional grid is not square")
        result.append(MAVOSFocalValueResidual(
            attention.kv_in_dim, (side, side), focal_window=3,
            focal_level=2, focal_factor=2,
        ).to(device=attention.v_proj.weight.device, dtype=attention.v_proj.weight.dtype))
    return torch.nn.ModuleList(result)


def seed_adapters(model, seed):
    torch.manual_seed(seed)
    adapters = make_adapters(model)
    return adapters


def train_sequence(model, adapters, optimizer, frames, name, bank, start, out, arm, weight,
                   target_by_frame,
                   source_hashes, completed_updates, checkpoint_callback):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    index, updates = start, list(completed_updates)
    while index < len(frames):
        stop = min(index + 4, len(frames))
        samples = []
        for current in range(index, stop):
            visual = capture_current_features(model, model._run_state, current)
            target = target_by_frame[current]
            samples.append({
                "frame": current, "num_frames": len(frames),
                "vision": visual["vision"], "positions": visual["positions"],
                "sizes": visual["sizes"], "target": target,
                "mask_inputs": None, "point_inputs": None,
            })
        optimizer.zero_grad(set_to_none=True)
        observer = NativeComplementaryRead(model, None, "regional", trace_only=True)
        integration = MAVOSMemoryValueModulation(
            model, adapters, record_value_receipts=True, capture_adapted_values=True,
        )
        auxiliary = MAVOSValueContrastive(integration, observer, target_by_frame, weight=weight)
        with observer, integration:
            loss, bank, writes = reader_clip_loss(model, samples, bank, auxiliary=auxiliary)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss in {name}/{index}:{stop}")
        loss.backward()
        grad_l1 = []
        for layer, adapter in enumerate(adapters):
            grads = [p.grad for p in adapter.parameters() if p.grad is not None]
            value = float(sum(g.detach().abs().sum() for g in grads))
            if not grads or any(not torch.isfinite(g).all() for g in grads) or value <= 0:
                raise RuntimeError(f"Missing/nonfinite adapter gradient at layer {layer}")
            grad_l1.append(value)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen SAM2 received parameter gradients")
        if len(observer.calls) != len(samples) or any(r["layers"] != [0, 1, 2, 3] for r in observer.calls):
            raise RuntimeError("Training clip did not read all four native attention layers")
        if len(integration.value_calls) != 4 * len(samples):
            raise RuntimeError("Actual SAM2 value input missed one or more layers")
        if not all(r["pointer_tail_bit_identical"] for r in integration.value_calls):
            raise RuntimeError("A pointer token was changed")
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 3.0)
        optimizer.step()
        bank = detach_tensors(bank)
        row = {
            "sequence": name, "frames": [index, stop - 1], "loss": float(loss.detach()),
            "contrastive_weight": weight, "contrastive": auxiliary.records,
            "gradient_l1_by_layer": grad_l1, "memory_reads": observer.calls,
            "value_calls": integration.value_calls, "writes": writes,
            "ground_truth_loss_only": True,
        }
        updates.append(row)
        index = stop
        guard_sources(source_hashes)
        checkpoint_callback(index, bank, updates)
        atomic_json(out / "progress.json", {
            "phase": "training", "arm": arm, "sequence": name,
            "next_frame": index, "sequence_frames": len(frames),
            "optimizer_chunks_in_sequence": len(updates),
            "contrastive_active_layers": sum(x["contrastive_active"] for x in auxiliary.records),
        })
        guard_sources(source_hashes)
    return bank, updates


def evaluate_video(model, frames, adapters, name, out, setup_hash):
    model_hash = parameter_digest(model)
    directory = out / "evaluation" / name
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "frames.jsonl"
    summary_path = directory / "summary.json"
    if final.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("setup_core_sha256") != setup_hash or summary.get("adapter_sha256") != (
            "native" if adapters is None else state_digest(adapters.state_dict())
        ):
            raise RuntimeError(f"Existing evaluation has incompatible provenance: {directory}")
        rows = [json.loads(line) for line in final.read_text().splitlines() if line]
        expected = len(frames) - summary["prompt_frame"] - 1
        if len(rows) != expected or [r["frame"] for r in rows] != list(
            range(summary["prompt_frame"] + 1, len(frames))
        ):
            raise RuntimeError(f"Cached frame coverage is corrupt: {directory}")
        for row in rows:
            mask_path = directory / "masks" / f"{row['frame']:06d}.png"
            if not mask_path.is_file() or digest(mask_path) != row.get("mask_sha256"):
                raise RuntimeError(f"Cached prediction mask integrity failed: {mask_path}")
        return summary
    masks_dir = directory / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=directory) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        model._run_state = state
        observer = NativeComplementaryRead(model, None, "regional", inference_state=state, trace_only=True)
        integration = None if adapters is None else MAVOSMemoryValueModulation(model, adapters, record_value_receipts=True)
        iterator = None
        started = time.monotonic()
        try:
            with observer:
                context = integration if integration is not None else _Null()
                with context, torch.inference_mode():
                    iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                    for frame, ids, logits in iterator:
                        if ids != [1]:
                            raise RuntimeError(f"Unexpected object IDs on {name}/{frame}")
                        if frame == prompt:
                            continue
                        if not torch.isfinite(logits).all():
                            raise RuntimeError(f"Nonfinite mask logits on {name}/{frame}")
                        pred = logits[0, 0].detach().float().cpu().numpy() > 0
                        truth = load_label_mask(frames[frame].mask_path, [1]) == 1
                        row = {
                            "frame": frame, "present": bool(truth.any()),
                            "scores": {**calculate_scores(pred, truth),
                                       "boundary_f": float(db_eval_boundary(pred, truth))},
                            "absent_fp": bool(pred.any()) if not truth.any() else None,
                            "prediction_sha256": tensor_hash(logits),
                        }
                        mask_path = masks_dir / f"{frame:06d}.png"
                        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_path)
                        row["mask_sha256"] = digest(mask_path)
                        if adapters is not None:
                            if len(observer.calls) != len(rows) + 1:
                                raise RuntimeError("Missing native memory trace")
                            row["memory_read"] = observer.calls[-1]
                            row["value_calls"] = integration.value_calls[-4:]
                        rows.append(row)
        finally:
            if iterator is not None:
                iterator.close()
            with torch.inference_mode():
                model.reset_state(state)
            model._run_state = None
    if len(rows) != len(frames) - prompt - 1:
        raise RuntimeError(f"Incomplete evaluation coverage on {name}")
    if adapters is not None and len(observer.calls) != len(rows):
        raise RuntimeError(f"Memory read count differs from target count on {name}")
    if parameter_digest(model) != model_hash:
        raise RuntimeError("SAM2 parameters changed during evaluation")
    present = [r for r in rows if r["present"]]
    elapsed = time.monotonic() - started
    summary = {
        "sequence": name, "prompt_frame": prompt, "target_frames": len(rows),
        "present_frames": len(present), "gt_empty_frames": len(rows) - len(present),
        "present_dice": float(np.mean([r["scores"]["dice"] for r in present])) if present else None,
        "present_iou": float(np.mean([r["scores"]["iou"] for r in present])) if present else None,
        "present_boundary_f": float(np.mean([r["scores"]["boundary_f"] for r in present])) if present else None,
        "gt_empty_false_alarm_count": sum(bool(r["absent_fp"]) for r in rows if not r["present"]),
        "gt_empty_false_alarm_rate": float(np.mean(
            [bool(r["absent_fp"]) for r in rows if not r["present"]]
        )) if any(not r["present"] for r in rows) else None,
        "propagation": propagation_summary(rows),
        "elapsed_seconds": elapsed,
        "fps": len(rows) / elapsed if elapsed else None,
        "adapter_sha256": "native" if adapters is None else state_digest(adapters.state_dict()),
        "sam2_sha256": model_hash, "setup_core_sha256": setup_hash,
    }
    atomic_jsonl(final, rows)
    atomic_json(summary_path, summary)
    return summary


class _Null:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def summarize(groups, names, out, protocol, *, filename="results.json", report_name="results.md",
              apply_release_rule=False):
    means = {}
    for arm, seqs in groups.items():
        means[arm] = {
            metric: (float(np.mean([seqs[n][metric] for n in names if seqs[n][metric] is not None]))
                     if any(seqs[n][metric] is not None for n in names) else None)
            for metric in ("present_dice", "present_iou", "present_boundary_f", "gt_empty_false_alarm_rate")
        }
    contrasts = {}
    for learned in ("segmentation_only", "segmentation_plus_contrastive"):
        for control in (("native",) if learned == "segmentation_only" else ("native", "segmentation_only")):
            delta = {n: groups[learned][n]["present_dice"] - groups[control][n]["present_dice"] for n in names}
            contrasts[f"{learned}_minus_{control}"] = {
                "mean": float(np.mean(list(delta.values()))), "sequence_delta": delta,
                "wins": sum(v > 0 for v in delta.values()), "ties": sum(v == 0 for v in delta.values()),
                "losses": sum(v < 0 for v in delta.values()),
                "bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=protocol["training"]["seed"]),
                "wilcoxon": paired_wilcoxon(
                    {n: groups[learned][n]["present_dice"] for n in names},
                    {n: groups[control][n]["present_dice"] for n in names},
                ),
            }
    result = {"per_sequence": groups, "equal_sequence_means": means, "contrasts": contrasts}
    # Pre-registered validation-only release check; test split remains unopened on failure.
    c2 = means["segmentation_plus_contrastive"]
    c0, c1 = means["native"], means["segmentation_only"]
    release = protocol["evaluation"]["release_rule"]
    native_delta = contrasts["segmentation_plus_contrastive_minus_native"]
    seg_delta = contrasts["segmentation_plus_contrastive_minus_segmentation_only"]
    fa_values = [c2["gt_empty_false_alarm_rate"], c0["gt_empty_false_alarm_rate"],
                 c1["gt_empty_false_alarm_rate"]]
    fa_increase = (max(fa_values[0] - fa_values[1], fa_values[0] - fa_values[2])
                   if all(value is not None for value in fa_values) else None)
    passed = (fa_increase is not None and native_delta["mean"] >= .01
              and seg_delta["mean"] >= .01 and native_delta["wins"] >= 2
              and seg_delta["wins"] >= 2 and fa_increase <= .02)
    result["validation_release_screen"] = ({
        "passed": bool(passed), "false_alarm_increase_max": fa_increase,
        "thresholds": release,
        "interpretation": "screen only; not significance or external validation",
    } if apply_release_rule else None)
    result["holm_pvalues"] = holm_adjust({k: v["wilcoxon"]["p"] for k, v in contrasts.items()})
    atomic_json(out / filename, result)
    lines = ["# Memory-value foreground/background contrastive pilot", "",
             "Exploratory PolypGen study; one training pass and one seed. No claim of convergence or independent validation.", "",
             "| Condition | Present Dice | Present IoU | Boundary F | GT-empty false-alarm rate |",
             "|---|---:|---:|---:|---:|"]
    for arm, row in means.items():
        metrics = [row["present_dice"], row["present_iou"], row["present_boundary_f"],
                   row["gt_empty_false_alarm_rate"]]
        formatted = [f"{value:.6f}" if value is not None else "N/A" for value in metrics]
        lines.append(f"| {arm} | " + " | ".join(formatted) + " |")
    lines += ["", "| Sequence | Native | Segmentation-only adapter | Adapter + contrastive |", "|---|---:|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{groups[a][name]['present_dice']:.6f}" for a in ("native", "segmentation_only", "segmentation_plus_contrastive")) + " |")
    for label, row in contrasts.items():
        lines.append(f"\n{label}: {row['mean']:+.6f}, 95% sequence-bootstrap CI {row['bootstrap_95_ci']}, wins/ties/losses {row['wins']}/{row['ties']}/{row['losses']}.")
    if apply_release_rule:
        lines.append(f"\nValidation release screen: {'PASS' if passed else 'FAIL'}. Test split is {'eligible for exploratory evaluation' if passed else 'not run'}.")
    else:
        lines.append("\nValidation release screen is not applied to this test-set summary.")
    (out / report_name).write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    os.environ["TMPDIR"] = str(out)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(PROTOCOL.read_text())
        if not (args.gate / "completion.json").is_file():
            raise RuntimeError("Missing real-frame engineering gate completion receipt")
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed"):
            raise RuntimeError("Real-frame engineering gate did not pass")
        gate_results = json.loads((args.gate / "results.json").read_text())
        if gate.get("results_sha256") != digest(args.gate / "results.json"):
            raise RuntimeError("Engineering gate receipt does not match its result file")
        if (gate_results.get("protocol_sha256") != digest(PROTOCOL)
                or gate_results.get("checkpoint_sha256") != digest(args.checkpoint)
                or gate_results.get("manifest_sha256") != digest(MANIFEST)):
            raise RuntimeError("Engineering gate protocol/data/checkpoint differs")
        if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from frozen protocol")
        if digest(MANIFEST) != protocol["data"]["manifest_sha256_at_freeze"]:
            # Older parent protocol hashes are not silently accepted.
            raise RuntimeError("Manifest differs from frozen contrastive protocol")
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["sam2_revision"]:
            raise RuntimeError("SAM2 revision differs from protocol")
        videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}
        if set(TRAIN + VALIDATION + TEST) - set(videos) or set(TRAIN) & set(VALIDATION + TEST) or set(VALIDATION) & set(TEST):
            raise RuntimeError("Frozen data splits overlap or sequence is missing")
        post_prompt = {
            name: len(videos[name]) - prompt_index(
                videos[name], {"initialization": "first_labeled_frame"}
            ) - 1
            for name in TRAIN + VALIDATION + TEST
        }
        train_targets = sum(post_prompt[name] for name in TRAIN)
        expected_updates = sum(math.ceil(post_prompt[name] / 4) for name in TRAIN)
        if (train_targets != protocol["data"]["training_post_prompt_frames"]
                or expected_updates != protocol["data"]["training_updates_per_arm"]):
            raise RuntimeError(
                f"Frozen training coverage changed: {train_targets} targets/{expected_updates} updates"
            )
        sources = source_receipt()
        core = {
            "protocol_sha256": digest(PROTOCOL), "protocol_commit": os.popen(
                f"git -C {ROOT} log -1 --format=%H -- {PROTOCOL.relative_to(ROOT)}"
            ).read().strip(),
            "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(MANIFEST),
            "sam2_revision": revision, "sources": sources,
            "gate_results_sha256": digest(args.gate / "results.json"),
            "train_sequences": TRAIN, "validation_sequences": VALIDATION, "test_sequences": TEST,
            "seed": protocol["training"]["seed"], "device": "cpu",
        }
        core_hash = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        if not args.resume:
            if not core["protocol_commit"]:
                raise RuntimeError("Protocol must be committed before efficacy training")
            atomic_json(out / "setup.json", {"core": core, "core_sha256": core_hash,
                          "command": sys.argv, "created_at": datetime.now(timezone.utc).isoformat(),
                          "python": sys.version, "torch": torch.__version__})
            (out / "protocol.json").write_bytes(PROTOCOL.read_bytes())
            with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in SOURCES:
                    archive.write(ROOT / name, name)
            atomic_json(out / "source_hashes.json", sources)
        else:
            saved = json.loads((out / "setup.json").read_text())
            if saved["core_sha256"] != core_hash:
                raise RuntimeError("Resume protocol/source/data identity differs")

        model = build_author_predictor(argparse.Namespace(
            method="sam2", device="cpu", checkpoint=args.checkpoint,
            model_config=protocol["model"]["configuration"],
        ), repo)
        model.requires_grad_(False)
        model.eval()
        if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
            raise RuntimeError("Native SAM2 memory settings differ from protocol")
        original_model_hash = parameter_digest(model)
        setup_hash = core_hash

        groups = {"native": {}, "segmentation_only": {}, "segmentation_plus_contrastive": {}}
        # Native validation first, then both trained adapters, so the test split
        # cannot influence the conditional go/no-go decision.
        for name in VALIDATION:
            status(out, "validation_native", sequence=name)
            groups["native"][name] = evaluate_video(model, videos[name], None, name,
                                                        out / "native_validation", setup_hash)
        for arm, weight in ARMS.items():
            arm_dir = out / "training" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(protocol["training"]["seed"])
            adapters = seed_adapters(model, protocol["training"]["seed"])
            initial_hash = state_digest(adapters.state_dict())
            if arm == "segmentation_only":
                paired_initial_hash = initial_hash
            elif initial_hash != paired_initial_hash:
                raise RuntimeError("Trainable arms did not start from identical adapter weights")
            optimizer = torch.optim.AdamW(adapters.parameters(), lr=1e-4, weight_decay=1e-3,
                                          betas=(.9,.999), eps=1e-8)
            completed_sequences, start_sequence = [], 0
            checkpoint_path = arm_dir / "resume.pt"
            saved = None
            if args.resume and checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if saved["core_hash"] != core_hash or saved["arm"] != arm:
                    raise RuntimeError("Resume checkpoint provenance mismatch")
                adapters.load_state_dict(saved["adapters"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                torch.random.set_rng_state(saved["rng"])
                completed_sequences = saved["completed_sequences"]
                start_sequence = saved["next_sequence_index"]
            for sequence_index in range(start_sequence, len(TRAIN)):
                name = TRAIN[sequence_index]
                frames = videos[name]
                with tempfile.TemporaryDirectory(prefix=f"train_{name}_", dir=arm_dir) as temp:
                    state, prompt = prepare(model, frames, Path(temp))
                    with torch.inference_mode():
                        model.propagate_in_video_preflight(state)
                    bank = clone_tensors(state["output_dict_per_obj"][0])
                    model._run_state = state
                    # The loss reads labels for every stored causal frame,
                    # including frames from earlier TBPTT chunks. These masks
                    # are kept outside SAM2 state and used only after reads.
                    target_by_frame = {
                        frame_index: target_for(frame)[0]
                        for frame_index, frame in enumerate(frames)
                    }
                    start_frame = prompt + 1
                    if args.resume and checkpoint_path.exists():
                        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                        if saved["sequence"] == name:
                            bank = saved["bank"]
                            start_frame = saved["next_frame"]
                    chunk_dir = arm_dir / "chunks" / name
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    sequence_updates = []
                    for receipt_path in sorted(chunk_dir.glob("frame_*.json")):
                        row = json.loads(receipt_path.read_text())
                        if row["frames"][1] < start_frame:
                            sequence_updates.append(row)
                    expected_frame = prompt + 1
                    for row in sequence_updates:
                        if row["frames"][0] != expected_frame:
                            raise RuntimeError("Saved per-chunk receipts are not contiguous")
                        expected_frame = row["frames"][1] + 1
                    if expected_frame != start_frame:
                        raise RuntimeError("Saved receipts do not cover the resumable sequence prefix")
                    before_hash = parameter_digest(model)
                    def checkpoint_callback(next_frame, current_bank, rows):
                        last_row = rows[-1]
                        atomic_json(chunk_dir / f"frame_{last_row['frames'][0]:06d}.json", last_row)
                        atomic_torch_save(checkpoint_path, {
                            "core_hash": core_hash, "arm": arm, "sequence": name,
                            "next_sequence_index": sequence_index, "next_frame": next_frame,
                            "adapters": adapters.state_dict(), "optimizer": optimizer.state_dict(),
                            "bank": current_bank, "rng": torch.random.get_rng_state(),
                            "completed_sequences": completed_sequences,
                        })
                    bank, sequence_updates = train_sequence(
                        model, adapters, optimizer, frames, name, bank, start_frame,
                        out, arm, weight, target_by_frame, sources, sequence_updates,
                        checkpoint_callback,
                    )
                    with torch.inference_mode():
                        model.reset_state(state)
                    model._run_state = None
                    if parameter_digest(model) != before_hash:
                        raise RuntimeError("Frozen SAM2 changed during training")
                    receipt = {"sequence": name, "prompt_frame": prompt,
                               "post_prompt_frames": len(frames)-prompt-1,
                               "updates": sequence_updates,
                               "update_count": len(sequence_updates)}
                    atomic_json(arm_dir / f"{name}.json", receipt)
                    completed_sequences.append(name)
                    atomic_torch_save(checkpoint_path, {
                        "core_hash": core_hash, "arm": arm,
                        "next_sequence_index": sequence_index+1, "sequence": None,
                        "next_frame": None, "adapters": adapters.state_dict(),
                        "optimizer": optimizer.state_dict(), "rng": torch.random.get_rng_state(),
                        "completed_sequences": completed_sequences,
                    })
                    status(out, "training", arm=arm, completed_sequences=len(completed_sequences),
                           expected_sequences=len(TRAIN), last_sequence=name)
            if completed_sequences != TRAIN:
                raise RuntimeError("Training sequence coverage is incomplete")
            completed_updates = sum(
                json.loads((arm_dir / f"{name}.json").read_text())["update_count"]
                for name in TRAIN
            )
            expected_updates = protocol["data"]["training_updates_per_arm"]
            if completed_updates != expected_updates:
                raise RuntimeError(f"Expected {expected_updates} optimizer updates, got {completed_updates}")
            if parameter_digest(model) != original_model_hash:
                raise RuntimeError("SAM2 checkpoint changed during training")
            active_contrastive_calls = sum(
                row["contrastive_active"]
                for name in TRAIN
                for update in json.loads((arm_dir / f"{name}.json").read_text())["updates"]
                for row in update["contrastive"]
            )
            if weight > 0 and active_contrastive_calls == 0:
                raise RuntimeError("Contrastive arm had no valid foreground/background positive pairs")
            weights = arm_dir / "adapters.pt"
            atomic_torch_save(weights, adapters.state_dict())
            atomic_json(arm_dir / "completion.json", {
                "complete": True, "sequences": completed_sequences,
                "initial_adapter_sha256": initial_hash,
                "adapter_sha256": digest(weights), "adapter_state_sha256": state_digest(adapters.state_dict()),
                "sam2_sha256": original_model_hash, "optimizer_updates": completed_updates,
                "active_contrastive_layer_calls": active_contrastive_calls,
                "core_hash": core_hash,
            })
            # Validation uses a fresh predictor state per sequence and the
            # trained adapters; the model weights remain the frozen checkpoint.
            for name in VALIDATION:
                status(out, "validation", arm=arm, sequence=name)
                groups[arm][name] = evaluate_video(model, videos[name], adapters, name,
                                                   out / f"validation_{arm}", setup_hash)
            del optimizer, adapters
            gc.collect()

        if parameter_digest(model) != original_model_hash:
            raise RuntimeError("SAM2 weights changed before validation decision")
        validation = summarize(
            groups, VALIDATION, out, protocol,
            filename="validation_results.json", report_name="validation_results.md",
            apply_release_rule=True,
        )
        if not validation["validation_release_screen"]["passed"]:
            atomic_json(out / "completion.json", {
                "complete": True, "test_run": False, "reason": "validation release screen failed",
                "validation_results_sha256": digest(out / "validation_results.json"),
                "interpretation": "No PolypGen test inference performed.",
            })
            status(out, "complete_validation_screen_failed")
            return
        # Open test only after the validation screen passes.
        groups = {"native": {}, "segmentation_only": {}, "segmentation_plus_contrastive": {}}
        for arm in groups:
            adapter_state = None
            if arm != "native":
                adapter_state = torch.load(out / "training" / arm / "adapters.pt",
                                           map_location="cpu", weights_only=True)
            adapters = None if adapter_state is None else make_adapters(model)
            if adapters is not None:
                adapters.load_state_dict(adapter_state, strict=True)
                adapters.eval()
            for name in TEST:
                status(out, "test_evaluation", arm=arm, sequence=name)
                groups[arm][name] = evaluate_video(model, videos[name], adapters, name,
                                                    out / f"test_{arm}", setup_hash)
            del adapters
            gc.collect()
        result = summarize(groups, TEST, out, protocol)
        atomic_json(out / "completion.json", {"complete": True, "test_run": True,
                    "results_sha256": digest(out / "results.json"),
                    "validation_results_sha256": digest(out / "validation_results.json")})
        status(out, "complete", validation_gate_passed=True)
    except BaseException as error:
        atomic_json(out / "failure.json", {"error": repr(error)})
        status(out, "failed", error=repr(error))
        raise
    finally:
        if "model" in locals():
            del model
            gc.collect()


if __name__ == "__main__":
    main()
