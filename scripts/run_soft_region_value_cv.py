"""Train and evaluate continuous region-role values with fixed sequence folds."""
import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.native_region_readout import NativeRegionalReadout
from modeling.paired_readout_training import clip_loss
from modeling.soft_region_roles import predicted_soft_regions
from modeling.soft_region_value_adapter import SoftRegionValueHeads
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.verify_on_demand_clip import capture_clip
from scripts.verify_soft_region_value import MODEL_CONFIG, SOURCES as GATE_SOURCES

PROTOCOL = ROOT / "protocols/soft_region_value_polypgen_v1.json"
SEQUENCES = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
BASE_SOURCES = [
    "protocols/soft_region_value_polypgen_v1.json",
    "scripts/run_soft_region_value_cv.py",
    *GATE_SOURCES[1:],
    "scripts/capture_presence_clip.py",
    "scripts/false_alarm_statistics.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "external/RMNet/models/rmnet.py",
    "external/RMNet/LICENSE",
]


def verify_folds(folds):
    flattened = [name for fold in folds for name in fold]
    if len(folds) != 3 or any(len(fold) != 7 for fold in folds):
        raise RuntimeError("Expected three fixed seven-sequence validation folds")
    if len(set(flattened)) != 21 or set(flattened) != set(SEQUENCES):
        raise RuntimeError("The folds must partition all promptable sequences exactly once")


def summarize(rows):
    summary = propagation_summary(rows)
    present = [row for row in rows if row["present"]]
    summary.update({key: float(np.mean([row["scores"][key] for row in present]))
                    for key in ("iou", "boundary_f")})
    return summary


def score_frame(logits, frames, frame):
    target = load_label_mask(frames[frame].mask_path, [1]) == 1
    prediction = logits[0, 0].detach().float().cpu().numpy() > 0
    return {
        "frame": frame,
        "present": bool(target.any()),
        "scores": {**calculate_scores(prediction, target),
                   "boundary_f": float(db_eval_boundary(prediction, target))},
        "absent_fp": bool(prediction.any()) if not target.any() else None,
        "prediction_sha256": tensor_hash(logits),
    }


def load_native_baseline(path, videos, checkpoint_sha, manifest_sha):
    setup = json.loads((path / "setup.json").read_text())
    completion = json.loads((path / "completion.json").read_text())
    if not completion.get("complete") or completion.get("sequences") != len(SEQUENCES):
        raise RuntimeError("Native SAM2 baseline is not complete for the cohort")
    if setup.get("checkpoint_sha256") != checkpoint_sha or setup.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Native baseline checkpoint or manifest differs")
    if setup.get("sam2_revision") != "2b90b9f5ceec907a1c18123530e92e794ad901a4":
        raise RuntimeError("Native baseline is not from the pinned SAM2 checkout")
    summaries, records = {}, {}
    for name in SEQUENCES:
        frames_path = path / "native" / f"{name}_frames.json"
        receipt_path = path / "native" / f"{name}_complete.json"
        receipt = json.loads(receipt_path.read_text())
        if digest(frames_path) != receipt["frames_sha256"]:
            raise RuntimeError(f"Native baseline digest mismatch: {name}")
        rows = json.loads(frames_path.read_text())
        prompt = prompt_index(videos[name], {"initialization": "first_labeled_frame"})
        if receipt.get("prompt") != prompt:
            raise RuntimeError(f"Native prompt mismatch: {name}")
        if [row["frame"] for row in rows] != list(range(prompt + 1, len(videos[name]))):
            raise RuntimeError(f"Native frame coverage mismatch: {name}")
        summaries[name], records[name] = summarize(rows), rows
    return summaries, records


def run_regional_sequence(model, heads, frames, name, native_rows, out):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    rows, elapsed = [], 0.0
    with tempfile.TemporaryDirectory(prefix=f"soft_{name}_", dir=out) as temp:
        state, actual_prompt = prepare(model, frames, Path(temp))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt mismatch: {name}")
        started = time.monotonic()
        with NativeRegionalReadout(
            model, heads, "regional_value", name,
            region_encoder=predicted_soft_regions,
        ) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs: {name}/{frame}")
                if frame == prompt:
                    continue
                native = native_by_frame.get(frame)
                if native is None:
                    raise RuntimeError(f"Native baseline missing {name}/{frame}")
                read = observer.observer.trace[-1]
                layers = observer.calls[-4:]
                if read["frame"] != frame or [row["layer"] for row in layers] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Incomplete four-layer regional read: {name}/{frame}")
                if any(row["spatial_keys"] <= 0 or row["pointer_keys"] < 0 for row in layers):
                    raise RuntimeError(f"Invalid key accounting: {name}/{frame}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite prediction: {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row.update(native_prediction_sha256=native["prediction_sha256"],
                           attention=read, attention_layers=layers)
                rows.append(row)
                write(out / "progress.json", {"phase": "validation", "sequence": name, "frame": frame})
        elapsed = time.monotonic() - started
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Regional trajectory incomplete: {name}")
    summary = summarize(rows)
    summary["inference_wall_seconds"] = elapsed
    summary["mean_spatial_memory_keys"] = float(np.mean([
        sum(call["spatial_keys"] for call in row["attention_layers"]) / 4 for row in rows]))
    summary["mean_pointer_tokens"] = float(np.mean([
        sum(call["pointer_keys"] for call in row["attention_layers"]) / 4 for row in rows]))
    return rows, summary


def report(result):
    lines = [
        "# Continuous region-role memory values — PolypGen sequence-grouped CV", "",
        "Exploratory only; this cohort informed method development and patient independence is unverified.", "",
        "| Sequence | Native Dice | Soft-role Dice | Delta | Native empty-GT FP | Soft-role empty-GT FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SEQUENCES:
        native, method = result["native"][name], result["soft_region_value"][name]
        lines.append(f"| {name} | {native['present_dice']:.6f} | {method['present_dice']:.6f} | "
                     f"{method['present_dice'] - native['present_dice']:+.6f} | "
                     f"{native['gt_empty_false_alarm_count']} | {method['gt_empty_false_alarm_count']} |")
    lines += [
        "", f"Equal-sequence mean present Dice: {result['mean_native']:.6f} native, "
        f"{result['mean_soft_region_value']:.6f} soft-role (delta {result['mean_delta']:+.6f}).",
        f"Sequence wins/losses/ties: {result['wins']}/{result['losses']}/{result['ties']}.",
        f"Sequence-bootstrap 95% interval: {result['delta_ci95']}.",
        f"Exploratory paired Wilcoxon p: {result['wilcoxon']['p']:.6g} (n={result['wilcoxon']['n']}).",
        f"Pooled empty-GT false alarms: {result['native_false_alarms']} native, "
        f"{result['soft_false_alarms']} soft-role.",
        f"Frozen meaningful-improvement screen passed: {result['meaningful_signal_screen']}.",
        "A passing screen is still exploratory and requires independent data and another prespecified seed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(PROTOCOL.read_text())
        folds = protocol["training"]["folds"]
        verify_folds(folds)
        manifest = ROOT / protocol["data"]["manifest"]
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed") or gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("A passing engineering gate for this checkpoint is required")
        if gate["manifest_sha256"] != manifest_sha or gate.get("model_config") != MODEL_CONFIG:
            raise RuntimeError("Engineering gate model/data identity differs")
        sources = {path: digest(ROOT / path) for path in BASE_SOURCES}
        for path, expected in gate["sources"].items():
            if path in sources and sources[path] != expected:
                raise RuntimeError(f"Engineering-gate source changed: {path}")
        rmnet_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "external/RMNet"), "rev-parse", "HEAD"], text=True).strip()
        if rmnet_revision != protocol["prior_work"]["RMNet"]["revision"]:
            raise RuntimeError("Pinned RMNet source changed")
        baseline = args.native_run
        repo, revision = activate_author_repo()
        if revision != gate["sam2_revision"]:
            raise RuntimeError("SAM2 checkout differs from the real-data gate")
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native, native_rows = load_native_baseline(baseline, videos, checkpoint_sha, manifest_sha)
        setup = {
            "protocol": protocol, "protocol_sha256": digest(PROTOCOL), "sources": sources,
            "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
            "sam2_revision": revision, "rmnet_revision": rmnet_revision,
            "model_config": MODEL_CONFIG, "gate": str(args.gate),
            "gate_completion_sha256": digest(args.gate / "completion.json"),
            "native_run": str(baseline), "native_setup_sha256": digest(baseline / "setup.json"),
            "folds": folds, "patient_independence_verified": False,
        }
        write(args.output_dir / "setup.json", setup)
        regional, fold_summaries, training_records = {}, [], []
        total_updates = protocol["training"]["epochs"] * 14 * len(folds)
        global_step = 0
        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Fold must train on exactly fourteen disjoint sequences")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            model = build_author_predictor(
                argparse.Namespace(method="sam2", device="cpu", model_config=MODEL_CONFIG,
                                   checkpoint=args.checkpoint), repo)
            model.requires_grad_(False)
            heads = SoftRegionValueHeads().train()
            optimizer = torch.optim.AdamW(
                heads.parameters(), lr=protocol["training"]["learning_rate"],
                weight_decay=protocol["training"]["weight_decay"])
            fold_logs, seen_layers = [], set()
            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = (protocol["training"]["seed"] + fold_index * 100000
                            + epoch * len(training) + sequence_index)
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    samples = capture_clip(model, clip, args.output_dir)
                    optimizer.zero_grad(set_to_none=True)
                    loss, calls = clip_loss(
                        model, heads, samples, "regional_value", name,
                        check_native=(epoch == 0 and sequence_index == 0),
                        return_trace=True, region_encoder=predicted_soft_regions)
                    if len(calls) != 4 * len(samples) or [row["layer"] for row in calls] != [0, 1, 2, 3] * len(samples):
                        raise RuntimeError(f"Training missed a layer in fold{fold_index}/{name}")
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite training loss: fold{fold_index}/{name}")
                    loss.backward()
                    gradient_l1 = []
                    for layer_index, head in enumerate(heads.layers):
                        grad = head.role_embeddings.grad
                        if grad is None or not torch.isfinite(grad).all():
                            raise RuntimeError(f"Invalid role gradient in layer {layer_index}")
                        norm = float(grad.abs().sum())
                        if norm <= 0:
                            raise RuntimeError(f"Zero role gradient in layer {layer_index}")
                        gradient_l1.append(norm)
                        seen_layers.add(layer_index)
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise RuntimeError("Frozen SAM2 acquired parameter gradients")
                    torch.nn.utils.clip_grad_norm_(
                        heads.parameters(), protocol["training"]["max_gradient_norm"])
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                        raise RuntimeError("Nonfinite role adapter parameter")
                    record = {"fold": fold_index, "epoch": epoch, "sequence": name,
                              "seed": seed, "frames": [frame.frame_index for frame in clip],
                              "loss": float(loss.detach()), "gradient_l1": gradient_l1,
                              "attention_calls": len(calls)}
                    fold_logs.append(record)
                    training_records.append(record)
                    global_step += 1
                    write(fold_dir / "training.json", fold_logs)
                    write(args.output_dir / "progress.json", {
                        "phase": "training", "fold": fold_index, "step": global_step,
                        "total": total_updates, "epoch": epoch, "sequence": name})
                    del samples, loss, calls
                    gc.collect()
            if seen_layers != {0, 1, 2, 3}:
                raise RuntimeError("Not all attention layers received training gradients")
            heads_path = fold_dir / "heads.pt"
            save_tensor(heads.state_dict(), heads_path)
            write(fold_dir / "training_complete.json", {
                "updates": len(fold_logs), "checkpoint_sha256": digest(heads_path),
                "training_sequences": training, "validation_sequences": validation,
                "layers_with_gradients": sorted(seen_layers)})
            heads.eval()
            for name in validation:
                rows, summary = run_regional_sequence(
                    model, heads, videos[name], name, native_rows[name], args.output_dir)
                regional[name] = summary
                write(fold_dir / f"{name}_regional_frames.json", rows)
                write(fold_dir / f"{name}_complete.json", {
                    "frames_sha256": digest(fold_dir / f"{name}_regional_frames.json"),
                    "native_sha256": digest(baseline / "native" / f"{name}_frames.json"),
                    "fold": fold_index})
            fold_summaries.append({
                "fold": fold_index, "training_sequences": training,
                "validation_sequences": validation,
                "native_mean": float(np.mean([native[name]["present_dice"] for name in validation])),
                "soft_region_mean": float(np.mean([regional[name]["present_dice"] for name in validation]))})
            write(fold_dir / "completion.json", {
                "complete": True, "fold": fold_index, "updates": len(fold_logs),
                "validation_sequences": validation})
            del model, heads, optimizer
            gc.collect()
            write(args.output_dir / "progress.json", {"phase": "fold_complete", "fold": fold_index})

        delta = {name: regional[name]["present_dice"] - native[name]["present_dice"]
                 for name in SEQUENCES}
        mean_native = float(np.mean([native[name]["present_dice"] for name in SEQUENCES]))
        mean_regional = float(np.mean([regional[name]["present_dice"] for name in SEQUENCES]))
        native_fp = sum(native[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        regional_fp = sum(regional[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        wins = sum(value > 0 for value in delta.values())
        losses = sum(value < 0 for value in delta.values())
        result = {
            "complete": True, "native": native, "soft_region_value": regional,
            "delta_by_sequence": delta, "mean_native": mean_native,
            "mean_soft_region_value": mean_regional, "mean_delta": mean_regional - mean_native,
            "wins": wins, "losses": losses, "ties": len(SEQUENCES) - wins - losses,
            "delta_ci95": sequence_bootstrap_interval(delta, seed=protocol["training"]["seed"]),
            "wilcoxon": paired_wilcoxon(
                {name: regional[name]["present_dice"] for name in SEQUENCES},
                {name: native[name]["present_dice"] for name in SEQUENCES}),
            "native_false_alarms": native_fp, "soft_false_alarms": regional_fp,
            "folds": fold_summaries, "training_updates": len(training_records),
            "meaningful_signal_screen": (
                mean_regional - mean_native >= 0.01 and wins >= 14 and regional_fp <= native_fp),
            "patient_independence_verified": False,
            "scope": protocol["data"]["scope"],
        }
        if sources != {path: digest(ROOT / path) for path in BASE_SOURCES}:
            raise RuntimeError("Source or protocol changed during the CV run")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(report(result))
        write(args.output_dir / "completion.json", {
            "complete": True, "sequences": len(SEQUENCES), "folds": len(folds),
            "training_updates": len(training_records),
            "meaningful_signal_screen": result["meaningful_signal_screen"],
            "sources": sources})
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
