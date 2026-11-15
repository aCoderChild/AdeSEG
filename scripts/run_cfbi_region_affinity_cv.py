"""Train and evaluate CFBI regional matching adapters on fixed sequence folds."""

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.cfbi_region_affinity import RegionalAffinityHeads
from modeling.cfbi_region_affinity_training import cfbi_region_clip_loss
from modeling.native_cfbi_region_affinity import NativeCFBIRegionAffinity
from scripts.capture_presence_clip import capture_presence_clip
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_temporal_region_fusion_cv import SEQUENCES, summarize, verify_folds

PROTOCOL = ROOT / "protocols/cfbi_regional_affinity_polypgen_v2.json"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SOURCES = [
    "protocols/cfbi_regional_affinity_polypgen_v2.json",
    "scripts/run_cfbi_region_affinity_cv.py",
    "scripts/verify_cfbi_region_affinity.py",
    "modeling/cfbi_region_affinity.py",
    "modeling/native_cfbi_region_affinity.py",
    "modeling/cfbi_region_affinity_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/run_temporal_region_fusion_cv.py",
    "scripts/memory_token_io.py",
    "scripts/false_alarm_statistics.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "external/CFBI/networks/layers/matching.py",
]


def score_frame(logits, frames, frame):
    from datasets.io import load_label_mask

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


def checked_native_run(path, videos, checkpoint_sha, manifest_sha):
    setup = json.loads((path / "setup.json").read_text())
    receipt = json.loads((path / "completion.json").read_text())
    if not receipt.get("complete") or receipt.get("sequences") != len(SEQUENCES):
        raise RuntimeError("Archived native run is incomplete")
    if setup.get("checkpoint_sha256") != checkpoint_sha or setup.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Native baseline checkpoint or manifest differs")
    native, rows = {}, {}
    for name in SEQUENCES:
        frames_path = path / "native" / f"{name}_frames.json"
        target_receipt = json.loads((path / "native" / f"{name}_complete.json").read_text())
        if digest(frames_path) != target_receipt["frames_sha256"]:
            raise RuntimeError(f"Native baseline digest mismatch: {name}")
        sequence_rows = json.loads(frames_path.read_text())
        prompt = prompt_index(videos[name], {"initialization": "first_labeled_frame"})
        if target_receipt.get("prompt") != prompt:
            raise RuntimeError(f"Native prompt mismatch: {name}")
        if [row["frame"] for row in sequence_rows] != list(range(prompt + 1, len(videos[name]))):
            raise RuntimeError(f"Native baseline frame coverage mismatch: {name}")
        native[name] = summarize(sequence_rows)
        rows[name] = sequence_rows
    return native, rows


def run_regional_sequence(args, model, heads, frames, name, native_rows):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    rows, elapsed = [], 0.0
    with tempfile.TemporaryDirectory(prefix=f"cfbi_{name}_", dir=args.output_dir) as temp:
        state, actual_prompt = prepare(model, frames, Path(temp))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt-frame mismatch: {name}")
        with NativeCFBIRegionAffinity(model, heads, name) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs: {name}/{frame}")
                if frame == prompt:
                    continue
                if frame not in native_by_frame:
                    raise RuntimeError(f"Native baseline missing {name}/{frame}")
                calls = observer.calls[-4:]
                if [row["layer"] for row in calls] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Missing memory-attention layer: {name}/{frame}")
                if any(row["spatial_keys"] <= 0 or row["pointer_keys"] < 0
                       for row in calls):
                    raise RuntimeError(f"Invalid memory key accounting: {name}/{frame}")
                if any(not np.isfinite(row["evidence_mean"]).all() for row in calls):
                    raise RuntimeError(f"Nonfinite regional evidence: {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row.update(native_prediction_sha256=native_by_frame[frame]["prediction_sha256"],
                           attention_layers=calls)
                rows.append(row)
                write(args.output_dir / "progress.json", {
                    "phase": "regional_evaluation", "sequence": name, "frame": frame,
                })
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Regional trajectory incomplete: {name}")
    result = summarize(rows)
    present = [row for row in rows if row["present"]]
    result.update({key: float(np.mean([row["scores"][key] for row in present]))
                   for key in ("iou", "boundary_f")})
    return rows, result, elapsed


def report(result):
    lines = ["# CFBI regional nearest-neighbour evidence — PolypGen CV", "",
             "Exploratory sequence-grouped cross-validation; patient independence is unverified.", "",
             "| Sequence | Native Dice | Regional Dice | Δ Dice | Empty-GT FP: native → regional |",
             "|---|---:|---:|---:|---:|"]
    for name in SEQUENCES:
        native, regional = result["native"][name], result["regional"][name]
        lines.append(f"| {name} | {native['present_dice']:.6f} | {regional['present_dice']:.6f} | "
                     f"{regional['present_dice'] - native['present_dice']:+.6f} | "
                     f"{native['gt_empty_false_alarm_count']} → {regional['gt_empty_false_alarm_count']} |")
    lines += ["", f"Equal-sequence mean present Dice: {result['mean_native']:.6f} native, "
              f"{result['mean_regional']:.6f} regional (Δ {result['mean_delta']:+.6f}).",
              f"Sequence wins/losses/ties: {result['wins']}/{result['losses']}/{result['ties']}.",
              f"Sequence-bootstrap 95% interval: {result['delta_ci95']}.",
              f"Pooled empty-GT false alarms: {result['native_false_alarms']} native, "
              f"{result['regional_false_alarms']} regional.",
              f"Frozen meaningful-improvement screen passed: {result['meaningful_signal_screen']}.",
              "This PolypGen cohort has informed prior development and is not independent confirmation."]
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
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed") or gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("A passing gate for this checkpoint is required")
        if gate["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Engineering gate manifest differs")
        for path, expected in gate["sources"].items():
            if path in SOURCES and digest(ROOT / path) != expected:
                raise RuntimeError(f"Engineering-gate source changed: {path}")
        sources = {path: digest(ROOT / path) for path in SOURCES}
        repo, sam2_revision = activate_author_repo()
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native, native_rows = checked_native_run(args.native_run, videos, checkpoint_sha, manifest_sha)
        setup = {
            "protocol_sha256": digest(PROTOCOL), "protocol": protocol,
            "sources": sources, "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha, "sam2_revision": sam2_revision,
            "cfbi_revision": protocol["prior_method"]["revision"],
            "engineering_gate": str(args.gate), "native_run": str(args.native_run),
            "folds": folds, "patient_independence_verified": False,
        }
        write(args.output_dir / "setup.json", setup)
        write(args.output_dir / "progress.json", {"phase": "training", "fold": 1})
        regional, fold_summaries, training_records = {}, [], []
        global_step = 0
        total_steps = protocol["training"]["epochs"] * 14 * len(folds)

        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14:
                raise RuntimeError("Each fold must contain fourteen training sequences")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            model = build_author_predictor(args, repo)
            model.requires_grad_(False)
            heads = RegionalAffinityHeads(channels=model.memory_attention.d_model).train()
            optimizer = torch.optim.AdamW(
                heads.parameters(), lr=protocol["training"]["learning_rate"],
                weight_decay=protocol["training"]["weight_decay"],
            )
            fold_logs = []
            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = (protocol["training"]["seed"] + fold_index * 100000
                            + epoch * len(training) + sequence_index)
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    samples = capture_presence_clip(model, clip, args.output_dir)
                    optimizer.zero_grad(set_to_none=True)
                    loss, calls, _ = cfbi_region_clip_loss(
                        model, heads, samples, name, check_native=(epoch == 0 and sequence_index == 0)
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss in fold {fold_index}/{name}")
                    loss.backward()
                    gradients = []
                    for layer, head in enumerate(heads.layers):
                        grad = head.projection.weight.grad
                        if grad is None or not torch.isfinite(grad).all():
                            raise RuntimeError(f"Invalid projection gradient at layer {layer}")
                        if (epoch > 0 or sequence_index > 0) and grad.abs().sum() == 0:
                            raise RuntimeError(f"Zero regional gradient at layer {layer}")
                        gradients.append(float(grad.abs().sum()))
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise RuntimeError("Frozen SAM2 received parameter gradients")
                    torch.nn.utils.clip_grad_norm_(heads.parameters(), protocol["training"]["gradient_norm"])
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                        raise RuntimeError("Regional adapter contains nonfinite parameters")
                    row = {"fold": fold_index, "epoch": epoch, "sequence": name, "seed": seed,
                           "clip_frames": [frame.frame_index for frame in clip],
                           "loss": float(loss.detach()), "gradient_l1": gradients,
                           "attention_calls": len(calls)}
                    fold_logs.append(row)
                    training_records.append(row)
                    global_step += 1
                    write(fold_dir / "training.json", fold_logs)
                    write(args.output_dir / "progress.json", {
                        "phase": "training", "fold": fold_index, "step": global_step,
                        "total": total_steps, "sequence": name, "epoch": epoch,
                    })
                    del samples, loss, calls
                    gc.collect()

            heads_path = fold_dir / "heads.pt"
            save_tensor(heads.state_dict(), heads_path)
            write(fold_dir / "training_complete.json", {
                "updates": len(fold_logs), "checkpoint_sha256": digest(heads_path),
                "training_sequences": training, "validation_sequences": validation,
                "layers": len(heads.layers),
            })
            heads.eval()
            for name in validation:
                rows, summary, _ = run_regional_sequence(
                    args, model, heads, videos[name], name, native_rows[name]
                )
                regional[name] = summary
                write(fold_dir / f"{name}_regional_frames.json", rows)
                write(fold_dir / f"{name}_complete.json", {
                    "frames_sha256": digest(fold_dir / f"{name}_regional_frames.json"),
                    "native_sha256": digest(args.native_run / "native" / f"{name}_frames.json"),
                    "fold": fold_index,
                })
            fold_summaries.append({
                "fold": fold_index, "training_sequences": training,
                "validation_sequences": validation,
                "native_mean": float(np.mean([native[name]["present_dice"] for name in validation])),
                "regional_mean": float(np.mean([regional[name]["present_dice"] for name in validation])),
            })
            write(fold_dir / "completion.json", {"complete": True, "fold": fold_index,
                                                   "updates": len(fold_logs),
                                                   "validation_sequences": validation})
            write(args.output_dir / "progress.json", {"phase": "fold_complete", "fold": fold_index})
            del model, heads, optimizer
            gc.collect()

        delta = {name: regional[name]["present_dice"] - native[name]["present_dice"]
                 for name in SEQUENCES}
        mean_native = float(np.mean([native[name]["present_dice"] for name in SEQUENCES]))
        mean_regional = float(np.mean([regional[name]["present_dice"] for name in SEQUENCES]))
        native_fp = sum(native[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        regional_fp = sum(regional[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        wins, losses = sum(value > 0 for value in delta.values()), sum(value < 0 for value in delta.values())
        result = {
            "complete": True, "native": native, "regional": regional,
            "delta_by_sequence": delta, "mean_native": mean_native,
            "mean_regional": mean_regional, "mean_delta": mean_regional - mean_native,
            "wins": wins, "losses": losses, "ties": len(SEQUENCES) - wins - losses,
            "delta_ci95": sequence_bootstrap_interval(delta, seed=protocol["training"]["seed"]),
            "exploratory_wilcoxon": paired_wilcoxon(
                {name: regional[name]["present_dice"] for name in SEQUENCES},
                {name: native[name]["present_dice"] for name in SEQUENCES}),
            "native_false_alarms": native_fp, "regional_false_alarms": regional_fp,
            "folds": fold_summaries, "training_updates": len(training_records),
            "meaningful_signal_screen": (mean_regional - mean_native >= 0.01
                and wins >= 14 and regional_fp <= native_fp),
            "patient_independence_verified": False,
            "scope": "exploratory sequence-grouped PolypGen cross-validation; not independent confirmation",
        }
        if sources != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError("Source or protocol changed during run")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(report(result))
        write(args.output_dir / "completion.json", {
            "complete": True, "sequences": len(SEQUENCES), "folds": len(folds),
            "training_updates": len(training_records),
            "meaningful_signal_screen": result["meaningful_signal_screen"],
            "sources": sources,
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
