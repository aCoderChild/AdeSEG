"""Train and evaluate uncertainty-gated regional memory fusion by sequence folds."""

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
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.native_temporal_region_fusion import NativeTemporalRegionFusion, TemporalRegionHeads
from modeling.temporal_region_training import temporal_region_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write

PROTOCOL = ROOT / "protocols/temporal_region_fusion_polypgen_v1.json"
GATE = Path.home() / "Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/94_temporal_region_fusion_engineering_gate_v1_retry4"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SEQUENCES = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
SOURCES = [
    "protocols/temporal_region_fusion_polypgen_v1.json",
    "scripts/run_temporal_region_fusion_cv.py",
    "scripts/verify_temporal_region_fusion.py",
    "scripts/capture_presence_clip.py",
    "modeling/matanyone_temporal_uncertainty.py",
    "modeling/temporal_region_fusion.py",
    "modeling/native_temporal_region_fusion.py",
    "modeling/temporal_region_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_readout.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "scripts/false_alarm_statistics.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "external/MatAnyone/matanyone/model/big_modules.py",
]


def verify_folds(folds):
    flattened = [name for fold in folds for name in fold]
    if len(folds) != 3 or any(len(fold) != 7 for fold in folds):
        raise RuntimeError("Expected three fixed seven-sequence folds")
    if len(set(flattened)) != 21 or set(flattened) != set(SEQUENCES):
        raise RuntimeError("Fold groups must partition all 21 promptable sequences")


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


def summarize(rows):
    summary = propagation_summary(rows)
    present = [row for row in rows if row["present"]]
    summary.update({key: float(np.mean([row["scores"][key] for row in present]))
                    for key in ("iou", "boundary_f")})
    return summary


def run_native_sequence(args, model, frames, name, out):
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"native_{name}_", dir=out) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        expected_prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        if prompt != expected_prompt:
            raise RuntimeError(f"Prompt-frame mismatch: {name}")
        with torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs in native {name}/{frame}")
                if frame != prompt:
                    rows.append(score_frame(logits, frames, frame))
                    write(out / "progress.json", {"phase": "native", "sequence": name,
                                                    "frame": frame})
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Native baseline coverage incomplete: {name}")
    return prompt, rows, summarize(rows)


def run_regional_sequence(args, model, heads, frames, name, native_rows, out):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"regional_{name}_", dir=out) as temp:
        state, actual_prompt = prepare(model, frames, Path(temp))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt-frame mismatch: {name}")
        with NativeTemporalRegionFusion(model, heads, name) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs in regional {name}/{frame}")
                if frame == prompt:
                    continue
                if frame not in native_by_frame:
                    raise RuntimeError(f"Native reference missing {name}/{frame}")
                trace = observer.observer.trace[-1]
                layers = observer.calls[-4:]
                if trace["frame"] != frame or trace["layers"] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Incomplete native memory layer trace: {name}/{frame}")
                if [row["layer"] for row in layers] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Regional attention trace missing a layer: {name}/{frame}")
                if any(row["spatial_keys"] <= 0 or row["pointer_keys"] < 0 for row in layers):
                    raise RuntimeError(f"Invalid actual key accounting: {name}/{frame}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite prediction: {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row.update(native_prediction_sha256=native_by_frame[frame]["prediction_sha256"],
                           attention=trace, attention_layers=layers)
                rows.append(row)
                write(out / "progress.json", {"phase": "regional_evaluation",
                    "sequence": name, "frame": frame})
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Regional trajectory incomplete: {name}")
    return rows, summarize(rows)


def report(results, names, signal):
    lines = ["# Temporal-uncertainty-gated regional memory residual — PolypGen CV", "",
             "Exploratory sequence-grouped cross-validation; no shuffled arm; patient independence is unverified.", "",
             "| Sequence | Native Dice | Regional Dice | Delta | Native empty-frame FP | Regional empty-frame FP |", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        native = results["native"][name]
        region = results["regional"][name]
        lines.append(f"| {name} | {native['present_dice']:.6f} | {region['present_dice']:.6f} | "
                     f"{region['present_dice']-native['present_dice']:+.6f} | "
                     f"{native['gt_empty_false_alarm_count']} | {region['gt_empty_false_alarm_count']} |")
    lines += ["", f"Equal-sequence native mean Dice: {results['mean_native']:.6f}.",
              f"Equal-sequence regional mean Dice: {results['mean_regional']:.6f}.",
              f"Mean paired delta: {results['mean_delta']:+.6f}; sequence wins: {results['wins']}/{len(names)}.",
              f"Sequence-bootstrap 95% interval: {results['delta_ci95']}.",
              f"Pooled empty-GT false alarms: {results['native_false_alarms']} native, {results['regional_false_alarms']} regional.",
              f"Pre-set exploratory signal screen passed: {signal}.",
              "PolypGen has informed prior method development, so this result is not independent confirmation."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                         help="Continue a failed run after verifying saved native and fold outputs")
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    if args.resume:
        if not args.output_dir.is_dir():
            raise RuntimeError("Resume output directory does not exist")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(PROTOCOL.read_text())
        folds = protocol["evaluation_plan"]["folds"]
        verify_folds(folds)
        gate = json.loads((args.gate / "completion.json").read_text())
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        if not gate.get("passed") or gate["checkpoint_sha256"] != digest(args.checkpoint):
            raise RuntimeError("A passing real-data gate for this checkpoint is required")
        if gate["manifest_sha256"] != digest(manifest):
            raise RuntimeError("Engineering gate manifest differs")
        for path, expected in gate["sources"].items():
            if digest(ROOT / path) != expected:
                raise RuntimeError(f"Engineering-gate source changed: {path}")
        sources = {path: digest(ROOT / path) for path in SOURCES}
        repo, revision = activate_author_repo()
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        for name in SEQUENCES:
            if name not in videos:
                raise RuntimeError(f"Manifest lacks {name}")
        setup = {
            "protocol": protocol, "protocol_sha256": digest(PROTOCOL),
            "sources": sources, "checkpoint_sha256": digest(args.checkpoint),
            "manifest_sha256": digest(manifest), "sam2_revision": revision,
            "matanyone_revision": protocol["prior_implementation"]["revision"],
            "gate": str(args.gate), "gate_sha256": digest(args.gate / "completion.json"),
            "folds": folds, "patient_independence_verified": False,
        }
        setup_path = args.output_dir / "setup.json"
        native_dir = args.output_dir / "native"
        native_summaries = {}
        regional_summaries = {}
        fold_summaries = []
        training_records = []
        total_updates = protocol["training"]["epochs"] * 14 * len(folds)
        global_step = 0
        first_fold = 1
        if args.resume:
            if not setup_path.is_file() or not (args.output_dir / "failure.json").is_file():
                raise RuntimeError("Resume requires an existing setup and recorded failed attempt")
            old_setup = json.loads(setup_path.read_text())
            for key in ("protocol_sha256", "checkpoint_sha256", "manifest_sha256", "folds"):
                if old_setup.get(key) != setup.get(key):
                    raise RuntimeError(f"Resume {key} differs from original run")
            for path, expected in old_setup["sources"].items():
                if path != "scripts/run_temporal_region_fusion_cv.py" and sources.get(path) != expected:
                    raise RuntimeError(f"Resume source changed: {path}")
            if set(old_setup["sources"]) != set(sources):
                raise RuntimeError("Resume source list differs from original run")
            continuation_path = args.output_dir / "continuation.json"
            if continuation_path.exists():
                raise RuntimeError("This run already has a continuation record")
            fold1 = args.output_dir / "fold1"
            fold1_done = json.loads((fold1 / "completion.json").read_text())
            if fold1_done.get("validation_sequences") != folds[0]:
                raise RuntimeError("Completed fold 1 does not match the frozen split")
            for name in SEQUENCES:
                frames_path = native_dir / f"{name}_frames.json"
                receipt = json.loads((native_dir / f"{name}_complete.json").read_text())
                if digest(frames_path) != receipt["frames_sha256"]:
                    raise RuntimeError(f"Saved native baseline digest mismatch: {name}")
                if receipt.get("prompt") != prompt_index(videos[name], {"initialization": "first_labeled_frame"}):
                    raise RuntimeError(f"Saved native prompt mismatch: {name}")
                rows = json.loads(frames_path.read_text())
                if [row["frame"] for row in rows] != list(range(receipt["prompt"] + 1, len(videos[name]))):
                    raise RuntimeError(f"Saved native baseline coverage incomplete: {name}")
                native_summaries[name] = summarize(rows)
            for name in folds[0]:
                regional_path = fold1 / f"{name}_regional_frames.json"
                receipt = json.loads((fold1 / f"{name}_complete.json").read_text())
                if receipt.get("fold") != 1 or digest(regional_path) != receipt["frames_sha256"]:
                    raise RuntimeError(f"Saved fold 1 regional result digest mismatch: {name}")
                if receipt.get("native_sha256") != digest(native_dir / f"{name}_frames.json"):
                    raise RuntimeError(f"Saved native/regional pairing mismatch: {name}")
                regional_rows = json.loads(regional_path.read_text())
                native_rows = json.loads((native_dir / f"{name}_frames.json").read_text())
                if len(regional_rows) != len(native_rows) or any(
                    regional["native_prediction_sha256"] != native["prediction_sha256"]
                    for regional, native in zip(regional_rows, native_rows)
                ):
                    raise RuntimeError(f"Saved regional replay does not match native: {name}")
                regional_summaries[name] = summarize(regional_rows)
            fold_summaries.append({"fold": 1,
                                   "training_sequences": [n for n in SEQUENCES if n not in folds[0]],
                                   "validation_sequences": folds[0],
                                   "native_mean": float(np.mean([native_summaries[n]["present_dice"] for n in folds[0]])),
                                   "regional_mean": float(np.mean([regional_summaries[n]["present_dice"] for n in folds[0]]))})
            training_records = json.loads((fold1 / "training.json").read_text())
            if len(training_records) != fold1_done["updates"]:
                raise RuntimeError("Fold 1 training log length does not match its completion receipt")
            global_step = len(training_records)
            fold2 = args.output_dir / "fold2"
            if fold2.exists() and any(fold2.iterdir()):
                raise RuntimeError("Incomplete fold 2 has artifacts; refusing an unsafe resume")
            failure_path = args.output_dir / "failure.json"
            failure_record = json.loads(failure_path.read_text())
            failure_path.rename(args.output_dir / "failure_attempt1.json")
            write(continuation_path, {
                "continued": True, "reason": failure_record["error"],
                "prior_runner_sha256": old_setup["sources"]["scripts/run_temporal_region_fusion_cv.py"],
                "continuation_runner_sha256": sources["scripts/run_temporal_region_fusion_cv.py"],
                "new_gate": str(args.gate), "new_gate_sha256": digest(args.gate / "completion.json"),
                "reused_native_sequences": SEQUENCES, "reused_completed_folds": [1],
                "resumed_at_fold": 2,
            })
            first_fold = 2
        else:
            write(setup_path, setup)
            native_dir.mkdir()
            args.output_dir.joinpath("progress.json").write_text(json.dumps({"phase": "native_baseline"}))
            model = build_author_predictor(args, repo)
            model.requires_grad_(False)
            for name in SEQUENCES:
                prompt, rows, summary = run_native_sequence(args, model, videos[name], name, args.output_dir)
                native_summaries[name] = summary
                write(native_dir / f"{name}_frames.json", rows)
                write(native_dir / f"{name}_complete.json", {"prompt": prompt,
                     "frames_sha256": digest(native_dir / f"{name}_frames.json")})
            del model
            gc.collect()

        for fold_index, validation in enumerate(folds, start=1):
            if fold_index < first_fold:
                continue
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Invalid fold training/evaluation split")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir(exist_ok=bool(args.resume and fold_index == first_fold))
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            model = build_author_predictor(args, repo)
            model.requires_grad_(False)
            channels = model.memory_attention.d_model
            if channels != 256:
                raise RuntimeError(f"Unexpected attention width: {channels}")
            heads = TemporalRegionHeads(channels=channels).train()
            optimizer = torch.optim.AdamW(
                heads.parameters(), lr=protocol["training"]["learning_rate"],
                weight_decay=protocol["training"]["weight_decay"])
            fold_records = []
            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = protocol["training"]["seed"] + fold_index * 100000 + epoch * len(training) + sequence_index
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    samples = capture_presence_clip(model, clip, args.output_dir)
                    optimizer.zero_grad(set_to_none=True)
                    loss, calls = temporal_region_clip_loss(
                        model, heads, samples, name,
                        check_native=(global_step == 0), return_trace=True)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss in fold{fold_index}/{name}")
                    loss.backward()
                    projection_grads, uncertainty_grads = [], []
                    for layer, head in enumerate(heads.layers):
                        grad = head.regional_residual.weight.grad
                        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                            raise RuntimeError(f"Missing regional residual gradient at layer {layer}")
                        projection_grads.append(float(grad.abs().sum()))
                        uncertainty_grads.append(sum(float(p.grad.abs().sum()) for p in head.uncertainty.parameters()
                                                      if p.grad is not None))
                    if (epoch > 0 or sequence_index > 0) and not all(value > 0 for value in uncertainty_grads):
                        raise RuntimeError("Temporal uncertainty predictor received no gradient")
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise RuntimeError("Frozen SAM2 received parameter gradients")
                    torch.nn.utils.clip_grad_norm_(heads.parameters(), protocol["training"]["max_gradient_norm"])
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                        raise RuntimeError("Nonfinite regional adapter parameters")
                    row = {"fold": fold_index, "epoch": epoch, "sequence": name,
                           "seed": seed, "frames": [frame.frame_index for frame in clip],
                           "loss": float(loss.detach()), "projection_gradient_l1": projection_grads,
                           "uncertainty_gradient_l1": uncertainty_grads,
                           "attention_calls": len(calls)}
                    fold_records.append(row)
                    training_records.append(row)
                    global_step += 1
                    write(fold_dir / "training.json", fold_records)
                    write(args.output_dir / "progress.json", {"phase": "training",
                         "fold": fold_index, "step": global_step, "total": total_updates,
                         "sequence": name, "epoch": epoch})
                    del samples, loss, calls
                    gc.collect()
            head_path = fold_dir / "heads.pt"
            save_tensor(heads.state_dict(), head_path)
            write(fold_dir / "training_complete.json", {
                "updates": len(fold_records), "checkpoint_sha256": digest(head_path),
                "validation_sequences": validation, "training_sequences": training,
                "layers": 4,
            })
            heads.eval()
            for name in validation:
                rows, summary = run_regional_sequence(
                    args, model, heads, videos[name], name,
                    json.loads((native_dir / f"{name}_frames.json").read_text()), args.output_dir)
                regional_summaries[name] = summary
                write(fold_dir / f"{name}_regional_frames.json", rows)
                write(fold_dir / f"{name}_complete.json", {
                    "frames_sha256": digest(fold_dir / f"{name}_regional_frames.json"),
                    "native_sha256": digest(native_dir / f"{name}_frames.json"),
                    "fold": fold_index,
                })
            fold_summaries.append({"fold": fold_index, "training_sequences": training,
                                   "validation_sequences": validation,
                                   "native_mean": float(np.mean([native_summaries[n]["present_dice"] for n in validation])),
                                   "regional_mean": float(np.mean([regional_summaries[n]["present_dice"] for n in validation]))})
            write(fold_dir / "completion.json", {"complete": True, "fold": fold_index,
                 "validation_sequences": validation, "updates": len(fold_records)})
            write(args.output_dir / "progress.json", {"phase": "fold_complete", "fold": fold_index})
            del model, heads, optimizer
            gc.collect()

        delta = {name: regional_summaries[name]["present_dice"] - native_summaries[name]["present_dice"]
                 for name in SEQUENCES}
        native_mean = float(np.mean([native_summaries[name]["present_dice"] for name in SEQUENCES]))
        regional_mean = float(np.mean([regional_summaries[name]["present_dice"] for name in SEQUENCES]))
        native_fp = sum(native_summaries[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        regional_fp = sum(regional_summaries[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        wins = sum(value > 0 for value in delta.values())
        signal = (regional_mean - native_mean >= 0.01 and wins >= 14 and regional_fp <= native_fp)
        result = {
            "complete": True, "native": native_summaries, "regional": regional_summaries,
            "delta_by_sequence": delta, "mean_native": native_mean,
            "mean_regional": regional_mean, "mean_delta": regional_mean - native_mean,
            "wins": wins, "losses": sum(value < 0 for value in delta.values()),
            "ties": sum(value == 0 for value in delta.values()),
            "delta_ci95": sequence_bootstrap_interval(delta, seed=protocol["training"]["seed"]),
            "exploratory_wilcoxon": paired_wilcoxon(
                {name: regional_summaries[name]["present_dice"] for name in SEQUENCES},
                {name: native_summaries[name]["present_dice"] for name in SEQUENCES}),
            "native_false_alarms": native_fp, "regional_false_alarms": regional_fp,
            "folds": fold_summaries, "meaningful_signal_screen": signal,
            "training_updates": len(training_records), "no_shuffled_arm": True,
            "patient_independence_verified": False,
            "scope": "exploratory sequence-grouped PolypGen cross-validation; not independent confirmation",
        }
        if sources != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError("Source/protocol changed during the experiment")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(report(result, SEQUENCES, signal))
        write(args.output_dir / "completion.json", {
            "complete": True, "sequences": len(SEQUENCES), "targets": sum(
                len(json.loads((args.output_dir / "native" / f"{name}_frames.json").read_text()))
                for name in SEQUENCES), "folds": 3, "training_updates": len(training_records),
            "meaningful_signal_screen": signal, "sources": sources,
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
