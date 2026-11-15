"""Train and evaluate prediction-conditioned regional SAM2 memory reads."""

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.prediction_conditioned_affinity import (
    PredictionConditionedTrackStep,
    PredictionRoleScales,
)
from modeling.prediction_conditioned_training import clip_loss
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_soft_role_affinity_cv import load_native_baseline, paired_delta, score_frame, summarize
from scripts.verify_on_demand_clip import capture_clip
from scripts.verify_prediction_conditioned_affinity import PROTOCOL, SOURCES

SEQUENCES = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
def verify_folds(folds):
    flat = [name for fold in folds for name in fold]
    if len(folds) != 3 or any(len(fold) != 7 for fold in folds):
        raise RuntimeError("Expected three folds with seven held-out sequences each")
    if len(set(flat)) != 21 or set(flat) != set(SEQUENCES):
        raise RuntimeError("Folds must partition all promptable sequences once")


def training_sources():
    return {path: digest(ROOT / path) for path in SOURCES}


def evaluate_sequence(model, scales, mode, frames, name, native_rows, output_dir, fold):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"pc_affinity_{name}_", dir=output_dir) as temp:
        state, actual_prompt = prepare(model, frames, Path(temp))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt mismatch for {name}")
        started = time.monotonic()
        with PredictionConditionedTrackStep(model, scales, mode) as tracker, torch.inference_mode():
            for frame, object_ids, logits in model.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if object_ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs for {name}/{frame}")
                if frame == prompt:
                    continue
                native = native_by_frame.get(frame)
                if native is None:
                    raise RuntimeError(f"Native reference missing {name}/{frame}")
                read = tracker.reads[-1]
                layers = read["layers"]
                if [row["layer"] for row in layers] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Incomplete second read at {name}/{frame}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite mask logits at {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row.update({
                    "fold": fold,
                    "native_prediction_sha256": native["prediction_sha256"],
                    "attention": {
                        "memory_frames": read["memory_frames"],
                        "spatial_keys": layers[0]["spatial_keys"],
                        "pointer_keys": layers[0]["pointer_keys"],
                    },
                    "attention_layers": layers,
                    "query_role_mean": read["query_role_mean"],
                    "memory_role_mean": read["memory_role_mean"],
                })
                rows.append(row)
                write(output_dir / "progress.json", {
                    "phase": "evaluation", "fold": fold, "mode": mode,
                    "sequence": name, "frame": int(frame),
                })
        elapsed = time.monotonic() - started
        model.reset_state(state)

    expected = list(range(prompt + 1, len(frames)))
    if [row["frame"] for row in rows] != expected:
        raise RuntimeError(f"Incomplete {name}/{mode} trajectory")
    summary = summarize(rows)
    summary["inference_wall_seconds"] = elapsed
    summary["mean_spatial_memory_keys"] = float(np.mean([
        np.mean([layer["spatial_keys"] for layer in row["attention_layers"]])
        for row in rows
    ]))
    summary["mean_pointer_tokens"] = float(np.mean([
        np.mean([layer["pointer_keys"] for layer in row["attention_layers"]])
        for row in rows
    ]))
    return rows, summary


def compare_read_schedules(regional, global_rows, name):
    if len(regional) != len(global_rows):
        raise RuntimeError(f"Regional/global frame counts differ for {name}")
    for left, right in zip(regional, global_rows, strict=True):
        if left["frame"] != right["frame"]:
            raise RuntimeError(f"Regional/global frame order differs for {name}")
        for key in ("memory_frames", "spatial_keys", "pointer_keys"):
            if left["attention"].get(key) != right["attention"].get(key):
                raise RuntimeError(f"Regional/global {key} differs at {name}/{left['frame']}")
        if any(layer["pointer_bias_max_abs"] != 0.0 for layer in left["attention_layers"] + right["attention_layers"]):
            raise RuntimeError(f"Pointer bias is nonzero for {name}/{left['frame']}")


def report(result):
    lines = [
        "# Prediction-conditioned regional memory affinity — PolypGen exploratory CV",
        "",
        "PolypGen informed prior method development; patient-to-sequence independence is unverified.",
        "This result is exploratory and is not an untouched external confirmation.",
        "",
        "| Sequence | Native Dice | Global within-frame control | Prediction-conditioned regional | Regional − native | Regional − global |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SEQUENCES:
        native = result["native"][name]["present_dice"]
        global_score = result["global"][name]["present_dice"]
        regional = result["regional"][name]["present_dice"]
        lines.append(
            f"| {name} | {native:.6f} | {global_score:.6f} | {regional:.6f} | "
            f"{regional - native:+.6f} | {regional - global_score:+.6f} |"
        )
    lines.extend(["", "## Cohort summaries", "", "| Arm | Present Dice | IoU | Boundary F | GT-empty false alarms | Runtime (s) |", "|---|---:|---:|---:|---:|---:|"])
    for mode, title in (("native", "Native SAM2"), ("global", "Global within-frame control"), ("regional", "Prediction-conditioned regional")):
        summary = result["summary"][mode]
        elapsed = summary.get("inference_wall_seconds")
        runtime = "N/A" if elapsed is None else f"{elapsed:.1f}"
        lines.append(
            f"| {title} | {summary['present_dice']:.6f} | {summary['iou']:.6f} | "
            f"{summary['boundary_f']:.6f} | {summary['gt_empty_false_alarm_count']} | "
            f"{runtime} |"
        )
    for key, title in (("regional_vs_native", "native"), ("regional_vs_global", "global control")):
        delta = result[key]
        lines.append(
            f"\nRegional − {title}: {delta['mean_delta']:+.6f}; "
            f"wins/losses/ties {delta['wins']}/{delta['losses']}/{delta['ties']}; "
            f"sequence-bootstrap 95% CI {delta['ci95']}; "
            f"paired Wilcoxon p={delta['wilcoxon']['p']:.6g}."
        )
    lines.append(f"\nFrozen meaningful-improvement screen passed: {result['meaningful_signal_screen']}.")
    return "\n".join(lines) + "\n"


def summarize_arm(rows_by_sequence):
    per_sequence = [rows_by_sequence[name] for name in SEQUENCES]
    runtimes = [row.get("inference_wall_seconds") for row in per_sequence]
    return {
        "present_dice": float(np.mean([row["present_dice"] for row in per_sequence])),
        "iou": float(np.mean([row["iou"] for row in per_sequence])),
        "boundary_f": float(np.mean([row["boundary_f"] for row in per_sequence])),
        "gt_empty_false_alarm_count": int(sum(row["gt_empty_false_alarm_count"] for row in per_sequence)),
        "inference_wall_seconds": None if any(value is None for value in runtimes) else float(sum(runtimes)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(PROTOCOL.read_text())
        folds = protocol["training"]["folds"]
        verify_folds(folds)
        manifest = ROOT / protocol["data"]["manifest"]
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate_path = args.gate / "completion.json"
        gate = json.loads(gate_path.read_text())
        if not gate.get("passed") or gate.get("protocol_sha256") != digest(PROTOCOL):
            raise RuntimeError("A passing gate for the frozen protocol is required")
        if gate.get("checkpoint_sha256") != checkpoint_sha or gate.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Gate checkpoint or manifest differs")
        sources = training_sources()
        gate_sources = gate.get("source_hashes", {})
        if set(gate_sources) != set(sources):
            raise RuntimeError("Gate source inventory differs from the full-run source inventory")
        for path in sources:
            if gate_sources[path] != sources[path]:
                raise RuntimeError(f"Source changed since the real-data gate: {path}")

        repo, sam2_revision = activate_author_repo()
        if sam2_revision != gate.get("sam2_revision"):
            raise RuntimeError("SAM2 checkout differs from the gate")
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native, native_rows = load_native_baseline(
            args.native_run, videos, checkpoint_sha, manifest_sha
        )
        setup = {
            "protocol": protocol,
            "protocol_sha256": digest(PROTOCOL),
            "source_hashes": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": sam2_revision,
            "model_config": MODEL_CONFIG,
            "gate": str(args.gate),
            "gate_completion_sha256": digest(gate_path),
            "native_run": str(args.native_run),
            "native_setup_sha256": digest(args.native_run / "setup.json"),
            "folds": folds,
            "device": "cpu",
            "patient_independence_verified": False,
        }
        write(output / "setup.json", setup)
        results = {"native": native, "global": {}, "regional": {}}
        training_records = []
        fold_summaries = []
        total_pairs = protocol["training"]["epochs"] * 14 * len(folds)
        progress_update = 0

        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Invalid sequence-grouped fold")
            fold_dir = output / f"fold{fold_index}"
            fold_dir.mkdir()
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            model = build_author_predictor(SimpleNamespace(
                method="sam2", device="cpu", checkpoint=args.checkpoint,
                model_config=MODEL_CONFIG,
            ), repo)
            model.requires_grad_(False)
            arms = {
                mode: PredictionRoleScales(max_strength=2.0).train()
                for mode in ("regional", "global")
            }
            optimizers = {
                mode: torch.optim.AdamW(
                    arms[mode].parameters(),
                    lr=protocol["training"]["learning_rate"],
                    weight_decay=protocol["training"]["weight_decay"],
                )
                for mode in arms
            }
            layer_gradients = {mode: set() for mode in arms}
            training_log = []

            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = (protocol["training"]["seed"] + fold_index * 100000
                            + epoch * len(training) + sequence_index)
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    if prompt_index(clip, {"initialization": "first_labeled_frame"}) != 0:
                        raise RuntimeError(f"Training clip prompt is not first: {name}")
                    samples = capture_clip(model, clip, output)
                    record = {
                        "fold": fold_index,
                        "epoch": epoch,
                        "sequence": name,
                        "seed": seed,
                        "frames": [int(frame.frame_index) for frame in clip],
                        "arms": {},
                    }
                    for mode in ("regional", "global"):
                        optimizer = optimizers[mode]
                        optimizer.zero_grad(set_to_none=True)
                        loss, frame_records = clip_loss(
                            model, arms[mode], samples, mode,
                            check_native=(epoch == 0 and sequence_index == 0),
                        )
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss for {mode}/{name}")
                        loss.backward()
                        gradient = arms[mode].raw_strength.grad
                        if gradient is None or not torch.isfinite(gradient).all():
                            raise RuntimeError(f"Invalid regional-scale gradient for {mode}")
                        per_layer = gradient.abs().sum(-1)
                        layer_gradients[mode].update(
                            index for index, value in enumerate(per_layer) if value > 0
                        )
                        if any(parameter.grad is not None for parameter in model.parameters()):
                            raise RuntimeError("Frozen SAM2 parameters acquired gradients")
                        torch.nn.utils.clip_grad_norm_(
                            arms[mode].parameters(), protocol["training"]["max_gradient_norm"]
                        )
                        optimizer.step()
                        strengths = arms[mode].strengths().detach()
                        if not torch.isfinite(strengths).all() or strengths.abs().max() > 2.0:
                            raise RuntimeError(f"Invalid bounded update in {mode}")
                        record["arms"][mode] = {
                            "loss": float(loss.detach()),
                            "gradient_l1_by_layer": per_layer.cpu().tolist(),
                            "strengths_by_layer_role": strengths.cpu().tolist(),
                            "frames": frame_records,
                        }
                        progress_update += 1
                        write(output / "progress.json", {
                            "phase": "training", "fold": fold_index,
                            "update": progress_update, "total": 2 * total_pairs,
                            "epoch": epoch, "sequence": name,
                        })
                    training_log.append(record)
                    training_records.append(record)
                    write(fold_dir / "training.json", training_log)
                    del samples, loss, frame_records
                    gc.collect()

            for mode in arms:
                if layer_gradients[mode] != {0, 1, 2, 3}:
                    raise RuntimeError(f"Not all layers trained for {mode}")
                save_tensor(arms[mode].state_dict(), fold_dir / f"{mode}_scales.pt")
                write(fold_dir / f"{mode}_training_complete.json", {
                    "updates": len(training_log),
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "layers_with_gradients": sorted(layer_gradients[mode]),
                })
                arms[mode].eval()

            for name in validation:
                for mode in ("regional", "global"):
                    rows, summary = evaluate_sequence(
                        model, arms[mode], mode, videos[name], name,
                        native_rows[name], output, fold_index,
                    )
                    results[mode][name] = summary
                    write(fold_dir / f"{name}_{mode}_frames.json", rows)
                    write(fold_dir / f"{name}_{mode}_complete.json", {
                        "frames_sha256": digest(fold_dir / f"{name}_{mode}_frames.json"),
                        "native_sha256": digest(args.native_run / "native" / f"{name}_frames.json"),
                        "fold": fold_index,
                    })
                compare_read_schedules(
                    json.loads((fold_dir / f"{name}_regional_frames.json").read_text()),
                    json.loads((fold_dir / f"{name}_global_frames.json").read_text()),
                    name,
                )
                write(output / "progress.json", {
                    "phase": "evaluation", "fold": fold_index,
                    "sequence": name, "complete": True,
                })

            fold_summaries.append({
                "fold": fold_index,
                "training_sequences": training,
                "validation_sequences": validation,
                "training_updates_per_arm": len(training_log),
                **{
                    mode: float(np.mean([results[mode][name]["present_dice"] for name in validation]))
                    for mode in ("regional", "global")
                },
                "native": float(np.mean([native[name]["present_dice"] for name in validation])),
            })
            write(fold_dir / "completion.json", {
                "complete": True,
                "fold": fold_index,
                "training_updates_per_arm": len(training_log),
                "validation_sequences": validation,
                "training_sha256": digest(fold_dir / "training.json"),
            })
            del model, arms, optimizers
            gc.collect()

        summaries = {
            mode: summarize_arm(results[mode])
            for mode in ("native", "global", "regional")
        }
        regional_vs_native = paired_delta(
            {"regional": results["regional"], "native": results["native"]},
            "regional", "native", protocol["training"]["seed"],
        )
        regional_vs_global = paired_delta(
            {"regional": results["regional"], "global": results["global"]},
            "regional", "global", protocol["training"]["seed"] + 1,
        )
        result = {
            "complete": True,
            "native": native,
            "global": results["global"],
            "regional": results["regional"],
            "summary": summaries,
            "regional_vs_native": regional_vs_native,
            "regional_vs_global": regional_vs_global,
            "folds": fold_summaries,
            "training_updates_per_arm": len(training_records),
            "patient_independence_verified": False,
            "scope": protocol["data"]["scope"],
        }
        regional_fa = summaries["regional"]["gt_empty_false_alarm_count"]
        result["meaningful_signal_screen"] = all(
            result[key]["mean_delta"] >= 0.01
            and result[key]["wins"] >= 14
            and regional_fa <= summaries[control]["gt_empty_false_alarm_count"]
            for key, control in (("regional_vs_native", "native"),
                                 ("regional_vs_global", "global"))
        )
        if sources != training_sources():
            raise RuntimeError("A frozen source changed during the cohort run")
        write(output / "training.json", training_records)
        write(output / "results.json", result)
        (output / "results.md").write_text(report(result))
        write(output / "completion.json", {
            "complete": True,
            "sequences": len(SEQUENCES),
            "folds": len(folds),
            "training_updates_per_arm": len(training_records),
            "meaningful_signal_screen": result["meaningful_signal_screen"],
            "source_hashes": sources,
            "protocol_sha256": digest(PROTOCOL),
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": sam2_revision,
            "fold_receipts": [
                {"fold": index, "sha256": digest(output / f"fold{index}" / "completion.json")}
                for index in range(1, 4)
            ],
        })
        write(output / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
