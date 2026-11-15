"""Train and evaluate SimVOS-style regional prototypes on fixed PolypGen folds."""

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
from modeling.native_simvos_prototype_read import NativeSimVOSPrototypeRead
from modeling.simvos_prototype_training import prototype_clip_loss
from modeling.simvos_region_prototypes import RegionPrototypeHeads, SIMVOS_REVISION
from scripts.capture_presence_clip import capture_presence_clip
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write

PROTOCOL = ROOT / "protocols/simvos_region_prototypes_polypgen_v1.json"
SEQUENCES = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
ARMS = ("regional", "global")
SOURCES = [
    "scripts/run_simvos_region_prototypes_cv.py",
    "scripts/verify_simvos_region_prototypes.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/false_alarm_statistics.py",
    "scripts/memory_token_io.py",
    "modeling/simvos_region_prototypes.py",
    "modeling/native_simvos_prototype_read.py",
    "modeling/simvos_prototype_training.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/region_bias.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "external/SimVOS/model/models_vit.py",
]


def score_frame(logits, frames, frame):
    target = load_label_mask(frames[frame].mask_path, [1]) == 1
    prediction = logits[0, 0].detach().float().cpu().numpy() > 0
    return {
        "frame": frame,
        "present": bool(target.any()),
        "scores": {
            **calculate_scores(prediction, target),
            "boundary_f": float(db_eval_boundary(prediction, target)),
        },
        "absent_fp": bool(prediction.any()) if not target.any() else None,
        "prediction_sha256": tensor_hash(logits),
    }


def summarize(rows):
    result = propagation_summary(rows)
    present = [row for row in rows if row["present"]]
    result["iou"] = (
        float(np.mean([row["scores"]["iou"] for row in present])) if present else None
    )
    result["boundary_f"] = (
        float(np.mean([row["scores"]["boundary_f"] for row in present]))
        if present else None
    )
    return result


def verify_folds(folds):
    flattened = [name for fold in folds for name in fold]
    if len(folds) != 3 or any(len(fold) != 7 for fold in folds):
        raise RuntimeError("Expected three fixed seven-sequence folds")
    if len(set(flattened)) != 21 or set(flattened) != set(SEQUENCES):
        raise RuntimeError("Folds must partition the 21 promptable sequences")


def source_hashes(protocol_path):
    protocol_name = protocol_path.relative_to(ROOT).as_posix()
    return {
        protocol_name: digest(protocol_path),
        **{path: digest(ROOT / path) for path in SOURCES},
    }


def train_arm(model, videos, training_sequences, protocol, arm, output_dir, fold):
    heads = RegionPrototypeHeads(
        prototypes=protocol["method"]["prototype_count"]
    ).to("cpu")
    heads.train()
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=protocol["training"]["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    epochs = protocol["training"]["epochs"]
    total_steps = epochs * len(training_sequences)
    records = []
    for epoch in range(epochs):
        for sequence_index, name in enumerate(training_sequences):
            seed = (
                protocol["training"]["seed"]
                + fold * 100000
                + epoch * len(training_sequences)
                + sequence_index
            )
            clip = sample_clip(
                videos[name], protocol["training"]["clip_length"], seed
            )
            samples = capture_presence_clip(model, clip, output_dir)
            optimizer.zero_grad(set_to_none=True)
            loss, calls, deltas = prototype_clip_loss(
                model,
                heads,
                samples,
                name,
                arm,
                check_native=(epoch == 0 and sequence_index == 0),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {arm} loss: fold{fold}/{name}")
            loss.backward()
            gradient_l1 = []
            for parameter_name, parameter in heads.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(
                        f"Nonfinite prototype gradient: {parameter_name}"
                    )
                gradient_l1.append(float(gradient.detach().abs().sum()))
            if not gradient_l1 or not any(value > 0 for value in gradient_l1):
                raise RuntimeError("No prototype parameter received a gradient")
            torch.nn.utils.clip_grad_norm_(
                heads.parameters(), protocol["training"]["gradient_norm"]
            )
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                raise RuntimeError("Nonfinite prototype parameter")
            record = {
                "arm": arm,
                "fold": fold,
                "epoch": epoch,
                "sequence": name,
                "seed": seed,
                "clip_frames": [frame.frame_index for frame in clip],
                "loss": float(loss.detach()),
                "gradient_l1": gradient_l1,
                "attention_calls": len(calls),
                "max_logit_delta_from_native": max(deltas),
            }
            records.append(record)
            write(output_dir / "training.json", records)
            write(output_dir.parent / "progress.json", {
                "phase": "training",
                "fold": fold,
                "arm": arm,
                "step": epoch * len(training_sequences) + sequence_index + 1,
                "total": total_steps,
                "sequence": name,
            })
            del samples, loss, calls, clip
            gc.collect()
    heads.eval()
    return heads, records


def run_native_sequence(model, frames, name, output_dir):
    started = time.monotonic()
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"native_{name}_", dir=output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        expected = prompt_index(frames, {"initialization": "first_labeled_frame"})
        if prompt != expected:
            raise RuntimeError(f"Native prompt mismatch: {name}")
        with torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs: {name}/{frame}")
                if frame > prompt:
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite native prediction: {name}/{frame}")
                    rows.append(score_frame(logits, frames, frame))
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Native frame coverage incomplete: {name}")
    summary = summarize(rows)
    summary["seconds"] = time.monotonic() - started
    summary["fps"] = len(rows) / summary["seconds"] if summary["seconds"] else None
    return prompt, rows, summary


def run_prototype_sequence(model, heads, frames, name, arm, output_dir):
    started = time.monotonic()
    rows = []
    expected_prompt = prompt_index(
        frames, {"initialization": "first_labeled_frame"}
    )
    with tempfile.TemporaryDirectory(prefix=f"{arm}_{name}_", dir=output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        if prompt != expected_prompt:
            raise RuntimeError(f"Prompt mismatch: {name}/{arm}")
        observer = NativeSimVOSPrototypeRead(model, heads, arm, name)
        with observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object IDs: {name}/{frame}")
                if frame == prompt:
                    continue
                layers = observer.calls[-4:]
                if [call["layer"] for call in layers] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Missing memory-attention layer: {name}/{frame}")
                if any(call["frame"] != frame for call in layers):
                    raise RuntimeError(f"Stale attention trace: {name}/{frame}")
                if any(call["prototype_keys"] <= 0 for call in layers):
                    raise RuntimeError(f"No prototype keys read: {name}/{frame}")
                if any(call["spatial_keys"] <= 0 or call["pointer_keys"] < 0
                       for call in layers):
                    raise RuntimeError(f"Invalid native key counts: {name}/{frame}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite prediction: {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row["attention_layers"] = layers
                rows.append(row)
                write(output_dir / "progress.json", {
                    "phase": "evaluation",
                    "arm": arm,
                    "sequence": name,
                    "frame": frame,
                })
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Prediction coverage incomplete: {name}/{arm}")
    summary = summarize(rows)
    summary["seconds"] = time.monotonic() - started
    summary["fps"] = len(rows) / summary["seconds"] if summary["seconds"] else None
    return rows, summary


def report(result):
    rows = [
        "# SimVOS-style regional memory prototypes — PolypGen sequence CV",
        "",
        "Exploratory sequence-grouped CV. PolypGen informed prior development; patient-to-sequence independence is unverified.",
        "",
        "| Sequence | Native SAM2 | Global prototypes | Regional prototypes | Regional − native | Regional − global |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SEQUENCES:
        native = result["native"][name]["present_dice"]
        global_value = result["global"][name]["present_dice"]
        regional = result["regional"][name]["present_dice"]
        rows.append(
            f"| {name} | {native:.6f} | {global_value:.6f} | {regional:.6f} | "
            f"{regional-native:+.6f} | {regional-global_value:+.6f} |"
        )
    rows.extend([
        "",
        "| Arm | Equal-sequence present Dice | Mean IoU | Mean Boundary F | GT-empty false-alarm frames |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ("native", "global", "regional"):
        rows.append(
            f"| {arm} | {result['means'][arm]:.6f} | "
            f"{result['mean_iou'][arm]:.6f} | {result['mean_boundary_f'][arm]:.6f} | "
            f"{result['false_alarms'][arm]} |"
        )
    rows.extend([
        "",
        f"Regional − native: {result['delta_regional_native']:+.6f}; "
        f"sequence-bootstrap 95% CI {result['ci_regional_native']}; "
        f"wins/losses/ties {result['wins_regional_native']}/"
        f"{result['losses_regional_native']}/{result['ties_regional_native']}; "
        f"paired Wilcoxon p={result['wilcoxon_regional_native']['p']:.6g}.",
        f"Regional − global: {result['delta_regional_global']:+.6f}; "
        f"sequence-bootstrap 95% CI {result['ci_regional_global']}; "
        f"wins/losses/ties {result['wins_regional_global']}/"
        f"{result['losses_regional_global']}/{result['ties_regional_global']}; "
        f"paired Wilcoxon p={result['wilcoxon_regional_global']['p']:.6g}.",
        f"Frozen screen versus native passed: {result['screen_vs_native']}; "
        f"regional attribution screen versus global control passed: {result['screen_vs_global']}.",
        "All full native keys and pointers remain active; prototype reads add keys and compute.",
    ])
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    if ROOT not in args.protocol.parents:
        parser.error("--protocol must be inside the project")
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads(args.protocol.read_text())
        folds = protocol["data"]["folds"]
        verify_folds(folds)
        manifest = ROOT / protocol["data"]["manifest"]
        checkpoint_sha = digest(args.checkpoint)
        manifest_sha = digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed"):
            raise RuntimeError("A passing real-data engineering gate is required")
        if gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("Gate checkpoint differs from full-run checkpoint")
        if gate["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Gate manifest differs from full-run manifest")
        if gate["protocol_sha256"] != digest(args.protocol):
            raise RuntimeError("Gate protocol differs from the frozen full-run protocol")
        for path, expected in gate["sources"].items():
            if digest(ROOT / path) != expected:
                raise RuntimeError(f"Gate source changed: {path}")
        sources = source_hashes(args.protocol)
        args.method = "sam2"
        args.device = "cpu"
        args.model_config = protocol["training"]["model_config"]
        repo, sam2_revision = activate_author_repo()
        if sam2_revision != gate["sam2_revision"]:
            raise RuntimeError("SAM2 source differs from the passing gate")
        if SIMVOS_REVISION != protocol["prior_work"]["SimVOS"]["revision"]:
            raise RuntimeError("SimVOS source revision differs from protocol")
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        setup = {
            "protocol": protocol,
            "protocol_sha256": digest(args.protocol),
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": sam2_revision,
            "simvos_revision": SIMVOS_REVISION,
            "engineering_gate": str(args.gate),
            "folds": folds,
            "device": "cpu",
            "patient_independence_verified": False,
        }
        write(args.output_dir / "setup.json", setup)

        native, native_rows = {}, {}
        native_dir = args.output_dir / "native"
        native_dir.mkdir()
        for name in SEQUENCES:
            model = build_author_predictor(args, repo)
            prompt, rows, summary = run_native_sequence(
                model, videos[name], name, args.output_dir
            )
            native[name], native_rows[name] = summary, rows
            path = native_dir / f"{name}_frames.json"
            write(path, rows)
            write(native_dir / f"{name}_complete.json", {
                "frames_sha256": digest(path),
                "prompt": prompt,
                "checkpoint_sha256": checkpoint_sha,
            })
            del model
            gc.collect()

        outputs = {arm: {} for arm in ARMS}
        training_records, fold_summaries = [], []
        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Invalid fold partition")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            for arm in ARMS:
                torch.manual_seed(protocol["training"]["seed"] + fold_index)
                model = build_author_predictor(args, repo)
                arm_dir = fold_dir / arm
                arm_dir.mkdir()
                started = time.monotonic()
                heads, records = train_arm(
                    model, videos, training, protocol, arm, arm_dir, fold_index
                )
                training_records.extend(records)
                save_tensor(heads.state_dict(), arm_dir / "prototype_heads.pt")
                write(arm_dir / "training_complete.json", {
                    "complete": True,
                    "fold": fold_index,
                    "arm": arm,
                    "updates": len(records),
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "prototype_heads_sha256": digest(arm_dir / "prototype_heads.pt"),
                    "training_seconds": time.monotonic() - started,
                })
                for name in validation:
                    frame_rows, summary = run_prototype_sequence(
                        model, heads, videos[name], name, arm, args.output_dir
                    )
                    for row in frame_rows:
                        row["native_prediction_sha256"] = next(
                            item["prediction_sha256"]
                            for item in native_rows[name]
                            if item["frame"] == row["frame"]
                        )
                    outputs[arm][name] = summary
                    path = arm_dir / f"{name}_frames.json"
                    write(path, frame_rows)
                    write(arm_dir / f"{name}_complete.json", {
                        "frames_sha256": digest(path),
                        "native_frames_sha256": digest(native_dir / f"{name}_frames.json"),
                        "fold": fold_index,
                        "arm": arm,
                    })
                write(arm_dir / "completion.json", {
                    "complete": True,
                    "fold": fold_index,
                    "arm": arm,
                    "updates": len(records),
                    "validation_sequences": validation,
                })
                fold_summaries.append({
                    "fold": fold_index,
                    "arm": arm,
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "mean_dice": float(np.mean([
                        outputs[arm][name]["present_dice"] for name in validation
                    ])),
                })
                del model, heads, records
                gc.collect()

        means = {
            "native": float(np.mean([native[n]["present_dice"] for n in SEQUENCES])),
            **{
                arm: float(np.mean([outputs[arm][n]["present_dice"] for n in SEQUENCES]))
                for arm in ARMS
            },
        }
        means_iou = {
            arm: float(np.mean([
                value
                for n in SEQUENCES
                for value in [native[n]["iou"] if arm == "native" else outputs[arm][n]["iou"]]
                if value is not None
            ]))
            for arm in ("native", *ARMS)
        }
        means_boundary = {
            arm: float(np.mean([
                value
                for n in SEQUENCES
                for value in [native[n]["boundary_f"] if arm == "native" else outputs[arm][n]["boundary_f"]]
                if value is not None
            ]))
            for arm in ("native", *ARMS)
        }
        false_alarms = {
            "native": sum(native[n]["gt_empty_false_alarm_count"] for n in SEQUENCES),
            **{
                arm: sum(outputs[arm][n]["gt_empty_false_alarm_count"] for n in SEQUENCES)
                for arm in ARMS
            },
        }
        delta_native = {
            n: outputs["regional"][n]["present_dice"] - native[n]["present_dice"]
            for n in SEQUENCES
        }
        delta_global = {
            n: outputs["regional"][n]["present_dice"] - outputs["global"][n]["present_dice"]
            for n in SEQUENCES
        }
        wins_native = sum(x > 0 for x in delta_native.values())
        losses_native = sum(x < 0 for x in delta_native.values())
        wins_global = sum(x > 0 for x in delta_global.values())
        losses_global = sum(x < 0 for x in delta_global.values())
        result = {
            "complete": True,
            "native": native,
            "regional": outputs["regional"],
            "global": outputs["global"],
            "means": means,
            "mean_iou": means_iou,
            "mean_boundary_f": means_boundary,
            "false_alarms": false_alarms,
            "delta_regional_native": means["regional"] - means["native"],
            "delta_regional_global": means["regional"] - means["global"],
            "delta_by_sequence_regional_native": delta_native,
            "delta_by_sequence_regional_global": delta_global,
            "wins_regional_native": wins_native,
            "losses_regional_native": losses_native,
            "ties_regional_native": len(SEQUENCES) - wins_native - losses_native,
            "wins_regional_global": wins_global,
            "losses_regional_global": losses_global,
            "ties_regional_global": len(SEQUENCES) - wins_global - losses_global,
            "ci_regional_native": sequence_bootstrap_interval(
                delta_native, seed=protocol["training"]["seed"]
            ),
            "ci_regional_global": sequence_bootstrap_interval(
                delta_global, seed=protocol["training"]["seed"] + 1
            ),
            "wilcoxon_regional_native": paired_wilcoxon(
                {n: outputs["regional"][n]["present_dice"] for n in SEQUENCES},
                {n: native[n]["present_dice"] for n in SEQUENCES},
            ),
            "wilcoxon_regional_global": paired_wilcoxon(
                {n: outputs["regional"][n]["present_dice"] for n in SEQUENCES},
                {n: outputs["global"][n]["present_dice"] for n in SEQUENCES},
            ),
            "folds": fold_summaries,
            "updates_per_arm": protocol["training"]["epochs"] * 14 * len(folds),
            "screen_vs_native": (
                means["regional"] - means["native"] >= 0.01
                and wins_native >= 14
                and false_alarms["regional"] <= false_alarms["native"]
            ),
            "screen_vs_global": (
                means["regional"] - means["global"] >= 0.01
                and wins_global >= 14
                and false_alarms["regional"] <= false_alarms["global"]
            ),
            "scope": "Exploratory sequence-grouped PolypGen CV; patient independence unverified.",
        }
        if sources != source_hashes(args.protocol):
            raise RuntimeError("Code, protocol, or author source changed during run")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(report(result))
        write(args.output_dir / "completion.json", {
            "complete": True,
            "sequences": len(SEQUENCES),
            "folds": len(folds),
            "training_updates": len(training_records),
            "native_coverage": {name: len(native_rows[name]) for name in SEQUENCES},
            "arm_coverage": {
                arm: sorted(outputs[arm]) for arm in ARMS
            },
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "protocol_sha256": digest(args.protocol),
            "sam2_revision": sam2_revision,
            "simvos_revision": SIMVOS_REVISION,
            "screens": {
                "versus_native": result["screen_vs_native"],
                "regional_attribution_vs_global": result["screen_vs_global"],
            },
        })
        print((args.output_dir / "results.md").read_text())
    except BaseException as error:
        write(args.output_dir / "failure.json", {
            "failed": True,
            "type": type(error).__name__,
            "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
