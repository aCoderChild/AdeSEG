"""Compare CFBI-region-trained SAM2 q/k projections with a matched q/k control."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.cfbi_region_affinity import RegionalAffinityHeads
from modeling.cfbi_region_affinity_training import cfbi_region_clip_loss
from modeling.cfbi_region_qk_training import enable_memory_qk_training, native_qk_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_affinity_cv import (
    MODEL_CONFIG,
    checked_native_run,
    run_regional_sequence,
)
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_temporal_region_fusion_cv import SEQUENCES, summarize, verify_folds

PROTOCOL = ROOT / "protocols/cfbi_region_qk_tuning_polypgen_v1.json"
ARMS = ("qk_only", "regional")
SOURCE_PATHS = [
    "protocols/cfbi_region_qk_tuning_polypgen_v1.json",
    "scripts/run_cfbi_region_qk_cv.py",
    "scripts/verify_cfbi_region_qk.py",
    "modeling/cfbi_region_qk_training.py",
    "modeling/cfbi_region_affinity.py",
    "modeling/native_cfbi_region_affinity.py",
    "modeling/cfbi_region_affinity_training.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_cfbi_region_affinity_cv.py",
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


def trainable_state(model, names):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def train_arm(model, heads, videos, training_sequences, protocol, arm, output_dir, fold):
    qk_names, qk_parameters = enable_memory_qk_training(model)
    if arm == "regional":
        heads.requires_grad_(True)
        parameters = [
            {"params": qk_parameters, "lr": protocol["training"]["qk_learning_rate"]},
            {
                "params": list(heads.parameters()),
                "lr": protocol["training"]["regional_projection_learning_rate"],
            },
        ]
    else:
        heads.requires_grad_(False)
        parameters = [{"params": qk_parameters, "lr": protocol["training"]["qk_learning_rate"]}]

    optimizer = torch.optim.AdamW(
        parameters,
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
            clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
            samples = capture_presence_clip(model, clip, output_dir)
            optimizer.zero_grad(set_to_none=True)
            if arm == "regional":
                loss, calls, _ = cfbi_region_clip_loss(
                    model,
                    heads,
                    samples,
                    name,
                    check_native=(epoch == 0 and sequence_index == 0),
                )
            else:
                loss = native_qk_clip_loss(
                    model,
                    samples,
                    check_native=(epoch == 0 and sequence_index == 0),
                )
                calls = []
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {arm} loss: fold{fold}/{name}")
            loss.backward()

            qk_gradient_l1 = []
            named_parameters = dict(model.named_parameters())
            for name_key in qk_names:
                grad = named_parameters[name_key].grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError(f"Missing/nonfinite q/k gradient: {name_key}")
                qk_gradient_l1.append(float(grad.abs().sum()))
            if arm == "regional":
                regional_gradient_l1 = []
                for layer_index, head in enumerate(heads.layers):
                    grad = head.projection.weight.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise RuntimeError(f"Missing regional gradient at layer {layer_index}")
                    if grad.abs().sum() <= 0:
                        raise RuntimeError(f"Zero regional gradient at layer {layer_index}")
                    regional_gradient_l1.append(float(grad.abs().sum()))
            else:
                regional_gradient_l1 = []

            allowed = set(qk_names)
            model_grad_names = {
                name_key for name_key, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if model_grad_names != allowed:
                raise RuntimeError("Unexpected SAM2 parameters received gradients")

            torch.nn.utils.clip_grad_norm_(
                qk_parameters + (list(heads.parameters()) if arm == "regional" else []),
                protocol["training"]["gradient_norm"],
            )
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in qk_parameters):
                raise RuntimeError("Nonfinite q/k projection parameter")
            if arm == "regional" and not all(
                torch.isfinite(parameter).all() for parameter in heads.parameters()
            ):
                raise RuntimeError("Nonfinite regional adapter parameter")

            record = {
                "arm": arm,
                "fold": fold,
                "epoch": epoch,
                "sequence": name,
                "seed": seed,
                "clip_frames": [frame.frame_index for frame in clip],
                "loss": float(loss.detach()),
                "qk_gradient_l1": qk_gradient_l1,
                "regional_gradient_l1": regional_gradient_l1,
                "attention_calls": len(calls),
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
            del samples, loss, calls
            gc.collect()
    return records, qk_names


def build_report(result):
    lines = [
        "# CFBI region-supervised SAM2 memory q/k tuning — PolypGen CV",
        "",
        "Exploratory sequence-grouped cross-validation; PolypGen informed prior development and patient independence is unverified.",
        "",
        "| Sequence | Native Dice | q/k-only Dice | Regional q/k Dice | Δ regional − native | Δ regional − q/k-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SEQUENCES:
        native = result["native"][name]["present_dice"]
        control = result["qk_only"][name]["present_dice"]
        regional = result["regional"][name]["present_dice"]
        lines.append(
            f"| {name} | {native:.6f} | {control:.6f} | {regional:.6f} | "
            f"{regional-native:+.6f} | {regional-control:+.6f} |"
        )
    for arm in ARMS:
        lines.append("")
        lines.append(
            f"{arm}: mean Dice {result['means'][arm]:.6f}; "
            f"false alarms {result['false_alarms'][arm]} pooled frames."
        )
    lines.extend([
        "",
        f"Regional − native: {result['delta_regional_native']:+.6f}; "
        f"sequence-bootstrap 95% CI {result['ci_regional_native']}; "
        f"wins/losses/ties {result['wins_regional_native']}/{result['losses_regional_native']}/{result['ties_regional_native']}; "
        f"paired Wilcoxon p={result['wilcoxon_regional_native']['p']:.6g}.",
        f"Regional − q/k-only: {result['delta_regional_qk']:+.6f}; "
        f"sequence-bootstrap 95% CI {result['ci_regional_qk']}; "
        f"wins/losses/ties {result['wins_regional_qk']}/{result['losses_regional_qk']}/{result['ties_regional_qk']}; "
        f"paired Wilcoxon p={result['wilcoxon_regional_qk']['p']:.6g}.",
        f"Frozen screen versus native passed: {result['screen_vs_native']}; "
        f"region-attribution screen versus q/k-only passed: {result['screen_vs_qk_only']}.",
        "This PolypGen experiment is exploratory, not external confirmation.",
    ])
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
        folds = protocol["data"]["folds"]
        verify_folds(folds)
        manifest = ROOT / protocol["data"]["manifest"]
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed") or gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("A passing q/k engineering gate for this checkpoint is required")
        if gate["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Gate manifest differs from the full run")
        for path, expected in gate["sources"].items():
            if path in SOURCE_PATHS and digest(ROOT / path) != expected:
                raise RuntimeError(f"Gate source changed: {path}")
        sources = {path: digest(ROOT / path) for path in SOURCE_PATHS}
        repo, sam2_revision = activate_author_repo()
        if sam2_revision != gate["sam2_revision"]:
            raise RuntimeError("SAM2 checkout differs from engineering gate")
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native, native_rows = checked_native_run(args.native_run, videos, checkpoint_sha, manifest_sha)
        setup = {
            "protocol": protocol,
            "protocol_sha256": digest(PROTOCOL),
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": sam2_revision,
            "cfbi_revision": protocol["prior_work"]["CFBI"]["revision"],
            "engineering_gate": str(args.gate),
            "native_run": str(args.native_run),
            "folds": folds,
            "patient_independence_verified": False,
        }
        write(args.output_dir / "setup.json", setup)
        outputs = {arm: {} for arm in ARMS}
        training_records, fold_summaries = [], []
        total_steps = protocol["training"]["epochs"] * 14 * len(folds) * len(ARMS)
        global_step = 0

        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Fold training/validation split is invalid")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            for arm in ARMS:
                torch.manual_seed(protocol["training"]["seed"] + fold_index)
                model = build_author_predictor(args, repo)
                heads = RegionalAffinityHeads(channels=model.memory_attention.d_model)
                arm_dir = fold_dir / arm
                arm_dir.mkdir()
                rows, qk_names = train_arm(
                    model, heads, videos, training, protocol, arm, arm_dir, fold_index
                )
                training_records.extend(rows)
                qk_state = trainable_state(model, qk_names)
                save_tensor(qk_state, arm_dir / "qk_state.pt")
                if arm == "regional":
                    save_tensor(heads.state_dict(), arm_dir / "regional_heads.pt")
                else:
                    heads.eval()
                write(arm_dir / "training_complete.json", {
                    "updates": len(rows),
                    "arm": arm,
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "trainable_qk_parameters": qk_names,
                    "regional_adapter_trained": arm == "regional",
                    "qk_sha256": digest(arm_dir / "qk_state.pt"),
                    "regional_head_sha256": digest(arm_dir / "regional_heads.pt")
                    if arm == "regional" else None,
                })
                heads.eval()
                for name in validation:
                    rows_eval, summary, _ = run_regional_sequence(
                        args, model, heads, videos[name], name, native_rows[name]
                    )
                    outputs[arm][name] = summary
                    write(arm_dir / f"{name}_frames.json", rows_eval)
                    write(arm_dir / f"{name}_complete.json", {
                        "frames_sha256": digest(arm_dir / f"{name}_frames.json"),
                        "native_sha256": digest(args.native_run / "native" / f"{name}_frames.json"),
                        "fold": fold_index,
                        "arm": arm,
                    })
                write(arm_dir / "completion.json", {
                    "complete": True,
                    "fold": fold_index,
                    "arm": arm,
                    "updates": len(rows),
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
                global_step += len(rows)
                write(args.output_dir / "progress.json", {
                    "phase": "fold_arm_complete",
                    "fold": fold_index,
                    "arm": arm,
                    "training_updates": global_step,
                    "total_updates": total_steps,
                })
                del model, heads
                gc.collect()

        means = {
            "native": float(np.mean([native[name]["present_dice"] for name in SEQUENCES])),
            **{
                arm: float(np.mean([outputs[arm][name]["present_dice"] for name in SEQUENCES]))
                for arm in ARMS
            },
        }
        delta_native = {
            name: outputs["regional"][name]["present_dice"] - native[name]["present_dice"]
            for name in SEQUENCES
        }
        delta_qk = {
            name: outputs["regional"][name]["present_dice"] - outputs["qk_only"][name]["present_dice"]
            for name in SEQUENCES
        }
        wins_native = sum(value > 0 for value in delta_native.values())
        losses_native = sum(value < 0 for value in delta_native.values())
        ties_native = len(SEQUENCES) - wins_native - losses_native
        wins_qk = sum(value > 0 for value in delta_qk.values())
        losses_qk = sum(value < 0 for value in delta_qk.values())
        ties_qk = len(SEQUENCES) - wins_qk - losses_qk
        false_alarms = {
            "native": sum(native[name]["gt_empty_false_alarm_count"] for name in SEQUENCES),
            **{
                arm: sum(outputs[arm][name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
                for arm in ARMS
            },
        }
        result = {
            "complete": True,
            "native": native,
            "qk_only": outputs["qk_only"],
            "regional": outputs["regional"],
            "means": means,
            "delta_regional_native": means["regional"] - means["native"],
            "delta_regional_qk": means["regional"] - means["qk_only"],
            "delta_by_sequence_regional_native": delta_native,
            "delta_by_sequence_regional_qk": delta_qk,
            "wins_regional_native": wins_native,
            "losses_regional_native": losses_native,
            "ties_regional_native": ties_native,
            "wins_regional_qk": wins_qk,
            "losses_regional_qk": losses_qk,
            "ties_regional_qk": ties_qk,
            "ci_regional_native": sequence_bootstrap_interval(
                delta_native, seed=protocol["training"]["seed"]
            ),
            "ci_regional_qk": sequence_bootstrap_interval(
                delta_qk, seed=protocol["training"]["seed"] + 1
            ),
            "wilcoxon_regional_native": paired_wilcoxon(
                {name: outputs["regional"][name]["present_dice"] for name in SEQUENCES},
                {name: native[name]["present_dice"] for name in SEQUENCES},
            ),
            "wilcoxon_regional_qk": paired_wilcoxon(
                {name: outputs["regional"][name]["present_dice"] for name in SEQUENCES},
                {name: outputs["qk_only"][name]["present_dice"] for name in SEQUENCES},
            ),
            "false_alarms": false_alarms,
            "folds": fold_summaries,
            "training_updates_per_arm": len(training_records) // 2,
            "screen_vs_native": (
                means["regional"] - means["native"] >= 0.01
                and wins_native >= 14
                and false_alarms["regional"] <= false_alarms["native"]
            ),
            "screen_vs_qk_only": (
                means["regional"] > means["qk_only"]
                and wins_qk >= 14
                and false_alarms["regional"] <= false_alarms["qk_only"]
            ),
            "scope": "Exploratory sequence-grouped PolypGen CV; patient independence unverified.",
        }
        if sources != {path: digest(ROOT / path) for path in SOURCE_PATHS}:
            raise RuntimeError("Source or frozen protocol changed during experiment")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(build_report(result))
        write(args.output_dir / "completion.json", {
            "complete": True,
            "sequences": len(SEQUENCES),
            "folds": len(folds),
            "arms": list(ARMS),
            "training_updates_per_arm": len(training_records) // 2,
            "screen_vs_native": result["screen_vs_native"],
            "screen_vs_qk_only": result["screen_vs_qk_only"],
            "sources": sources,
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
