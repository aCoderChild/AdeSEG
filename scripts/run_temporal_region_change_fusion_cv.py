"""Train and evaluate change-supervised regional fusion on fixed PolypGen folds."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.native_temporal_region_fusion import NativeTemporalRegionFusion
from modeling.temporal_region_fusion_change import (
    ChangeSupervisedRegionHeads,
    temporal_change_loss,
)
from modeling.temporal_region_training import temporal_region_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_temporal_region_fusion_cv import (
    SEQUENCES,
    report as base_report,
    run_regional_sequence,
    summarize,
    verify_folds,
)

PROTOCOL = ROOT / "protocols/temporal_region_change_fusion_polypgen_v1.json"
SOURCES = [
    "protocols/temporal_region_change_fusion_polypgen_v1.json",
    "scripts/run_temporal_region_change_fusion_cv.py",
    "scripts/run_temporal_region_fusion_cv.py",
    "scripts/verify_temporal_region_change_fusion.py",
    "modeling/temporal_region_fusion_change.py",
    "modeling/matanyone_temporal_uncertainty.py",
    "modeling/native_temporal_region_fusion.py",
    "modeling/temporal_region_fusion.py",
    "modeling/temporal_region_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_readout.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
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


def load_native_baselines(directory, videos, checkpoint_sha, manifest_sha):
    setup = json.loads((directory / "setup.json").read_text())
    if setup["checkpoint_sha256"] != checkpoint_sha or setup["manifest_sha256"] != manifest_sha:
        raise RuntimeError("Native baseline checkpoint or manifest differs")
    summaries = {}
    rows_by_sequence = {}
    for name in SEQUENCES:
        frames_path = directory / "native" / f"{name}_frames.json"
        receipt = json.loads((directory / "native" / f"{name}_complete.json").read_text())
        if digest(frames_path) != receipt["frames_sha256"]:
            raise RuntimeError(f"Native baseline digest mismatch: {name}")
        rows = json.loads(frames_path.read_text())
        prompt = prompt_index(videos[name], {"initialization": "first_labeled_frame"})
        if receipt["prompt"] != prompt or [row["frame"] for row in rows] != list(range(prompt + 1, len(videos[name]))):
            raise RuntimeError(f"Native baseline coverage or prompt mismatch: {name}")
        summaries[name] = summarize(rows)
        rows_by_sequence[name] = rows
    return summaries, rows_by_sequence


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
        folds = protocol["evaluation"]["folds"]
        verify_folds(folds)
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed") or gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("A passing engineering gate for this checkpoint is required")
        if gate["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Engineering gate manifest differs")
        for path, expected in gate["sources"].items():
            if digest(ROOT / path) != expected:
                raise RuntimeError(f"Gate source changed: {path}")
        repo, revision = activate_author_repo()
        args.method, args.device = "sam2", "cpu"
        args.model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native_summaries, native_rows = load_native_baselines(
            args.native_run, videos, checkpoint_sha, manifest_sha
        )
        sources = {path: digest(ROOT / path) for path in SOURCES}
        setup = {
            "protocol": protocol,
            "protocol_sha256": digest(PROTOCOL),
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "native_run": str(args.native_run),
            "native_run_results_sha256": digest(args.native_run / "native" / "seq2_complete.json"),
            "gate": str(args.gate),
            "gate_sha256": digest(args.gate / "completion.json"),
            "sam2_revision": revision,
            "matanyone_revision": protocol["prior_implementation"]["revision"],
            "folds": folds,
            "patient_independence_verified": False,
        }
        write(args.output_dir / "setup.json", setup)
        write(args.output_dir / "progress.json", {"phase": "training", "fold": 1})

        regional_summaries, fold_summaries, training_records = {}, [], []
        total_updates = protocol["training"]["epochs"] * 14 * len(folds)
        global_step = 0
        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14:
                raise RuntimeError("Fold must train on fourteen sequences")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            model = build_author_predictor(args, repo)
            model.requires_grad_(False)
            heads = ChangeSupervisedRegionHeads(model.memory_attention.d_model).train()
            optimizer = torch.optim.AdamW(
                heads.parameters(), lr=protocol["training"]["learning_rate"],
                weight_decay=protocol["training"]["weight_decay"],
            )
            fold_records = []
            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = (protocol["training"]["seed"] + fold_index * 100000
                            + epoch * len(training) + sequence_index)
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    samples = capture_presence_clip(model, clip, args.output_dir)
                    prompt_mask = load_label_mask(clip[0].mask_path, [1]) == 1
                    prompt_target = torch.from_numpy(prompt_mask.copy())[None, None].float()
                    for head in heads.layers:
                        head.change_logits_history.clear()
                    optimizer.zero_grad(set_to_none=True)
                    segmentation_loss, calls = temporal_region_clip_loss(
                        model, heads, samples, name,
                        check_native=(epoch == 0 and sequence_index == 0), return_trace=True,
                    )
                    change_loss = temporal_change_loss(heads, samples, prompt_target)
                    loss = segmentation_loss + protocol["training"]["temporal_change_loss_weight"] * change_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss in fold{fold_index}/{name}")
                    loss.backward()
                    projection_grads, change_grads, mix_grads = [], [], []
                    for layer, head in enumerate(heads.layers):
                        projection = head.regional_residual.weight.grad
                        if fold_records and (projection is None or not torch.isfinite(projection).all()
                                             or projection.abs().sum() == 0):
                            raise RuntimeError(f"Regional projection gradient missing at layer {layer}")
                        projection_grads.append(0.0 if projection is None else float(projection.abs().sum()))
                        uncertainty = [parameter.grad for parameter in head.uncertainty.parameters()]
                        uncertainty = [gradient for gradient in uncertainty if gradient is not None]
                        change_norm = sum(float(gradient.abs().sum()) for gradient in uncertainty)
                        mix = head.mix_logit.grad
                        if not uncertainty or change_norm <= 0 or mix is None or not torch.isfinite(mix):
                            raise RuntimeError(f"Change/fusion gradient missing at layer {layer}")
                        change_grads.append(change_norm)
                        mix_grads.append(float(mix.abs()))
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise RuntimeError("Frozen SAM2 received parameter gradients")
                    torch.nn.utils.clip_grad_norm_(heads.parameters(), protocol["training"]["max_gradient_norm"])
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in heads.parameters()):
                        raise RuntimeError("Nonfinite regional adapter parameters")
                    row = {
                        "fold": fold_index, "epoch": epoch, "sequence": name, "seed": seed,
                        "frames": [frame.frame_index for frame in clip],
                        "segmentation_loss": float(segmentation_loss.detach()),
                        "change_loss": float(change_loss.detach()),
                        "total_loss": float(loss.detach()),
                        "regional_projection_gradient_l1": projection_grads,
                        "change_gradient_l1": change_grads,
                        "mix_gradient_l1": mix_grads,
                        "attention_calls": len(calls),
                    }
                    fold_records.append(row)
                    training_records.append(row)
                    global_step += 1
                    write(fold_dir / "training.json", fold_records)
                    write(args.output_dir / "progress.json", {
                        "phase": "training", "fold": fold_index,
                        "step": global_step, "total": total_updates,
                        "sequence": name, "epoch": epoch,
                    })
                    del samples, loss, calls
                    gc.collect()
            heads_path = fold_dir / "heads.pt"
            save_tensor(heads.state_dict(), heads_path)
            write(fold_dir / "training_complete.json", {
                "updates": len(fold_records), "checkpoint_sha256": digest(heads_path),
                "validation_sequences": validation, "training_sequences": training, "layers": 4,
            })
            heads.eval()
            for name in validation:
                regional_rows, summary = run_regional_sequence(
                    args, model, heads, videos[name], name, native_rows[name], args.output_dir
                )
                regional_summaries[name] = summary
                write(fold_dir / f"{name}_regional_frames.json", regional_rows)
                write(fold_dir / f"{name}_complete.json", {
                    "frames_sha256": digest(fold_dir / f"{name}_regional_frames.json"),
                    "native_sha256": digest(args.native_run / "native" / f"{name}_frames.json"),
                    "fold": fold_index,
                })
            native_mean = float(np.mean([native_summaries[name]["present_dice"] for name in validation]))
            regional_mean = float(np.mean([regional_summaries[name]["present_dice"] for name in validation]))
            fold_summaries.append({"fold": fold_index, "training_sequences": training,
                                   "validation_sequences": validation, "native_mean": native_mean,
                                   "regional_mean": regional_mean})
            write(fold_dir / "completion.json", {"complete": True, "fold": fold_index,
                 "validation_sequences": validation, "updates": len(fold_records)})
            write(args.output_dir / "progress.json", {"phase": "fold_complete", "fold": fold_index})
            del model, heads, optimizer
            gc.collect()

        deltas = {name: regional_summaries[name]["present_dice"] - native_summaries[name]["present_dice"]
                  for name in SEQUENCES}
        mean_native = float(np.mean([native_summaries[name]["present_dice"] for name in SEQUENCES]))
        mean_regional = float(np.mean([regional_summaries[name]["present_dice"] for name in SEQUENCES]))
        native_fp = sum(native_summaries[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        regional_fp = sum(regional_summaries[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
        signal = mean_regional - mean_native >= 0.01 and sum(value > 0 for value in deltas.values()) >= 14 and regional_fp <= native_fp
        result = {
            "complete": True, "native": native_summaries, "regional": regional_summaries,
            "delta_by_sequence": deltas, "mean_native": mean_native, "mean_regional": mean_regional,
            "mean_delta": mean_regional - mean_native,
            "wins": sum(value > 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "delta_ci95": sequence_bootstrap_interval(deltas, seed=protocol["training"]["seed"]),
            "exploratory_wilcoxon": paired_wilcoxon(
                {n: regional_summaries[n]["present_dice"] for n in SEQUENCES},
                {n: native_summaries[n]["present_dice"] for n in SEQUENCES}),
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
        report = base_report(result, SEQUENCES, signal)
        report = report.replace("# Temporal-uncertainty-gated regional memory residual",
                                "# Change-supervised region-adaptive memory fusion")
        (args.output_dir / "results.md").write_text(report)
        write(args.output_dir / "completion.json", {
            "complete": True, "sequences": len(SEQUENCES), "folds": 3,
            "training_updates": len(training_records), "meaningful_signal_screen": signal,
            "sources": sources,
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
