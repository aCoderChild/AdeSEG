"""Train and evaluate RegionContrast-supervised SAM2 memory Q/K projections."""

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import paired_wilcoxon, sequence_bootstrap_interval
from modeling.cfbi_region_qk_training import (
    enable_memory_qk_training,
    native_qk_clip_loss,
)
from modeling.region_contrast import REGIONCONTRAST_REVISION, RegionContrastBank
from modeling.region_contrast_training import region_contrast_clip_loss
from scripts.capture_presence_clip import capture_presence_clip
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_affinity_cv import (
    MODEL_CONFIG,
    checked_native_run,
    score_frame,
)
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_temporal_region_fusion_cv import SEQUENCES, summarize, verify_folds
from scripts.false_alarm_statistics import propagation_summary

PROTOCOL = ROOT / "protocols/regioncontrast_memory_qk_polypgen_v1.json"
ARMS = ("qk_only", "regional")
SOURCES = [
    "protocols/regioncontrast_memory_qk_polypgen_v1.json",
    "scripts/run_regioncontrast_memory_qk_cv.py",
    "scripts/verify_regioncontrast_memory_qk.py",
    "modeling/region_contrast.py",
    "modeling/region_contrast_training.py",
    "modeling/cfbi_region_qk_training.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/region_definitions.py",
    "modeling/region_tokens.py",
    "modeling/regional_refinement.py",
    "scripts/capture_presence_clip.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_cfbi_region_affinity_cv.py",
    "scripts/run_temporal_region_fusion_cv.py",
    "scripts/run_regional_refinement.py",
    "scripts/memory_token_io.py",
    "scripts/false_alarm_statistics.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "external/RegionContrast/pyseg/models/decoder_contrast.py",
]


def _initial_qk_state(model, names):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def _match_initial_states(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def train_arm(model, videos, training_sequences, protocol, arm, output_dir, fold):
    qk_names, qk_parameters = enable_memory_qk_training(model)
    initial_state = _initial_qk_state(model, qk_names)
    banks = None
    if arm == "regional":
        banks = [
            RegionContrastBank(
                capacity=protocol["training"]["contrast_queue_capacity_per_role_layer"],
                temperature=protocol["training"]["contrast_temperature"],
            ).to("cpu")
            for _ in range(4)
        ]
    optimizer = torch.optim.AdamW(
        qk_parameters,
        lr=protocol["training"]["qk_learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    epochs = protocol["training"]["epochs"]
    total_steps = epochs * len(training_sequences)
    records = []
    step = 0
    for epoch in range(epochs):
        for sequence_index, sequence in enumerate(training_sequences):
            seed = (
                protocol["training"]["seed"]
                + fold * 100000
                + epoch * len(training_sequences)
                + sequence_index
            )
            clip = sample_clip(
                videos[sequence], protocol["training"]["clip_length"], seed
            )
            samples = capture_presence_clip(model, clip, output_dir)
            optimizer.zero_grad(set_to_none=True)
            check_native = epoch == 0 and sequence_index == 0
            if arm == "regional":
                loss, contrast_records, trace = region_contrast_clip_loss(
                    model,
                    banks,
                    samples,
                    sequence,
                    contrast_weight=protocol["training"]["contrast_weight"],
                    check_native=check_native,
                )
                contrast_loss = float(
                    np.mean([row["contrast_loss"] for row in contrast_records])
                )
            else:
                loss = native_qk_clip_loss(
                    model, samples, check_native=check_native
                )
                contrast_records, trace, contrast_loss = [], [], None
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {arm} loss in fold {fold}/{sequence}")
            loss.backward()

            named_parameters = dict(model.named_parameters())
            gradient_by_layer = {index: 0.0 for index in range(4)}
            for name in qk_names:
                gradient = named_parameters[name].grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Missing or nonfinite Q/K gradient: {name}")
                layer_index = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                gradient_by_layer[layer_index] += float(gradient.abs().sum())
            if any(value <= 0 for value in gradient_by_layer.values()):
                raise RuntimeError(f"Missing Q/K gradient in a layer: {gradient_by_layer}")
            if {
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None
            } != set(qk_names):
                raise RuntimeError("Unexpected SAM2 parameters received gradients")
            if arm == "regional" and any(
                row["contrast_roles"] < 1
                for frame in contrast_records
                for row in frame["layers"]
            ):
                raise RuntimeError("A training read had no valid regional contrast")

            torch.nn.utils.clip_grad_norm_(
                qk_parameters, protocol["training"]["gradient_norm"]
            )
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in qk_parameters):
                raise RuntimeError("Nonfinite Q/K projection after optimizer update")
            step += 1
            record = {
                "arm": arm,
                "fold": fold,
                "epoch": epoch,
                "sequence": sequence,
                "seed": seed,
                "clip_frames": [frame.frame_index for frame in clip],
                "loss": float(loss.detach()),
                "contrast_loss": contrast_loss,
                "qk_gradient_l1_by_layer": gradient_by_layer,
                "attention_reads": len(trace),
                "regional_role_records": contrast_records,
            }
            records.append(record)
            write(output_dir / "training.json", records)
            write(output_dir.parent.parent / "progress.json", {
                "phase": "training",
                "fold": fold,
                "arm": arm,
                "step": step,
                "total": total_steps,
                "sequence": sequence,
                "epoch": epoch,
            })
            del samples, loss, contrast_records, trace
            gc.collect()
    return records, qk_names, initial_state, banks


def run_sequence(args, model, frames, sequence, native_rows, out):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    reads = []

    def memory_read(module, args, kwargs):
        if args or "memory" not in kwargs:
            raise RuntimeError("Unexpected native SAM2 memory-attention call")
        memory = kwargs["memory"]
        pointer_count = int(kwargs.get("num_obj_ptr_tokens", 0))
        spatial_count = int(memory.shape[0]) - pointer_count
        if memory.ndim != 3 or memory.shape[1] != 1:
            raise RuntimeError("Unexpected memory-attention tensor layout")
        if spatial_count <= 0 or spatial_count % 4096:
            raise RuntimeError("Spatial memory keys do not align to 64x64 frames")
        reads.append({
            "spatial_keys": spatial_count,
            "pointer_keys": pointer_count,
            "layers": [],
        })

    def layer_read(index):
        def hook(module, args, output):
            if not reads:
                raise RuntimeError("Attention layer ran before memory-read trace")
            reads[-1]["layers"].append(index)
        return hook

    handles = [model.memory_attention.register_forward_pre_hook(
        memory_read, with_kwargs=True
    )]
    handles.extend(
        layer.cross_attn_image.register_forward_hook(layer_read(index))
        for index, layer in enumerate(model.memory_attention.layers)
    )
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"regioncontrast_{sequence}_", dir=out) as temp:
            state, actual_prompt = prepare(model, frames, Path(temp))
            if actual_prompt != prompt:
                raise RuntimeError(f"Prompt index changed for {sequence}")
            with torch.inference_mode():
                for frame, object_ids, logits in model.propagate_in_video(
                    state, start_frame_idx=prompt
                ):
                    if object_ids != [1]:
                        raise RuntimeError(f"Unexpected objects in {sequence}/{frame}")
                    if frame == prompt:
                        continue
                    if frame not in native_by_frame:
                        raise RuntimeError(f"Native reference misses {sequence}/{frame}")
                    read_index = len(rows)
                    if read_index >= len(reads):
                        raise RuntimeError(f"Missing memory read for {sequence}/{frame}")
                    read = reads[read_index]
                    if read["layers"] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Incomplete four-layer trace: {sequence}/{frame}")
                    row = score_frame(logits, frames, frame)
                    row.update(
                        native_prediction_sha256=native_by_frame[frame]["prediction_sha256"],
                        attention_read={**read, "frame": frame},
                    )
                    rows.append(row)
                    write(out / "progress.json", {
                        "phase": "evaluation",
                        "sequence": sequence,
                        "frame": frame,
                    })
            model.reset_state(state)
    finally:
        for handle in handles:
            handle.remove()
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Incomplete full propagation for {sequence}")
    if len(reads) != len(rows):
        raise RuntimeError(f"Unmatched memory-read count for {sequence}")
    summary = summarize(rows)
    present = [row for row in rows if row["present"]]
    summary.update({
        key: float(np.mean([row["scores"][key] for row in present]))
        for key in ("iou", "boundary_f")
    })
    summary["re_detections"] = propagation_summary(rows)["returns"]
    return rows, summary


def build_report(result):
    lines = [
        "# RegionContrast-supervised SAM2 memory Q/K — PolypGen sequence CV",
        "",
        "Exploratory sequence-grouped cross-validation; prior PolypGen analyses informed this method, and patient-to-sequence independence is unverified.",
        "",
        "| Sequence | Native SAM2 | Q/K-only | Regional Q/K | Regional − native | Regional − Q/K-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sequence in SEQUENCES:
        native = result["native"][sequence]["present_dice"]
        control = result["qk_only"][sequence]["present_dice"]
        regional = result["regional"][sequence]["present_dice"]
        lines.append(
            f"| {sequence} | {native:.6f} | {control:.6f} | {regional:.6f} | "
            f"{regional-native:+.6f} | {regional-control:+.6f} |"
        )
    lines.extend([
        "",
        "| Arm | Equal-sequence Dice | IoU | Boundary F | GT-empty false-alarm frames |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ("native", *ARMS):
        lines.append(
            f"| {arm} | {result['means'][arm]:.6f} | "
            f"{result['means_iou'][arm]:.6f} | {result['means_boundary_f'][arm]:.6f} | "
            f"{result['false_alarms'][arm]} |"
        )
    lines.extend([
        "",
        "| Fold | Native Dice | Q/K-only Dice | Regional Q/K Dice | Regional − Q/K-only |",
        "|---|---:|---:|---:|---:|",
    ])
    for fold in result["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['native_mean']:.6f} | "
            f"{fold['qk_only_mean']:.6f} | {fold['regional_mean']:.6f} | "
            f"{fold['regional_mean'] - fold['qk_only_mean']:+.6f} |"
        )
    for contrast in ("native", "qk_only"):
        key = "regional_native" if contrast == "native" else "regional_qk_only"
        row = result["statistics"][key]
        lines.extend([
            "",
            f"Regional − {contrast}: {row['mean_difference']:+.6f}, "
            f"sequence-bootstrap 95% CI {row['ci95']}, paired Wilcoxon "
            f"p={row['wilcoxon']['p']:.6g}, wins/losses/ties "
            f"{row['wins']}/{row['losses']}/{row['ties']}.",
        ])
    lines.extend([
        "",
        f"Frozen screens passed: versus native={result['screens']['versus_native']}; "
        f"regional attribution versus Q/K-only={result['screens']['regional_attribution'] }.",
        "The experiment retains dense native memory reads and all pointers; it does not select frames or prune/gate tokens.",
        "",
    ])
    return "\n".join(lines)


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
        if protocol["prior_work"]["RegionContrast"]["revision"] != REGIONCONTRAST_REVISION:
            raise RuntimeError("RegionContrast source revision differs from protocol")
        manifest = ROOT / protocol["data"]["manifest"]
        checkpoint_sha, manifest_sha = digest(args.checkpoint), digest(manifest)
        gate = json.loads((args.gate / "completion.json").read_text())
        if not gate.get("passed") or gate["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("A passing engineering gate for this checkpoint is required")
        if gate["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Gate manifest differs from full-run manifest")
        for path, expected in gate["sources"].items():
            if path in SOURCES and digest(ROOT / path) != expected:
                raise RuntimeError(f"Engineering-gate source changed: {path}")
        sources = {path: digest(ROOT / path) for path in SOURCES}
        repo, sam2_revision = activate_author_repo()
        if sam2_revision != gate["sam2_revision"]:
            raise RuntimeError("SAM2 source revision differs from engineering gate")
        args.method, args.device, args.model_config = "sam2", "cpu", MODEL_CONFIG
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native, native_rows = checked_native_run(
            args.native_run, videos, checkpoint_sha, manifest_sha
        )
        write(args.output_dir / "setup.json", {
            "protocol": protocol,
            "protocol_sha256": digest(PROTOCOL),
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": sam2_revision,
            "regioncontrast_revision": REGIONCONTRAST_REVISION,
            "engineering_gate": str(args.gate),
            "native_run": str(args.native_run),
            "folds": folds,
            "patient_independence_verified": False,
        })

        outputs = {arm: {} for arm in ARMS}
        training_records = []
        fold_summaries = []
        total_updates = protocol["training"]["epochs"] * 14 * len(folds) * len(ARMS)
        completed_updates = 0
        for fold_index, validation in enumerate(folds, start=1):
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Fold training/validation split is invalid")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            initial_states = {}
            schedules = {}
            for arm in ARMS:
                torch.manual_seed(protocol["training"]["seed"] + fold_index)
                model = build_author_predictor(args, repo)
                arm_dir = fold_dir / arm
                arm_dir.mkdir()
                records, qk_names, initial_state, banks = train_arm(
                    model, videos, training, protocol, arm, arm_dir, fold_index
                )
                initial_states[arm] = initial_state
                schedules[arm] = [
                    (row["epoch"], row["sequence"], row["seed"], tuple(row["clip_frames"]))
                    for row in records
                ]
                training_records.extend(records)
                qk_path = arm_dir / "qk_state.pt"
                save_tensor({
                    name: parameter.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if name in set(qk_names)
                }, qk_path)
                if banks is not None:
                    save_tensor(
                        [bank.state_dict() for bank in banks],
                        arm_dir / "regioncontrast_queues.pt",
                    )
                write(arm_dir / "training_complete.json", {
                    "complete": True,
                    "updates": len(records),
                    "arm": arm,
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "trainable_qk_parameters": qk_names,
                    "qk_state_sha256": digest(qk_path),
                    "regioncontrast_queue_sha256": digest(arm_dir / "regioncontrast_queues.pt")
                    if banks is not None else None,
                })
                if arm == "regional" and not _match_initial_states(
                    initial_states["qk_only"], initial_state
                ):
                    raise RuntimeError("Q/K arms did not share identical initialization")
                if arm == "regional" and schedules["qk_only"] != schedules[arm]:
                    raise RuntimeError("Q/K arms received different sampled training clips")

                model.eval()
                for sequence in validation:
                    rows, summary = run_sequence(
                        args, model, videos[sequence], sequence,
                        native_rows[sequence], args.output_dir,
                    )
                    outputs[arm][sequence] = summary
                    frames_path = arm_dir / f"{sequence}_frames.json"
                    write(frames_path, rows)
                    write(arm_dir / f"{sequence}_complete.json", {
                        "complete": True,
                        "frames_sha256": digest(frames_path),
                        "native_sha256": digest(
                            args.native_run / "native" / f"{sequence}_frames.json"
                        ),
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
                completed_updates += len(records)
                write(args.output_dir / "progress.json", {
                    "phase": "fold_arm_complete",
                    "fold": fold_index,
                    "arm": arm,
                    "training_updates": completed_updates,
                    "total_updates": total_updates,
                })
                del model, banks
                gc.collect()

            fold_summaries.append({
                "fold": fold_index,
                "training_sequences": training,
                "validation_sequences": validation,
                "native_mean": float(np.mean([
                    native[name]["present_dice"] for name in validation
                ])),
                "qk_only_mean": float(np.mean([
                    outputs["qk_only"][name]["present_dice"] for name in validation
                ])),
                "regional_mean": float(np.mean([
                    outputs["regional"][name]["present_dice"] for name in validation
                ])),
            })
            write(fold_dir / "completion.json", {
                "complete": True,
                "fold": fold_index,
                "training_sequences": training,
                "validation_sequences": validation,
            })

        means = {
            arm: float(np.mean([results[name]["present_dice"] for name in SEQUENCES]))
            for arm, results in (("native", native), *outputs.items())
        }
        means_iou = {
            arm: float(np.mean([results[name]["iou"] for name in SEQUENCES]))
            for arm, results in (("native", native), *outputs.items())
        }
        means_boundary_f = {
            arm: float(np.mean([results[name]["boundary_f"] for name in SEQUENCES]))
            for arm, results in (("native", native), *outputs.items())
        }
        false_alarms = {
            arm: sum(results[name]["gt_empty_false_alarm_count"] for name in SEQUENCES)
            for arm, results in (("native", native), *outputs.items())
        }
        regional_native = {
            name: outputs["regional"][name]["present_dice"] - native[name]["present_dice"]
            for name in SEQUENCES
        }
        regional_qk = {
            name: outputs["regional"][name]["present_dice"] - outputs["qk_only"][name]["present_dice"]
            for name in SEQUENCES
        }

        def contrast_stats(values, right):
            left_values = {
                name: outputs["regional"][name]["present_dice"] for name in SEQUENCES
            }
            right_values = {
                name: right[name]["present_dice"] for name in SEQUENCES
            }
            wins = sum(value > 0 for value in values.values())
            losses = sum(value < 0 for value in values.values())
            return {
                "mean_difference": float(np.mean(list(values.values()))),
                "ci95": sequence_bootstrap_interval(
                    values, seed=protocol["training"]["seed"]
                ),
                "wilcoxon": paired_wilcoxon(left_values, right_values),
                "wins": wins,
                "losses": losses,
                "ties": len(SEQUENCES) - wins - losses,
            }

        stats = {
            "regional_native": contrast_stats(regional_native, native),
            "regional_qk_only": contrast_stats(regional_qk, outputs["qk_only"]),
        }
        evaluation = protocol["evaluation"]
        native_screen = (
            means["regional"] - means["native"] >= 0.01
            and stats["regional_native"]["wins"] >= 14
            and false_alarms["regional"] <= false_alarms["native"]
        )
        attribution_screen = (
            means["regional"] - means["qk_only"] >= 0.01
            and stats["regional_qk_only"]["wins"] >= 14
            and false_alarms["regional"] <= false_alarms["qk_only"]
        )
        result = {
            "complete": True,
            "scope": protocol["data"]["independence"],
            "native": native,
            "qk_only": outputs["qk_only"],
            "regional": outputs["regional"],
            "means": means,
            "means_iou": means_iou,
            "means_boundary_f": means_boundary_f,
            "false_alarms": false_alarms,
            "delta_by_sequence_regional_native": regional_native,
            "delta_by_sequence_regional_qk_only": regional_qk,
            "statistics": stats,
            "folds": fold_summaries,
            "training_updates_per_arm": len(training_records) // 2,
            "screens": {
                "versus_native": native_screen,
                "regional_attribution": attribution_screen,
                "frozen_native_rule": evaluation["screen_vs_native"],
                "frozen_attribution_rule": evaluation["region_attribution_screen"],
            },
            "all_native_frames_and_tokens_retained": True,
            "patient_independence_verified": False,
        }
        if sources != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError("Source or frozen protocol changed during CV")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(build_report(result) + "\n")
        write(args.output_dir / "completion.json", {
            "complete": True,
            "sequences": len(SEQUENCES),
            "folds": len(folds),
            "arms": list(ARMS),
            "training_updates_per_arm": len(training_records) // 2,
            "screens": result["screens"],
            "sources": sources,
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
