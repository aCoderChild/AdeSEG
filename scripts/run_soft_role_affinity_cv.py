"""Train and compare soft role-conditioned dense SAM2 memory reads."""

import argparse
import copy
import gc
import json
import shutil
import sys
import time
import tempfile
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
from modeling.native_region_readout import NativeRegionalReadout
from modeling.soft_role_affinity import SoftRoleAffinityHeads, mask_logits_to_soft_roles
from modeling.soft_role_attention_training import clip_loss
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import save_tensor, write
from scripts.verify_on_demand_clip import capture_clip
from scripts.verify_soft_role_affinity import MODEL_CONFIG, SOURCES as GATE_SOURCES

PROTOCOL = ROOT / "protocols/soft_role_affinity_polypgen_v1.json"
SEQUENCES = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
BASE_SOURCES = [
    "protocols/soft_role_affinity_polypgen_v1.json",
    "scripts/run_soft_role_affinity_cv.py",
    "scripts/verify_soft_role_affinity.py",
    "modeling/soft_role_affinity.py",
    "modeling/soft_role_attention_training.py",
    "modeling/native_region_readout.py",
    "modeling/region_readout.py",
    "modeling/region_mass_attention.py",
    "modeling/cfbi_region_gate.py",
    "modeling/cfbi_training.py",
    "modeling/decoder_training_mask.py",
    "modeling/regional_refinement.py",
    "datasets/sam2_clip_sampler.py",
    "datasets/video.py",
    "datasets/io.py",
    "evaluation/segmentation.py",
    "evaluation/davis2017/metrics.py",
    "evaluation/statistics.py",
    "scripts/false_alarm_statistics.py",
    "scripts/memory_token_io.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_regional_refinement.py",
    "scripts/verify_on_demand_clip.py",
    "external/HODOR/hodor/modelling/encoder/soft_masked_attention.py",
]


def verify_folds(folds):
    flat = [name for fold in folds for name in fold]
    if len(folds) != 3 or any(len(fold) != 7 for fold in folds):
        raise RuntimeError("Expected three seven-sequence folds")
    if len(set(flat)) != 21 or set(flat) != set(SEQUENCES):
        raise RuntimeError("Folds must partition all promptable sequences exactly once")


def summarize(rows):
    summary = propagation_summary(rows)
    present = [row for row in rows if row["present"]]
    summary.update({
        key: float(np.mean([row["scores"][key] for row in present]))
        for key in ("iou", "boundary_f")
    })
    return summary


def score_frame(logits, frames, frame):
    from datasets.io import load_label_mask

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


def load_native_baseline(path, videos, checkpoint_sha, manifest_sha):
    setup = json.loads((path / "setup.json").read_text())
    completion = json.loads((path / "completion.json").read_text())
    if not completion.get("complete") or completion.get("sequences") != len(SEQUENCES):
        raise RuntimeError("Native baseline is not complete for all promptable sequences")
    if setup.get("checkpoint_sha256") != checkpoint_sha or setup.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Native baseline checkpoint or manifest differs")
    if setup.get("sam2_revision") != "2b90b9f5ceec907a1c18123530e92e794ad901a4":
        raise RuntimeError("Native baseline SAM2 checkout differs")
    summaries, records = {}, {}
    for name in SEQUENCES:
        frame_path = path / "native" / f"{name}_frames.json"
        receipt_path = path / "native" / f"{name}_complete.json"
        receipt = json.loads(receipt_path.read_text())
        if digest(frame_path) != receipt["frames_sha256"]:
            raise RuntimeError(f"Native baseline digest mismatch: {name}")
        rows = json.loads(frame_path.read_text())
        prompt = prompt_index(videos[name], {"initialization": "first_labeled_frame"})
        if receipt.get("prompt") != prompt:
            raise RuntimeError(f"Native prompt mismatch: {name}")
        if [row["frame"] for row in rows] != list(range(prompt + 1, len(videos[name]))):
            raise RuntimeError(f"Native frame coverage mismatch: {name}")
        summaries[name], records[name] = summarize(rows), rows
    return summaries, records


def restore_completed_folds(prior_run, output_dir, folds, videos, native_path,
                            native, native_rows, setup):
    prior_setup = json.loads((prior_run / "setup.json").read_text())
    for key in (
        "protocol_sha256", "checkpoint_sha256", "manifest_sha256", "sam2_revision",
        "model_config", "hodor_revision", "gate_completion_sha256",
        "native_setup_sha256", "native_run",
    ):
        if prior_setup.get(key) != setup.get(key):
            raise RuntimeError(f"Prior run differs in {key}")
    if set(prior_setup.get("sources", {})) != set(setup["sources"]):
        raise RuntimeError("Prior run source inventory differs")
    runner = "scripts/run_soft_role_affinity_cv.py"
    changed_sources = [
        path for path, sha in setup["sources"].items()
        if prior_setup["sources"].get(path) != sha
    ]
    if changed_sources != [runner]:
        raise RuntimeError(f"Prior run sources differ: {changed_sources}")

    restored = {"global": {}, "regional": {}}
    summaries = []
    training_records = []
    receipts = []
    for fold_index in (1, 2):
        source = prior_run / f"fold{fold_index}"
        destination = output_dir / f"fold{fold_index}"
        expected_validation = folds[fold_index - 1]
        completion = json.loads((source / "completion.json").read_text())
        if (not completion.get("complete")
                or completion.get("fold") != fold_index
                or completion.get("training_updates_per_arm") != 140
                or completion.get("validation_sequences") != expected_validation):
            raise RuntimeError(f"Prior fold {fold_index} is not complete under this protocol")
        records = json.loads((source / "training.json").read_text())
        if (len(records) != 140
                or any(row.get("fold") != fold_index
                       or set(row.get("arms", {})) != {"regional", "global"}
                       for row in records)):
            raise RuntimeError(f"Prior fold {fold_index} training log is incomplete")
        for mode in ("regional", "global"):
            training_complete = json.loads(
                (source / f"{mode}_training_complete.json").read_text()
            )
            if (training_complete.get("updates") != 140
                    or training_complete.get("validation_sequences") != expected_validation
                    or training_complete.get("layers_with_gradients") != [0, 1, 2, 3]):
                raise RuntimeError(f"Prior fold {fold_index} {mode} training receipt is invalid")

        for name in expected_validation:
            native_by_frame = {row["frame"]: row for row in native_rows[name]}
            prompt = prompt_index(videos[name], {"initialization": "first_labeled_frame"})
            expected_frames = list(range(prompt + 1, len(videos[name])))
            native_file = native_path / "native" / f"{name}_frames.json"
            for mode in ("regional", "global"):
                frames_path = source / f"{name}_{mode}_frames.json"
                receipt_path = source / f"{name}_{mode}_complete.json"
                receipt = json.loads(receipt_path.read_text())
                if digest(frames_path) != receipt.get("frames_sha256"):
                    raise RuntimeError(f"Prior frame-file digest mismatch: {name}/{mode}")
                if digest(native_file) != receipt.get("native_sha256"):
                    raise RuntimeError(f"Prior native-file digest mismatch: {name}/{mode}")
                rows = json.loads(frames_path.read_text())
                if [row.get("frame") for row in rows] != expected_frames:
                    raise RuntimeError(f"Prior frame coverage mismatch: {name}/{mode}")
                if receipt.get("fold") != fold_index:
                    raise RuntimeError(f"Prior fold receipt mismatch: {name}/{mode}")
                for row in rows:
                    native_row = native_by_frame.get(row["frame"])
                    layers = row.get("attention_layers", [])
                    roles = row.get("query_role_mean", [])
                    if (native_row is None
                            or row.get("native_prediction_sha256") != native_row.get("prediction_sha256")
                            or [layer.get("layer") for layer in layers] != [0, 1, 2, 3]
                            or len(roles) != 4):
                        raise RuntimeError(f"Prior prediction or layer trace mismatch: {name}/{mode}")
                    for layer, role in zip(layers, roles):
                        if (layer.get("spatial_keys") != row["attention"].get("spatial_keys")
                                or layer.get("pointer_keys") != row["attention"].get("pointer_keys")
                                or len(role) != 3
                                or not np.isfinite(role).all()
                                or abs(sum(role) - 1.0) > 1e-4):
                            raise RuntimeError(f"Prior attention receipt invalid: {name}/{mode}")
                restored[mode][name] = summarize(rows)
        training_records.extend(records)
        summaries.append({
            "fold": fold_index,
            "training_sequences": [name for name in SEQUENCES if name not in expected_validation],
            "validation_sequences": expected_validation,
            "training_clips_per_arm": len(records),
            "optimizer_updates_total": 2 * len(records),
            "global": float(np.mean([
                restored["global"][name]["present_dice"] for name in expected_validation
            ])),
            "regional": float(np.mean([
                restored["regional"][name]["present_dice"] for name in expected_validation
            ])),
            "native": float(np.mean([
                native[name]["present_dice"] for name in expected_validation
            ])),
        })
        receipts.append({
            "fold": fold_index,
            "completion_sha256": digest(source / "completion.json"),
            "training_sha256": digest(source / "training.json"),
        })
    for fold_index in (1, 2):
        shutil.copytree(prior_run / f"fold{fold_index}", output_dir / f"fold{fold_index}")
    return restored, summaries, training_records, receipts, prior_setup["sources"][runner]


def run_sequence(model, heads, mode, frames, name, native_rows, output_dir):
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native_by_frame = {row["frame"]: row for row in native_rows}
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"soft_affinity_{name}_", dir=output_dir) as temp:
        state, actual_prompt = prepare(model, frames, Path(temp))
        if actual_prompt != prompt:
            raise RuntimeError(f"Prompt mismatch for {name}")
        started = time.monotonic()
        with NativeRegionalReadout(
            model, heads, mode, name, region_encoder=mask_logits_to_soft_roles
        ) as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError(f"Unexpected object ids for {name}/{frame}")
                if frame == prompt:
                    continue
                native = native_by_frame.get(frame)
                if native is None:
                    raise RuntimeError(f"Native baseline missing {name}/{frame}")
                read = observer.observer.trace[-1]
                layers = observer.calls[-4:]
                if read["frame"] != frame or [row["layer"] for row in layers] != [0, 1, 2, 3]:
                    raise RuntimeError(f"Incomplete four-layer read for {name}/{frame}")
                if any(row["spatial_keys"] != read["spatial_keys"] for row in layers):
                    raise RuntimeError(f"Spatial-key count changed across layers for {name}/{frame}")
                if any(row["pointer_keys"] < 0 for row in layers):
                    raise RuntimeError(f"Invalid pointer count for {name}/{frame}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite prediction for {name}/{frame}")
                row = score_frame(logits, frames, frame)
                row.update(
                    native_prediction_sha256=native["prediction_sha256"],
                    attention=read,
                    attention_layers=layers,
                    query_role_mean=[
                        layer.last_query_roles.detach().mean(dim=(0, 1, 2)).cpu().tolist()
                        for layer in heads.layers
                    ],
                    role_bias=[
                        (layer.max_bias * layer.role_bias.detach().tanh()).cpu().tolist()
                        for layer in heads.layers
                    ],
                )
                rows.append(row)
                write(output_dir / "progress.json", {
                    "phase": "evaluation", "mode": mode,
                    "sequence": name, "frame": frame,
                })
        elapsed = time.monotonic() - started
        model.reset_state(state)
    expected_frames = list(range(prompt + 1, len(frames)))
    if [row["frame"] for row in rows] != expected_frames:
        raise RuntimeError(f"Incomplete trajectory for {name}/{mode}")
    summary = summarize(rows)
    summary["inference_wall_seconds"] = elapsed
    summary["mean_spatial_memory_keys"] = float(np.mean([
        sum(call["spatial_keys"] for call in row["attention_layers"]) / 4 for row in rows
    ]))
    summary["mean_pointer_tokens"] = float(np.mean([
        sum(call["pointer_keys"] for call in row["attention_layers"]) / 4 for row in rows
    ]))
    return rows, summary


def report(result):
    names = {
        "native": "Native SAM2",
        "global": "Globalized-role control",
        "regional": "Soft role-conditioned read",
    }
    lines = [
        "# Soft role-conditioned memory attention — PolypGen exploratory CV",
        "",
        "PolypGen informed method development; patient-to-sequence independence is unverified.",
        "The evaluation is exploratory and does not establish external generalization.",
        "",
        "| Sequence | Native Dice | Globalized-role Dice | Regional Dice | Regional − native | Regional − global |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SEQUENCES:
        native = result["native"][name]["present_dice"]
        global_score = result["global"][name]["present_dice"]
        regional = result["regional"][name]["present_dice"]
        lines.append(
            f"| {name} | {native:.6f} | {global_score:.6f} | {regional:.6f} | "
            f"{regional-native:+.6f} | {regional-global_score:+.6f} |"
        )
    lines += ["", "## Cohort summaries", "", "| Arm | Present Dice | IoU | Boundary F | GT-empty false alarms |",
              "|---|---:|---:|---:|---:|"]
    for arm, title in names.items():
        row = result["summary"][arm]
        lines.append(
            f"| {title} | {row['present_dice']:.6f} | {row['iou']:.6f} | "
            f"{row['boundary_f']:.6f} | {row['gt_empty_false_alarm_count']} |"
        )
    lines += [
        "",
        f"Regional − native Dice: {result['regional_vs_native']['mean_delta']:+.6f}; "
        f"sequence wins/losses/ties {result['regional_vs_native']['wins']}/"
        f"{result['regional_vs_native']['losses']}/{result['regional_vs_native']['ties']}; "
        f"95% sequence-bootstrap interval {result['regional_vs_native']['ci95']}; "
        f"Wilcoxon p={result['regional_vs_native']['wilcoxon']['p']:.6g}.",
        f"Regional − globalized-role Dice: {result['regional_vs_global']['mean_delta']:+.6f}; "
        f"sequence wins/losses/ties {result['regional_vs_global']['wins']}/"
        f"{result['regional_vs_global']['losses']}/{result['regional_vs_global']['ties']}; "
        f"95% sequence-bootstrap interval {result['regional_vs_global']['ci95']}; "
        f"Wilcoxon p={result['regional_vs_global']['wilcoxon']['p']:.6g}.",
        f"Frozen meaningful-improvement screen passed: {result['meaningful_signal_screen']}.",
        "",
        "The globalized-role arm retains the learned query classifier and bias parameters but removes "
        "per-token role location. It controls for a generic role-dependent pointer/spatial read effect.",
        "The three attention branches perform dense reads over the full memory; no frame or key is dropped.",
    ]
    return "\n".join(lines) + "\n"


def paired_delta(result, left, right, seed):
    delta = {
        name: result[left][name]["present_dice"] - result[right][name]["present_dice"]
        for name in SEQUENCES
    }
    wins = sum(value > 0 for value in delta.values())
    losses = sum(value < 0 for value in delta.values())
    ties = len(SEQUENCES) - wins - losses
    ci = sequence_bootstrap_interval(delta, seed=seed)
    wilcoxon = paired_wilcoxon(
        {name: result[left][name]["present_dice"] for name in SEQUENCES},
        {name: result[right][name]["present_dice"] for name in SEQUENCES},
    )
    return {
        "mean_delta": float(np.mean(list(delta.values()))),
        "delta_by_sequence": delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ci95": ci,
        "wilcoxon": wilcoxon,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-folds-from", type=Path)
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
        gate_protocol_sha = gate["source_hashes"].get("protocols/soft_role_affinity_polypgen_v1.json")
        if gate_protocol_sha != digest(PROTOCOL):
            raise RuntimeError("Engineering gate used a different frozen protocol")
        if not all(gate.get(key) for key in (
            "native_zero_bias_replay_bit_identical",
            "spatial_and_pointer_keys_preserved",
            "all_four_layers_observed",
        )):
            raise RuntimeError("Engineering gate did not verify all required invariants")

        sources = {path: digest(ROOT / path) for path in BASE_SOURCES}
        for path, expected in gate["source_hashes"].items():
            if path in sources and sources[path] != expected:
                raise RuntimeError(f"Engineering-gate source changed: {path}")
        hodor_revision = protocol["prior_work"]["HODOR_soft_masked_attention"]["revision"]
        _, _ = __import__("modeling.region_tokens", fromlist=["_verified_source"])._verified_source(
            "HODOR", hodor_revision,
            "hodor/modelling/encoder/soft_masked_attention.py",
        )
        repo, revision = activate_author_repo()
        if revision != gate["sam2_revision"]:
            raise RuntimeError("SAM2 checkout differs from the real-data gate")
        videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
        native_path = args.native_run
        native, native_rows = load_native_baseline(
            native_path, videos, checkpoint_sha, manifest_sha
        )
        setup = {
            "protocol": protocol,
            "protocol_sha256": digest(PROTOCOL),
            "sources": sources,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "sam2_revision": revision,
            "model_config": MODEL_CONFIG,
            "hodor_revision": hodor_revision,
            "engineering_gate": str(args.gate),
            "gate_completion_sha256": digest(args.gate / "completion.json"),
            "native_run": str(native_path),
            "native_setup_sha256": digest(native_path / "setup.json"),
            "folds": folds,
            "patient_independence_verified": False,
        }
        results = {"native": native, "global": {}, "regional": {}}
        fold_summaries, training_records = [], []
        total_updates = 2 * protocol["training"]["epochs"] * 14 * len(folds)
        update = 0
        if args.resume_folds_from:
            prior_run = args.resume_folds_from.resolve()
            if prior_run == args.output_dir.resolve():
                raise RuntimeError("Continuation output must differ from the prior run")
            (restored, fold_summaries, training_records, prior_receipts,
             prior_runner_sha) = restore_completed_folds(
                prior_run, args.output_dir, folds, videos, native_path,
                native, native_rows, setup,
            )
            results.update(restored)
            setup["continuation"] = {
                "prior_run": str(prior_run),
                "completed_folds_reused": [1, 2],
                "fold_receipts": prior_receipts,
                "prior_runner_sha256": prior_runner_sha,
                "current_runner_sha256": sources["scripts/run_soft_role_affinity_cv.py"],
                "fold3_retrained_from_frozen_seed": True,
            }
            update = 2 * protocol["training"]["epochs"] * 14 * 2
            fold_indices = (3,)
        else:
            fold_indices = (1, 2, 3)
        write(args.output_dir / "setup.json", setup)

        for fold_index in fold_indices:
            validation = folds[fold_index - 1]
            training = [name for name in SEQUENCES if name not in validation]
            if len(training) != 14 or set(training) & set(validation):
                raise RuntimeError("Each fold must train on 14 disjoint sequences")
            fold_dir = args.output_dir / f"fold{fold_index}"
            fold_dir.mkdir()
            torch.manual_seed(protocol["training"]["seed"] + fold_index)
            build_args = argparse.Namespace(
                method="sam2", device="cpu", model_config=MODEL_CONFIG,
                checkpoint=args.checkpoint,
            )
            model = build_author_predictor(build_args, repo)
            model.requires_grad_(False)
            attention = model.memory_attention.layers[0].cross_attn_image
            template = SoftRoleAffinityHeads(
                layers=4,
                heads=attention.num_heads,
                head_dim=attention.q_proj.out_features // attention.num_heads,
                max_bias=protocol["training"]["max_attention_bias"],
            )
            initial_state = copy.deepcopy(template.state_dict())
            heads = {
                mode: SoftRoleAffinityHeads(
                    layers=4,
                    heads=attention.num_heads,
                    head_dim=attention.q_proj.out_features // attention.num_heads,
                    max_bias=protocol["training"]["max_attention_bias"],
                ).train()
                for mode in ("regional", "global")
            }
            for head in heads.values():
                head.load_state_dict(initial_state)
            optimizers = {
                mode: torch.optim.AdamW(
                    heads[mode].parameters(),
                    lr=protocol["training"]["learning_rate"],
                    weight_decay=protocol["training"]["weight_decay"],
                )
                for mode in heads
            }
            training_log, seen_layers = [], {mode: set() for mode in heads}
            for epoch in range(protocol["training"]["epochs"]):
                for sequence_index, name in enumerate(training):
                    seed = (
                        protocol["training"]["seed"] + fold_index * 100000
                        + epoch * len(training) + sequence_index
                    )
                    clip = sample_clip(videos[name], protocol["training"]["clip_length"], seed)
                    if prompt_index(clip, {"initialization": "first_labeled_frame"}) != 0:
                        raise RuntimeError(f"Sampled training prompt is not first: {name}")
                    samples = capture_clip(model, clip, args.output_dir)
                    row = {
                        "fold": fold_index,
                        "epoch": epoch,
                        "sequence": name,
                        "seed": seed,
                        "frames": [frame.frame_index for frame in clip],
                        "arms": {},
                    }
                    for mode in ("regional", "global"):
                        optimizer = optimizers[mode]
                        optimizer.zero_grad(set_to_none=True)
                        read_mode = "soft_role_attention" if mode == "regional" else "soft_role_global"
                        loss, records, calls = clip_loss(
                            model, heads[mode], samples, read_mode, name,
                            check_native=(epoch == 0 and sequence_index == 0),
                            role_loss_weight=protocol["training"]["role_loss_weight"],
                        )
                        if len(calls) != 4 * len(samples) or [c["layer"] for c in calls] != [0, 1, 2, 3] * len(samples):
                            raise RuntimeError(f"Missing four-layer reads for fold{fold_index}/{name}")
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss for {mode}/{name}")
                        loss.backward()
                        gradients = []
                        for layer_index, layer in enumerate(heads[mode].layers):
                            for param_name, parameter in (
                                ("bias", layer.role_bias),
                                ("query_role", layer.query_role.weight),
                            ):
                                grad = parameter.grad
                                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
                                    raise RuntimeError(f"Invalid {param_name} gradient in {mode} layer {layer_index}")
                            gradients.append({
                                "role_bias_l1": float(layer.role_bias.grad.abs().sum()),
                                "query_role_l1": float(layer.query_role.weight.grad.abs().sum()),
                            })
                            seen_layers[mode].add(layer_index)
                        if any(parameter.grad is not None for parameter in model.parameters()):
                            raise RuntimeError("Frozen SAM2 acquired gradients")
                        torch.nn.utils.clip_grad_norm_(
                            heads[mode].parameters(), protocol["training"]["max_gradient_norm"]
                        )
                        optimizer.step()
                        if not all(torch.isfinite(parameter).all() for parameter in heads[mode].parameters()):
                            raise RuntimeError(f"Nonfinite update in {mode}")
                        row["arms"][mode] = {
                            "loss": float(loss.detach()),
                            "frame_records": records,
                            "gradient_l1_by_layer": gradients,
                            "calls": len(calls),
                        }
                        update += 1
                        write(args.output_dir / "progress.json", {
                            "phase": "training", "fold": fold_index,
                            "update": update, "total": total_updates,
                            "epoch": epoch, "sequence": name,
                        })
                    training_log.append(row)
                    training_records.append(row)
                    write(fold_dir / "training.json", training_log)
                    del samples, loss, calls
                    gc.collect()

            for mode in heads:
                if seen_layers[mode] != {0, 1, 2, 3}:
                    raise RuntimeError(f"Not all layers trained for {mode}")
                save_tensor(heads[mode].state_dict(), fold_dir / f"{mode}_heads.pt")
                write(fold_dir / f"{mode}_training_complete.json", {
                    "updates": len(training_log),
                    "training_sequences": training,
                    "validation_sequences": validation,
                    "layers_with_gradients": sorted(seen_layers[mode]),
                })

            for name in validation:
                for mode in ("regional", "global"):
                    read_mode = "soft_role_attention" if mode == "regional" else "soft_role_global"
                    rows, summary = run_sequence(
                        model, heads[mode].eval(), read_mode,
                        videos[name], name, native_rows[name], args.output_dir,
                    )
                    results[mode][name] = summary
                    write(fold_dir / f"{name}_{mode}_frames.json", rows)
                    write(fold_dir / f"{name}_{mode}_complete.json", {
                        "frames_sha256": digest(fold_dir / f"{name}_{mode}_frames.json"),
                        "native_sha256": digest(native_path / "native" / f"{name}_frames.json"),
                        "fold": fold_index,
                    })
                write(args.output_dir / "progress.json", {
                    "phase": "evaluation", "fold": fold_index,
                    "sequence": name, "complete": True,
                })

            fold_summaries.append({
                "fold": fold_index,
                "training_sequences": training,
                "validation_sequences": validation,
                "training_clips_per_arm": len(training_log),
                "optimizer_updates_total": 2 * len(training_log),
                **{
                    mode: float(np.mean([
                        results[mode][name]["present_dice"] for name in validation
                    ]))
                    for mode in ("global", "regional")
                },
                "native": float(np.mean([
                    native[name]["present_dice"] for name in validation
                ])),
            })
            write(fold_dir / "completion.json", {
                "complete": True,
                "fold": fold_index,
                "training_updates_per_arm": len(training_log),
                "validation_sequences": validation,
            })
            del model, heads, optimizers, template
            gc.collect()

        summaries = {
            mode: {
                "present_dice": float(np.mean([results[mode][name]["present_dice"] for name in SEQUENCES])),
                "iou": float(np.mean([results[mode][name]["iou"] for name in SEQUENCES])),
                "boundary_f": float(np.mean([results[mode][name]["boundary_f"] for name in SEQUENCES])),
                "gt_empty_false_alarm_count": sum(results[mode][name]["gt_empty_false_alarm_count"] for name in SEQUENCES),
            }
            for mode in ("native", "global", "regional")
        }
        result = {
            "complete": True,
            "native": native,
            "global": results["global"],
            "regional": results["regional"],
            "summary": summaries,
            "regional_vs_native": paired_delta(
                {"regional": results["regional"], "native": results["native"]},
                "regional", "native", protocol["training"]["seed"],
            ),
            "regional_vs_global": paired_delta(
                {"regional": results["regional"], "global": results["global"]},
                "regional", "global", protocol["training"]["seed"] + 1,
            ),
            "folds": fold_summaries,
            "training_clips_per_arm": len(training_records),
            "optimizer_updates_per_arm": len(training_records),
            "optimizer_updates_total": sum(
                len(row["arms"]) for row in training_records
            ),
            "meaningful_signal_screen": False,
            "patient_independence_verified": False,
            "scope": protocol["data"]["scope"],
        }
        regional_fp = summaries["regional"]["gt_empty_false_alarm_count"]
        native_fp = summaries["native"]["gt_empty_false_alarm_count"]
        global_fp = summaries["global"]["gt_empty_false_alarm_count"]
        result["meaningful_signal_screen"] = all(
            result[f"regional_vs_{control}"]["mean_delta"] >= 0.01
            and result[f"regional_vs_{control}"]["wins"] >= 14
            and regional_fp <= control_fp
            for control, control_fp in (("native", native_fp), ("global", global_fp))
        )
        if sources != {path: digest(ROOT / path) for path in BASE_SOURCES}:
            raise RuntimeError("Source or protocol changed during CV")
        write(args.output_dir / "training.json", training_records)
        write(args.output_dir / "results.json", result)
        (args.output_dir / "results.md").write_text(report(result))
        write(args.output_dir / "completion.json", {
            "complete": True,
            "sequences": len(SEQUENCES),
            "folds": len(folds),
            "training_clips_per_arm": len(training_records),
            "optimizer_updates_per_arm": len(training_records),
            "optimizer_updates_total": sum(
                len(row["arms"]) for row in training_records
            ),
            "meaningful_signal_screen": result["meaningful_signal_screen"],
            "sources": sources,
            "continuation": setup.get("continuation"),
        })
        write(args.output_dir / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(args.output_dir / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
