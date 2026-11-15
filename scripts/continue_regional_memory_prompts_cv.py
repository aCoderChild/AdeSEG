"""Continue an interrupted, source-frozen regional memory-prompt CV run."""

import argparse
import gc
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from datasets.sam2_clip_sampler import sample_clip
from evaluation.statistics import holm_adjust
from modeling.native_history_cache import capture_cached_block, load_bank
from modeling.regional_memory_prompts import RegionalMemoryPrompts
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_prediction_conditioned_affinity_cv import SEQUENCES, summarize_arm
from scripts.run_regional_memory_prompts_cv import PROTOCOL, schedule, train_step
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import paired_delta, summarize
from scripts.verify_memory_prompt_propagation import evaluate


def atomic_torch_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_sources(setup):
    sources = setup["sources"]
    for relative, expected in sources.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
        snapshot = Path(setup["run_dir"]) / "source_snapshot" / relative
        if not snapshot.is_file() or digest(snapshot) != expected:
            raise RuntimeError(f"Frozen source snapshot changed: {relative}")
    if digest(ROOT / PROTOCOL) != sources[PROTOCOL]:
        raise RuntimeError("Current protocol differs from the frozen run")


def validate_prerequisites(args, setup):
    protocol = setup["protocol"]
    verify_sources(setup)
    if digest(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint digest differs from the frozen run")
    native_completion = args.native_run / "completion.json"
    gate_completion = args.propagation_gate / "completion.json"
    history_completion = args.history_gate / "completion.json"
    if digest(native_completion) != setup["native_sha256"]:
        raise RuntimeError("Native completion receipt differs from the frozen run")
    if digest(gate_completion) != setup["propagation_sha256"]:
        raise RuntimeError("Propagation gate differs from the frozen run")
    if digest(history_completion) != setup["history_sha256"]:
        raise RuntimeError("History gate differs from the frozen run")
    native = json.loads(native_completion.read_text())
    gate = json.loads(gate_completion.read_text())
    history = json.loads(history_completion.read_text())
    native_setup = json.loads((args.native_run / "setup.json").read_text())
    audit = json.loads((args.native_run / "independent_audit.json").read_text())
    if (not native.get("passed") or native.get("scope") != "cohort" or not audit.get("passed")
            or not gate.get("passed") or not gate.get("engineering_only")
            or not history.get("passed") or not history.get("engineering_only")):
        raise RuntimeError("A frozen native or engineering prerequisite no longer passes")
    for name, expected in gate.get("artifacts", {}).items():
        if digest(args.propagation_gate / name) != expected:
            raise RuntimeError(f"Propagation artifact changed: {name}")
    history_audit_path = args.history_gate / "independent_receipt_audit.json"
    history_audit = json.loads(history_audit_path.read_text())
    if (not history_audit.get("passed")
            or history_audit.get("completion_sha256") != digest(history_completion)):
        raise RuntimeError("Independent native-history receipt audit changed or failed")
    if (native_setup["checkpoint_sha256"] != protocol["checkpoint_sha256"]
            or native_setup["model_config"] != protocol["model_config"]
            or native_setup["manifest_sha256"] != digest(ROOT / protocol["manifest"])):
        raise RuntimeError("Native model or manifest identity differs")
    if set(native["entries"]) != set(SEQUENCES) or set(audit["sequences"]) != set(SEQUENCES):
        raise RuntimeError("Native reference coverage differs")
    if any(history.get("sources", {}).get(path) != sha for path, sha in gate["sources"].items()):
        raise RuntimeError("History gate does not preserve all propagation-gate sources")
    if history.get("sources") != setup["sources"]:
        raise RuntimeError("Frozen run and engineering-gate source inventories differ")
    repo, _ = activate_author_repo()
    return native, audit, native_setup, repo


def read_native(native, audit):
    references, receipts = {}, {}
    for name, entry in native["entries"].items():
        folder = Path(entry)
        if (digest(folder / "receipt.json") != audit["sequences"][name]["receipt_sha256"]
                or digest(folder / "completion.json") != audit["sequences"][name]["completion_sha256"]):
            raise RuntimeError(f"Native receipt changed: {name}")
        complete = json.loads((folder / "completion.json").read_text())
        if digest(folder / "frames.json") != complete["frames_sha256"]:
            raise RuntimeError(f"Native trajectory changed: {name}")
        references[name] = json.loads((folder / "frames.json").read_text())
        receipts[name] = json.loads((folder / "receipt.json").read_text())
    return references, receipts


def validate_completed_arm(folder, protocol, validation):
    complete_path = folder / "training_completion.json"
    weights_path = folder / "prompts.pt"
    training_path = folder / "training.json"
    if not all(path.is_file() for path in (complete_path, weights_path, training_path)):
        return None
    complete = json.loads(complete_path.read_text())
    if (complete.get("updates") != protocol["updates_per_fold_per_arm"]
            or complete.get("validation") != validation
            or digest(weights_path) != complete.get("weights_sha256")
            or digest(training_path) != complete.get("training_sha256")):
        raise RuntimeError(f"Completed training receipt invalid: {folder}")
    records = json.loads(training_path.read_text())
    if len(records) != protocol["updates_per_fold_per_arm"]:
        raise RuntimeError(f"Training log coverage invalid: {folder}")
    return complete


def validate_completed_sequence(folder, name, training_receipt, expected_frames,
                                tokens_per_region):
    frames_path = folder / f"{name}_frames.json"
    receipt_path = folder / f"{name}_completion.json"
    if not receipt_path.exists() and not frames_path.exists():
        return None
    if not receipt_path.is_file() or not frames_path.is_file():
        raise RuntimeError(f"Partial sequence output needs manual review: {folder}/{name}")
    receipt = json.loads(receipt_path.read_text())
    if (digest(frames_path) != receipt.get("frames_sha256")
            or digest(folder / "training_completion.json") != receipt.get("training_receipt_sha256")):
        raise RuntimeError(f"Sequence output receipt invalid: {folder}/{name}")
    if receipt.get("training_receipt_sha256") != digest(folder / "training_completion.json"):
        raise RuntimeError(f"Sequence was evaluated with different learned weights: {folder}/{name}")
    rows = json.loads(frames_path.read_text())
    if [row.get("frame") for row in rows] != expected_frames:
        raise RuntimeError(f"Sequence frame coverage differs from native manifest: {folder}/{name}")
    for row in rows:
        layers = row.get("attention_layers", [])
        reads = row.get("auxiliary_reads", [])
        if ([layer.get("layer") for layer in layers] != [0, 1, 2, 3]
                or [read.get("layer") for read in reads] != [0, 1, 2, 3]):
            raise RuntimeError(f"Sequence is missing a memory-attention layer: {folder}/{name}")
        for layer, read in zip(layers, reads, strict=True):
            expected_aux = len(read["memory_frames"]) * 3 * tokens_per_region
            if (read["frame"] != row["frame"] or read["object_batch"] != 1
                    or read["auxiliary_keys"] != expected_aux
                    or layer["keys"] != read["spatial_keys"] + read["pointer_keys"]
                    or layer["pointer_keys"] != read["pointer_keys"]
                    or any(span["frame"] >= row["frame"] for span in read["memory_frames"])):
                raise RuntimeError(f"Saved attention receipt is invalid: {folder}/{name}/{row['frame']}")
    recomputed = summarize(rows)
    for key, value in recomputed.items():
        actual = receipt["summary"].get(key)
        if actual is None and value is None:
            continue
        if actual is None or value is None:
            raise RuntimeError(f"Saved summary differs from frame records: {folder}/{name}/{key}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            equal = isinstance(actual, (int, float)) and abs(actual - value) <= 1e-12
        else:
            equal = actual == value
        if not equal:
            raise RuntimeError(f"Saved summary differs from frame records: {folder}/{name}/{key}")
    return receipt["summary"]


def train_arm(out, fold, mode, folder, model, heads, backbone, protocol, videos,
              native, receipts, sources, initial_sha, validation):
    updates = schedule(protocol, fold)
    train_receipt = validate_completed_arm(folder, protocol, validation)
    if train_receipt:
        heads.load_state_dict(torch.load(folder / "prompts.pt", map_location="cpu", weights_only=True))
        heads.eval()
        if parameter_digest(model) != backbone or train_receipt["original_sam2_sha256"] != backbone:
            raise RuntimeError("Frozen SAM2 weights changed before evaluation")
        return heads, train_receipt

    training_path = folder / "training.json"
    checkpoint_path = folder / "latest_training.pt"
    logs = json.loads(training_path.read_text()) if training_path.is_file() else []
    start = 0
    optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required_state = {"rng_state", "python_rng_state", "numpy_rng_state", "initial_sha256"}
        if state.get("sources") != sources or not required_state.issubset(state):
            raise RuntimeError(f"Training checkpoint is not safely resumable: {folder}")
        if state["initial_sha256"] != initial_sha:
            raise RuntimeError(f"Training checkpoint initialization differs: {folder}")
        start = int(state["updates"])
        if start <= 0 or start > len(updates) or start > len(logs) or start % 14:
            raise RuntimeError(f"Training checkpoint/log alignment is invalid: {folder}")
        for index, record in enumerate(logs[:start]):
            epoch, name, seed = updates[index]
            if (record.get("epoch"), record.get("sequence"), record.get("seed")) != (epoch, name, seed):
                raise RuntimeError(f"Saved training log schedule differs at update {index + 1}")
        logs = logs[:start]
        write(training_path, logs)
        heads.load_state_dict(state["prompts"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
    elif logs:
        write(folder / "uncheckpointed_training.json", logs)
        logs = []
        write(training_path, logs)

    if parameter_digest(heads) != initial_sha and start == 0:
        raise RuntimeError(f"Matched initial prompt tensors differ: fold {fold}/{mode}")
    for index in range(start, len(updates)):
        epoch, name, seed = updates[index]
        clip = sample_clip(videos[name], protocol["clip_length"], seed)
        samples, capture = capture_cached_block(model, videos[name],
            [frame.frame_index for frame in clip], out,
            load_bank(Path(native["entries"][name]) / "bank.pt", receipts[name],
                protocol["checkpoint_sha256"], digest(ROOT / protocol["manifest"]), protocol["model_config"]))
        before = [fingerprint(sample["output_dict"])["sha256"] for sample in samples]
        record = train_step(model, heads, samples, mode, optimizer, protocol)
        if (before != [fingerprint(sample["output_dict"])["sha256"] for sample in samples]
                or parameter_digest(model) != backbone):
            raise RuntimeError("Training changed the native teacher bank or frozen SAM2 weights")
        logs.append({"epoch": epoch, "sequence": name, "seed": seed, "capture": capture, **record})
        write(training_path, logs)
        write(out / "progress.json", {"phase": "training", "fold": fold, "mode": mode,
            "update": index + 1, "total": len(updates), "sequence": name, "epoch": epoch})
        if (index + 1) % 14 == 0:
            atomic_torch_save({"prompts": heads.state_dict(), "optimizer": optimizer.state_dict(),
                "updates": index + 1, "sources": sources, "rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                "initial_sha256": initial_sha}, checkpoint_path)
        del samples
        gc.collect()

    atomic_torch_save(heads.state_dict(), folder / "prompts.pt")
    anchor = {"weights_sha256": digest(folder / "prompts.pt"),
        "training_sha256": digest(training_path), "updates": len(logs),
        "initial_sha256": initial_sha, "original_sam2_sha256": backbone,
        "training": sorted({name for _, name, _ in updates}), "validation": validation,
        "selection": "Final epoch anchored before validation"}
    write(folder / "training_completion.json", anchor)
    heads.eval()
    return heads, anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--propagation-gate", type=Path, required=True)
    parser.add_argument("--history-gate", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    out = validate_output(args.run_dir)
    setup_path = out / "setup.json"
    setup = json.loads(setup_path.read_text())
    setup["run_dir"] = str(out)
    protocol = setup["protocol"]
    native, audit, native_setup, repo = validate_prerequisites(args, setup)
    references, receipts = read_native(native, audit)
    manifest_sha = digest(ROOT / protocol["manifest"])
    videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
              **load_manifest(ROOT / protocol["manifest"], "test")}
    plan = []
    for fold, validation in enumerate(protocol["folds"], 1):
        for mode in protocol["arms"]:
            folder = out / f"fold{fold}" / mode
            training_done = validate_completed_arm(folder, protocol, validation) is not None
            completed_sequences = []
            if training_done:
                training_receipt = json.loads((folder / "training_completion.json").read_text())
                for name in validation:
                    expected = list(range(prompt_index(videos[name],
                        {"initialization": "first_labeled_frame"}) + 1, len(videos[name])))
                    if validate_completed_sequence(folder, name, training_receipt, expected,
                                                   protocol["tokens_per_region_per_frame"]):
                        completed_sequences.append(name)
            plan.append({"fold": fold, "mode": mode, "training_complete": training_done,
                         "validation_complete": completed_sequences,
                         "validation_remaining": [n for n in validation if n not in completed_sequences]})
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    continuation_sources = {p: digest(ROOT / p) for p in setup["sources"]}
    write(out / "continuation_setup.json", {"script_sha256": digest(Path(__file__),),
        "frozen_setup_sha256": digest(setup_path), "protocol_sha256": digest(ROOT / PROTOCOL),
        "native_sha256": digest(args.native_run / "completion.json"),
        "propagation_sha256": digest(args.propagation_gate / "completion.json"),
        "history_sha256": digest(args.history_gate / "completion.json"),
        "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": manifest_sha,
        "source_hashes_rechecked": continuation_sources,
        "action": "Resume only missing validation and folds; never retrain completed arms"})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.history_gate and json.loads((args.history_gate / "completion.json").read_text())["sources"] != continuation_sources:
        raise RuntimeError("Current frozen source inventory differs from the history gate")

    results = {"native": {n: summarize(rows) for n, rows in references.items()},
               "regional": {}, "global": {}}
    write(out / "progress.json", {"phase": "continuation_started", "plan": plan})
    try:
        for fold, validation in enumerate(protocol["folds"], 1):
            initial_sha = None
            for mode in protocol["arms"]:
                folder = out / f"fold{fold}" / mode
                folder.mkdir(parents=True, exist_ok=True)
                model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
                    checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
                model.requires_grad_(False)
                backbone = parameter_digest(model)
                torch.manual_seed(protocol["seed"] + fold)
                heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"])
                if sum(p.numel() for p in heads.parameters()) != protocol["trainable_parameters"]:
                    raise RuntimeError("Auxiliary trainable scope differs from frozen protocol")
                initial = parameter_digest(heads)
                if initial_sha is not None and initial != initial_sha:
                    raise RuntimeError("Matched regional/global initialization differs")
                initial_sha = initial
                heads, training_receipt = train_arm(out, fold, mode, folder, model, heads,
                    backbone, protocol, videos, native, receipts, setup["sources"], initial_sha, validation)
                if training_receipt["initial_sha256"] != initial_sha:
                    raise RuntimeError(f"Training initialization receipt differs: fold {fold}/{mode}")
                if model.training or any(p.requires_grad for p in model.parameters()):
                    raise RuntimeError("SAM2 is not frozen in eval mode")
                for name in validation:
                    expected = list(range(prompt_index(videos[name],
                        {"initialization": "first_labeled_frame"}) + 1, len(videos[name])))
                    existing = validate_completed_sequence(folder, name, training_receipt, expected,
                                                           protocol["tokens_per_region_per_frame"])
                    if existing is not None:
                        results[mode][name] = existing
                        continue
                    rows, summary = evaluate(model, videos[name], name, references[name], out,
                                             heads, mode)
                    summary["inference_wall_seconds"] = summary.pop("wall_seconds")
                    results[mode][name] = summary
                    write(folder / f"{name}_frames.json", rows)
                    write(folder / f"{name}_completion.json", {"summary": summary,
                        "frames_sha256": digest(folder / f"{name}_frames.json"),
                        "training_receipt_sha256": digest(folder / "training_completion.json")})
                write(folder / "completion.json", {"complete": True, **training_receipt})
                if parameter_digest(model) != backbone:
                    raise RuntimeError("Validation changed frozen SAM2 weights")
                del model, heads
                gc.collect()
                write(out / "progress.json", {"phase": "fold_complete", "fold": fold, "mode": mode})

        for fold, validation in enumerate(protocol["folds"], 1):
            for mode in protocol["arms"]:
                folder = out / f"fold{fold}" / mode
                receipt = json.loads((folder / "training_completion.json").read_text())
                for name in validation:
                    expected = list(range(prompt_index(videos[name],
                        {"initialization": "first_labeled_frame"}) + 1, len(videos[name])))
                    summary = validate_completed_sequence(folder, name, receipt, expected,
                                                         protocol["tokens_per_region_per_frame"])
                    if summary is None:
                        raise RuntimeError(f"Missing final sequence result: fold{fold}/{mode}/{name}")
                    results[mode][name] = summary

        for fold, validation in enumerate(protocol["folds"], 1):
            for name in validation:
                regional = json.loads((out / f"fold{fold}" / "regional" / f"{name}_frames.json").read_text())
                global_rows = json.loads((out / f"fold{fold}" / "global" / f"{name}_frames.json").read_text())
                if len(regional) != len(global_rows):
                    raise RuntimeError(f"Regional/global frame counts differ: {name}")
                for left, right in zip(regional, global_rows, strict=True):
                    if left["frame"] != right["frame"]:
                        raise RuntimeError(f"Regional/global frame order differs: {name}")
                    for left_read, right_read in zip(left["auxiliary_reads"], right["auxiliary_reads"], strict=True):
                        for key in ("layer", "spatial_keys", "pointer_keys", "auxiliary_keys", "memory_frames"):
                            if left_read[key] != right_read[key]:
                                raise RuntimeError(f"Regional/global memory schedule differs: {name}/{left['frame']}")

        verify_sources(setup)
        if digest(args.checkpoint) != protocol["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during the continuation")

        result = {**results, "summary": {mode: {**summarize_arm(values),
            "gt_empty_targets": sum(row["gt_empty_targets"] for row in values.values())}
            for mode, values in results.items()}}
        contrasts = {f"regional_vs_{other}": paired_delta(results, "regional", other, protocol["seed"])
                     for other in ("native", "global")}
        result.update(contrasts)
        result["holm_p"] = holm_adjust({key: value["wilcoxon"]["p"]
                                        for key, value in contrasts.items()})
        result["meaningful_signal_screen"] = (
            contrasts["regional_vs_native"]["mean_delta"] >= .01
            and contrasts["regional_vs_native"]["wins"] >= 14
            and contrasts["regional_vs_global"]["mean_delta"] > 0
            and all(value["ci95"][0] > 0 and result["holm_p"][key] < .05
                    for key, value in contrasts.items())
            and result["summary"]["regional"]["gt_empty_false_alarm_count"]
                <= result["summary"]["native"]["gt_empty_false_alarm_count"])
        write(out / "results.json", result)
        lines = ["# Frozen-SAM2 regional memory prompts — exploratory CV", "",
            "| Sequence | Native Dice | Global Dice | Regional Dice |", "|---|---:|---:|---:|"]
        for name in SEQUENCES:
            lines.append("| " + name + " | " + " | ".join(
                f"{results[mode][name]['present_dice']:.6f}" for mode in ("native", "global", "regional")) + " |")
        lines += ["", "| Arm | Present Dice | IoU | Boundary F | GT-empty false alarms | Runtime (s) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for mode, row in result["summary"].items():
            runtime = "N/A" if row["inference_wall_seconds"] is None else f"{row['inference_wall_seconds']:.1f}"
            lines.append(f"| {mode} | {row['present_dice']:.6f} | {row['iou']:.6f} | {row['boundary_f']:.6f} | "
                         f"{row['gt_empty_false_alarm_count']} | {runtime} |")
        lines += ["", f"Meaningful improvement screen: {result['meaningful_signal_screen']}.",
            f"Regional − native Dice: {contrasts['regional_vs_native']['mean_delta']:+.6f}; "
            f"wins/losses/ties {contrasts['regional_vs_native']['wins']}/"
            f"{contrasts['regional_vs_native']['losses']}/{contrasts['regional_vs_native']['ties']}; "
            f"95% CI {contrasts['regional_vs_native']['ci95']}; "
            f"Holm-adjusted p={result['holm_p']['regional_vs_native']:.6g}.",
            f"Regional − global Dice: {contrasts['regional_vs_global']['mean_delta']:+.6f}; "
            f"wins/losses/ties {contrasts['regional_vs_global']['wins']}/"
            f"{contrasts['regional_vs_global']['losses']}/{contrasts['regional_vs_global']['ties']}; "
            f"95% CI {contrasts['regional_vs_global']['ci95']}; "
            f"Holm-adjusted p={result['holm_p']['regional_vs_global']:.6g}.", ""]
        (out / "results.md").write_text("\n".join(lines))
        write(out / "completion.json", {"complete": True,
            "continuation_setup_sha256": digest(out / "continuation_setup.json"),
            "results_sha256": digest(out / "results.json")})
        write(out / "progress.json", {"phase": "complete"})
    except BaseException as error:
        write(out / "continuation_failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
