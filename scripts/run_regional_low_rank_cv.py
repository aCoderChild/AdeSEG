"""Gate, train and evaluate dense regional low-rank SAM2 memory transforms."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import holm_adjust
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from modeling.native_history_cache import capture_cached_block, load_bank
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_prediction_conditioned_affinity_cv import SEQUENCES, summarize_arm, verify_folds
from scripts.run_regional_refinement import save_tensor, write
from scripts.run_soft_role_affinity_cv import load_native_baseline, paired_delta, score_frame, summarize
from scripts.verify_on_demand_clip import capture_clip
from training.tracking import start_wandb

PROTOCOL = ROOT / "protocols/regional_low_rank_polypgen_cv_v1.json"


def verified_history_cache(directory, provenance, capacity_sha256):
    completed = json.loads((directory / "completion.json").read_text())
    setup = json.loads((directory / "setup.json").read_text())
    gate = json.loads((directory / "gate_completion.json").read_text())
    if (not completed.get("complete") or not gate.get("passed")
            or completed.get("sequences") != len(SEQUENCES)
            or completed.get("native_targets") != 2012
            or setup.get("capacity_completion_sha256") != capacity_sha256
            or completed.get("sources") != setup.get("sources")
            or completed.get("sources") != gate.get("sources")):
        raise RuntimeError("Native-history cache is incomplete or prerequisites differ")
    if any(record.get(key) != value for record in (completed, setup, gate)
           for key, value in provenance.items()):
        raise RuntimeError("Native-history cache provenance differs")
    if (digest(directory / "cache_index.json") != completed["cache_index_sha256"]
            or digest(directory / "gate_completion.json") != completed["gate_completion_sha256"]):
        raise RuntimeError("Native-history cache index or gate changed")
    index = json.loads((directory / "cache_index.json").read_text())
    if set(index) != set(SEQUENCES) or sum(row["native_frames_checked"] for row in index.values()) != 2012:
        raise RuntimeError("Native-history cache cohort differs")
    for name, entry in index.items():
        if entry["bank"] != f"{name}/bank.pt" or entry["receipt"] != f"{name}/receipt.json":
            raise RuntimeError("Native-history cache path differs")
        if digest(directory / entry["receipt"]) != entry["receipt_sha256"]:
            raise RuntimeError("Native-history cache receipt changed")
        receipt = json.loads((directory / entry["receipt"]).read_text())
        if (receipt["bank_sha256"] != entry["bank_sha256"]
                or receipt["native_frames_checked"] != entry["native_frames_checked"]
                or any(receipt.get(key) != value for key, value in provenance.items())):
            raise RuntimeError("Native-history cache receipt differs")
    return index, completed["sources"]


def evaluate(model, heads, mode, frames, name, reference, output, fold, *, check_native=False):
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    for module in model.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache.clear()
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    native = {row["frame"]: row for row in reference}
    rows = []
    with tempfile.TemporaryDirectory(prefix="low_rank_", dir=output) as directory:
        state, actual_prompt = prepare(model, frames, Path(directory))
        if actual_prompt != prompt:
            raise RuntimeError("Prompt differs from native reference")
        context = (NativeRegionalLowRank(model, heads, mode, inference_state=state)
                   if heads is not None else nullcontext())
        started = time.monotonic()
        with context as observer, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1] or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid native object identity or prediction")
                if frame == prompt:
                    continue
                layers = observer.calls[-4:] if heads is not None else []
                if heads is not None and (
                    [c["layer"] for c in layers] != [0, 1, 2, 3]
                    or any(c["frame"] != frame or c["object_id"] != 1 for c in layers)
                ):
                    raise RuntimeError("Wrong native layer or instance schedule")
                row = score_frame(logits, frames, frame)
                if check_native and row["prediction_sha256"] != native[frame]["prediction_sha256"]:
                    raise RuntimeError(f"Archived native replay differs at {name}/{frame}")
                bank = state["output_dict_per_obj"][state["obj_id_to_idx"][1]]
                stored = bank["non_cond_frame_outputs"][frame]
                row.update(fold=fold, native_prediction_sha256=native[frame]["prediction_sha256"],
                           attention_layers=layers, stored_sha256=fingerprint(stored)["sha256"])
                rows.append(row)
                write(output / "progress.json", {"phase": "evaluation", "fold": fold,
                    "mode": mode, "sequence": name, "frame": frame})
        elapsed = time.monotonic() - started
        model.reset_state(state)
    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError("Incomplete validation trajectory")
    summary = summarize(rows)
    summary["inference_wall_seconds"] = elapsed
    if heads is not None:
        summary["mean_spatial_keys"] = float(np.mean([c["spatial_keys"] for r in rows for c in r["attention_layers"]]))
        summary["mean_pointer_keys"] = float(np.mean([c["pointer_keys"] for r in rows for c in r["attention_layers"]]))
    return rows, summary


def matched_schedules(left, right):
    if len(left) != len(right) or any(a["frame"] != b["frame"] or a["attention_layers"] != b["attention_layers"]
                                     for a, b in zip(left, right)):
        raise RuntimeError("Regional and global native read schedules differ")


def result_table(result):
    lines = ["# Regional low-rank memory — exploratory PolypGen CV", "",
             "Sequence-grouped evaluation; patient independence and external generalization unverified.", "",
             "| Sequence | Native Dice | Global | Regional | Regional − native | Regional − global |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in SEQUENCES:
        a, b, c = [result[mode][name]["present_dice"] for mode in ("native", "global", "regional")]
        lines.append(f"| {name} | {a:.6f} | {b:.6f} | {c:.6f} | {c-a:+.6f} | {c-b:+.6f} |")
    lines += ["", "| Arm | Mean Dice | IoU | Boundary F | False alarms | Runtime (s) |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode, row in result["summary"].items():
        runtime = row.get("inference_wall_seconds")
        runtime = "N/A" if runtime is None else f"{runtime:.1f}"
        lines.append(f"| {mode} | {row['present_dice']:.6f} | {row['iou']:.6f} | {row['boundary_f']:.6f} | {row['gt_empty_false_alarm_count']} | {runtime} |")
    for contrast in ("regional_vs_native", "regional_vs_global"):
        row = result[contrast]
        lines.append(f"\n{contrast}: Δ={row['mean_delta']:+.6f}, CI={row['ci95']}, wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']}, Holm p={result['holm_p'][contrast]:.6g}.")
    lines.append(f"\nFrozen meaningful-improvement screen passed: {result['meaningful_signal_screen']}.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "engineering-gate", "capacity-run", "native-run", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--cohort-gate", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    args = parser.parse_args()
    if not args.gate_only and args.cohort_gate is None:
        parser.error("Full CV requires --cohort-gate completion.json")
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    run = None
    try:
        protocol = json.loads(args.protocol.read_text())
        history = protocol.get("context", "fresh_prompt") == "native_history"
        if history != (args.history_cache is not None):
            raise RuntimeError("Protocol context and --history-cache must agree")
        verify_folds(protocol["folds"])
        gate = json.loads(args.engineering_gate.read_text())
        capacity = json.loads((args.capacity_run / "completion.json").read_text())
        if not gate.get("passed") or not capacity.get("passed_integrity") or not all(capacity["results"]["mean_loss_decreased"].values()):
            raise RuntimeError("Engineering/capacity prerequisites not met")
        for collection in (gate["sources"], capacity["sources"]):
            if any(digest(ROOT / name) != sha for name, sha in collection.items()):
                raise RuntimeError("Source changed since verified prerequisites")
        checkpoint_sha = digest(args.checkpoint)
        manifest_sha = digest(ROOT / protocol["manifest"])
        capacity_setup = json.loads((args.capacity_run / "setup.json").read_text())
        if (gate["checkpoint_sha256"] != checkpoint_sha or gate["model_config"] != protocol["model_config"]
                or capacity_setup["checkpoint_sha256"] != checkpoint_sha or capacity_setup["manifest_sha256"] != manifest_sha):
            raise RuntimeError("Prerequisite checkpoint, config or manifest differs")
        inherited = capacity["sources"]
        cache_index = None
        if history:
            if capacity_setup.get("context") != "native_history":
                raise RuntimeError("Native-history CV requires its matching capacity gate")
            cache_index, inherited = verified_history_cache(args.history_cache,
                {"checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
                 "model_config": protocol["model_config"]}, digest(args.capacity_run / "completion.json"))
        if any(digest(ROOT / name) != sha for name, sha in inherited.items()):
            raise RuntimeError("Sources changed since cache verification")
        sources = {**inherited, **{name: digest(ROOT / name) for name in (
            "scripts/run_regional_low_rank_cv.py", str(args.protocol.resolve().relative_to(ROOT)),
            "tests/test_regional_low_rank_cv.py")}}
        if args.cohort_gate:
            completed_gate = json.loads(args.cohort_gate.read_text())
            if not completed_gate.get("passed") or completed_gate.get("sources") != sources:
                raise RuntimeError("Full-sequence gate does not match current sources")
            if completed_gate.get("checkpoint_sha256") != checkpoint_sha or completed_gate.get("manifest_sha256") != manifest_sha:
                raise RuntimeError("Full-sequence gate checkpoint or manifest differs")
        for name in sources:
            target = out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write(out / "protocol.json", protocol)
        write(out / "setup.json", {"sources": sources, "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha, "model_config": protocol["model_config"],
            "engineering_gate_sha256": digest(args.engineering_gate),
            "capacity_completion_sha256": digest(args.capacity_run / "completion.json"),
            "native_setup_sha256": digest(args.native_run / "setup.json"),
            "cohort_gate_sha256": digest(args.cohort_gate) if args.cohort_gate else None,
            "context": "native_history" if history else "fresh_prompt",
            "cache_completion_sha256": digest(args.history_cache / "completion.json") if history else None})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, _ = activate_author_repo()
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        native, native_rows = load_native_baseline(args.native_run, videos, checkpoint_sha, manifest_sha)
        settings = protocol["training"]

        def teacher_bank(name):
            entry = cache_index[name]
            receipt = json.loads((args.history_cache / entry["receipt"]).read_text())
            if digest(args.history_cache / entry["receipt"]) != entry["receipt_sha256"]:
                raise RuntimeError("Native-history receipt changed before loading")
            return load_bank(args.history_cache / entry["bank"], receipt,
                             checkpoint_sha, manifest_sha, protocol["model_config"])

        def new_model():
            return build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
                checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)

        if args.gate_only:
            model = new_model()
            backbone = parameter_digest(model)
            for name in ("seq2", "seq3"):
                if history:
                    selected = sample_clip(videos[name], settings["clip_length"], settings["seed"])
                    samples, metadata = capture_cached_block(model, videos[name],
                        [frame.frame_index for frame in selected], out, teacher_bank(name))
                    with torch.no_grad():
                        native_loss = native_qk_clip_loss(model, samples, check_native=True)
                        zero_heads = RegionalLowRankHeads(model.mem_dim, settings["rank"])
                        with NativeRegionalLowRank(model, zero_heads, "regional"):
                            zero_loss = native_qk_clip_loss(model, samples, check_native=True)
                    if not torch.equal(native_loss, zero_loss):
                        raise RuntimeError("Cached block zero-adapter loss differs")
                    write(out / f"{name}_cached_block.json", {**metadata,
                        "native_loss": float(native_loss), "zero_loss": float(zero_loss)})
                    del samples
                reference, _ = evaluate(model, None, "native", videos[name], name, native_rows[name], out, 0, check_native=True)
                write(out / f"{name}_native.json", reference)
                zero = RegionalLowRankHeads(model.mem_dim, settings["rank"])
                zero_rows, _ = evaluate(model, zero, "regional", videos[name], name, native_rows[name], out, 0, check_native=True)
                if [(r["frame"], r["stored_sha256"]) for r in reference] != [(r["frame"], r["stored_sha256"]) for r in zero_rows]:
                    raise RuntimeError("Full-sequence zero adapter changed stored native outputs")
                write(out / f"{name}_zero.json", zero_rows)
                paired = {}
                for mode in ("regional", "global"):
                    heads = RegionalLowRankHeads(model.mem_dim, settings["rank"])
                    heads.load_state_dict(torch.load(args.capacity_run / f"{mode}.pt", weights_only=True), strict=True)
                    paired[mode], _ = evaluate(model, heads, mode, videos[name], name, native_rows[name], out, 0)
                    write(out / f"{name}_{mode}.json", paired[mode])
                matched_schedules(paired["regional"], paired["global"])
            if parameter_digest(model) != backbone:
                raise RuntimeError("Native weights changed during sequence gate")
            if sources != {name: digest(ROOT / name) for name in sources}:
                raise RuntimeError("Sources changed during sequence gate")
            write(out / "completion.json", {"passed": True, "sources": sources,
                "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
                "sequences": ["seq2", "seq3"], "scope": "Full-sequence engineering gate, not CV accuracy"})
            write(out / "progress.json", {"phase": "gate_complete"})
            return

        for key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
            os.environ[key] = str(out / key.lower())
        run = start_wandb("ade-seg", "adenoid-hypertrophy", out.name, args.wandb_mode,
                          None, protocol, out, ["regional-low-rank", "polypgen-cv", "exploratory"])
        results = {"native": native, "global": {}, "regional": {}}
        for fold, validation in enumerate(protocol["folds"], 1):
            fold_dir = out / f"fold{fold}"
            fold_dir.mkdir()
            model = new_model()
            backbone = parameter_digest(model)
            arms = {}
            for mode in ("regional", "global"):
                torch.manual_seed(settings["seed"] + fold)
                arms[mode] = RegionalLowRankHeads(model.mem_dim, settings["rank"])
            optimizers = {mode: torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],
                lr=settings["learning_rate"], weight_decay=settings["weight_decay"]) for mode, heads in arms.items()}
            training = [name for name in SEQUENCES if name not in validation]
            banks = {name: teacher_bank(name) for name in training} if history else {}
            log = []
            layers_trained = {mode: set() for mode in arms}
            for epoch in range(settings["epochs"]):
                for index, name in enumerate(training):
                    seed = settings["seed"] + fold * 100000 + epoch * len(training) + index
                    frames = sample_clip(videos[name], settings["clip_length"], seed)
                    if history:
                        samples, capture = capture_cached_block(model, videos[name],
                            [frame.frame_index for frame in frames], out, banks[name])
                    else:
                        samples, capture = capture_clip(model, frames, out), None
                    record = {"fold": fold, "epoch": epoch, "sequence": name, "seed": seed,
                              "frames": [f.frame_index for f in frames], "capture": capture, "arms": {}}
                    for mode, heads in arms.items():
                        optimizers[mode].zero_grad(set_to_none=True)
                        with NativeRegionalLowRank(model, heads, mode) as observer:
                            loss = native_qk_clip_loss(model, samples)
                        if not torch.isfinite(loss) or len(observer.calls) != 4 * len(samples):
                            raise RuntimeError("Nonfinite loss or incomplete native read")
                        loss.backward()
                        norms = []
                        for layer, head in enumerate(heads.layers):
                            grads = [p.grad for p in head.parameters() if p.requires_grad]
                            if any(g is None or not torch.isfinite(g).all() for g in grads):
                                raise RuntimeError("Invalid adapter gradient")
                            norm = sum(float(g.abs().sum()) for g in grads)
                            norms.append(norm)
                            if norm > 0:
                                layers_trained[mode].add(layer)
                        torch.nn.utils.clip_grad_norm_(heads.parameters(), settings["max_gradient_norm"])
                        optimizers[mode].step()
                        if any(not torch.isfinite(p).all() for p in heads.parameters()):
                            raise RuntimeError("Nonfinite adapter weights")
                        record["arms"][mode] = {"loss": float(loss.detach()), "gradient_l1_by_layer": norms,
                                                "reads": observer.calls}
                    if record["arms"]["regional"]["reads"] != record["arms"]["global"]["reads"]:
                        raise RuntimeError("Training read schedules differ")
                    if any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError("Native weights received gradients")
                    log.append(record)
                    write(fold_dir / "training.json", log)
                    write(out / "progress.json", {"phase": "training", "fold": fold,
                        "paired_updates": len(log), "total_per_fold": len(training) * settings["epochs"]})
                    run.log({f"fold{fold}/{mode}/loss": values["loss"] for mode, values in record["arms"].items()})
                    del samples, loss
                    gc.collect()
            if parameter_digest(model) != backbone or any(layers_trained[mode] != {0, 1, 2, 3} for mode in arms):
                raise RuntimeError("Frozen model changed or a layer did not train")
            for mode, heads in arms.items():
                heads.eval()
                save_tensor(heads.state_dict(), fold_dir / f"{mode}.pt")
            for name in validation:
                paired = {}
                for mode, heads in arms.items():
                    paired[mode], results[mode][name] = evaluate(model, heads, mode, videos[name], name, native_rows[name], out, fold)
                    path = fold_dir / f"{name}_{mode}_frames.json"
                    write(path, paired[mode])
                    write(fold_dir / f"{name}_{mode}_complete.json", {"frames_sha256": digest(path),
                        "summary": results[mode][name], "native_sha256": digest(args.native_run / "native" / f"{name}_frames.json")})
                matched_schedules(paired["regional"], paired["global"])
            write(fold_dir / "completion.json", {"complete": True, "training_sha256": digest(fold_dir / "training.json"),
                "validation": validation, "training": training, "updates_per_arm": len(log)})
            del model, arms, optimizers, banks
            gc.collect()
        result = {**results, "summary": {mode: summarize_arm(rows) for mode, rows in results.items()}}
        for index, control in enumerate(("native", "global")):
            result[f"regional_vs_{control}"] = paired_delta(results, "regional", control, settings["seed"] + index)
        result["holm_p"] = holm_adjust({key: result[key]["wilcoxon"]["p"] for key in ("regional_vs_native", "regional_vs_global")})
        result["meaningful_signal_screen"] = all(
            result[f"regional_vs_{control}"]["mean_delta"] >= 0.01
            and result[f"regional_vs_{control}"]["wins"] >= 14
            and result[f"regional_vs_{control}"]["ci95"][0] > 0
            and result["holm_p"][f"regional_vs_{control}"] < 0.05
            and result["summary"]["regional"]["gt_empty_false_alarm_count"] <= result["summary"][control]["gt_empty_false_alarm_count"]
            for control in ("native", "global"))
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise RuntimeError("Sources changed during CV")
        write(out / "results.json", result)
        (out / "results.md").write_text(result_table(result))
        write(out / "completion.json", {"complete": True, "sources": sources, "sequences": len(SEQUENCES),
                                       "results_sha256": digest(out / "results.json")})
        write(out / "progress.json", {"phase": "complete"})
        run.finish(quiet=True)
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        if run is not None:
            run.finish(exit_code=1, quiet=True)
        raise


if __name__ == "__main__":
    main()
