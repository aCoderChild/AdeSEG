"""Train only auxiliary memory prompts; keep original SAM2 frozen."""
import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import holm_adjust
from modeling.decoder_core_training import decoder_clip_loss
from modeling.native_history_cache import capture_cached_block, load_bank
from modeling.regional_memory_prompts import NativeRegionalMemoryPrompts, RegionalMemoryPrompts
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_prediction_conditioned_affinity_cv import SEQUENCES, summarize_arm, verify_folds
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import paired_delta, summarize
from scripts.verify_memory_prompt_propagation import evaluate
from training.tracking import start_wandb

PROTOCOL = "protocols/regional_memory_prompts_cv_v1.json"


def schedule(protocol, fold):
    verify_folds(protocol["folds"])
    validation = protocol["folds"][fold - 1]
    training = [name for name in SEQUENCES if name not in validation]
    updates = [(epoch, name, protocol["seed"] + fold * 100000 + epoch * len(training) + index)
               for epoch in range(protocol["epochs"]) for index, name in enumerate(training)]
    if len(updates) != protocol["updates_per_fold_per_arm"] or set(training) & set(validation):
        raise RuntimeError("Invalid prospective training schedule")
    return updates


def train_step(model, heads, samples, mode, optimizer, protocol):
    optimizer.zero_grad(set_to_none=True)
    with NativeRegionalMemoryPrompts(model, heads, mode) as observer:
        loss, info = decoder_clip_loss(model, samples, retain_first_write=True)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite author loss")
    temporal, = torch.autograd.grad(info["last_loss"], info["first_write"], retain_graph=True)
    if not torch.isfinite(temporal).all():
        raise RuntimeError("Nonfinite temporal gradient")
    loss.backward()
    gradients = [p.grad for p in heads.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing/nonfinite auxiliary gradient")
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError("Original SAM2 trainable scope changed")
    expected = [(s["frame"], i) for s in samples for i in range(4)]
    if [(r["frame"], r["layer"]) for r in observer.calls] != expected:
        raise RuntimeError("Incomplete training layer/read schedule")
    for record in observer.calls:
        if record["auxiliary_keys"] != len(record["memory_frames"]) * 3 * protocol["tokens_per_region_per_frame"]:
            raise RuntimeError("Auxiliary memory count differs")
        if any(span["frame"] >= record["frame"] for span in record["memory_frames"]):
            raise RuntimeError("Future memory entered training")
    norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), protocol["max_gradient_norm"], error_if_nonfinite=True)
    if norm == 0:
        raise RuntimeError("All auxiliary gradients are zero")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in heads.parameters()):
        raise RuntimeError("Nonfinite auxiliary update")
    return {"loss": float(loss.detach()), "terms": info["terms"], "gradient_norm": float(norm),
            "temporal_gradient_l1": float(temporal.abs().sum()), "reads": observer.calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "native-run", "propagation-gate", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--history-gate", type=Path)
    args = parser.parse_args()
    if not args.gate_only and args.history_gate is None:
        parser.error("CV requires a passed --history-gate")
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    run = None
    try:
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        gate = json.loads((args.propagation_gate / "completion.json").read_text())
        native = json.loads((args.native_run / "completion.json").read_text())
        audit = json.loads((args.native_run / "independent_audit.json").read_text())
        native_setup = json.loads((args.native_run / "setup.json").read_text())
        if (not gate.get("passed") or not gate.get("engineering_only") or not native.get("passed")
                or not audit.get("passed") or native.get("scope") != "cohort"):
            raise RuntimeError("Audited engineering/native prerequisites missing")
        for filename, sha in gate["artifacts"].items():
            if digest(args.propagation_gate / filename) != sha:
                raise RuntimeError("Propagation evidence changed")
        if (digest(args.checkpoint) != protocol["checkpoint_sha256"]
                or native_setup["checkpoint_sha256"] != protocol["checkpoint_sha256"]
                or native_setup["model_config"] != protocol["model_config"]
                or native_setup["manifest_sha256"] != digest(ROOT / protocol["manifest"])):
            raise RuntimeError("Official asset provenance differs")
        sources = {**gate["sources"], **{p: digest(ROOT / p) for p in (
            PROTOCOL, "scripts/run_regional_memory_prompts_cv.py", "tests/test_memory_prompt_cv.py",
            "modeling/native_history_cache.py", "modeling/native_history_training.py",
            "modeling/cfbi_compact_snapshot.py", "modeling/native_stage_capture.py")}}
        if any(digest(ROOT / p) != sha for p, sha in sources.items()):
            raise RuntimeError("Verified source changed")
        history = None
        if args.history_gate:
            history = json.loads((args.history_gate / "completion.json").read_text())
            if not history.get("passed") or not history.get("engineering_only") or history["sources"] != sources:
                raise RuntimeError("Native-history training gate missing or changed")
            if history["propagation_sha256"] != digest(args.propagation_gate / "completion.json"):
                raise RuntimeError("History gate used different propagation evidence")
        verify_folds(protocol["folds"])
        for fold in range(1, 4):
            schedule(protocol, fold)
        for p in sources:
            path = out / "source_snapshot" / p
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / p, path)
        write(out / "setup.json", {"protocol": protocol, "sources": sources,
            "propagation_sha256": digest(args.propagation_gate / "completion.json"),
            "history_sha256": digest(args.history_gate / "completion.json") if history else None,
            "native_sha256": digest(args.native_run / "completion.json")})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, _ = activate_author_repo()
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        references, receipts = {}, {}
        if set(native["entries"]) != set(SEQUENCES) or set(audit["sequences"]) != set(SEQUENCES):
            raise RuntimeError("Native cohort coverage differs")
        for name, entry in native["entries"].items():
            folder = Path(entry)
            if (digest(folder / "receipt.json") != audit["sequences"][name]["receipt_sha256"]
                    or digest(folder / "completion.json") != audit["sequences"][name]["completion_sha256"]):
                raise RuntimeError("Native receipts changed")
            completed = json.loads((folder / "completion.json").read_text())
            if digest(folder / "frames.json") != completed["frames_sha256"]:
                raise RuntimeError("Native trajectory changed")
            references[name] = json.loads((folder / "frames.json").read_text())
            receipts[name] = json.loads((folder / "receipt.json").read_text())
        def new_model():
            model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
                checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
            model.requires_grad_(False)
            return model.eval()
        def block(model, name, seed):
            folder = Path(native["entries"][name])
            bank = load_bank(folder / "bank.pt", receipts[name], protocol["checkpoint_sha256"],
                             native_setup["manifest_sha256"], protocol["model_config"])
            clip = sample_clip(videos[name], protocol["clip_length"], seed)
            return capture_cached_block(model, videos[name], [f.frame_index for f in clip], out, bank)
        if args.gate_only:
            evidence = {}
            for name in ("seq2", "seq13"):
                model = new_model()
                backbone = parameter_digest(model)
                samples, capture = block(model, name, protocol["seed"])
                before = [fingerprint(s["output_dict"])["sha256"] for s in samples]
                evidence[name] = {"capture": capture, "arms": {}}
                for mode in protocol["arms"]:
                    torch.manual_seed(protocol["seed"])
                    heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"])
                    with torch.no_grad(), NativeRegionalMemoryPrompts(model, heads, enabled=False):
                        decoder_clip_loss(model, samples, check_native=True)
                    optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["learning_rate"],
                                                 weight_decay=protocol["weight_decay"])
                    evidence[name]["arms"][mode] = train_step(model, heads, samples, mode, optimizer, protocol)
                if before != [fingerprint(s["output_dict"])["sha256"] for s in samples] or parameter_digest(model) != backbone:
                    raise RuntimeError("Training gate changed native banks or original parameters")
                write(out / f"{name}.json", evidence[name])
                del model, samples, heads, optimizer
                gc.collect()
            if sources != {p: digest(ROOT / p) for p in sources}:
                raise RuntimeError("Sources changed during history verification")
            write(out / "completion.json", {"passed": True, "engineering_only": True, "sources": sources,
                "propagation_sha256": digest(args.propagation_gate / "completion.json"), "results": evidence})
            return
        for key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR", "TMPDIR"):
            folder = out / key.lower()
            folder.mkdir()
            os.environ[key] = str(folder)
        run = start_wandb("ade-seg", "adenoid-hypertrophy", out.name, "offline", None, protocol, out,
                          ["frozen-sam2", "regional-memory-prompts", "exploratory-cv"])
        results = {"native": {n: summarize(r) for n, r in references.items()}, "regional": {}, "global": {}}
        for fold, validation in enumerate(protocol["folds"], 1):
            initial_sha = None
            for mode in protocol["arms"]:
                folder = out / f"fold{fold}" / mode
                folder.mkdir(parents=True)
                model = new_model()
                backbone = parameter_digest(model)
                torch.manual_seed(protocol["seed"] + fold)
                heads = RegionalMemoryPrompts(model.mem_dim, protocol["tokens_per_region_per_frame"])
                if sum(p.numel() for p in heads.parameters()) != protocol["trainable_parameters"]:
                    raise RuntimeError("External trainable scope differs")
                initial = parameter_digest(heads)
                if initial_sha is not None and initial != initial_sha:
                    raise RuntimeError("Matched initialization differs")
                initial_sha = initial
                optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["learning_rate"],
                                              weight_decay=protocol["weight_decay"])
                logs = []
                for epoch, name, seed in schedule(protocol, fold):
                    samples, capture = block(model, name, seed)
                    before = [fingerprint(s["output_dict"])["sha256"] for s in samples]
                    record = train_step(model, heads, samples, mode, optimizer, protocol)
                    if before != [fingerprint(s["output_dict"])["sha256"] for s in samples] or parameter_digest(model) != backbone:
                        raise RuntimeError("Training changed source banks or original SAM2 parameters")
                    logs.append({"epoch": epoch, "sequence": name, "seed": seed, "capture": capture, **record})
                    write(folder / "training.json", logs)
                    write(out / "progress.json", {"phase": "training", "fold": fold, "mode": mode,
                        "update": len(logs), "total": protocol["updates_per_fold_per_arm"]})
                    run.log({f"fold{fold}/{mode}/loss": record["loss"],
                             f"fold{fold}/{mode}/gradient_norm": record["gradient_norm"]})
                    if len(logs) % 14 == 0:
                        torch.save({"prompts": heads.state_dict(), "optimizer": optimizer.state_dict(),
                                    "updates": len(logs), "sources": sources}, folder / "latest_training.pt")
                    del samples
                    gc.collect()
                torch.save(heads.state_dict(), folder / "prompts.pt")
                anchor = {"weights_sha256": digest(folder / "prompts.pt"),
                    "training_sha256": digest(folder / "training.json"), "updates": len(logs),
                    "initial_sha256": initial, "original_sam2_sha256": backbone,
                    "training": sorted({n for _, n, _ in schedule(protocol, fold)}),
                    "validation": validation, "selection": "Final epoch anchored before validation"}
                write(folder / "training_completion.json", anchor)
                heads.eval()
                for name in validation:
                    rows, summary = evaluate(model, videos[name], name, references[name], out, heads, mode)
                    summary["inference_wall_seconds"] = summary.pop("wall_seconds")
                    results[mode][name] = summary
                    write(folder / f"{name}_frames.json", rows)
                    write(folder / f"{name}_completion.json", {"summary": summary,
                        "frames_sha256": digest(folder / f"{name}_frames.json"),
                        "training_receipt_sha256": digest(folder / "training_completion.json")})
                write(folder / "completion.json", {"complete": True, **anchor})
                del model, heads, optimizer
                gc.collect()
        result = {**results, "summary": {m: {**summarize_arm(values),
            "gt_empty_targets": sum(s["gt_empty_targets"] for s in values.values())} for m, values in results.items()}}
        contrasts = {f"regional_vs_{other}": paired_delta(results, "regional", other, protocol["seed"])
                     for other in ("native", "global")}
        result.update(contrasts)
        result["holm_p"] = holm_adjust({k: v["wilcoxon"]["p"] for k, v in contrasts.items()})
        result["meaningful_signal_screen"] = (
            contrasts["regional_vs_native"]["mean_delta"] >= .01
            and contrasts["regional_vs_native"]["wins"] >= 14
            and contrasts["regional_vs_global"]["mean_delta"] > 0
            and all(v["ci95"][0] > 0 and result["holm_p"][k] < .05 for k, v in contrasts.items())
            and result["summary"]["regional"]["gt_empty_false_alarm_count"] <= result["summary"]["native"]["gt_empty_false_alarm_count"])
        if sources != {p: digest(ROOT / p) for p in sources}:
            raise RuntimeError("Sources changed during CV")
        write(out / "results.json", result)
        lines = ["# Frozen-SAM2 regional memory prompts — exploratory CV", "",
            "| Sequence | Native Dice | Global Dice | Regional Dice |", "|---|---:|---:|---:|"]
        for name in SEQUENCES:
            lines.append("| " + name + " | " + " | ".join(f"{results[m][name]['present_dice']:.6f}" for m in ("native", "global", "regional")) + " |")
        lines += ["", f"Meaningful improvement screen: {result['meaningful_signal_screen']}.",
                  "See results.json for paired intervals, adjusted p values, false alarms and secondary metrics."]
        (out / "results.md").write_text("\n".join(lines) + "\n")
        write(out / "completion.json", {"complete": True, "sources": sources,
            "results_sha256": digest(out / "results.json")})
        run.finish()
        run = None
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
