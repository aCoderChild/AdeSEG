"""Measure fixed training-clip optimization before regional-adapter CV."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_region_qk_training import native_qk_clip_loss
from modeling.regional_low_rank import NativeRegionalLowRank, RegionalLowRankHeads
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip
from training.tracking import start_wandb

PROTOCOL = ROOT / "protocols/regional_low_rank_capacity_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "engineering-gate", "schedule-run", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run = None
    try:
        protocol = json.loads(args.protocol.read_text())
        context = protocol.get("context", "fresh_prompt")
        if context not in ("fresh_prompt", "native_history"):
            raise ValueError("Unknown training context")
        if context == "native_history" and args.native_run is None:
            raise ValueError("Native-history capture requires --native-run")
        gate = json.loads(args.engineering_gate.read_text())
        if (gate.get("passed") is not True or gate.get("checkpoint_sha256") != digest(args.checkpoint)
                or gate.get("model_config") != protocol["model_config"]):
            raise RuntimeError("Engineering gate checkpoint or config differs")
        for name, expected in gate["sources"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Source differs from engineering gate: {name}")
        added_sources = [str(args.protocol.resolve().relative_to(ROOT)), "scripts/diagnose_regional_low_rank_capacity.py"]
        if context == "native_history":
            added_sources += ["modeling/native_history_training.py", "modeling/cfbi_compact_snapshot.py",
                              "tests/test_native_history_training.py", "scripts/run_soft_role_affinity_cv.py"]
        sources = {**gate["sources"], **{name: digest(ROOT / name) for name in added_sources}}
        for name in sources:
            target = out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        history = json.loads((args.schedule_run / "training.json").read_text())
        selected = []
        for row in history:
            if row["fold"] == 3 and row["epoch"] == 0 and row["sequence"] not in [c["sequence"] for c in selected]:
                selected.append({k: row[k] for k in ("sequence", "seed", "frames")})
            if len(selected) == 4:
                break
        if selected != protocol["clips"] or any(c["sequence"] in protocol["excluded_fold3_sequences"] for c in selected):
            raise RuntimeError("Training-only clip selection differs from the recorded rule")
        write(out / "protocol.json", protocol)
        write(out / "setup.json", {"sources": sources, "gate_sha256": digest(args.engineering_gate),
            "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(ROOT / protocol["manifest"]),
            "schedule_sha256": digest(args.schedule_run / "training.json"), "context": context,
            "native_setup_sha256": digest(args.native_run / "setup.json") if args.native_run else None})
        for name in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
            os.environ[name] = str(out / name.lower())
        run = start_wandb("ade-seg", "adenoid-hypertrophy", out.name, args.wandb_mode,
                          None, protocol, out, ["regional-low-rank", "training-capacity", "exploratory"])
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        repo, _ = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(method="sam2", device="cpu",
            checkpoint=args.checkpoint, model_config=protocol["model_config"]), repo)
        model.requires_grad_(False)
        backbone = parameter_digest(model)
        videos = {**load_manifest(ROOT / protocol["manifest"], "dev"),
                  **load_manifest(ROOT / protocol["manifest"], "test")}
        native_rows = None
        if context == "native_history":
            from scripts.run_soft_role_affinity_cv import load_native_baseline
            _, native_rows = load_native_baseline(args.native_run, videos,
                digest(args.checkpoint), digest(ROOT / protocol["manifest"]))
        clips = []
        capture_records = []
        for spec in selected:
            frames = sample_clip(videos[spec["sequence"]], 5, spec["seed"])
            if [f.frame_index for f in frames] != spec["frames"]:
                raise RuntimeError("Author sampler does not reproduce recorded frames")
            if context == "native_history":
                from modeling.native_history_training import capture_history_block
                write(out / "progress.json", {"phase": "capture", "sequence": spec["sequence"],
                                               "clips": len(capture_records), "total": 4})
                full, samples, record = capture_history_block(model, videos[spec["sequence"]],
                    spec["frames"], out, native_rows[spec["sequence"]])
                with torch.no_grad():
                    full_loss = native_qk_clip_loss(model, full, check_native=True)
                    compact_loss = native_qk_clip_loss(model, samples, check_native=True)
                if not torch.equal(full_loss, compact_loss):
                    raise RuntimeError("Full and compact native history losses differ")
                record.update(sequence=spec["sequence"], full_compact_replay_identical=True,
                              initial_native_loss=float(full_loss))
                capture_records.append(record)
                write(out / "capture.json", capture_records)
                write(out / "progress.json", {"phase": "capture", "clips": len(capture_records), "total": 4})
                del full, full_loss, compact_loss
            else:
                samples = capture_clip(model, frames, out)
                with torch.inference_mode():
                    for sample in samples:
                        reference = model.track_step(frame_idx=sample["frame"], is_init_cond_frame=False,
                            current_vision_feats=sample["vision"], current_vision_pos_embeds=sample["positions"],
                            feat_sizes=sample["sizes"], point_inputs=None, mask_inputs=None,
                            output_dict=sample["output_dict"], num_frames=sample["num_frames"],
                            track_in_reverse=False, run_mem_encoder=False)
                        sample["native_object_score"] = reference["object_score_logits"].clone()
            clips.append(samples)
        settings = protocol["training"]
        arms = {}
        for mode in ("regional", "global"):
            torch.manual_seed(settings["seed"])
            arms[mode] = RegionalLowRankHeads(model.mem_dim, settings["rank"])
        if parameter_digest(arms["regional"]) != parameter_digest(arms["global"]):
            raise RuntimeError("Arms do not have identical initialization")

        def measure(heads, mode, *, native=False):
            values = []
            for clip_index, samples in enumerate(clips):
                with torch.no_grad(), NativeRegionalLowRank(model, heads, mode) as observer:
                    loss = native_qk_clip_loss(model, samples, check_native=native)
                if not torch.isfinite(loss) or len(observer.calls) != 4 * len(samples):
                    raise RuntimeError("Invalid fixed-clip measurement")
                if [(call["frame"], call["layer"]) for call in observer.calls] != [
                    (sample["frame"], layer) for sample in samples for layer in range(4)
                ] or any(not call["pointer_tokens_unchanged"] for call in observer.calls):
                    raise RuntimeError("Invalid layer or pointer schedule")
                if context == "native_history" and any(
                    call["memory_frames"] != capture_records[clip_index]["records"][index // 4]["spatial_frames"]
                    for index, call in enumerate(observer.calls)
                ):
                    raise RuntimeError("Replay changed the native memory-frame schedule")
                values.append(float(loss))
            return values

        before = {mode: measure(heads, mode, native=True) for mode, heads in arms.items()}
        if before["regional"] != before["global"]:
            raise RuntimeError("Initial losses differ")
        optimizers = {mode: torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],
            lr=settings["learning_rate"], weight_decay=settings["weight_decay"]) for mode, heads in arms.items()}
        records = []
        for step in range(settings["updates_per_arm"]):
            clip_index = step % len(clips)
            record = {"step": step, "clip": selected[clip_index], "arms": {}}
            for mode, heads in arms.items():
                optimizer = optimizers[mode]
                optimizer.zero_grad(set_to_none=True)
                with NativeRegionalLowRank(model, heads, mode) as observer:
                    loss = native_qk_clip_loss(model, clips[clip_index])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                norms = {}
                for name, parameter in heads.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    grad = parameter.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise RuntimeError(f"Invalid gradient: {name}")
                    norms[name] = float(grad.abs().sum())
                    if name.endswith("lora_B") and norms[name] == 0:
                        raise RuntimeError(f"Missing expert gradient: {name}")
                torch.nn.utils.clip_grad_norm_(heads.parameters(), settings["clip_gradient_norm"])
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in heads.parameters()):
                    raise RuntimeError("Nonfinite adapter weights")
                record["arms"][mode] = {"loss": float(loss.detach()), "gradient_l1": norms,
                    "adapter_parameter_l2": float(torch.stack([p.detach().square().sum()
                        for p in heads.parameters() if p.requires_grad]).sum().sqrt()), "reads": observer.calls}
            if record["arms"]["regional"]["reads"] != record["arms"]["global"]["reads"]:
                raise RuntimeError("Paired native read schedules differ")
            records.append(record)
            write(out / "training.json", records)
            write(out / "progress.json", {"phase": "training", "paired_updates": step + 1,
                                          "total_paired_updates": settings["updates_per_arm"]})
            run.log({f"{mode}/loss": values["loss"] for mode, values in record["arms"].items()}, step=step)
        after = {mode: measure(heads, mode) for mode, heads in arms.items()}
        if parameter_digest(model) != backbone or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen native model changed")
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise RuntimeError("Sources changed during diagnostic")
        result = {"before": before, "after": after, "sequences": [c["sequence"] for c in selected],
            "mean_loss_decreased": {mode: sum(after[mode]) < sum(before[mode]) for mode in arms},
            "scope": protocol["scope"]}
        for mode, heads in arms.items():
            torch.save(heads.state_dict(), out / f"{mode}.pt")
        write(out / "results.json", result)
        write(out / "completion.json", {"passed_integrity": True, "sources": sources,
            "seconds": time.monotonic() - started, "backbone_sha256": backbone, "results": result})
        write(out / "progress.json", {"phase": "complete"})
        run.summary.update({"integrity_passed": True, **{f"{mode}/mean_loss_decreased": value
                                                        for mode, value in result["mean_loss_decreased"].items()}})
        try:
            run.finish(quiet=True)
        except Exception as error:
            write(out / "tracking_delivery_failure.json", {"error": repr(error)})
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error)})
        if run is not None:
            run.finish(exit_code=1, quiet=True)
        raise


if __name__ == "__main__":
    main()
