"""One pruning read followed by native propagation; frozen exploratory pilot."""

import argparse
import gc
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.contribution_region_memory import contribution_cells
from modeling.matched_region_memory import random_cells
from modeling.region_definitions import mixed_cell_regions
from modeling.sparse_memory import SparseMemoryRead
from scripts import run_author_baseline as author
from scripts.analyze_v2_trajectories import archive_rows, CORE, SEEDS, longest_lower_run
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import validate_output
from scripts.run_long_history_screen import require_e2
from scripts.run_v2_target_replay import verify_layers
from training.tracking import start_wandb, log_results_artifact, finalize_tracking

PROTOCOL = "protocols/v2_trajectory_pulse_v1.json"
TARGETS = "protocols/v2_pulse_targets_v1.json"
SOURCES = [PROTOCOL, TARGETS, "scripts/analyze_v2_trajectories.py", "scripts/run_v2_pulse.py",
           "tests/test_v2_trajectory_pulse.py", "modeling/contribution_region_memory.py",
           "modeling/matched_region_memory.py", "modeling/sparse_memory.py",
           "modeling/region_definitions.py", "modeling/region_tokens.py",
           "scripts/run_author_baseline.py", "scripts/bank_integrity.py",
           "scripts/false_alarm_statistics.py", "scripts/run_v2_target_replay.py",
           "scripts/memory_token_io.py", "evaluation/segmentation.py",
           "evaluation/davis2017/metrics.py", "training/tracking.py"]


def pulse_bank(state, prompt, target, sequence, condition, seed=None):
    """Same selector and random context as frozen source26, actual stored logits."""
    if condition not in ("sham", "regional", "random"):
        raise ValueError("Unknown pulse condition")
    if (condition == "random") != (seed is not None):
        raise ValueError("Random pulse requires a fixed seed only")
    outputs = state["output_dict_per_obj"][0]
    frames = [prompt]+list(range(max(prompt+1, target-6), target))
    bank = {}
    for frame in frames:
        stored = outputs["cond_frame_outputs"].get(frame, outputs["non_cond_frame_outputs"].get(frame))
        if stored is None or stored["maskmem_features"] is None:
            raise ValueError("Missing pre-target native memory")
        keep = contribution_cells(mixed_cell_regions(stored["pred_masks"][0, 0])["labels"])
        if condition == "random":
            keep = random_cells(int(keep.sum()), sequence, frame, seed)
        if condition == "sham" or frame in (prompt, target-1):
            keep = torch.ones(4096, dtype=torch.bool)
        bank[frame] = {"features": stored["maskmem_features"].detach(),
                       "pos": stored["maskmem_pos_enc"][-1].detach(), "keep": keep}
    return bank


def assert_native_forward(model):
    for layer in model.memory_attention.layers:
        attention = layer.cross_attn_image
        if "forward" in attention.__dict__ or hasattr(attention, "_sparse_mask") or hasattr(attention, "_sparse_actual_keys"):
            raise RuntimeError("Sparse attention override remained outside the pulse")


def next_at_target(iterator, model, state, index, prompt, target, sequence, condition, seed=None):
    assert_native_forward(model)
    if index != target:
        return next(iterator), None
    bank = pulse_bank(state, prompt, target, sequence, condition, seed)
    controller = SparseMemoryRead(model, prompt, "all_native" if condition == "sham" else "region_native")
    controller.frame, controller.bank = index, bank
    # Normal writeback is intentional here, unlike the source28 read-only probe.
    with patch.object(model, "_encode_new_memory", wraps=model._encode_new_memory) as writer:
        with controller:
            result = next(iterator)
        if writer.call_count != 1:
            raise RuntimeError("Pulse did not use exactly one native memory write")
    assert_native_forward(model)
    counts = verify_layers(controller.layers, list(bank), bank)
    if condition != "sham" and sum(counts) == len(bank)*4096:
        raise RuntimeError("Frozen pulse removed no actual tokens")
    receipt = {"frame": target, "condition": condition, "seed": seed, "layers": controller.layers,
               "per_frame_counts": counts, "native_memory_writes": 1,
               "keep_bits": {str(f): np.packbits(v["keep"].numpy()).tobytes().hex() for f,v in bank.items()}}
    return result, receipt


def metrics(rows):
    present = [r for r in rows if r["present"]]
    return {**propagation_summary(rows),
            "present_iou": float(np.mean([r["scores"]["iou"] for r in present])) if present else None,
            "present_boundary_f": float(np.mean([r["scores"]["boundary_f"] for r in present])) if present else None}


@torch.inference_mode()
def rollout(args, repo, name, frames, condition, seed, folder, references, repeat=None):
    folder.mkdir(parents=True)
    chosen = args.targets["targets"][name]
    prompt, target = chosen["prompt"], chosen["frame"]
    native_rows, native_integrity, _ = references["native"]
    arm = "native" if condition == "sham" else ("v2" if condition == "regional" else f"random_{seed}")
    continuous_rows, continuous_integrity, _ = references[arm]
    inputs_path = args.target_replay_dir/f"{name}.zip"
    with zipfile.ZipFile(inputs_path) as packed:
        inputs = json.loads(packed.read("inputs.json"))
    if len(inputs) != len(frames):
        raise ValueError("Input count changed")
    for frame, old in zip(frames, inputs):
        if (frame.frame_index != old["frame"] or author.digest(frame.image_path) != old["image_sha256"]
                or author.digest(frame.mask_path) != old["mask_sha256"]):
            raise ValueError("Image/GT/chronology differs from archived native replay")
    model = author.build_author_predictor(args, repo)
    parameters = author.parameter_fingerprints(model)
    rows, receipts, native_reads = [], {}, []
    def observe(module, positional, kwargs):
        if positional:
            raise RuntimeError("Unexpected positional attention inputs")
        pointers = int(kwargs.get("num_obj_ptr_tokens", 0))
        native_reads.append({"selected_spatial_keys": int(kwargs["memory"].shape[0])-pointers,
                             "pointer_keys": pointers})
    handle = model.memory_attention.register_forward_pre_hook(observe, with_kwargs=True)
    native_masks = zipfile.ZipFile(chosen["archives"]["native"]["path"])
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="frames_", dir=folder) as staged:
            stage_frames(frames, Path(staged))
            state = model.init_state(video_path=staged, offload_video_to_cpu=True, offload_state_to_cpu=False)
            initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
            model.add_new_mask(state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial))
            iterator = model.propagate_in_video(state, start_frame_idx=prompt)
            try:
                for index in range(prompt, len(frames)):
                    native_reads.clear()
                    (actual, ids, logits), pulse = next_at_target(iterator, model, state, index, prompt, target, name, condition, seed)
                    if actual != index or ids != [1]:
                        raise RuntimeError("Chronology/object identity changed")
                    if native_reads != native_integrity[index]["trace"]["native_reads"]:
                        raise RuntimeError("Raw native spatial/pointer memory assembly changed")
                    outputs = state["output_dict_per_obj"][0]
                    stored = outputs["cond_frame_outputs"].get(index, outputs["non_cond_frame_outputs"].get(index))
                    if stored is None or stored["maskmem_features"] is None:
                        raise RuntimeError("Native writeback missing")
                    core = {"prediction_sha256": author.tensor_hash(logits),
                            "low_res_sha256": author.tensor_hash(stored["pred_masks"]),
                            "bank_sha256": fingerprint(state["output_dict_per_obj"])["sha256"],
                            "cache_sha256": fingerprint(state["cached_features"])["sha256"]}
                    expected = (native_integrity[index] if condition == "sham" or index < target
                                else continuous_integrity[index] if index == target else None)
                    if expected is not None and any(core[k] != expected[k] for k in CORE):
                        raise RuntimeError(f"Archived exact prefix/pulse/sham mismatch: {name}/{condition}/{seed}/{index}")
                    if pulse is not None and condition != "sham":
                        verify_layers(pulse["layers"], pulse["layers"][0]["frames"],
                                      pulse_bank_for_check(pulse), continuous_integrity[index]["trace"]["layers"])
                        expected_counts = continuous_integrity[index]["trace"]["allocation"]["per_frame_spatial_keys"]
                        if pulse["per_frame_counts"] != expected_counts:
                            raise RuntimeError("Pulse counts differ from continuous reference")
                        for frame in pulse["layers"][0]["frames"][1:-1]:
                            if pulse["keep_bits"][str(frame)] != continuous_integrity[frame]["trace"]["keep_bits_hex"]:
                                raise RuntimeError("Pulse token positions differ from continuous reference")
                    record = {"frame": index, **core, "pulse": pulse, "native_reads": native_reads.copy()}
                    if repeat is not None and record != repeat[index]:
                        raise RuntimeError("Fresh pulse rollout not bit-identical")
                    receipts[index] = record
                    with (folder/"integrity.jsonl").open("a") as stream:
                        stream.write(json.dumps(record)+"\n")
                    if index > prompt:
                        predicted = logits[0, 0].numpy()>0
                        native_mask = np.asarray(Image.open(BytesIO(native_masks.read(f"frame_{index:06d}.png")))) > 0
                        truth = load_label_mask(frames[index].mask_path, [1]) == 1
                        row = {"frame": index, "source_frame_index": frames[index].frame_index,
                               "prediction_sha256": core["prediction_sha256"], "present": bool(truth.any()),
                               "scores": {**calculate_scores(predicted, truth), "boundary_f": float(db_eval_boundary(predicted, truth))},
                               "absent_fp": bool(predicted.any()) if not truth.any() else None,
                               "different_pixels_from_native": int(np.count_nonzero(predicted != native_mask)),
                               "object_score": float(stored["object_score_logits"].item())}
                        rows.append(row)
                        Image.fromarray(predicted.astype(np.uint8)*255).save(folder/f"frame_{index:06d}.png")
                        with (folder/"frames.jsonl").open("a") as stream:
                            stream.write(json.dumps(row)+"\n")
                    progress = {"state": "running", "phase": folder.relative_to(args.output_dir).parts[0],
                                "sequence": name, "condition": condition, "seed": seed, "frame": index,
                                "last_frame": len(frames)-1, "pulse_frame": target}
                    (args.output_dir/"progress.json").write_text(json.dumps(progress, indent=2)+"\n")
                    if index == target or index % 10 == 0 or index == len(frames)-1:
                        print(f"[pulse] {name} {condition} seed={seed} frame={index} target={target}", flush=True)
                try:
                    next(iterator)
                    raise RuntimeError("Unexpected extra propagated frame")
                except StopIteration:
                    pass
            finally:
                iterator.close()
            model.reset_state(state)
    finally:
        handle.remove()
        native_masks.close()
    assert_native_forward(model)
    author.assert_parameter_integrity(model, parameters, folder, "pulse")
    if len(rows) != len(frames)-prompt-1 or sum(r["pulse"] is not None for r in receipts.values()) != 1:
        raise RuntimeError("Pulse or target coverage differs")
    summary = {"sequence": name, "condition": condition, "seed": seed, "pulse_frame": target,
               "whole": metrics(rows), "after_pulse": metrics([r for r in rows if r["frame"] > target]),
               "native": metrics(native_rows), "continuous": metrics(continuous_rows),
               "native_after_pulse": metrics([r for r in native_rows if r["frame"] > target]),
               "continuous_after_pulse": metrics([r for r in continuous_rows if r["frame"] > target]),
               "first_postpulse_logit_difference": next((r["frame"] for r in rows if r["frame"]>target and r["prediction_sha256"] != native_integrity[r["frame"]]["prediction_sha256"]), None),
               "first_postpulse_binary_difference": next((r["frame"] for r in rows if r["frame"]>target and r["different_pixels_from_native"]), None),
               "longest_lower_dice_run": longest_lower_run([r["frame"] for r in rows], [r["present"] for r in rows],
                    np.array([r["scores"]["dice"]-n["scores"]["dice"] for r,n in zip(rows,native_rows)])),
               "wall_seconds": time.monotonic()-started}
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    del model
    gc.collect()
    return summary, receipts


def pulse_bank_for_check(receipt):
    return {int(f): {"keep": torch.from_numpy(np.unpackbits(np.frombuffer(bytes.fromhex(bits), dtype=np.uint8)).astype(bool))}
            for f,bits in receipt["keep_bits"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "e2-dir", "offline-dir", "target-replay-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    args = parser.parse_args()
    e2 = require_e2(args.e2_dir)
    prior = json.loads((args.target_replay_dir/"delivery.json").read_text())
    if not prior["completed"] or not prior["wandb_artifact_upload_complete"]:
        raise ValueError("Completed source28 required")
    args.targets = json.loads((ROOT/TARGETS).read_text())
    if args.targets["offline_targets_sha256"] != author.digest(args.offline_dir/"pulse_targets.json"):
        raise ValueError("Committed target receipt differs from offline analysis")
    if author.digest(args.checkpoint) != e2["setup"]["checkpoint_sha256"] or author.digest(args.manifest) != e2["setup"]["manifest_sha256"]:
        raise ValueError("Checkpoint/manifest identity differs")
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *SOURCES], cwd=ROOT, check=True, capture_output=True)
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR", "TMPDIR"):
        os.environ[key] = str(output/key.lower())
        Path(os.environ[key]).mkdir()
    args.method, args.device = "sam2", "cpu"
    repo, revision = author.source_identity("sam2")
    sys.path.insert(0, str(repo))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = {n: author.digest(ROOT/n) for n in set(SOURCES) | set(e2["setup"]["sources"])}
    setup = {"protocol": json.loads((ROOT/PROTOCOL).read_text()), "targets": args.targets, "sources": sources,
             "author_revision": revision, "checkpoint_sha256": author.digest(args.checkpoint),
             "manifest_sha256": author.digest(args.manifest), "torch": torch.__version__,
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (output/"setup.json").write_text(json.dumps(setup, indent=2)+"\n")
    with zipfile.ZipFile(output/"sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(ROOT/name, arcname=name)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "29-single pruning pulse propagation", "online",
                     "https://forge.coreweave.com/api/wandb", setup, output, ["pulse", "v2", "exploratory"])
    try:
        videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
        references = {}
        for name, target in args.targets["targets"].items():
            references[name] = {}
            for arm, archive in target["archives"].items():
                path = Path(archive["path"])
                if author.digest(path) != archive["sha256"]:
                    raise ValueError("Frozen reference archive changed")
                references[name][arm] = archive_rows(path)
        results = {}
        conditions = [("regional",None)]+[("random",seed) for seed in SEEDS]
        def execute(name, condition, seed, folder, repeat=None):
            if any(author.digest(ROOT/n) != sha for n,sha in sources.items()):
                raise RuntimeError("Frozen source changed during pilot")
            summary, receipts = rollout(args, repo, name, videos[name], condition, seed, folder, references[name], repeat)
            filename = "_".join(folder.relative_to(output).parts)+".zip"
            with zipfile.ZipFile(output/filename, "w", zipfile.ZIP_DEFLATED) as z:
                for path in folder.iterdir():
                    z.write(path, arcname=path.name)
            log_results_artifact(run, output, [filename])
            return summary, receipts
        execute("seq2", "sham", None, output/"gate/seq2/sham")
        results["seq2"] = []
        for condition, seed in conditions:
            label = condition if seed is None else f"random_{seed}"
            summary, first = execute("seq2", condition, seed, output/"gate/seq2"/f"{label}_first")
            execute("seq2", condition, seed, output/"gate/seq2"/f"{label}_repeat", repeat=first)
            results["seq2"].append(summary)
        (output/"validation.json").write_text(json.dumps({"passed": True, "sequence": "seq2", "native_sham_exact": True,
            "all_four_conditions_fresh_repeat_exact": True, "prefix_and_pulse_match_archives": True,
            "one_writeback_pulse_then_native": True, "sources": sources}, indent=2)+"\n")
        log_results_artifact(run, output, ["validation.json", "setup.json", "sources.zip"])
        results["seq13"] = []
        for condition, seed in conditions:
            label = condition if seed is None else f"random_{seed}"
            summary, _ = execute("seq13", condition, seed, output/"pilot/seq13"/label)
            results["seq13"].append(summary)
        (output/"results.json").write_text(json.dumps({"completed": True, "exploratory": True, "sequences": results}, indent=2)+"\n")
        lines = ["# One-time pruning followed by native propagation", "", "Post-hoc selected two-sequence diagnostic. Pulse frame fixed by first actual pruning, not outcome optimization. The pulse prediction is written normally; subsequent reading is native. No population inference.", "",
                 "| Sequence | Condition | Pulse | Whole present Dice | After-pulse present Dice | Whole Boundary F | False alarms |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for name, summaries in results.items():
            for s in summaries:
                lines.append(f"| {name} | {s['condition']} {s['seed'] or ''} | {s['pulse_frame']} | {s['whole']['present_dice']:.6f} | {s['after_pulse']['present_dice']:.6f} | {s['whole']['present_boundary_f']:.6f} | {s['whole']['gt_empty_false_alarm_count']} |")
        (output/"results.md").write_text("\n".join(lines)+"\n")
        finalize_tracking(run, output, {"completed": True, "gate_passed": True, "sequences": 2}, ["results.json","results.md","validation.json","setup.json"])
    except BaseException as error:
        (output/"failure.json").write_text(json.dumps({"error": repr(error), "time": time.time()}, indent=2)+"\n")
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
