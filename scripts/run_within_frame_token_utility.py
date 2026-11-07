"""Frozen native-history within-frame diagnostic, reusing Run171 replay.

First invocation completes seq2 then stops intentionally. --resume regenerates
its native history, checks saved outputs and repeats saved interventions before
allowing the remaining cohort. All runtime artifacts belong on Drive.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cfbi_training import capture_training_read
from modeling.within_frame_token_utility import ARMS, WithinFrameTokenRead, choose_source, removal_plan
from scripts import run_complementary_memory_pilot as base
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_replay import check_exact, check_layout
from scripts.run_regional_token_selection import EXTRA_SOURCES, SEQUENCES, replay, scored
from scripts.run_token_contribution import native_signatures
from scripts.verify_within_frame_token_utility import SOURCES as ENGINEERING_SOURCES
from scripts.verify_within_frame_token_utility import RUNTIME_SOURCES
from scripts.within_frame_utility_records import read_receipt, sample_targets, summarize, write_receipt, write_report

PROTOCOL = "protocols/within_frame_token_utility_v1.json"
SOURCES = (*ENGINEERING_SOURCES, "scripts/run_within_frame_token_utility.py",
           "scripts/within_frame_utility_records.py", "tests/test_within_frame_utility_records.py",
           "tests/test_within_frame_utility_runner.py")


def status(out, phase, **values):
    base.atomic_json(out/"run_status.json", {"pid": os.getpid(), "phase": phase,
        "time_utc": datetime.now(timezone.utc).isoformat(), **values})


def guard(core, videos, name=None):
    if core["sources"] != {p: digest(ROOT/p) for p in core["sources"]}:
        raise RuntimeError("Frozen source changed")
    names = [name] if name is not None else SEQUENCES
    if {n: core["inputs"][n] for n in names} != base.input_hashes(videos, names):
        raise RuntimeError("Frozen input changed")


def verify_engineering(folder, checkpoint):
    completion = json.loads((folder/"completion.json").read_text())
    setup = json.loads((folder/"setup.json").read_text())
    if not completion["passed"] or completion["sequences"] != ["seq2", "seq3"]:
        raise RuntimeError("Missing full engineering verification")
    if completion["setup_sha256"] != digest(folder/"setup.json") or setup["checkpoint_sha256"] != digest(checkpoint):
        raise RuntimeError("Engineering identity differs")
    if any(digest(ROOT/p) != h for p, h in setup["sources"].items()):
        raise RuntimeError("Engineering verified different sources")
    for name in ("seq2", "seq3"):
        receipt = json.loads((folder/name/"receipt.json").read_text())
        if (not receipt["passed"] or len(receipt["interventions"]) != 35
                or digest(folder/name/"receipt.json") != completion["receipt_sha256"][name]):
            raise RuntimeError("Incomplete source-isolation engineering receipt")
    return {"folder": str(folder.resolve()), "completion_sha256": digest(folder/"completion.json"),
            "receipts": {n: digest(folder/n/"receipt.json") for n in ("seq2", "seq3")}}


def measured_replay(model, state, sample, observer, frames, frame, folder, reference):
    started = time.monotonic()
    result, trace = replay(model, sample, observer)
    elapsed = time.monotonic()-started
    check_layout(trace, reference)
    for layer in trace["selection_layers"]:
        expected = [4096-int(removal_plan(observer.arm).sum())
                    if observer.arm != "native" and f == observer.source_frame else 4096
                    for f in layer["frame_ids"]]
        if layer["counts"] != expected:
            raise RuntimeError("Source/count isolation failed")
    if not all(torch.isfinite(result[k]).all() for k in ("pred_masks", "obj_ptr", "object_score_logits")):
        raise RuntimeError("Nonfinite replay outputs")
    full = model._get_orig_video_res_output(state, result["pred_masks"])[1]
    row = scored(full, frames, frame, folder, trace)
    row["object_score_logit"] = float(result["object_score_logits"].item())
    row["output_hashes"] = {k: tensor_hash(result[k]) for k in ("pred_masks", "obj_ptr", "object_score_logits")}
    row["foreground_pixels"] = int((full > 0).sum())
    return row, elapsed


@torch.inference_mode()
def sequence(args, videos, name, model, out, core):
    frames = videos[name]
    folder = out/"sequences"/name
    folder.mkdir(parents=True, exist_ok=True)
    observer = WithinFrameTokenRead(model)
    weights = parameter_digest(model)
    targets = core["target_plan"][name]["targets"]
    quality, pending, main_records, followups = {}, {}, [], []
    resumed_targets = 0
    with tempfile.TemporaryDirectory(dir=folder, prefix="native_frames_") as temp:
        state, prompt = base.prepare(model, frames, Path(temp))
        if prompt != core["target_plan"][name]["prompt"]:
            raise RuntimeError("Prompt changed")
        iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
        seen = []
        while True:
            observer.arm, observer.source_frame = "native", None
            with observer:
                try:
                    frame, ids, logits = next(iterator)
                except StopIteration:
                    break
            if ids != [1] or frame != prompt+len(seen):
                raise RuntimeError("Native object/order mismatch")
            seen.append(frame)
            # Saved native quality is from the source frame's own original output.
            signature = native_signatures(state, frame, prompt, logits)
            native_path = folder/"native_masks"/f"{frame:06d}.json"
            native_context = {"frame": frame, "setup_sha256": core["setup_sha256"]}
            if native_path.exists():
                saved = read_receipt(native_path, native_context)
                if saved["signatures"] != signature:
                    raise RuntimeError("Regenerated native trajectory changed")
                quality[frame] = saved["source_quality"]
            else:
                qrow = scored(logits, frames, frame, folder/"native_masks", {})
                quality[frame] = {k: qrow[k] for k in ("present", "scores", "prediction_sha256", "source_frame_index")}
                quality[frame]["stored_low_mask_sha256"] = signature["pred_masks"]
                write_receipt(native_path, {"context": native_context,
                    "artifacts": {f"{frame:06d}.png": qrow["mask_sha256"]},
                    "signatures": signature, "source_quality": quality[frame]})
            requests = pending.pop(frame, [])
            if frame in targets:
                plan = choose_source(observer.calls[-1]["memory_frames"], name, frame)
                requests.append({"anchor_target": frame, "kind": "main", **plan})
                if frame+1 < len(frames):
                    pending.setdefault(frame+1, []).append({**requests[-1], "kind": "followup"})
                else:
                    followups.append({"sequence": name, "anchor_target": frame, "eligible": False, "reason": "end_of_sequence"})
            if requests:
                collected = observer.calls[-1]
                sample = capture_training_read(model, state, frame)
                sample_sha = fingerprint(sample)["sha256"]
                bank_sha = lambda: fingerprint(bank_contents(state, frame, [], model.max_obj_ptrs_in_encoder))["sha256"]
                before = bank_sha()
                native, trace = replay(model, sample, observer)
                check_exact(native, state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame])
                check_layout(trace, collected)
                if tensor_hash(model._get_orig_video_res_output(state, native["pred_masks"])[1]) != tensor_hash(logits):
                    raise RuntimeError("Native full-resolution replay changed")
                for request in requests:
                    source = request["source_frame"]
                    if source not in [s["frame"] for s in trace["memory_frames"][1:-1]]:
                        if request["kind"] == "main":
                            raise RuntimeError("Planned source unexpectedly missing")
                        followups.append({"sequence": name, "anchor_target": request["anchor_target"], "eligible": False,
                                          "reason": "source_not_in_next_native_read"})
                        continue
                    observer.source_frame = source
                    bank = sample["output_dict"]
                    stored = bank["cond_frame_outputs"].get(source, bank["non_cond_frame_outputs"].get(source))
                    if tensor_hash(stored["pred_masks"]) != quality[source]["stored_low_mask_sha256"]:
                        raise RuntimeError("Source quality no longer describes the mask stored in replay")
                    dest = folder/f"{request['kind']}_{request['anchor_target']:06d}"
                    dest.mkdir(exist_ok=True)
                    arms = ARMS if request["kind"] == "main" else ARMS[:17]
                    rows, resumed = {}, False
                    for arm in arms:
                        observer.arm = arm
                        path = dest/f"{arm}.json"
                        context = {"setup_sha256": core["setup_sha256"], "sample_sha256": sample_sha,
                            "sequence": name, "target_frame": frame, "request": request, "arm": arm}
                        if path.exists():
                            saved = read_receipt(path, context)
                            rows[arm] = saved["row"]
                            if not resumed and arm != "native":  # repeat a cached altered arm, not just identity
                                with tempfile.TemporaryDirectory(dir=dest, prefix="resume_check_") as scratch:
                                    checked, _ = measured_replay(model, state, sample, observer, frames, frame, Path(scratch), trace)
                                if checked != rows[arm]:
                                    raise RuntimeError("Resumed replay differs from saved output")
                                resumed, resumed_targets = True, resumed_targets+1
                        else:
                            rows[arm], elapsed = measured_replay(model, state, sample, observer, frames, frame, dest/arm, trace)
                            mask_path = dest/arm/f"{frame:06d}.png"
                            write_receipt(path, {"context": context, "row": rows[arm], "replay_seconds": elapsed,
                                "artifacts": {str(mask_path.relative_to(dest)): digest(mask_path)}})
                        status(out, "diagnostic", sequence=name, frame=frame, kind=request["kind"], arm=arm)
                    record = {"sequence": name, "frame": frame, **request, "source_frame": source,
                        "source_age_processed": frame-source,
                        "source_age_original": frames[frame].frame_index-frames[source].frame_index,
                        "source_quality": quality[source], "arms": rows}
                    (main_records if request["kind"] == "main" else followups).append(record)
                if bank_sha() != before:
                    raise RuntimeError("Replay changed native bank/cache")
            observer.calls.clear()
            if frame % 10 == 0:
                print(f"[{name}] native {frame}/{len(frames)-1}; main targets {len(main_records)}/{len(targets)}", flush=True)
        if seen != list(range(prompt, len(frames))) or [r["frame"] for r in main_records] != targets or pending:
            raise RuntimeError("Incomplete trajectory/target coverage")
    if parameter_digest(model) != weights:
        raise RuntimeError("Pretrained weights changed")
    result = {"main": main_records, "followups": followups}
    if (folder/"results.json").exists() and json.loads((folder/"results.json").read_text()) != result:
        raise RuntimeError("Resumed sequence results differ; preserving originals")
    base.atomic_json(folder/"results.json", result)
    base.atomic_json(folder/"completion.json", {"complete": True, "resumed_targets_verified": resumed_targets,
        "results_sha256": digest(folder/"results.json"), "parameter_sha256": weights})
    return main_records, followups, resumed_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engineering-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT/PROTOCOL).read_text())
    if protocol.get("status") != "frozen_for_scientific_diagnostic":
        raise RuntimeError("Scientific protocol is not frozen")
    for p in SOURCES:
        if subprocess.check_output(["git", "show", f"HEAD:{p}"], cwd=ROOT) != (ROOT/p).read_bytes():
            raise RuntimeError(f"Commit unchanged scientific sources first: {p}")
    ref = json.loads((ROOT/"protocols/token_contribution_v1.json").read_text())
    manifest = ROOT/"data/polypgen_sequence.jsonl"
    if digest(args.checkpoint) != ref["checkpoint_sha256"] or digest(manifest) != ref["manifest_sha256"]:
        raise RuntimeError("Checkpoint/manifest changed")
    gate = verify_engineering(args.engineering_dir, args.checkpoint)
    videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
    planned = sample_targets(videos, ref["target_plan"])
    if planned != protocol.get("target_plan"):
        raise RuntimeError("Targets differ from frozen protocol")
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out/"process.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["TMPDIR"] = str(out)
    os.environ["MPLCONFIGDIR"] = str(out/"matplotlib_cache")
    tempfile.tempdir = str(out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths = sorted(set(base.SOURCE_REL) | set(EXTRA_SOURCES) | set(SOURCES) | set(RUNTIME_SOURCES) |
                   {"scripts/run_token_contribution.py", "protocols/token_contribution_v1.json"})
    core = {"protocol": protocol, "sources": {p: digest(ROOT/p) for p in paths},
            "inputs": base.input_hashes(videos, SEQUENCES), "target_plan": planned,
            "checkpoint_sha256": digest(args.checkpoint), "engineering": gate, "torch": torch.__version__}
    if args.resume:
        if core != json.loads((out/"setup.json").read_text()):
            raise RuntimeError("Resume configuration/source/input mismatch")
    else:
        base.atomic_json(out/"setup.json", core)
        with zipfile.ZipFile(out/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(ROOT/p, arcname=p)
    core["setup_sha256"] = digest(out/"setup.json")
    started = time.monotonic()
    try:
        repo, _ = base.activate_author_repo()
        records, followups = [], []
        for name in SEQUENCES:
            guard(core, videos, name)
            model = base.init_predictor(args.checkpoint, repo)
            rows, later, checked = sequence(args, videos, name, model, out, core)
            guard(core, videos, name)
            del model
            gc.collect()
            records.extend(rows)
            followups.extend(later)
            result = summarize(records, followups)
            base.atomic_json(out/"partial_results.json", result)
            write_report(out, result)
            if name == "seq2":
                if not args.resume:
                    status(out, "awaiting_planned_resume", complete_sequence="seq2")
                    return
                if not checked:
                    raise RuntimeError("Cohort refused: no regenerated resume comparisons")
                base.atomic_json(out/"resume_verification.json", {"passed": True, "sequence": name,
                    "verified_targets": checked, "setup_sha256": core["setup_sha256"]})
        guard(core, videos)
        base.atomic_json(out/"results.json", summarize(records, followups))
        base.atomic_json(out/"completion.json", {"complete": True, "results_sha256": digest(out/"results.json"),
            "setup_sha256": core["setup_sha256"], "invocation_elapsed_seconds": time.monotonic()-started})
        status(out, "complete", sequences=len(SEQUENCES))
    except BaseException as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base.atomic_json(out/f"failure_{stamp}.json", {"error": repr(exc), "no_bypass": True})
        status(out, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
