"""Run170: engineering, fixed-native diagnostic, conditional full propagation."""

import argparse
from datetime import datetime, timezone
import gc
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
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust
from modeling.cfbi_training import capture_training_read, clone_tensors
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.regional_token_selection import NativeRegionalTokenSelection, author_selectors
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts import run_complementary_memory_pilot as base
from scripts.run_complementary_memory_replay import check_exact, check_layout, arm_summary
from training.tracking import start_wandb

PROTOCOL = "protocols/regional_token_selection_v1.json"
ARMS = ("native", "usage", "foreground", "role_blind", "regional")
SEQUENCES = [f"seq{i}" for i in range(2, 24) if i != 7]
EXTRA_SOURCES = (
    PROTOCOL, "modeling/regional_token_selection.py", "scripts/run_regional_token_selection.py",
    "tests/test_regional_token_selection.py", "modeling/sparse_memory.py",
    "modeling/native_memory_layout.py", "modeling/region_attention.py",
    "scripts/run_complementary_memory_replay.py",
    "external/Lean-SAM2/sam2/modeling/sam2_base.py",
    "external/Efficient-SAM2/sam2/modeling/sam2_base.py",
    "external/Efficient-SAM2/build_model/window_bypass.py",
    "external/Efficient-SAM2/sam2/modeling/sam/transformer.py",
)


def update_status(out, phase, **values):
    base.atomic_json(out / "run_status.json", {"phase": phase, "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(), **values})


def setup(args, videos, out):
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    committed = subprocess.check_output(["git", "show", f"HEAD:{PROTOCOL}"], cwd=ROOT)
    if committed != (ROOT / PROTOCOL).read_bytes():
        raise RuntimeError("Commit the unchanged scientific protocol before real inference")
    if protocol['arms'] != list(ARMS) or protocol['selection']['keep_fraction_of_each_older_frame'] != .25:
        raise RuntimeError("Runtime selectors/budget differ from frozen protocol")
    if digest(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Official SAM2.1 Tiny checkpoint differs from protocol")
    if digest(ROOT / "data/polypgen_sequence.jsonl") != protocol["manifest_sha256"]:
        raise RuntimeError("Manifest differs from protocol")
    author_selectors()
    paths = set(base.SOURCE_REL) | set(EXTRA_SOURCES)
    sources = {name: digest(ROOT / name) for name in sorted(paths)}
    core = {"protocol": protocol, "sources": sources, "checkpoint": str(args.checkpoint),
        "inputs": base.input_hashes(videos, SEQUENCES),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "tracking_mode": args.wandb_mode}
    base.atomic_json(out / "setup.json", core)
    with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(paths):
            archive.write(ROOT / name, arcname=name)
    return core


def guard(core, videos, name=None):
    for path, sha in core["sources"].items():
        if digest(ROOT / path) != sha:
            raise RuntimeError(f"Source changed during experiment: {path}")
    if name and base.input_hashes(videos, [name])[name] != core["inputs"][name]:
        raise RuntimeError(f"Input changed during experiment: {name}")


def scored(logits, frames, frame, folder, trace):
    prediction = logits[0, 0].cpu().numpy() > 0
    truth = load_label_mask(frames[frame].mask_path, [1]) == 1
    if prediction.shape != truth.shape or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid prediction shape/logits")
    folder.mkdir(parents=True, exist_ok=True)
    mask = folder / f"{frame:06d}.png"
    Image.fromarray(prediction.astype(np.uint8) * 255).save(mask)
    return {"frame": frame, "source_frame_index": frames[frame].frame_index,
        "present": bool(truth.any()), "absent_fp": bool(prediction.any()) if not truth.any() else None,
        "scores": calculate_scores(prediction, truth) | {"boundary_f": float(db_eval_boundary(prediction, truth))},
        "prediction_sha256": tensor_hash(logits), "mask_sha256": digest(mask), "read": trace}


def replay(model, sample, controller):
    before = fingerprint(sample)["sha256"]
    local = clone_tensors(sample)
    old_count = len(controller.calls)
    with patch.object(model, "_encode_new_memory", side_effect=RuntimeError("Diagnostic wrote memory")):
        with controller:
            result = model.track_step(frame_idx=local["frame"], is_init_cond_frame=False,
                current_vision_feats=local["vision"], current_vision_pos_embeds=local["positions"],
                feat_sizes=local["sizes"], point_inputs=None, mask_inputs=None,
                output_dict=local["output_dict"], num_frames=local["num_frames"],
                track_in_reverse=False, run_mem_encoder=False)
    if (len(controller.calls) != old_count + 1 or result["maskmem_features"] is not None
            or fingerprint(local)["sha256"] != before or fingerprint(sample)["sha256"] != before):
        raise RuntimeError("Diagnostic replay changed its input or wrote memory")
    return result, controller.calls[-1]


def save_rows(folder, rows, controller=None):
    base.state_to_json_rows(folder / "frames.jsonl", rows)
    if controller is not None:
        torch.save({"selection_masks": controller.masks, "prompt_anchors": controller.anchors},
                   folder / "selection_cache.pt")


@torch.inference_mode()
def diagnostic(args, videos, repo, out, core):
    summaries = {arm: {} for arm in ARMS}
    different_reads, pruned_reads, targets = 0, 0, 0
    for name in ("seq2", "seq3"):
        guard(core, videos, name)
        model = base.init_predictor(args.checkpoint, repo)
        model_hash = parameter_digest(model)
        controllers = {arm: NativeRegionalTokenSelection(model, "identity" if arm == "native" else arm)
                       for arm in ARMS}
        rows = {arm: [] for arm in ARMS}
        target = out / "diagnostic" / name
        target.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=target, prefix="frames_") as temp:
            state, prompt = base.prepare(model, videos[name], Path(temp))
            expected = list(range(prompt+1, len(videos[name])))
            repeat = {expected[0], expected[len(expected)//2], expected[-1]}
            for frame, ids, original in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError("Unexpected persistent object ID")
                if frame == prompt:
                    continue
                if frame != expected[len(rows["native"])]:
                    raise RuntimeError("Diagnostic target order changed")
                sample = capture_training_read(model, state, frame)
                signature = lambda: fingerprint(bank_contents(state, frame, [], model.max_obj_ptrs_in_encoder))["sha256"]
                before = signature()
                results = {}
                for arm in ARMS:
                    controller = controllers[arm]
                    previous_masks, previous_anchors = dict(controller.masks), dict(controller.anchors)
                    result, trace = replay(model, sample, controller)
                    if arm == "native":
                        check_exact(result, state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame])
                        if tensor_hash(model._get_orig_video_res_output(state, result["pred_masks"])[1]) != tensor_hash(original):
                            raise RuntimeError("All-token full-resolution native parity failed")
                    else:
                        check_layout(trace, results["native"][1])
                    if frame in repeat:
                        controller.masks, controller.anchors = previous_masks, previous_anchors
                        repeated, repeated_trace = replay(model, sample, controller)
                        check_exact(result, repeated)
                        if trace != repeated_trace:
                            raise RuntimeError("Repeated selector replay changed indices or trace")
                    results[arm] = (result, trace)
                    logits = model._get_orig_video_res_output(state, result["pred_masks"])[1]
                    folder = target / arm
                    rows[arm].append(scored(logits, videos[name], frame, folder / "masks", trace))
                    rows[arm][-1]['object_score_logit'] = float(result['object_score_logits'].item())
                    with (folder / "frames.jsonl").open("a") as stream:
                        stream.write(json.dumps(rows[arm][-1], allow_nan=False) + "\n")
                if signature() != before:
                    raise RuntimeError("Diagnostic changed native bank or image cache")
                for left, right in zip(results["regional"][1]["selection_layers"], results["foreground"][1]["selection_layers"]):
                    if left["counts"] != right["counts"]:
                        raise RuntimeError("Published/regional selector budgets differ")
                    different_reads += left["selected_indices_sha256"] != right["selected_indices_sha256"]
                    pruned_reads += left["spatial_keys_after"] < left["spatial_keys_before"]
                targets += 1
                update_status(out, "diagnostic", sequence=name, completed_targets=targets,
                              expected_targets=73, frame=frame)
                if len(rows["native"]) % 10 == 0:
                    print(f"[diagnostic] {name} {len(rows['native'])}/{len(expected)}", flush=True)
            model.reset_state(state)
        if parameter_digest(model) != model_hash:
            raise RuntimeError("Diagnostic changed frozen SAM2 weights")
        for arm in ARMS:
            save_rows(target / arm, rows[arm], controllers[arm])
            summaries[arm][name] = arm_summary(rows[arm], model_hash)
        guard(core, videos, name)
        del model, state, sample, controllers, rows, results
        gc.collect()
    if targets != 73 or not pruned_reads or not different_reads:
        raise RuntimeError("Gate did not cover all targets, active pruning and distinct regional selection")
    result = summarize(summaries)
    continuation = diagnostic_continuation(result)
    result.update(engineering_passed=True, targets=targets, distinct_regional_reads=different_reads,
        pruned_reads=pruned_reads, continue_full_propagation=bool(continuation))
    base.atomic_json(out / "diagnostic_results.json", result)
    return result


@torch.inference_mode()
def propagate(args, videos, repo, out, core):
    summaries = {arm: {} for arm in ARMS}
    for name in SEQUENCES:
        reference_layout = None
        for arm in ARMS:
            guard(core, videos, name)
            update_status(out, "full_propagation", sequence=name, arm=arm,
                          completed_sequences=len(summaries[arm]), expected_sequences=len(SEQUENCES))
            model = base.init_predictor(args.checkpoint, repo)
            model_hash = parameter_digest(model)
            folder = out / "full" / name / arm
            folder.mkdir(parents=True)
            rows, layouts = [], []
            with tempfile.TemporaryDirectory(dir=folder, prefix="frames_") as temp:
                state, prompt = base.prepare(model, videos[name], Path(temp))
                controller = (NativeComplementaryRead(model, None, inference_state=state, trace_only=True)
                    if arm == "native" else NativeRegionalTokenSelection(model, arm, inference_state=state))
                started = time.monotonic()
                with controller:
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1]:
                            raise RuntimeError("Propagation object order differs")
                        if frame == prompt and not rows:
                            continue
                        if frame != prompt + len(rows) + 1:
                            raise RuntimeError("Propagation frame order differs")
                        trace = controller.calls[-1]
                        layouts.append((trace["memory_frames"], trace["pointer_keys"]))
                        row = scored(logits, videos[name], frame, folder / "masks", trace)
                        row['object_score_logit'] = float(state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]['object_score_logits'].item())
                        rows.append(row)
                        with (folder / "frames.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        base.atomic_json(out / "progress.json", {"phase": "full_propagation", "arm": arm,
                            "sequence": name, "frame": frame, "targets": len(rows),
                            "expected_targets": len(videos[name])-prompt-1, "pid": os.getpid(),
                            "updated_at": datetime.now(timezone.utc).isoformat()})
                seconds = time.monotonic() - started
                model.reset_state(state)
            if len(rows) != len(videos[name])-prompt-1 or parameter_digest(model) != model_hash:
                raise RuntimeError("Incomplete propagation or frozen parameter change")
            if reference_layout is None:
                reference_layout = layouts
            elif reference_layout != layouts:
                raise RuntimeError("Token method changed native memory-frame selection or pointer counts")
            summary = arm_summary(rows, model_hash) | {"elapsed_seconds": seconds, "sequence": name,
                "propagation": base.propagation_summary(rows)}
            summaries[arm][name] = summary
            save_rows(folder, rows, None if arm == "native" else controller)
            base.atomic_json(folder / "summary.json", summary)
            base.atomic_json(out / "partial_results.json", summaries)
            guard(core, videos, name)
            print(f"[full] {name} {arm} done; Dice={summary['present_dice']:.6f}", flush=True)
            del model, state, controller, rows
            gc.collect()
        sync_docs(out, f"Full propagation: {len(summaries['regional'])}/{len(SEQUENCES)} sequences complete.")
    if any(len(v) != 21 or sum(s["target_frames"] for s in v.values()) != 2012 for v in summaries.values()):
        raise RuntimeError("Full-cohort coverage differs from protocol")
    result = summarize(summaries)
    result["criterion_passed"] = full_screen(result, len(SEQUENCES))
    return result


def diagnostic_continuation(result):
    comparisons = result['contrasts']
    return bool(comparisons['native']['mean_dice_difference'] > 0
        and all(comparisons[a]['mean_dice_difference'] >= 0 for a in ('foreground', 'role_blind'))
        and all(comparisons[a]['equal_sequence_false_alarm_difference'] <= .05
                and comparisons[a]['pooled_false_alarm_difference'] <= .05
                for a in ('native', 'foreground', 'role_blind')))


def full_screen(result, sequence_count):
    return bool(all(
        result["contrasts"][arm]["mean_dice_difference"] >= margin
        and result["contrasts"][arm]["sequence_bootstrap_95_ci"][0] > 0
        and result["contrasts"][arm]["wins"] > sequence_count/2
        and result["contrasts"][arm]["equal_sequence_false_alarm_difference"] <= .05
        and result["contrasts"][arm]["pooled_false_alarm_difference"] <= .05
        for arm, margin in (("native", .02), ("foreground", .01), ("role_blind", .01))))


def summarize(summaries):
    names = list(summaries["native"])
    if not names or any(set(values) != set(names) for values in summaries.values()):
        raise RuntimeError("Paired summary coverage differs")
    for name in names:
        for key in ('target_frames', 'present_frames', 'gt_empty_frames', 'original_sam2_parameter_sha256'):
            if len({summaries[a][name][key] for a in ARMS}) != 1:
                raise RuntimeError(f'Paired summary denominator/weights differ: {name}/{key}')
        if summaries['native'][name]['present_frames'] <= 0:
            raise RuntimeError(f'No primary endpoint denominator: {name}')
    result = {"per_sequence": summaries, "means": {}, "contrasts": {}}
    for arm, values in summaries.items():
        result["means"][arm] = {key: float(np.mean([r[key] for r in values.values()]))
            for key in ("present_dice", "present_iou", "present_boundary_f")}
        result["means"][arm].update(false_alarms=sum(r["gt_empty_false_alarm_count"] for r in values.values()),
                                    gt_empty_frames=sum(r["gt_empty_frames"] for r in values.values()))
    pvalues = {}
    for arm in ("native", "usage", "foreground", "role_blind"):
        left = {n: summaries["regional"][n]["present_dice"] for n in names}
        right = {n: summaries[arm][n]["present_dice"] for n in names}
        differences = {n: left[n] - right[n] for n in names}
        fa = [summaries["regional"][n]["gt_empty_false_alarm_rate"] - summaries[arm][n]["gt_empty_false_alarm_rate"]
              for n in names if summaries[arm][n]["gt_empty_frames"]]
        empty = result["means"][arm]["gt_empty_frames"]
        pvalues[arm] = paired_wilcoxon(left, right)["p"]
        result["contrasts"][arm] = {"mean_dice_difference": float(np.mean(list(differences.values()))),
            "per_sequence_delta": differences,
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(differences, seed=20261107),
            "wins": sum(v > 0 for v in differences.values()), "ties": sum(v == 0 for v in differences.values()),
            "losses": sum(v < 0 for v in differences.values()),
            "equal_sequence_false_alarm_difference": float(np.mean(fa)) if fa else 0.,
            "pooled_false_alarm_difference": ((result["means"]["regional"]["false_alarms"]-result["means"][arm]["false_alarms"])/empty) if empty else 0.,
            "false_alarm_evaluable_sequences": len(fa)}
    result["primary_holm_pvalues"] = holm_adjust({k:v for k,v in pvalues.items() if k in ("native", "foreground", "role_blind")})
    result["usage_secondary_pvalue"] = pvalues["usage"]
    return result


def sync_docs(out, state, result=None):
    lines = ["# Regional memory-token selection: Run170", "", f"**Status:** {state}", "",
        "One fixed configuration; PolypGen is exploratory because it has already informed development.",
        "SAM2 weights and native memory-frame selection are fixed. Prompt and latest memory stay full;",
        "older frames retain 25% of tokens per layer. No new confidence threshold or training.", ""]
    if result:
        lines += ["| Condition | Present Dice | IoU | Boundary F | False alarms / empty frames |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            r = result["means"][arm]
            lines.append(f"| {arm} | {r['present_dice']:.6f} | {r['present_iou']:.6f} | {r['present_boundary_f']:.6f} | {r['false_alarms']}/{r['gt_empty_frames']} |")
        lines += ["", "| Sequence | Native | Attention only | Foreground anchors | Role-blind anchors | Regional anchors |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in result["per_sequence"]["native"]:
            lines.append(f"| {name} | " + " | ".join(f"{result['per_sequence'][a][name]['present_dice']:.6f}" for a in ARMS) + " |")
        for arm in ("native", "foreground", "role_blind"):
            row = result["contrasts"][arm]
            lines += ["", f"Regional minus {arm}: {row['mean_dice_difference']:+.6f}; 95% sequence-bootstrap CI {row['sequence_bootstrap_95_ci']}; wins/ties/losses {row['wins']}/{row['ties']}/{row['losses']}."]
    lines += ["", f"Artifacts: `{out}`. Mounted Drive presence does not independently prove remote upload.", "",
              "[Method](REGION_TOKENS.md) · [Frozen protocol](../protocols/regional_token_selection_v1.json)"]
    text = "\n".join(lines) + "\n"
    (ROOT / "docs/REGION_TOKENS_RESULTS.md").write_text(text)
    (out / "results.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(out)
    os.environ["WANDB_CACHE_DIR"] = str(out / "wandb_cache")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    videos = load_manifest(ROOT / "data/polypgen_sequence.jsonl", "dev") | load_manifest(ROOT / "data/polypgen_sequence.jsonl", "test")
    run, started = None, time.monotonic()
    try:
        update_status(out, "preflight")
        core = setup(args, videos, out)
        repo, _ = base.activate_author_repo()
        try:
            run = start_wandb("ade-seg", "adenoid-hypertrophy", "170 regional token selection",
                args.wandb_mode, None, {"protocol": core["protocol"]}, out, ["token-selection", "exploratory"])
        except Exception as exc:
            base.atomic_json(out / "tracking_error.json", {"error": repr(exc), "science_continues": True})
        sync_docs(out, "Engineering and fixed-native diagnostic running; no efficacy result yet.")
        result = diagnostic(args, videos, repo, out, core)
        if result["continue_full_propagation"]:
            sync_docs(out, "Engineering passed; diagnostic continuation criterion passed. Full propagation starting.", result)
            result = propagate(args, videos, repo, out, core)
            decision = "full propagation screen passed; exploratory only" if result["criterion_passed"] else "full propagation screen failed; stop this candidate"
        else:
            decision = "engineering passed; diagnostic continuation criterion failed; full propagation not launched"
        guard(core, videos)
        result["decision"] = decision
        base.atomic_json(out / "results.json", result)
        sync_docs(out, decision, result)
        base.atomic_json(out / "completion.json", {"complete": True, "decision": decision,
            "results_sha256": digest(out / "results.json"), "elapsed_seconds": time.monotonic()-started,
            "setup_sha256": digest(out / "setup.json")})
        if run:
            try:
                run.summary.update({"decision": decision, "regional_dice": result["means"]["regional"]["present_dice"]})
            except Exception as exc:
                base.atomic_json(out / "tracking_error.json", {"error": repr(exc), "science_complete": True})
        update_status(out, "complete", decision=decision)
    except BaseException as exc:
        base.atomic_json(out / "failure.json", {"error": repr(exc), "no_bypass": True,
            "elapsed_seconds": time.monotonic()-started})
        update_status(out, "failed", error=repr(exc))
        sync_docs(out, f"Stopped on integrity/implementation check: {exc}")
        raise
    finally:
        if run:
            try:
                run.finish()
            except Exception as exc:
                base.atomic_json(out / "tracking_error.json", {"error": repr(exc), "science_complete": (out / 'completion.json').exists()})


if __name__ == "__main__":
    main()
