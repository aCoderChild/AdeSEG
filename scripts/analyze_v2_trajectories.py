"""Offline, threshold-free trajectory audit of completed v2 and random runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import failure_episodes
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, validate_output

SEEDS = (20261025, 20261026, 20261027)
CORE = ("prediction_sha256", "low_res_sha256", "bank_sha256", "cache_sha256")
PROTOCOL = "protocols/v2_trajectory_pulse_v1.json"


def archive_rows(path):
    with zipfile.ZipFile(path) as packed:
        if packed.testzip() is not None:
            raise ValueError(f"Corrupt archive: {path}")
        rows = [json.loads(s) for s in packed.read("frames.jsonl").decode().splitlines()]
        receipts = [json.loads(s) for s in packed.read("integrity.jsonl").decode().splitlines()]
        parameter_files = [n for n in packed.namelist() if n.startswith("parameter_integrity_")]
        if len(parameter_files) != 1 or not json.loads(packed.read(parameter_files[0]))["unchanged"]:
            raise ValueError("Missing or failed parameter integrity")
        summary = json.loads(packed.read("summary.json"))
    if not rows or len(receipts) != len(rows)+1:
        raise ValueError("Incomplete archive")
    prompt = receipts[0]["frame"]
    if ([r["frame"] for r in rows] != list(range(prompt+1, receipts[-1]["frame"]+1))
            or [r["frame"] for r in receipts] != list(range(prompt, receipts[-1]["frame"]+1))):
        raise ValueError("Nonconsecutive archive")
    by_frame = {r["frame"]: r for r in receipts}
    if any(r["prediction_sha256"] != by_frame[r["frame"]]["prediction_sha256"] for r in rows):
        raise ValueError("Prediction receipt mismatch")
    computed = propagation_summary(rows)
    for key in ("targets", "present_targets", "gt_empty_targets", "gt_empty_false_alarm_count", "present_dice"):
        if computed[key] != summary[key]:
            raise ValueError(f"Archived summary mismatch: {key}")
    return rows, by_frame, summary


def longest_lower_run(frames, present, differences):
    values = np.where(np.asarray(present, bool), differences, 0.)
    episodes = failure_episodes(values, threshold=0., minimum_length=1)
    if not episodes:
        return None
    episode = max(episodes, key=lambda e: (e.length, -e.start))
    return {"start": int(frames[episode.start]), "end": int(frames[episode.end]),
            "length": episode.length, "summed_dice_deficit": float(-values[episode.start:episode.end+1].sum())}


def prediction_states(records):
    result = {}
    for arm in records[0]["arms"]:
        nonempty = [i for i,r in enumerate(records) if r["arms"][arm]["prediction_nonempty"]]
        last = nonempty[-1] if nonempty else -1
        result[arm] = {
            "empty_predictions_on_present": sum(r["present"] and not r["arms"][arm]["prediction_nonempty"] for r in records),
            "false_alarms": sum(not r["present"] and r["arms"][arm]["prediction_nonempty"] for r in records),
            "terminal_empty_start": records[last+1]["frame"] if last+1 < len(records) else None,
            "last_pruning_frame": next((r["frame"] for r in reversed(records) if r["arms"][arm]["removed_tokens"] > 0), None),
        }
    return result


def first_pruning(receipts):
    first = None
    for frame in sorted(receipts):
        layers = receipts[frame]["trace"]["layers"]
        if not layers:
            continue
        if [r["layer"] for r in layers] != list(range(4)):
            raise ValueError("Missing layers")
        values = [(r["raw_spatial_keys"], r["selected_spatial_keys"]) for r in layers]
        if len(set(values)) != 1 or values[0][1] > values[0][0]:
            raise ValueError("Inconsistent token counts")
        if first is None and values[0][0] > values[0][1]:
            first = frame
    return first


def analyse_sequence(name, baseline, regional, frames, output):
    paths = {"native": baseline/f"full_{name}_C0.zip", "v2": regional/f"full_{name}_C2.zip",
             **{f"random_{seed}": regional/f"full_{name}_C3_{seed}.zip" for seed in SEEDS}}
    data = {key: archive_rows(path) for key, path in paths.items()}
    native, native_receipts, _ = data["native"]
    indices = [r["frame"] for r in native]
    prompt = min(native_receipts)
    if indices != list(range(prompt+1, len(frames))):
        raise ValueError("Manifest/archive coverage differs")
    pulse = first_pruning(data["v2"][1])
    for arm, (rows, receipts, _) in data.items():
        if [(r["frame"], r["source_frame_index"], r["present"]) for r in rows] != [
            (r["frame"], r["source_frame_index"], r["present"]) for r in native]:
            raise ValueError("Unmatched frame/GT scope")
        for frame, receipt in receipts.items():
            if pulse is None or frame < pulse:
                if any(receipt[k] != native_receipts[frame][k] for k in CORE):
                    raise ValueError(f"Trajectory diverged before first pruning: {name}/{arm}/{frame}")
            if arm.startswith("random") and receipt["allocation"] != data["v2"][1][frame]["allocation"]:
                raise ValueError("Random frame/count/pointer allocation mismatch")
    decoded = {key: zipfile.ZipFile(path) for key, path in paths.items()}
    records = []
    boundary_checks = {indices[0], indices[len(indices)//2], indices[-1]}
    try:
        for offset, index in enumerate(indices):
            truth = load_label_mask(frames[index].mask_path, [1]) == 1
            if bool(truth.any()) != native[offset]["present"]:
                raise ValueError("GT presence changed")
            masks = {key: np.asarray(Image.open(BytesIO(z.read(f"frame_{index:06d}.png")))) > 0
                     for key, z in decoded.items()}
            row = {"frame": index, "source_frame_index": frames[index].frame_index, "present": bool(truth.any()), "arms": {}}
            for arm, (rows, receipts, _) in data.items():
                saved = rows[offset]
                scores = calculate_scores(masks[arm], truth)
                if any(not np.isclose(scores[m], saved["scores"][m], atol=1e-12, rtol=0) for m in ("dice", "iou")):
                    raise ValueError(f"PNG/GT scores differ: {name}/{index}/{arm}")
                if index in boundary_checks and not np.isclose(db_eval_boundary(masks[arm], truth), saved["scores"]["boundary_f"], atol=1e-12, rtol=0):
                    raise ValueError("Boundary F mismatch")
                receipt = receipts[index]
                layers = [] if arm == "native" else receipt["trace"]["layers"]
                removed = layers[0]["raw_spatial_keys"]-layers[0]["selected_spatial_keys"] if layers else 0
                row["arms"][arm] = {"dice": scores["dice"], "iou": scores["iou"],
                    "boundary_f": saved["scores"]["boundary_f"], "removed_tokens": removed,
                    "object_score": receipt["current_object_score"], "prediction_nonempty": bool(masks[arm].any()),
                    "logits_differ_from_native": receipt["prediction_sha256"] != native_receipts[index]["prediction_sha256"],
                    "different_pixels_from_native": int(np.count_nonzero(masks[arm] != masks["native"]))}
            records.append(row)
    finally:
        for z in decoded.values():
            z.close()
    folder = output/name
    folder.mkdir()
    (folder/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    summary = {"sequence": name, "prompt": prompt, "targets": len(indices), "first_pruning_frame": pulse,
               "archives": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
               "verified_pngs": len(indices)*len(paths), "verified_boundary_f": len(boundary_checks)*len(paths), "arms": {}}
    present = [r["present"] for r in records]
    for arm in paths:
        delta = np.array([r["arms"][arm]["dice"]-r["arms"]["native"]["dice"] for r in records])
        summary["arms"][arm] = {**data[arm][2],
            "present_boundary_f": float(np.mean([r["arms"][arm]["boundary_f"] for r in records if r["present"]])),
            "first_logit_difference": next((r["frame"] for r in records if r["arms"][arm]["logits_differ_from_native"]), None),
            "first_binary_difference": next((r["frame"] for r in records if r["arms"][arm]["different_pixels_from_native"]), None),
            "longest_lower_dice_run": longest_lower_run(indices, present, delta),
            "present_dice_difference": float(delta[np.asarray(present)].mean()),
            "mean_removed_tokens": float(np.mean([r["arms"][arm]["removed_tokens"] for r in records]))}
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"[offline] verified {name}: {summary['verified_pngs']} PNGs; first pruning={pulse}", flush=True)
    return summary, records


def plot_trajectory(summary, records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["frame"] for r in records]
    present = np.array([r["present"] for r in records])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, metric in zip(axes[:2], ("dice", "boundary_f")):
        for arm, color in (("native", "black"), ("v2", "tab:red")):
            axis.plot(x, np.where(present, [r["arms"][arm][metric] for r in records], np.nan), color=color, label=arm)
        seeds = np.array([[r["arms"][f"random_{seed}"][metric] for r in records] for seed in SEEDS])
        seeds[:, ~present] = np.nan
        axis.fill_between(x, seeds.min(0), seeds.max(0), color="tab:blue", alpha=.15, label="Random seed range")
        axis.plot(x, seeds.mean(0), color="tab:blue", label="Random mean")
        axis.set_ylabel(metric.replace('_', ' ').title())
        axis.set_ylim(-.02, 1.02)
    axes[2].plot(x, [r["arms"]["v2"]["removed_tokens"] for r in records], color="tab:purple")
    axes[2].set_ylabel("Removed spatial tokens")
    for axis in axes:
        for r in records:
            if not r["present"]:
                axis.axvspan(r["frame"]-.5, r["frame"]+.5, color="gray", alpha=.15, lw=0)
        if summary["first_pruning_frame"] is not None:
            axis.axvline(summary["first_pruning_frame"], color="tab:green", linestyle="--")
        axis.grid(alpha=.15)
    axes[0].legend(loc="best", fontsize=8)
    axes[2].set_xlabel("Sampled manifest frame (gray: GT absent; green: first pruning)")
    fig.suptitle(f"{summary['sequence']}: archived full propagation; exploratory")
    fig.tight_layout()
    fig.savefig(output/summary["sequence"]/"trajectory.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-dir", "regional-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/polypgen_sequence.jsonl")
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output/"matplotlib_cache")
    for folder in (args.baseline_dir, args.regional_dir):
        receipt = json.loads((folder/"delivery.json").read_text())
        if not receipt["completed"] or not receipt["wandb_artifact_upload_complete"]:
            raise ValueError("Source not completed/delivered")
        setup = json.loads((folder/"setup.json").read_text())
        if setup["E2_setup"]["manifest_sha256"] != digest(args.manifest):
            raise ValueError("Manifest identity changed")
    videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
    names = [f"seq{i}" for i in range(1, 24) if i not in (1, 7)]
    summaries = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(analyse_sequence, name, args.baseline_dir, args.regional_dir, videos[name], output) for name in names}
        for name in names:
            summary, records = futures[name].result()
            summary["prediction_states"] = prediction_states(records)
            summaries[name] = summary
            plot_trajectory(summary, records, output)
    if sum(s["targets"] for s in summaries.values()) != 2012:
        raise ValueError("Cohort incomplete")
    published = json.loads((args.regional_dir/"results.json").read_text())
    means = {arm: float(np.mean([s["arms"][arm]["present_dice"] for s in summaries.values()]))
             for arm in ("native", "v2", *(f"random_{seed}" for seed in SEEDS))}
    if not all(np.isclose(means[arm], published[key], atol=1e-12, rtol=0) for arm,key in (("native","C0_present_dice"),("v2","C2_present_dice"))):
        raise ValueError("Published means not reproduced")
    if not np.isclose(np.mean([means[f"random_{s}"] for s in SEEDS]), published["C3_seed_mean_present_dice"], atol=1e-12, rtol=0):
        raise ValueError("Published random mean not reproduced")
    (output/"results.json").write_text(json.dumps({"completed": True, "inference": False, "means": means, "sequences": summaries}, indent=2)+"\n")
    targets = {name: {"frame": summaries[name]["first_pruning_frame"], "prompt": summaries[name]["prompt"],
                      "archives": summaries[name]["archives"]} for name in ("seq2", "seq13")}
    if any(r["frame"] is None for r in targets.values()):
        raise ValueError("Fixed pilot lacks a nonzero pruning read")
    (output/"pulse_targets.json").write_text(json.dumps({"rule": "first actual pruning; no outcome threshold", "targets": targets}, indent=2)+"\n")
    lines = ["# Archived v2 trajectories", "", "Exploratory, no inference. PNG Dice/IoU recomputed for every condition/frame; Boundary F checked on first/middle/last targets of every sequence. Longest lower run means consecutive present frames with any negative Dice difference, not a clinical failure threshold.", "",
             "| Sequence | Native Dice | V2 Dice | Random mean | V2-native | First pruning | First binary change | Longest lower-Dice run | FA native→v2 |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name,s in summaries.items():
        n,v = s["arms"]["native"],s["arms"]["v2"]
        random = np.mean([s["arms"][f"random_{seed}"]["present_dice"] for seed in SEEDS])
        episode = v["longest_lower_dice_run"]
        run = f"{episode['start']}–{episode['end']} ({episode['length']})" if episode else "none"
        lines.append(f"| {name} | {n['present_dice']:.5f} | {v['present_dice']:.5f} | {random:.5f} | {v['present_dice_difference']:+.5f} | {s['first_pruning_frame']} | {v['first_binary_difference']} | {run} | {n['gt_empty_false_alarm_count']}→{v['gt_empty_false_alarm_count']} |")
    lines += ["", "Fixed pulse targets are recorded in pulse_targets.json and must be committed before inference. Each sequence folder contains the complete numeric trajectory, summary and trajectory.png. Timing refers to sampled manifest indices, not seconds. A small immediate effect plus a large propagation effect suggests a trajectory mechanism but does not identify it causally."]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    (output/"provenance.json").write_text(json.dumps({"protocol_sha256": digest(ROOT/PROTOCOL), "script_sha256": digest(Path(__file__)), "manifest_sha256": digest(args.manifest), "inference": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
