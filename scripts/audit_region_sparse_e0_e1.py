"""Frozen, no-inference E0/E1 audit of archived predictions and native labels."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from modeling.region_definitions import mixed_cell_regions  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.region_sparse_statistics import deficit_parts, pack_history, token_counts  # noqa: E402

PROTOCOL = ROOT / "protocols/region_sparse_e0_e1_v1.json"
PARTS = ("complete_miss", "edge", "undersegmentation", "excess_prediction")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def geometry(frames):
    records = []
    previous = None
    for index, frame in enumerate(frames):
        mask = load_label_mask(frame.mask_path, [1]) == 1
        ys, xs = np.nonzero(mask)
        row = {"frame": index, "source_frame_index": frame.frame_index,
               "present": bool(mask.any()), "area_fraction": float(mask.mean()),
               "centroid": [float(xs.mean()/mask.shape[1]), float(ys.mean()/mask.shape[0])] if len(xs) else None}
        if previous and row["present"] and previous["present"]:
            row["absolute_log_area_change"] = float(abs(np.log(row["area_fraction"]/previous["area_fraction"])))
            row["centroid_displacement"] = float(np.linalg.norm(np.subtract(row["centroid"], previous["centroid"])))
        records.append(row)
        previous = row
    return records


def e0(args, videos, protocol, output):
    opportunity = json.loads((args.archive_dir / "opportunities.json").read_text())
    results = {"methods": {}, "geometry": {}, "inference": False}
    for seq, frames in videos.items():
        rows = geometry(frames)
        present = [r["present"] for r in rows]
        returns = []
        start = None
        seen = False
        for i, state in enumerate(present):
            if not state and seen and start is None:
                start = i
            if state:
                if start is not None:
                    returns.append({"absent_start": start, "return": i, "sampled_absent_frames": i-start})
                    start = None
                seen = True
        gaps = np.diff([f.frame_index for f in frames])
        results["geometry"][seq] = {"frames": len(frames), "gt_present": sum(present), "gt_empty": len(frames)-sum(present),
            "source_gap_min": int(gaps.min()) if len(gaps) else None, "source_gap_max": int(gaps.max()) if len(gaps) else None,
            "observed_returns": returns, "terminal_absence": len(frames)-start if start is not None else 0,
            "mean_absolute_log_area_change": mean([r for r in rows if "absolute_log_area_change" in r], "absolute_log_area_change"),
            "mean_centroid_displacement": mean([r for r in rows if "centroid_displacement" in r], "centroid_displacement")}
    for method, source in opportunity["methods"].items():
        sequences = {}
        for entry in source["archive_provenance"]:
            sequence = entry["sequence"]
            path = args.archive_dir / "remote_sources" / method / entry["file"]
            if digest(path) != entry["sha256"]:
                raise RuntimeError("Native archive checksum mismatch")
            frames = videos[sequence]
            by_source = {f.frame_index: i for i, f in enumerate(frames)}
            targets, seen = [], set()
            with zipfile.ZipFile(path) as archive:
                if archive.testzip():
                    raise RuntimeError("Archive CRC failure")
                rows = [json.loads(line) for line in archive.read(f"{sequence}/frames.jsonl").splitlines()]
                prompt = next(i for i, f in enumerate(frames) if (load_label_mask(f.mask_path, [1]) == 1).any())
                for row in rows:
                    index = by_source[row["source_frame_index"]]
                    if index in seen or index <= prompt:
                        raise RuntimeError("Duplicate or preprompt archive target")
                    seen.add(index)
                    gt = load_label_mask(frames[index].mask_path, [1]) == 1
                    pred = np.asarray(Image.open(io.BytesIO(archive.read(f"{sequence}/frame_{row['frame']:06d}.png")))) > 0
                    if pred.shape != gt.shape or bool(gt.any()) != row["present"]:
                        raise RuntimeError("Prediction/manifest alignment mismatch")
                    record = {"frame": index, "source_frame_index": row["source_frame_index"],
                              "present": bool(gt.any()), "prediction_nonempty": bool(pred.any()), "prompt_age": index-prompt}
                    if gt.any():
                        record.update(deficit_parts(pred, gt, protocol["E0"].get("band_ratio", 0.02)))
                        if not np.isclose(record["dice"], row["scores"]["dice"], atol=1e-10, rtol=0):
                            raise RuntimeError("Recomputed Dice disagrees with archived metric")
                    targets.append(record)
                if seen != set(range(prompt+1, len(frames))):
                    raise RuntimeError("Incomplete native target coverage")
            positive = [r for r in targets if r["present"]]
            summary = {"targets": len(targets), "present": len(positive), "dice": mean(positive, "dice"),
                       **{k: mean(positive, k) for k in PARTS},
                       "complete_miss_frames": sum(r["complete_miss"] > 0 for r in positive),
                       "zero_overlap_nonempty_frames": sum(r["zero_overlap_nonempty"] for r in positive),
                       "after_six_targets": sum(r["prompt_age"] > 6 for r in targets),
                       "misses_after_six": sum(r["complete_miss"] > 0 and r["prompt_age"] > 6 for r in positive),
                       "false_alarm_frames": sum(not r["present"] and r["prediction_nonempty"] for r in targets),
                       "gt_empty_targets": sum(not r["present"] for r in targets)}
            deficit = sum(1-r["dice"] for r in positive)
            summary["deficit_fraction_after_six"] = sum(1-r["dice"] for r in positive if r["prompt_age"] > 6) / deficit if deficit else 0
            by_index = {r["frame"]: r for r in targets}
            recovery = []
            for event in results["geometry"][sequence]["observed_returns"]:
                returning = event["return"]
                end = next((i for i in range(returning, len(frames)) if i not in by_index or not by_index[i]["present"]), len(frames))
                detected = next((i for i in range(returning, end) if by_index[i]["prediction_nonempty"] and not by_index[i]["zero_overlap_nonempty"]), None)
                recovery.append({**event, "first_overlapping_prediction": detected,
                                 "delay_sampled_frames": detected-returning if detected is not None else None,
                                 "censored": detected is None})
            summary["return_recovery"] = recovery
            sequences[sequence] = summary
            write_json(output / f"E0_{method}_{sequence}.json", targets)
            print("E0", method, sequence, summary["targets"], flush=True)
        macro = {k: mean(list(sequences.values()), k) for k in ("dice", *PARTS)}
        macro["complete_miss_deficit_share"] = macro["complete_miss"] / (1-macro["dice"]) if macro["dice"] < 1 else 0
        results["methods"][method] = {"sequences": sequences, "equal_sequence": macro, "archive_provenance": source["archive_provenance"]}
    policy = protocol["E0"]["screening"]
    eligible = sum(s["targets"] >= policy["minimum_targets_for_history"] for s in results["methods"]["sam2"]["sequences"].values())
    share = results["methods"]["sam2"]["equal_sequence"]["complete_miss_deficit_share"]
    results["screening"] = {"eligible_sequences": eligible, "complete_miss_deficit_share": share,
                            "passes_pragmatic_screen": share >= policy["complete_miss_deficit_share_min"] and eligible >= policy["minimum_eligible_sequences"], "launch_authorized": False}
    write_json(output / "E0_results.json", results)
    lines = ["# E0: archived failure audit", "", "Exploratory; equal-sequence present-frame means. Components sum to 1−Dice.", "",
             "| Method | Dice | Complete miss deficit | Edge deficit | Undersegmentation deficit | Excess prediction deficit | Miss share of deficit |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, record in results["methods"].items():
        m = record["equal_sequence"]
        lines.append(f"| {method} | {m['dice']:.5f} | {m['complete_miss']:.5f} | {m['edge']:.5f} | {m['undersegmentation']:.5f} | {m['excess_prediction']:.5f} | {m['complete_miss_deficit_share']:.1%} |")
    lines += ["", f"Frozen pragmatic screening: {results['screening']}", "", "Source-index gaps are not elapsed seconds. Late failure is not proof that useful memory expired. Area/centroid changes are geometry proxies, not semantic appearance. Excess prediction cannot be identified specifically as mucosa leakage from binary GT. No E2+ is launched.", "", "Sequence details and full frame records: E0_results.json and E0_METHOD_SEQ.json."]
    (output / "E0_results.md").write_text("\n".join(lines)+"\n")


def e1(args, videos, output):
    sequences = {}
    for sequence, frames in sorted(videos.items(), key=lambda x: int(x[0][3:])):
        folder = args.assignment_dir / sequence
        if not folder.exists():
            if sequence not in ("seq1", "seq7"):
                raise RuntimeError("Missing promptable native assignments")
            continue
        receipts = {r["frame"]: r for r in map(json.loads, (folder / "assignment_frames.jsonl").read_text().splitlines())}
        rows = []
        for index in sorted(receipts):
            receipt = receipts[index]
            path = folder / f"assignments_{index:06d}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            logits = saved["native_low_res_logits"].detach().float().cpu().contiguous()
            if hashlib.sha256(logits.numpy().tobytes()).hexdigest() != receipt["native_mask_sha256"]:
                raise RuntimeError("Stored native logits disagree with original capture hash")
            labels = np.asarray(saved["schemes"]["mixed_cells"]["labels"])
            if not np.array_equal(labels, mixed_cell_regions(logits)["labels"]):
                raise RuntimeError("Saved labels disagree with pinned author occupancy")
            row = {"frame": index, "source_frame_index": frames[index].frame_index, **token_counts(labels), "assignment_sha256": digest(path)}
            rows.append(row)
            if len(rows) % 25 == 0:
                print("E1", sequence, "verified", len(rows), flush=True)
        prompt = min(receipts)
        if sorted(receipts) != list(range(prompt, len(frames))):
            raise RuntimeError("Incomplete stored assignment trajectory")
        # At target i, prompt and immediately previous frame remain dense;
        # older candidates are observed frames prompt+1 .. i-2.
        packed = []
        for target in range(2, len(rows)):
            counts = [r["retained"] for r in rows[1:target-1]]
            packed.append({"target": rows[target]["frame"], **pack_history(counts), "available_older_frames": len(counts)})
        sequences[sequence] = {"frames": len(rows), **{key: mean(rows, key) for key in ("interior", "boundary", "near_background", "far_background", "retained")},
            "retained_p95": float(np.quantile([r["retained"] for r in rows], .95)),
            "mean_actual_older_frames": mean(packed, "older_frames"),
            "max_actual_older_frames": max((p["older_frames"] for p in packed), default=0)}
        write_json(output / f"E1_{sequence}.json", {"frames": rows, "finite_history_packing": packed})
        print("E1", sequence, "complete", len(rows), flush=True)
        write_json(output / "E1_progress.json", {"complete": False, "sequences": sequences})
    if len(sequences) != 21 or sum(r["frames"] for r in sequences.values()) != 2033:
        raise RuntimeError("E1 full cohort coverage differs")
    results = {"complete": True, "inference": False, "native_logits_and_occupancy_verified": True,
               "sequences": sequences, "equal_sequence": {k: mean(list(sequences.values()), k) for k in ("interior", "boundary", "near_background", "far_background", "retained", "mean_actual_older_frames")}}
    write_json(output / "E1_results.json", results)
    lines = ["# E1: native predicted-region spatial-key budget", "", "21 sequences; 2,033 stored frames including prompts. Native capture hashes and author occupancy verified. Budget 28,672; dense prompt/latest reserve 8,192; older budget 20,480. Boundary retained; near-background ring fixed at one token.", "", "| Sequence | Stored frames | Interior | Boundary | Near BG | Far BG | Retained | Mean older frames fitting actual history |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seq, r in sequences.items():
        lines.append(f"| {seq} | {r['frames']} | {r['interior']:.1f} | {r['boundary']:.1f} | {r['near_background']:.1f} | {r['far_background']:.1f} | {r['retained']:.1f} | {r['mean_actual_older_frames']:.1f} |")
    lines += ["", "Counts are per-frame means, not a fixed deployment K. Actual recent-suffix packing uses finite past native trajectories; zero-token empty frames do not imply infinite history. Pointer keys, memory storage bytes and actual sparse attention compute are not included. Dropping far background is only budget accounting here, not validated as safe for segmentation."]
    (output / "E1_results.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("E0", "E1"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads(PROTOCOL.read_text())
    if digest(args.manifest) != protocol["manifest_sha256"]:
        raise RuntimeError("Manifest differs from frozen protocol")
    output = validate_output(args.output_dir) / args.stage
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", {"protocol": protocol, "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(Path(__file__)),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "inference": False})
    videos = {**load_manifest(args.manifest, "dev"), **load_manifest(args.manifest, "test")}
    if args.stage == "E0":
        e0(args, videos, protocol, output)
    else:
        e1(args, videos, output)


if __name__ == "__main__":
    main()
