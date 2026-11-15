"""Reanalyse saved source-local ablations against the original poor-frame hypothesis.

No model inference or new outcome selection is performed. Reuses the study's
record validator, verifies immutable receipts, and reports continuous source Dice.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.within_frame_utility_records import analyze_target, read_receipt


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_result_file(folder):
    result = folder / "results.json"
    completion = load(folder / "completion.json")
    if not completion.get("complete") or digest(result) != completion["results_sha256"]:
        raise ValueError(f"Incomplete or changed results: {folder}")
    return load(result)


def mask_dice(path, ground_truth):
    with Image.open(path) as image:
        prediction = np.asarray(image) > 0
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Mask shape changed: {path}")
    denominator = int(prediction.sum()) + int(ground_truth.sum())
    return 2 * np.logical_and(prediction, ground_truth).sum() / denominator if denominator else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_result_file(args.study)
    setup = load(args.study / "setup.json")
    setup_hash = digest(args.study / "setup.json")
    if setup_hash != load(args.study / "completion.json")["setup_sha256"]:
        raise ValueError("Study setup differs from its completion receipt")
    expected = {(r["sequence"], r["frame"]): r for r in result["targets"]}
    rows, raw, receipt_count = [], {}, 0
    for folder in sorted((args.study / "sequences").iterdir()):
        if not folder.is_dir():
            continue
        sequence = verify_result_file(folder)
        for record in sequence["main"]:
            identity = (record["sequence"], record["frame"])
            source_path = folder / "native_masks" / f"{record['source_frame']:06d}.json"
            source = load(source_path)["payload"]
            read_receipt(source_path, source["context"])
            if source["context"]["setup_sha256"] != setup_hash:
                raise ValueError(f"Source has a different study setup: {identity}")
            if source["source_quality"] != record["source_quality"]:
                raise ValueError(f"Source quality differs from stored native mask: {identity}")
            if source["signatures"]["pred_masks"] != record["source_quality"]["stored_low_mask_sha256"]:
                raise ValueError(f"Source quality does not describe its stored mask: {identity}")
            selection = hashlib.sha256(record["selection_context"].encode()).hexdigest()
            eligible = record["eligible_sources"]
            if selection != record["selection_sha256"] or eligible[int(selection, 16) % len(eligible)] != record["source_frame"]:
                raise ValueError(f"Prespecified source selection differs: {identity}")
            pointer_signatures = set()
            for arm, row in record["arms"].items():
                path = folder / f"main_{record['frame']:06d}" / f"{arm}.json"
                payload = load(path)["payload"]
                read_receipt(path, payload["context"])
                context = payload["context"]
                if context["setup_sha256"] != setup_hash:
                    raise ValueError(f"Arm has a different study setup: {path}")
                if (context["sequence"], context["target_frame"], context["arm"]) != (*identity, arm):
                    raise ValueError(f"Mismatched arm identity: {path}")
                if context["request"]["source_frame"] != record["source_frame"] or payload["row"] != row:
                    raise ValueError(f"Receipt and aggregate disagree: {path}")
                trace = row["read"]
                if not trace["bank_read_unchanged"] or not trace["native_positions_verified"] or trace["layers"] != [0, 1, 2, 3]:
                    raise ValueError(f"Recorded read invariants failed: {path}")
                pointer_signatures.add((trace["pointer_input_sha256"], trace["pointer_position_sha256"]))
                receipt_count += 1
            if len(pointer_signatures) != 1:
                raise ValueError(f"Pointer inputs differ between interventions: {identity}")
            analyzed = analyze_target(record)
            if analyzed != expected[identity]:
                raise ValueError(f"Recomputed target analysis differs: {identity}")
            raw[identity] = record
            if analyzed["present"] and analyzed["source_quality"]["present"]:
                utility = analyzed["tile_utility"]
                helpful = max(utility, key=utility.get)
                harmful = min(utility, key=utility.get)
                rows.append({
                    "sequence": identity[0], "target_frame": identity[1],
                    "source_frame": record["source_frame"],
                    "source_dice": record["source_quality"]["scores"]["dice"],
                    "target_native_dice": analyzed["native_dice"],
                    "most_helpful_group": helpful, "max_utility": utility[helpful],
                    "most_harmful_group": harmful, "min_utility": utility[harmful],
                    "whole_source_utility": analyzed["native_dice"] - analyzed["whole_source_dice"],
                })
    if set(raw) != set(expected):
        raise ValueError("Raw records do not cover the complete saved diagnostic")

    # These two illustrations are selected retrospectively; audit every saved
    # intervention in them and do not treat extrema as an online selector.
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    splits = {json.loads(line)["split"] for line in manifest.read_text().splitlines() if line.strip()}
    videos = {}
    for split in sorted(splits):
        split_videos = load_manifest(manifest, split)
        if videos.keys() & split_videos.keys():
            raise ValueError("Duplicate sequence IDs across manifest splits")
        videos.update(split_videos)
    mask_checks = []
    for identity in (("seq17", 19), ("seq22", 45)):
        record = raw[identity]
        folder = args.study / "sequences" / identity[0]
        for frame, arm, path, expected_dice in [
            (record["source_frame"], "source_native", folder / "native_masks" / f"{record['source_frame']:06d}.png", record["source_quality"]["scores"]["dice"]),
            *[(identity[1], arm, folder / f"main_{identity[1]:06d}" / arm / f"{identity[1]:06d}.png", row["scores"]["dice"]) for arm, row in record["arms"].items()],
        ]:
            gt_path = videos[identity[0]][frame].mask_path
            original_inputs = [item for item in setup["inputs"][identity[0]]["frames"] if item["local_frame"] == frame]
            if len(original_inputs) != 1 or digest(gt_path) != original_inputs[0]["mask_sha256"]:
                raise ValueError(f"Ground truth differs from original study: {gt_path}")
            actual = float(mask_dice(path, load_label_mask(gt_path, [1]) == 1))
            if abs(actual - expected_dice) > 1e-8:
                raise ValueError(f"Saved-mask Dice does not reproduce: {path}")
            mask_checks.append({"sequence": identity[0], "frame": frame, "arm": arm,
                                "dice": actual, "mask_sha256": digest(path), "gt_sha256": digest(gt_path)})

    rows.sort(key=lambda r: (r["source_dice"], int(r["sequence"][3:]), r["target_frame"]))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "all_present_source_target_pairs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"study_results_sha256": digest(args.study / "results.json"),
               "analyzer_sha256": digest(Path(__file__)), "raw_targets_verified": len(raw),
               "arm_receipts_verified": receipt_count, "present_source_present_target_pairs": len(rows),
               "sequences": len({row["sequence"] for row in rows}), "pairs": rows,
               "independent_mask_checks": mask_checks,
               "original_coverage": result["source_quality_coverage"],
               "scope": "Retrospective fixed-native-bank group-removal mechanism evidence; no new inference, fitted selector, or propagation gain."}
    (args.output / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].scatter([r["source_dice"] for r in rows], [100*r["max_utility"] for r in rows], label="Most helpful group", marker="^")
    axes[0].scatter([r["source_dice"] for r in rows], [100*r["min_utility"] for r in rows], label="Most harmful group", marker="v")
    axes[0].set(xlabel="Source frame's own Dice", ylabel="Conditional utility (Dice percentage points)", title=f"All {len(rows)} GT-present source/target pairs")
    axes[0].legend(fontsize=8)
    selected = expected[("seq17", 19)]
    values = [100*u for u in selected["tile_utility"].values()]
    axes[1].bar(range(16), values, color=["#178f82" if u>0 else "#c76a48" for u in values])
    axes[1].set(xlabel="Source-frame spatial tile (256 tokens each)", ylabel="Native Dice minus Dice after removal (pp)", title="Seq17: source 17 (Dice 0.0038) → target 19")
    axes[1].set_xticks(range(16))
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(args.output / "source_quality_and_group_utility.png", dpi=180)
    plt.close(fig)

    lines = ["# Original hypothesis: useful tokens in poorly predicted frames", "",
             "Saved-data reanalysis only. All source-quality values are retrospective GT measurements; no GT-based token selector is proposed.", "",
             f"Verified {len(raw)} target records and {receipt_count} arm receipts; reproduced {len(mask_checks)} source/target mask Dice values independently from PNGs and GT.", "",
             "| Sequence | Source → target | Source Dice | Target native Dice | Most helpful group utility | Most harmful group utility | Whole-source utility |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows[:6]:
        lines.append(f"| {row['sequence']} | {row['source_frame']} → {row['target_frame']} | {row['source_dice']:.6f} | {row['target_native_dice']:.6f} | {row['max_utility']:+.6f} | {row['min_utility']:+.6f} | {row['whole_source_utility']:+.6f} |")
    lines += ["", "Positive utility means removing a group worsened the later prediction. Negative utility means removal helped. The table shows the six lowest source-Dice pairs, including all four low-source cases with zero measured effects. Full continuous data are in the CSV and plot.", "",
              "Seq17 provides a direct local example of a poorly predicted source containing both helpful and harmful spatial groups. Seq22 provides a second example with much smaller effects. These conditional effects include attention renormalization and interaction with other memories; they do not establish semantic purity or intrinsic token usefulness.", "",
              "Coverage is limited: only two of the six lowest-source-Dice pairs have nonzero measured effects. Most sampled sources have high Dice. Selecting the extreme groups uses target GT after inference and is an oracle diagnostic. No reliable online selector or full-video gain follows from this observation.", "",
              f"The original equal-sequence uniform-removal effect is {result['contrasts']['uniform_dice']['mean']:+.6f} (95% CI {result['contrasts']['uniform_dice']['bootstrap_95_ci']}); the best-of-16 oracle effect is {result['contrasts']['best_tile_oracle_dice']['mean']:+.6f}. These are sampled immediate effects, with insufficient headroom to justify reviving the stopped coarse-tile pruning family as a +0.01 full-video method.", "",
              "The original hypothesis has local mechanism support. Its practical requirements—predicting useful subsets without GT and improving full-video Dice reliably—remain unmet."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verified_targets":len(raw), "verified_arm_receipts":receipt_count,
                      "mask_checks":len(mask_checks), "output":str(args.output)}))


if __name__ == "__main__":
    main()
