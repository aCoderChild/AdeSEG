"""Sampling, immutable receipts and sequence summaries for source-local replay.

Model inference remains in the existing SAM2/replay integrations. These are
experimental bookkeeping and analysis helpers, not a new token selector.
"""
import hashlib
import json

import numpy as np

from datasets.io import load_label_mask
from evaluation.statistics import sequence_bootstrap_interval
from modeling.within_frame_token_utility import ARMS
from scripts.memory_token_io import digest
from scripts.run_complementary_memory_pilot import atomic_json

TILES = ARMS[1:17]


def sample_targets(videos, archived_plan):
    """First/last present and first empty among the frozen Run171 target IDs."""
    result = {}
    for sequence, archived in archived_plan.items():
        frames = videos[sequence]
        ids = archived["targets"]
        if ids != sorted(set(ids)) or any(i <= archived["prompt"] or i >= len(frames) for i in ids):
            raise ValueError("Invalid archived target IDs")
        flags = {i: bool((load_label_mask(frames[i].mask_path, [1]) == 1).any()) for i in ids}
        present = [i for i in ids if flags[i]]
        empty = [i for i in ids if not flags[i]]
        chosen = sorted(set(present[:1] + present[-1:] + empty[:1]))
        result[sequence] = {"prompt": archived["prompt"], "targets": chosen,
            "present_targets": [i for i in chosen if flags[i]],
            "empty_targets": [i for i in chosen if not flags[i]],
            "source_frame_indices": {str(i): frames[i].frame_index for i in chosen}}
    return result


def record_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_receipt(path, context):
    """Resume only with exact input/protocol identity and intact output files."""
    record = json.loads(path.read_text())
    payload = record["payload"]
    if record_hash(payload) != record["payload_sha256"]:
        raise RuntimeError(f"Receipt content changed: {path}")
    if payload["context"] != context:
        raise RuntimeError(f"Replay context changed: {path}")
    for relative, expected in payload["artifacts"].items():
        artifact = (path.parent/relative).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or not artifact.is_file() or digest(artifact) != expected:
            raise RuntimeError(f"Replay artifact changed or missing: {relative}")
    return payload


def write_receipt(path, payload):
    """Never replace a completed scientific record during a resumed attempt."""
    if path.exists():
        if read_receipt(path, payload["context"]) != payload:
            raise RuntimeError(f"Refusing to overwrite a different replay: {path}")
        return
    atomic_json(path, {"payload": payload, "payload_sha256": record_hash(payload)})
    read_receipt(path, payload["context"])


def analyze_target(record):
    """Validate the complete arm pairing before computing conditional utility."""
    rows = record["arms"]
    if set(rows) != set(ARMS):
        raise ValueError("Incomplete main-target arms")
    native = rows["native"]
    if record["frame"] != native["frame"]:
        raise ValueError("Record and arm target frames differ")
    if any((r["frame"], r["present"]) != (native["frame"], native["present"]) for r in rows.values()):
        raise ValueError("Unmatched target identity or ground-truth denominator")
    result = {"sequence": record["sequence"], "frame": native["frame"],
        "source_frame": record["source_frame"], "source_quality": record["source_quality"],
        "present": native["present"]}
    if not native["present"]:
        if any(type(row["absent_fp"]) is not bool for row in rows.values()):
            raise ValueError("Missing empty-target false-alarm flag")
        result["false_alarms"] = {arm: bool(row["absent_fp"]) for arm, row in rows.items()}
        return result
    dice = {arm: float(row["scores"]["dice"]) for arm, row in rows.items()}
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in dice.values()):
        raise ValueError("Invalid Dice values")
    utility = {a: dice["native"]-dice[a] for a in TILES}
    best = max(TILES, key=lambda a: dice[a])  # fixed row-major tie breaking
    result.update(native_dice=dice["native"],
        uniform_dice=float(np.mean([dice[a] for a in TILES])),
        whole_source_dice=dice["whole_source"],
        best_tile_oracle_dice=dice[best], best_tile_oracle_arm=best,
        tile_utility=utility, mixed_signs=min(utility.values()) < 0 < max(utility.values()),
        interactions={"00_15": dice["tile_00"]+dice["tile_15"]-dice["union_00_15"]-dice["native"],
                      "03_12": dice["tile_03"]+dice["tile_12"]-dice["union_03_12"]-dice["native"]},
        metrics={metric: {"native": native["scores"][metric],
            "uniform": float(np.mean([rows[a]["scores"][metric] for a in TILES]))}
            for metric in ("iou", "boundary_f")})
    return result


def summarize(records, followups=()):
    """Equal-sequence reporting; GT-empty cases never enter present Dice."""
    targets = [analyze_target(r) for r in records]
    identities = [(r["sequence"], r["frame"]) for r in targets]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source-target record")
    per_sequence = {}
    for name in sorted({r["sequence"] for r in targets}, key=lambda s: int(s.removeprefix("seq"))):
        present = [r for r in targets if r["sequence"] == name and r["present"]]
        empty = [r for r in targets if r["sequence"] == name and not r["present"]]
        means = {field: float(np.mean([r[field] for r in present])) if present else None
                 for field in ("native_dice", "uniform_dice", "whole_source_dice", "best_tile_oracle_dice")}
        per_sequence[name] = {**means, "present_targets": len(present), "empty_targets": len(empty),
            "mixed_sign_source_fraction": float(np.mean([r["mixed_signs"] for r in present])) if present else None,
            "false_alarms_native": sum(r["false_alarms"]["native"] for r in empty),
            "expected_false_alarms_uniform": sum(float(np.mean([r["false_alarms"][a] for a in TILES])) for r in empty)}
        per_sequence[name]["metrics"] = {
            metric: {arm: float(np.mean([r["metrics"][metric][arm] for r in present])) if present else None
                     for arm in ("native", "uniform")}
            for metric in ("iou", "boundary_f")}
    contrasts = {}
    for arm in ("uniform_dice", "whole_source_dice", "best_tile_oracle_dice"):
        values = {s: r[arm]-r["native_dice"] for s, r in per_sequence.items() if r["present_targets"]}
        array = list(values.values())
        contrasts[arm] = {"sequence_deltas": values,
            "mean": float(np.mean(array)) if array else None,
            "median": float(np.median(array)) if array else None,
            "leave_one_out_means": {s: float(np.mean([v for t, v in values.items() if t != s])) for s in values}
                                   if len(values) > 1 else {},
            "bootstrap_95_ci": sequence_bootstrap_interval(values, seed=20261108)
                if len(values) > 1 and arm != "best_tile_oracle_dice" else None}
    qualities = [r["source_quality"]["scores"]["dice"] for r in targets if r["source_quality"]["present"]]
    return {"scope": "Exploratory fixed-native-history source-group utility, not propagation efficacy",
        "targets": targets, "per_sequence": per_sequence, "contrasts": contrasts,
        "persistence": persistence(records, followups),
        "source_quality_coverage": {"present_source_target_pairs": len(qualities),
            "empty_source_target_pairs": sum(not r["source_quality"]["present"] for r in targets),
            "min_median_max_present_dice": [float(np.min(qualities)), float(np.median(qualities)), float(np.max(qualities))] if qualities else None},
        "oracle_caveat": "Outcome-selected best tile among sixteen interventions. No deployment, statistical significance or global upper-bound claim.",
        "next": "No automatic selector fitting or efficacy launch; inspect coverage, headroom and outlier sensitivity first."}


def persistence(records, followups):
    main = {(r["sequence"], r["frame"]): r for r in records}
    pairs, missing = [], []
    for row in followups:
        if row.get("eligible") is False:
            missing.append({k: row[k] for k in ("sequence", "anchor_target", "reason")})
            continue
        original = main[(row["sequence"], row["anchor_target"])]
        if row["source_frame"] != original["source_frame"] or row["frame"] != original["frame"]+1:
            raise ValueError("Persistence source or timing changed")
        if set(row["arms"]) != {"native", *TILES}:
            raise ValueError("Incomplete persistence arms")
        later_native = row["arms"]["native"]
        if any((r["frame"], r["present"]) != (row["frame"], later_native["present"])
               for r in row["arms"].values()):
            raise ValueError("Unmatched persistence target denominator")
        if not (original["arms"]["native"]["present"] and row["arms"]["native"]["present"]):
            missing.append({"sequence": row["sequence"], "anchor_target": row["anchor_target"],
                            "reason": "present_Dice_not_defined_for_both_targets"})
            continue
        before = analyze_target(original)
        native = row["arms"]["native"]["scores"]["dice"]
        utility = {a: native-row["arms"][a]["scores"]["dice"] for a in TILES}
        pairs.append({"sequence": row["sequence"], "anchor_target": row["anchor_target"],
            "source_frame": row["source_frame"], "later_tile_utility": utility,
            "same_sign_fraction": float(np.mean([np.sign(utility[a]) == np.sign(before["tile_utility"][a]) for a in TILES])),
            "earlier_oracle_selected_tile_later_gain": -utility[before["best_tile_oracle_arm"]]})
    return {"paired_present_targets": pairs, "excluded_or_unavailable": missing,
            "scope": "Adjacent native queries, no treatment writes; not propagation persistence or online selection."}


def write_report(out, result):
    """Compact report plus source-quality/conditional-utility scientific plot."""
    lines = ["# Within-frame memory-token utility", "", result["scope"], "",
             "| Sequence | Present targets | Native Dice | Uniform tile removal | Best tile (oracle) | Whole-source spatial removal |",
             "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "N/A" if v is None else f"{v:.6f}"
    for name, row in result["per_sequence"].items():
        lines.append(f"| {name} | {row['present_targets']} | " + " | ".join(fmt(row[k]) for k in
            ("native_dice", "uniform_dice", "best_tile_oracle_dice", "whole_source_dice")) + " |")
    lines += ["", "Uniform is the exact expectation over sixteen equal-size tile removals.",
              "Whole-source removal deletes more keys; it is a mechanism reference, not a budget-matched control.",
              result["oracle_caveat"], "", "| Removal minus native | Equal-sequence Dice difference | Median | 95% sequence-bootstrap CI |",
              "|---|---:|---:|---|"]
    for name, row in result["contrasts"].items():
        lines.append(f"| {name} | {fmt(row['mean'])} | {fmt(row['median'])} | {row['bootstrap_95_ci']} |")
    lines += ["", "| Sequence | Empty targets | Native false alarms | Expected uniform-removal false alarms |",
              "|---|---:|---:|---:|"]
    for name, row in result["per_sequence"].items():
        lines.append(f"| {name} | {row['empty_targets']} | {row['false_alarms_native']} | {row['expected_false_alarms_uniform']:.3f} |")
    lines += ["", "| Sequence | Native IoU | Uniform IoU | Native boundary F | Uniform boundary F |",
              "|---|---:|---:|---:|---:|"]
    for name, row in result["per_sequence"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(row["metrics"][m][a])
                     for m in ("iou", "boundary_f") for a in ("native", "uniform")) + " |")
    coverage = result["source_quality_coverage"]
    lines += ["", f"GT-present source-target pairs: {coverage['present_source_target_pairs']}; "
              f"GT-empty source-target pairs: {coverage['empty_source_target_pairs']}.",
              f"Present-source Dice min/median/max: {coverage['min_median_max_present_dice']}.",
              "A poorly predicted source is not necessarily blurred, occluded, or poorly illuminated.",
              f"Adjacent-query comparable pairs: {len(result['persistence']['paired_present_targets'])}; "
              f"unavailable or non-comparable: {len(result['persistence']['excluded_or_unavailable'])}.",
              "", "Positive utility means removing the tile hurts; negative utility means removing it helps.",
              "Effects concern whole 256-token groups and include attention renormalization.",
              "No learned selector or segmentation improvement is established by an oracle result.",
              "", result["next"]]
    (out/"analysis.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in result["targets"]:
        if row["present"] and row["source_quality"]["present"]:
            ax.scatter([row["source_quality"]["scores"]["dice"]]*16,
                       list(row["tile_utility"].values()), s=8, alpha=.3, color="tab:blue")
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xlabel="Source frame's own prediction Dice (GT-present)",
           ylabel="Native target Dice − Dice without tile", title="Conditional tile utility within source frames")
    fig.tight_layout()
    fig.savefig(out/"source_quality_vs_utility.png", dpi=180)
    plt.close(fig)
