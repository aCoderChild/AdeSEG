#!/usr/bin/env python3
"""Produce evidence-only tables from completed local v2 diagnostic shards."""

# ruff: noqa: E402
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import (
    paired_wilcoxon,
    holm_adjust,
    sequence_bootstrap_interval,
)
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.run_region_token_pilot import write_json
from training.tracking import start_wandb, log_results_artifact


def condition_effects(rows, metric="dice", bank="recent"):
    """Pair targets first; average random seeds, then targets within sequence."""
    targets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["state"] != "complete" or row["bank"] != bank:
            continue
        category = row["category"]
        if category in {"boundary", "spatial_random"}:
            targets[(row["video"], row["frame"], row["condition"])][category].append(
                row["harm"][metric]
            )
    sequences = defaultdict(lambda: defaultdict(list))
    for (video, _frame, condition), values in targets.items():
        if values["boundary"] and values["spatial_random"]:
            sequences[condition][video].append(
                float(np.mean(values["boundary"]) - np.mean(values["spatial_random"]))
            )
    return {
        c: {v: float(np.mean(x)) for v, x in videos.items()}
        for c, videos in sequences.items()
    }


def query_region_effects(rows):
    """Local outcome harm after global key masking; not a pathway-isolation map."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["state"] != "complete" or row["category"] not in {
            "boundary",
            "foreground",
            "near_background",
            "far_background",
        }:
            continue
        for query, value in row.get("query_dice_harm", {}).items():
            if value is not None:
                groups[(row["condition"], row["bank"], query, row["category"])][
                    row["video"]
                ].append(value)
    result = []
    for (condition, bank, query, category), sequences in sorted(groups.items()):
        means = {video: float(np.mean(values)) for video, values in sequences.items()}
        result.append(
            {
                "condition": condition,
                "bank": bank,
                "query_region": query,
                "memory_category": category,
                "n": len(means),
                "mean_dice_harm": float(np.mean(list(means.values()))),
                "sequence_means": means,
            }
        )
    return result


def regional_attention_summary(records):
    """Sequence-weighted saliency; token-normalized, not causal importance."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        for layer in record["layers"]:
            total_keys = (
                sum(
                    sum(r["tokens"] for r in u["regions"].values())
                    for u in layer["units"]
                )
                + layer["pointer_tokens"]
            )
            for bank, conditioning in [("conditioning", True), ("recent", False)]:
                units = [u for u in layer["units"] if u["conditioning"] == conditioning]
                for category in (
                    "boundary",
                    "foreground",
                    "near_background",
                    "far_background",
                ):
                    tokens = sum(u["regions"][category]["tokens"] for u in units)
                    mass = sum(u["regions"][category]["mass"] for u in units)
                    if tokens:
                        groups[(record["condition"], layer["layer"], bank, category)][
                            record["video"]
                        ].append((mass, mass * total_keys / tokens))
    results = []
    for (condition, layer, bank, category), sequences in sorted(groups.items()):
        means = {v: np.mean(values, axis=0).tolist() for v, values in sequences.items()}
        values = np.mean(list(means.values()), axis=0)
        results.append(
            dict(
                condition=condition,
                layer=layer,
                bank=bank,
                category=category,
                n=len(means),
                mean_attention_mass=float(values[0]),
                mean_uniform_enrichment=float(values[1]),
                sequence_means=means,
            )
        )
    return results


def paired_regional_removal_summary(rows, bank="recent"):
    """Exploratory c0/natural regional excess harm versus boundary-reference random.

    Pair within target, average seeds then targets within sequence. Holm covers
    all eight region-by-condition tests separately for each endpoint. Random
    distance/frame strata match the boundary reference, NOT each other region.
    """
    targets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (
            row["state"] == "complete"
            and row["bank"] == bank
            and row["condition"] in {"c0", "natural"}
        ):
            targets[(row["condition"], row["video"], row["frame"])][
                row["category"]
            ].append(row)
    results = []
    for metric in ("dice", "boundary_f"):
        for condition in ("c0", "natural"):
            for category in (
                "boundary",
                "foreground",
                "near_background",
                "far_background",
            ):
                sequences = defaultdict(list)
                for (c, video, _), arms in targets.items():
                    if (
                        c != condition
                        or not arms[category]
                        or not arms["spatial_random"]
                    ):
                        continue
                    if (
                        len({r["k"] for r in arms[category] + arms["spatial_random"]})
                        != 1
                    ):
                        raise ValueError("Regional/random intervention budgets differ")
                    harm = np.mean([r["harm"][metric] for r in arms[category]])
                    random = np.mean(
                        [r["harm"][metric] for r in arms["spatial_random"]]
                    )
                    sequences[video].append((harm, harm - random))
                means = {
                    v: np.mean(values, axis=0).tolist()
                    for v, values in sequences.items()
                }
                differences = {v: x[1] for v, x in means.items()}
                test = (
                    paired_wilcoxon(differences, {v: 0 for v in differences})
                    if len(means) >= 2
                    else {"p": None}
                )
                results.append(
                    dict(
                        metric=metric,
                        condition=condition,
                        category=category,
                        n=len(means),
                        targets=sum(map(len, sequences.values())),
                        mean_harm=float(np.mean([x[0] for x in means.values()]))
                        if means
                        else None,
                        excess_harm=float(np.mean(list(differences.values())))
                        if means
                        else None,
                        ci=sequence_bootstrap_interval(differences)
                        if len(means) >= 2
                        else None,
                        p=float(test["p"]) if test["p"] is not None else None,
                        sequence_means=means,
                    )
                )
        family = {
            str(i): r["p"]
            for i, r in enumerate(results)
            if r["metric"] == metric and r["p"] is not None
        }
        adjusted = holm_adjust(family)
        for i, row in enumerate(results):
            if row["metric"] == metric:
                row["holm_p"] = adjusted.get(str(i))
    return results


def followup_statistics(frames, headroom):
    """Sequence-level inference for the registered disjoint and oracle diagnostics."""
    pairs = defaultdict(dict)
    for row in frames:
        if row["state"] == "complete":
            m = row["matching"]
            pairs[
                (row["video"], row["frame"], row["condition"], m["region"], row["seed"])
            ][m["role"]] = row
    geometry = []
    for metric in ("dice", "boundary_f"):
        for condition in ("natural", "c0"):
            for region in (
                "boundary",
                "foreground",
                "near_background",
                "far_background",
                "mixed_side",
            ):
                targets = defaultdict(list)
                for (video, frame, c, r, _), arms in pairs.items():
                    if (
                        c == condition
                        and r == region
                        and set(arms) == {"target", "control"}
                    ):
                        if arms["target"]["k"] != arms["control"]["k"]:
                            raise ValueError("Disjoint pair budgets differ")
                        targets[(video, frame)].append(
                            arms["target"]["harm"][metric]
                            - arms["control"]["harm"][metric]
                        )
                sequences = defaultdict(list)
                for (video, _), values in targets.items():
                    sequences[video].append(float(np.mean(values)))
                scores = {v: float(np.mean(x)) for v, x in sequences.items()}
                geometry.append(
                    dict(
                        metric=metric,
                        condition=condition,
                        region=region,
                        n=len(scores),
                        targets=len(targets),
                        mean=float(np.mean(list(scores.values()))) if scores else None,
                        ci=sequence_bootstrap_interval(scores)
                        if len(scores) >= 2
                        else None,
                        p=paired_wilcoxon(scores, {v: 0 for v in scores})["p"]
                        if len(scores) >= 2
                        else None,
                        sequence_means=scores,
                    )
                )
        family = {
            str(i): x["p"] if x["p"] is not None else 1.0
            for i, x in enumerate(geometry)
            if x["metric"] == metric
        }
        adjusted = holm_adjust(family)
        for i, row in enumerate(geometry):
            if row["metric"] == metric:
                row["holm_p"] = adjusted.get(str(i))
    searches, search_rows = defaultdict(dict), defaultdict(dict)
    for row in headroom:
        key = (row["video"], row["frame"], row["condition"])
        identity = (row["search"], row["seed"])
        if identity in search_rows[key]:
            raise ValueError(
                "Duplicate headroom target/search; do not pool validation repeats"
            )
        search_rows[key][identity] = row
        if len(row["candidates"]) != 8:
            raise ValueError("Oracle and null must each have eight candidates")
        # Baseline first wins ties: never select a worse candidate unnecessarily.
        best = max(
            [{"metrics": row["baseline"]}, *row["candidates"]],
            key=lambda x: x["metrics"]["dice"],
        )
        searches[key][(row["search"], row["seed"])] = {
            metric: best["metrics"][metric] - row["baseline"][metric]
            for metric in ("dice", "boundary_f")
        }
    headroom_results = []
    for metric in ("dice", "boundary_f"):
        for condition in ("natural", "c0"):
            sequences = defaultdict(list)
            for (video, _, c), arms in searches.items():
                if c != condition:
                    continue
                if set(arms) != {
                    ("regional", None),
                    ("random", 0),
                    ("random", 1),
                    ("random", 2),
                }:
                    raise ValueError("Incomplete equal-search null")
                source_rows = search_rows[(video, _, c)]
                real_row = source_rows[("regional", None)]
                for seed in range(3):
                    null_row = source_rows[("random", seed)]
                    if null_row["baseline"] != real_row["baseline"]:
                        raise ValueError("Search baselines differ")
                    for a, b in zip(real_row["candidates"], null_row["candidates"]):
                        if a["candidate"] != b["candidate"] or not np.isclose(
                            a["token_l2"], b["token_l2"], rtol=1e-8, atol=1e-8
                        ):
                            raise ValueError(
                                "Search candidate norms or identities differ"
                            )
                real = arms[("regional", None)][metric]
                null = float(np.mean([arms[("random", s)][metric] for s in range(3)]))
                sequences[video].append((real, null, real - null))
            means = {v: np.mean(x, axis=0).tolist() for v, x in sequences.items()}
            scores = {v: x[2] for v, x in means.items()}
            headroom_results.append(
                dict(
                    metric=metric,
                    condition=condition,
                    n=len(scores),
                    oracle_gain=float(np.mean([x[0] for x in means.values()]))
                    if scores
                    else None,
                    null_gain=float(np.mean([x[1] for x in means.values()]))
                    if scores
                    else None,
                    excess=float(np.mean(list(scores.values()))) if scores else None,
                    ci=sequence_bootstrap_interval(scores)
                    if len(scores) >= 2
                    else None,
                    p=paired_wilcoxon(scores, {v: 0 for v in scores})["p"]
                    if len(scores) >= 2
                    else None,
                    sequence_means=means,
                )
            )
        family = {
            str(i): x["p"]
            for i, x in enumerate(headroom_results)
            if x["metric"] == metric and x["p"] is not None
        }
        adjusted = holm_adjust(family)
        for i, row in enumerate(headroom_results):
            if row["metric"] == metric:
                row["holm_p"] = adjusted.get(str(i))
    return {"geometry": geometry, "headroom": headroom_results}


def write_followup_report(root, names, output_name):
    output = root / output_name
    output.mkdir(exist_ok=False)
    frames, headroom, shards = [], [], {}
    names = list(
        dict.fromkeys([*names, *[p.name for p in sorted(root.glob("dev_seq*"))]])
    )
    for name in names:
        directory = root / name
        status = directory / "summary.json"
        shards[name] = (
            json.loads(status.read_text())
            if status.exists()
            else {"completed": False, "reason": "missing summary"}
        )
        if shards[name].get("completed"):
            frames.extend(
                json.loads(x)
                for x in (directory / "frames.jsonl").read_text().splitlines()
            )
            headroom.extend(
                json.loads(x)
                for x in (directory / "headroom.jsonl").read_text().splitlines()
            )
    result = {
        "shards": shards,
        "full_study_complete": False,
        "rows": len(frames),
        "skipped_rows": sum(x["state"] != "complete" for x in frames),
        **followup_statistics(frames, headroom),
    }
    result["planned_cohort"] = json.loads(
        (ROOT / "protocols/region_tokens_v2_amendment.json").read_text()
    )["geometry_headroom_followup_2026_10_19"]["cohort"]
    result["completed_cohort"] = sorted({r["video"] for r in frames})
    result["incomplete_shards"] = [
        name for name, status in shards.items() if not status.get("completed")
    ]
    result["all_planned_sequences_complete"] = (
        set(result["planned_cohort"]) == set(result["completed_cohort"])
        and not result["incomplete_shards"]
    )
    natural = next(
        x
        for x in result["headroom"]
        if x["condition"] == "natural" and x["metric"] == "dice"
    )
    result["headroom_gate"] = bool(
        natural["n"] >= 10 and natural["oracle_gain"] >= 0.01 and natural["ci"][0] > 0
    )
    result["training_launched"] = False
    result["training_requirement"] = (
        "Headroom plus geometry evidence and a verified CUDA backward path; no automatic training from oracle-selected target scores"
    )
    write_json(output / "results.json", result)
    lines = [
        "# Disjoint regional controls and reweighting headroom",
        "",
        f"Completed sequences: {len(result['completed_cohort'])}/{len(result['planned_cohort'])}. Incomplete shards: {result['incomplete_shards']}. All planned sequences complete: {result['all_planned_sequences_complete']}.",
        "",
        "Exploratory development diagnostics, not deployed method scores. Incomplete shards excluded. Sequence-level Wilcoxon; 10,000-sequence bootstrap; CIs unadjusted. Geometry Holm family: ten comparisons per endpoint. Headroom Holm family: two conditions per endpoint.",
        "",
        "| Endpoint | Condition | Region | Sequences / targets | Excess removal harm [95% CI] | Holm p |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in result["geometry"]:
        if row["n"] >= 2:
            lines.append(
                f"| {row['metric']} | {row['condition']} | {row['region']} | {row['n']} / {row['targets']} | {row['mean']:.6f} [{row['ci'][0]:.6f}, {row['ci'][1]:.6f}] | {row['holm_p']:.5f} |"
            )
    lines += [
        "",
        "| Endpoint | Condition | Sequences | Regional oracle gain | Equal-search null gain | Excess [95% CI] | Holm p |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in result["headroom"]:
        if row["n"] >= 2:
            lines.append(
                f"| {row['metric']} | {row['condition']} | {row['n']} | {row['oracle_gain']:.6f} | {row['null_gain']:.6f} | {row['excess']:.6f} [{row['ci'][0]:.6f}, {row['ci'][1]:.6f}] | {row['holm_p']:.5f} |"
            )
    lines += [
        "",
        f"Headroom gate: {result['headroom_gate']}. Training launched: false. Missing common support is not imputed. Only the selected subsets—not entire regions—are tested. Oracle selection uses target GT and cannot be reported as deployable improvement.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "Region tokens | geometry and headroom summary",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"shards": names},
        output,
        ["region-tokens-v2", "geometry-headroom"],
    )
    try:
        log_results_artifact(run, output, ["results.json", "results.md"])
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--followup-shard", action="append", default=[])
    parser.add_argument("--output-name", default="analysis")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a single directory name")
    if args.followup_shard:
        if args.shard:
            parser.error("Do not pool prior regional rows with the disjoint follow-up")
        write_followup_report(args.root, args.followup_shard, args.output_name)
        return
    output = args.root / args.output_name
    output.mkdir(exist_ok=False)
    audit = json.loads((args.root / "feasibility/summary.json").read_text())
    counts = {
        k: sum(v[k] for v in audit["sequences"].values())
        for k in next(iter(audit["sequences"].values()))
    }
    side_audit_path = args.root / "feasibility_side_matched/summary.json"
    side_audit = (
        json.loads(side_audit_path.read_text()) if side_audit_path.exists() else None
    )
    side_counts = (
        {k: sum(v[k] for v in side_audit["sequences"].values()) for k in counts}
        if side_audit
        else None
    )
    alignment_path = args.root / "data_alignment/alignment.json"
    alignment = (
        json.loads(alignment_path.read_text()) if alignment_path.exists() else None
    )
    diagnosis_path = args.root / "bplus_diagnosis/summary.json"
    diagnosis = (
        json.loads(diagnosis_path.read_text())
        if diagnosis_path.exists()
        else {"completed": False}
    )
    # Presence-stratified reporting is retrospective evaluation only. Inputs to
    # the predictor remain one initial mask; no labels select corrections.
    videos = load_manifest(ROOT / "data/polypgen_sequence.jsonl", "dev")
    presence = {}
    for video, arms in diagnosis.get("sequences", {}).items():
        presence[video] = {}
        for arm, values in arms.items():
            records = [
                json.loads(line)
                for line in (
                    args.root / "bplus_diagnosis" / video / arm / "frames.jsonl"
                )
                .read_text()
                .splitlines()
                if line
            ]
            if len(records) != values["evaluated_frames"]:
                raise ValueError("Diagnosis frame coverage does not match summary")
            positive, absent = [], []
            for row in records:
                target = (
                    load_label_mask(videos[video][row["frame"]].mask_path, [1]) == 1
                )
                (positive if target.any() else absent).append(row["metrics"]["dice"])
            presence[video][arm] = {
                "positive_frames": len(positive),
                "absent_frames": len(absent),
                "positive_dice": float(np.mean(positive)) if positive else None,
                "absent_dice": float(np.mean(absent)) if absent else None,
            }
    persistent_path = args.root / "persistent_regions" / "summary.json"
    persistent = (
        json.loads(persistent_path.read_text()) if persistent_path.exists() else None
    )
    baseline_path = args.root / "baseline_all/summary.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    baseline_scores = (
        {
            video: arms["native"]["mean_dice"]
            for video, arms in baseline.get("sequences", {}).items()
            if arms["native"].get("full_sequence")
        }
        if baseline
        else {}
    )
    rows, shards, attention_records = [], {}, []
    for name in args.shard:
        directory = args.root / name
        summary_path = directory / "summary.json"
        shards[name] = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else {
                "completed": False,
                **json.loads((directory / "error.json").read_text()),
            }
        )
        if shards[name].get("completed"):
            attention_path = directory / "attention.jsonl"
            if attention_path.exists():
                attention_records.extend(
                    json.loads(line)
                    for line in attention_path.read_text().splitlines()
                    if line
                )
            rows.extend(
                json.loads(line)
                for line in (directory / "frames.jsonl").read_text().splitlines()
                if line
            )
    effects, comparisons = {}, {}
    for metric in ("dice", "boundary_f"):
        effects[metric] = condition_effects(rows, metric)
        tests = {}
        for condition in ("c1", "c2", "natural"):
            clean, other = (
                effects[metric].get("c0", {}),
                effects[metric].get(condition, {}),
            )
            common = sorted(clean.keys() & other.keys())
            differences = {v: other[v] - clean[v] for v in common}
            paired = paired_wilcoxon(clean, other)
            paired = {
                key: value if np.isfinite(value) else None
                for key, value in paired.items()
            }
            tests[condition] = {
                **paired,
                "mean_difference_vs_c0": float(np.mean(list(differences.values())))
                if differences
                else None,
                "ci": sequence_bootstrap_interval(differences)
                if len(differences) >= 2
                else None,
            }
        adjusted = holm_adjust({c: t["p"] for c, t in tests.items() if t["n"] >= 2})
        comparisons[metric] = {
            c: {**t, "p": t["p"] if t["n"] >= 2 else None, "holm_p": adjusted.get(c)}
            for c, t in tests.items()
        }
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "audit": counts,
        "audit_completed": audit["completed"],
        "side_matched_audit": side_counts,
        "mirror_alignment": alignment,
        "diagnosis": diagnosis,
        "diagnosis_presence_strata": presence,
        "persistent": persistent,
        "chronological_baseline": baseline,
        "baseline_full_sequence_count": len(baseline_scores),
        "baseline_sequence_mean_dice": float(np.mean(list(baseline_scores.values())))
        if baseline_scores
        else None,
        "shards": shards,
        "recent_boundary_excess_harm": effects,
        "comparisons": comparisons,
        "query_region_effects": query_region_effects(rows),
        "regional_attention": regional_attention_summary(attention_records),
        "paired_regional_removal": paired_regional_removal_summary(rows),
        "full_study_complete": False,
        "scope": "Sampled matched-bank perturbations; not accumulated corrupted rollouts; oracle fixed categories; frame-distance random may overlap boundary",
        "incomplete_shards": [
            name for name, value in shards.items() if not value.get("completed")
        ],
    }
    write_json(output / "results.json", result)
    lines = [
        "# Region v2: local diagnostic results",
        "",
        "The full study is not complete. Results below are scoped to saved shards.",
        f"Report timestamp: {result['created_at']}",
        f"Incomplete replay shards (excluded from inferential aggregation): {result['incomplete_shards']}",
        "",
        "## Sampling feasibility",
        "",
        f"{len(audit['sequences'])} sequences; {counts['frames']} frames; {counts['empty_masks']} empty masks.",
        "",
        f"Full-budget disjoint boundary matching feasible: {counts['boundary_feasible']} frames; mixed-coverage matching: {counts['mixed_feasible']} frames.",
        "",
        "## B+ single-prompt diagnosis",
        "",
        "Spatial suppression retains pointers; it is not complete memory-off. Two sequences are descriptive only.",
        "",
        "| Sequence | Arm | Evaluated frames | Mean Dice | Full sequence |",
        "|---|---|---:|---:|---|",
    ]
    for video, arms in diagnosis.get("sequences", {}).items():
        for arm, values in arms.items():
            lines.append(
                f"| {video} | {arm} | {values['evaluated_frames']} | {values['mean_dice']:.5f} | {values['full_sequence']} |"
            )
    lines.extend(
        [
            "",
            "## Sampled corruption replays",
            "",
            result["scope"],
            "",
            "| Endpoint | Condition minus c0 | Sequences | Difference | 95% CI | Holm p |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for metric, conditions in comparisons.items():
        for condition, values in conditions.items():
            p_label = (
                f"{values['holm_p']:.5f}"
                if values["holm_p"] is not None
                else "not tested"
            )
            lines.append(
                f"| {metric} | {condition} | {values['n']} | {values['mean_difference_vs_c0']} | {values['ci']} | {p_label} |"
            )
    if persistent is not None:
        lines.extend(
            [
                "",
                "## Persistent predicted-region ablations",
                "",
                "Two failure-selected sequences, descriptive only. Equal boundary-sized budgets; infeasible reads perform no intervention. Pointers and conditioning remain unchanged.",
                "",
                "| Sequence | Arm | Frames | Dice | Full sequence | Infeasible policy reads |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for video, arms in persistent.get("sequences", {}).items():
            for arm, values in arms.items():
                lines.append(
                    f"| {video} | {arm} | {values['evaluated_frames']} | {values['mean_dice']:.5f} | {values['full_sequence']} | {values['infeasible_policy_reads']} |"
                )
    query_lookup = {
        (r["condition"], r["query_region"], r["memory_category"]): r
        for r in result["query_region_effects"]
        if r["bank"] == "recent"
    }
    lines.extend(
        [
            "",
            "## Query-region outcome matrix: recent memory",
            "",
            "Signed local Dice harm after masking memory keys for all queries. This is not isolation of a query-specific attention pathway. Cells show equal-sequence means and feasible sequence counts; cohorts can differ.",
            "",
            "| Condition | Query outcome region | Boundary | Foreground | Near background | Far background |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for condition in ("c0", "c1", "c2", "natural"):
        for query in ("foreground", "boundary", "background"):
            cells = []
            for category in (
                "boundary",
                "foreground",
                "near_background",
                "far_background",
            ):
                value = query_lookup.get((condition, query, category))
                cells.append(
                    f"{value['mean_dice_harm']:+.5f} (n={value['n']})"
                    if value
                    else "not available"
                )
            lines.append(f"| {condition} | {query} | " + " | ".join(cells) + " |")
    if side_counts is not None:
        lines.extend(
            [
                "",
                "## Additional side-matched feasibility audit",
                "",
                f"Same-side and same-distance-bin disjoint matching: boundary feasible on {side_counts['boundary_feasible']}/{side_counts['frames']} frames; mixed coverage on {side_counts['mixed_feasible']}/{side_counts['frames']} frames. This is distinct from the initial distance-only audit.",
            ]
        )
    if alignment is not None:
        lines.extend(
            [
                "",
                "## Mirror chronology warning",
                "",
                "Hash-verified mirror seq13 samples 0,1,2,6 closely match original frames 0,100,104,12. The mirror's reindexed order must not be treated as chronological; these local results are not pooled with its temporal results.",
            ]
        )
    if baseline is not None:
        lines.extend(
            [
                "",
                "## Chronological B+ single-prompt baseline",
                "",
                f"Completed full sequences: {len(baseline_scores)}/21. Equal-sequence mean Dice: {result['baseline_sequence_mean_dice']}. This is not directly comparable to the detector-assisted 23-sequence benchmark.",
                "",
                "| Sequence | Post-prompt frames | Dice | Full sequence | Failure episodes |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for video, arms in baseline.get("sequences", {}).items():
            values = arms["native"]
            lines.append(
                f"| {video} | {values['evaluated_frames']} | {values['mean_dice']:.5f} | {values['full_sequence']} | {values['temporal']['episode_count']} |"
            )
    lines.extend(
        [
            "",
            "## Diagnosis: present versus absent frames",
            "",
            "| Sequence | Arm | Present frames | Present Dice | Absent frames | Absent Dice |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for video, arms in presence.items():
        for arm, values in arms.items():
            lines.append(
                f"| {video} | {arm} | {values['positive_frames']} | {values['positive_dice']} | {values['absent_frames']} | {values['absent_dice']} |"
            )
    lines.extend(
        [
            "",
            "Differences concern boundary-removal excess harm versus frame-distance random, not a method's segmentation gain. Confidence intervals are not multiplicity-adjusted. Incomplete/infeasible cells are not imputed.",
            "",
            "No trained-method improvement, full A3 completion or endoscopy-specific causal mechanism is claimed.",
        ]
    )
    lines += [
        "",
        "## Matched regional excess removal harm (recent bank)",
        "",
        "Exploratory; paired within target, then equal sequence weights. Holm over eight region × condition tests per endpoint; 10,000-sequence-bootstrap CIs are unadjusted. Positive excess means removal hurts more than boundary-referenced frame/distance random. Only the boundary comparison is distance matched; other regions share its token budget, not its distance distribution. Supports differ across regions.",
        "",
        "| Endpoint | Condition | Region | Sequences / targets | Removal harm | Excess harm [95% CI] | Holm p |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in result["paired_regional_removal"]:
        if row["n"] >= 2:
            lines.append(
                f"| {row['metric']} | {row['condition']} | {row['category']} | {row['n']} / {row['targets']} | {row['mean_harm']:.6f} | {row['excess_harm']:.6f} [{row['ci'][0]:.6f}, {row['ci'][1]:.6f}] | {row['holm_p']:.5f} |"
            )
    if result["regional_attention"]:
        lines += [
            "",
            "## Regional attention (read-only measurement)",
            "",
            "All native queries; whole-sequence weighting. Enrichment is mass divided by the region's fraction of all keys, including pointers in the denominator. Empty regional supports are excluded. Attention mass is not causal utility; use the removal experiments separately.",
            "",
            "| Condition | Layer | Bank | Region | Sequences | Mean mass | Uniform enrichment |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for row in result["regional_attention"]:
            lines.append(
                f"| {row['condition']} | {row['layer']} | {row['bank']} | {row['category']} | {row['n']} | {row['mean_attention_mass']:.6f} | {row['mean_uniform_enrichment']:.4f} |"
            )
    (output / "results.md").write_text("\n".join(lines) + "\n")
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "Region v2 | local evidence summary",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"shards": args.shard, "full_study_complete": False},
        output,
        ["region-tokens-v2", "analysis"],
    )
    try:
        log_results_artifact(run, output, ["results.json", "results.md"])
        run.summary["full_study_complete"] = False
    finally:
        run.finish()
    print(output / "results.md", flush=True)


if __name__ == "__main__":
    main()
