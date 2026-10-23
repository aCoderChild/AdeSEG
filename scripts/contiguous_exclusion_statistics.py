"""Sequence-level summaries for the committed rigid-placement null."""

import json
from pathlib import Path

import numpy as np

from evaluation.statistics import (
    paired_wilcoxon,
    sequence_bootstrap_interval,
    holm_adjust,
)


METRICS = ("dice", "iou", "boundary_f")


def summarize_sequence(rows):
    groups = {}
    for row in rows:
        if row["excluded"] is None:
            continue
        strata = [
            "all_valid",
            "present" if row["present"] else "absent",
            "size_" + row["size_bin"],
            "outcome_" + row["outcome"],
        ]
        if row["present"] and row["informative"]:
            strata.append("present_informative")
        for stratum in strata:
            groups.setdefault((row["definition"], row["region"], stratum), []).append(
                row
            )
    result = []
    for (definition, region, stratum), selected in sorted(groups.items()):
        item = {
            "definition": definition,
            "region": region,
            "stratum": stratum,
            "frames": len(selected),
            "metrics": {},
        }
        for metric in METRICS:
            baseline = np.mean([r["baseline"][metric] for r in selected])
            excluded = np.mean([r["excluded"][metric] for r in selected])
            control = np.mean(
                [
                    np.mean([c["scores"][metric] for c in r["controls"]])
                    for r in selected
                ]
            )
            item["metrics"][metric] = {
                "baseline": float(baseline),
                "excluded": float(excluded),
                "control": float(control),
                "loss": float(baseline - excluded),
                "control_loss": float(baseline - control),
                "excess_loss": float(control - excluded),
            }
        result.append(item)
    return {
        "rows": result,
        "cases": len(rows),
        "undefined_cases": sum(r["excluded"] is None for r in rows),
        "uninformative_cases": sum(not r["informative"] for r in rows),
    }


def report(output, sequences, protocol, scope):
    output = Path(output)
    valid = [r for r in sequences if r["status"] == "completed"]
    rows, tests = [], {}
    for definition in protocol["definitions"]:
        for region in protocol["regions"]:
            matches = {
                r["sequence"]: next(
                    (
                        s
                        for s in r["statistics"]["rows"]
                        if s["definition"] == definition
                        and s["region"] == region
                        and s["stratum"] == "present_informative"
                    ),
                    None,
                )
                for r in valid
            }
            matches = {
                k: v["metrics"]["dice"] for k, v in matches.items() if v is not None
            }
            test = paired_wilcoxon(
                {k: v["loss"] for k, v in matches.items()},
                {k: v["control_loss"] for k, v in matches.items()},
            )
            ci = (
                sequence_bootstrap_interval(
                    {k: v["excess_loss"] for k, v in matches.items()}
                )
                if len(matches) > 1
                else None
            )
            item = {
                "definition": definition,
                "region": region,
                "n": len(matches),
                "p": test["p"],
                "ci95": ci,
                "mean_excess_dice_loss": float(
                    np.mean([v["excess_loss"] for v in matches.values()])
                )
                if matches
                else None,
            }
            for key in ("baseline", "excluded", "control"):
                item[key] = (
                    float(np.mean([v[key] for v in matches.values()]))
                    if matches
                    else None
                )
            tests[definition + "/" + region] = item["p"]
            rows.append(item)
    adjusted = holm_adjust(tests)
    for row in rows:
        row["holm18_p"] = adjusted[row["definition"] + "/" + row["region"]]
        row["passes"] = bool(
            scope == "full"
            and row["mean_excess_dice_loss"] is not None
            and row["mean_excess_dice_loss"] > 0
            and row["ci95"] is not None
            and row["ci95"][0] > 0
            and row["holm18_p"] < 0.05
        )
    robust = {
        region: bool(
            scope == "full" and all(r["passes"] for r in rows if r["region"] == region)
        )
        for region in protocol["regions"]
    }
    result = {
        "scope": scope,
        "sequences": sequences,
        "primary_comparisons": rows,
        "robust_regional_effect": robust,
        "meaning": "Cohort-specific one-step spatial effects, not deployment gains; gate results cannot pass the paper claim",
    }
    (output / "exclusion_results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Contiguity-matched exclusions",
        "",
        f"Scope: {scope}; {len(valid)} completed sequences. Primary: informative present-target sequence means. Three rigid-control replicates averaged. Holm across18 fixed comparisons. Gate cannot establish the paper claim.",
        "",
        "| Definition | Region | N | Baseline Dice | Excluded | Rigid control | Excess loss [95% CI] | Holm18 p |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['definition']} | {row['region']} | {row['n']} | {row['baseline']} | {row['excluded']} | {row['control']} | {row['mean_excess_dice_loss']} {row['ci95']} | {row['holm18_p']} |"
        )
    lines += [
        "",
        f"Committed cross-definition rule: {robust}",
        "",
        "All secondary size/outcome/absent/metric strata and eligibility are retained in per-sequence JSON. Nulls preserve mask shape/count/connectivity/perimeter but not anatomical position or border proximity. Native memory trajectory is never changed.",
    ]
    (output / "exclusion_results.md").write_text("\n".join(lines) + "\n")
    return result
