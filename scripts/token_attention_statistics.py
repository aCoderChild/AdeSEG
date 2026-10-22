"""Committed pooled sequence statistics for saved per-token attention."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.region_definition_statistics import contour_coverage


def footprint_precision(binary, labels):
    coverage = contour_coverage(binary, labels)
    support = np.repeat(np.repeat(np.asarray(labels) == 2, 4, 0), 4, 1)
    result = {}
    for mode in ("including_border", "excluding_border"):
        selected = support.copy()
        if mode == "excluding_border":
            selected[[0, -1], :] = False
            selected[:, [0, -1]] = False
        denominator = int(selected.sum())
        hit = coverage[mode]["contour_pixels"] - coverage[mode]["missed_pixels"]
        result[mode] = {
            "contour_hits": hit,
            "support_pixels": denominator,
            "precision": hit / denominator if denominator else None,
        }
    return result


def pooled_enrichment(mass, spatial_mass, opportunity, query_weight):
    if spatial_mass <= 0 or opportunity <= 0 or query_weight <= 0:
        return None
    return (mass / spatial_mass) / (opportunity / query_weight)


def log_summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "median_log": None, "geometric_median": None, "ci95_log": None}
    logs = np.array([np.log(v) if v > 0 else -np.inf for v in values])
    median = float(np.median(logs))
    result = {
        "n": len(values),
        "zero_enrichments": int(np.isneginf(logs).sum()),
        "median_log": median if np.isfinite(median) else "-infinity",
        "geometric_median": float(np.exp(median)),
        "ci95_log": None,
    }
    if np.isfinite(logs).all() and len(logs) > 1:
        rng = np.random.default_rng(20261023)
        draws = np.median(rng.choice(logs, (10000, len(logs)), replace=True), axis=1)
        result["ci95_log"] = np.quantile(draws, [0.025, 0.975]).tolist()
    elif not np.isfinite(logs).all():
        result["ci_omitted_reason"] = (
            "Zero enrichment: no epsilon or nonfinite-log bootstrap"
        )
    return result


def sequence_summary(folder):
    folder = Path(folder)
    metadata = {
        r["frame"]: r
        for r in map(
            json.loads, (folder / "assignment_frames.jsonl").read_text().splitlines()
        )
    }
    scores = {
        r["frame"]: r
        for r in map(json.loads, (folder / "frames.jsonl").read_text().splitlines())
    }
    annotations = {}
    pool, dice = {}, {}
    for path in sorted(folder.glob("token_attention_*.npz")):
        with np.load(path) as saved:
            mass = saved["joint_mass"].astype(np.float64)
            counts = saved["query_counts"]
            frames = saved["memory_frames"].tolist()
            target, layer = int(saved["target"]), int(saved["layer"])
            pointer_count = int(saved["pointer_count"])
        record = metadata[target]
        status = record["outcome"]
        strata = ("all", "size_" + record["size_bin"], "outcome_" + status)
        for stratum in strata:
            dice.setdefault(stratum, {})[target] = scores[target]["scores"]["dice"]
        for frame in frames:
            if frame not in annotations:
                annotations[frame] = torch.load(
                    folder / f"assignments_{frame:06d}.pt", weights_only=False
                )["schemes"]
        spatial_count = len(frames) * 4096
        if mass.shape != (3, spatial_count + pointer_count) or not np.isclose(
            mass.sum(), 1, atol=1e-4
        ):
            raise RuntimeError("Saved key layout/mass invalid")
        for name in annotations[frames[0]]:
            labels = np.concatenate(
                [annotations[frame][name]["labels"].ravel() for frame in frames]
            )
            for query in range(4):
                key_mass = mass.sum(0) if query == 3 else mass[query]
                query_fraction = (
                    1.0 if query == 3 else float(counts[query] / counts.sum())
                )
                spatial_mass = float(key_mass[:spatial_count].sum())
                for region in range(3):
                    selected = labels == region
                    region_mass = float(key_mass[:spatial_count][selected].sum())
                    opportunity = query_fraction * float(selected.mean())
                    for stratum in strata:
                        for level in (str(layer), "all"):
                            key = (stratum, name, level, str(query), str(region))
                            item = pool.setdefault(key, np.zeros(8))
                            item += [
                                region_mass,
                                spatial_mass,
                                opportunity,
                                query_fraction,
                                float(key_mass[spatial_count:].sum()),
                                1,
                                query_fraction == 0,
                                not selected.any(),
                            ]
    rows = []
    for key, values in sorted(pool.items()):
        rows.append(
            {
                "stratum": key[0],
                "definition": key[1],
                "layer": key[2],
                "query_region": key[3],
                "memory_region": key[4],
                "enrichment": pooled_enrichment(*values[:4]),
                "spatial_attention_share": values[0] / values[1] if values[1] else None,
                "token_opportunity_share": values[2] / values[3] if values[3] else None,
                "pointer_attention_share": values[4] / values[3] if values[3] else None,
                "region_joint_mass_sum": values[0],
                "spatial_joint_mass_sum": values[1],
                "opportunity_sum": values[2],
                "query_weight_sum": values[3],
                "pointer_joint_mass_sum": values[4],
                "reads": int(values[5]),
                "empty_query_reads": int(values[6]),
                "empty_memory_region_reads": int(values[7]),
            }
        )
    precisions = {}
    for name in next(iter(metadata.values()))["schemes"]:
        precisions[name] = {}
        for reference in ("self_precision", "gt_precision"):
            precisions[name][reference] = {}
            for mode in ("including_border", "excluding_border"):
                rows_ = [
                    r["schemes"][name][reference][mode]
                    for r in metadata.values()
                    if not r["conditioning"]
                ]
                hits = sum(r["contour_hits"] for r in rows_)
                support = sum(r["support_pixels"] for r in rows_)
                contour_key = (
                    "self_contour" if reference == "self_precision" else "gt_contour"
                )
                contour = sum(
                    r["schemes"][name][contour_key][mode]["contour_pixels"]
                    for r in metadata.values()
                    if not r["conditioning"]
                )
                precisions[name][reference][mode] = {
                    "hits": hits,
                    "support": support,
                    "precision": hits / support if support else None,
                    "contour_pixels": contour,
                    "coverage": hits / contour if contour else None,
                }
    return {
        "sequence": folder.name,
        "rows": rows,
        "precision": precisions,
        "dice_strata": {
            s: {
                "frames": len(v),
                "median": float(np.median(list(v.values()))),
                "minimum": min(v.values()),
                "maximum": max(v.values()),
            }
            for s, v in dice.items()
        },
    }


def cohort_summary(output):
    output = Path(output)
    sequences = [
        json.loads(path.read_text())
        for path in sorted(output.glob("seq*/token_summary.json"))
    ]
    grouped = {}
    for sequence in sequences:
        for row in sequence["rows"]:
            key = tuple(
                row[k]
                for k in (
                    "stratum",
                    "definition",
                    "layer",
                    "query_region",
                    "memory_region",
                )
            )
            grouped.setdefault(key, []).append(row["enrichment"])
    rows = [
        {
            **dict(
                zip(
                    ("stratum", "definition", "layer", "query_region", "memory_region"),
                    key,
                )
            ),
            **log_summary(values),
        }
        for key, values in sorted(grouped.items())
    ]
    result = {
        "sequences": sequences,
        "statistics": rows,
        "unit": "sequence",
        "inference": "descriptive, not causal",
    }
    (output / "token_results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Native memory-token attention",
        "",
        "Sequence-pooled ratios; natural-log enrichment medians. Query regions: fixed GT reference; memory regions: stored predictions. Pointers excluded from spatial enrichment. All-query index=3. Geometry, attention allocation and causal utility are distinct.",
        "",
        "| Definition | Layer | Memory region | N sequences | Median log enrichment [95% CI] | exp(median log) |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        if row["stratum"] != "all" or row["query_region"] != "3":
            continue
        lines.append(
            f"| {row['definition']} | {row['layer']} | {row['memory_region']} | {row['n']} | {row['median_log']} {row['ci95_log']} | {row['geometric_median']} |"
        )
    lines += [
        "",
        "Region indices: 0 background, 1 foreground interior, 2 boundary-associated. Full query-region, size and outcome strata, undefined counts, sequence estimates, Dice distributions and footprint precision are in token_results.json. No unexpected result triggers a definition or code change. Contiguity-matched exclusions remain pending a separately committed protocol.",
    ]
    lines += [
        "",
        "## Footprint precision beside contour coverage",
        "",
        "Pooled pixel counts, descriptive geometry only; native one-pixel contours, border included. Not a tolerance-based Boundary F score.",
        "",
        "| Definition | Self coverage | Self footprint precision | GT coverage | GT footprint precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sequences[0]["precision"] if sequences else []:
        values = []
        for reference in ("self_precision", "gt_precision"):
            items = [
                s["precision"][name][reference]["including_border"] for s in sequences
            ]
            hits = sum(r["hits"] for r in items)
            for denominator in ("contour_pixels", "support"):
                total = sum(r[denominator] for r in items)
                values.append(f"{hits / total:.6f}" if total else "N/A")
        lines.append(f"| {name} | {' | '.join(values)} |")
    (output / "token_results.md").write_text("\n".join(lines) + "\n")
    return result
