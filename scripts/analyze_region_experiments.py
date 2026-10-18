#!/usr/bin/env python3
"""E2--E6 reporting with matched frames and sequence-level statistics.

Reuses the registered Wilcoxon, Holm, bootstrap and failure-episode helpers.
Missing interventions, infeasible budgets and failed models stay explicit.
"""

# ruff: noqa: E402
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.utility_retrieval import (
    failure_episodes,
    holm_adjust,
    paired_wilcoxon,
    rank_agreement,
    sequence_bootstrap_interval,
)
from modeling.region_tokens import CATEGORIES
from scripts.run_region_token_pilot import write_json
from scripts.run_region_experiments import cholec_method_split


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def analysis_variant(rows):
    """Reject silent pooling of prospective controls with historical results."""
    variants = {
        (r.get("boundary_definition", "mixed_coverage"),
         r.get("spatial_match", "pooled_distance")) for r in rows
    }
    if len(variants) > 1:
        raise ValueError("Analyze taxonomy/spatial-control variants separately")
    boundary, spatial = next(iter(variants), ("mixed_coverage", "pooled_distance"))
    return {"boundary_definition": boundary, "spatial_match": spatial}


def cluster(row):
    # Cholec clips/classes from one source video cannot be resampled as
    # independent patients/videos. DAVIS follows sequence-object analysis.
    return (
        row["source_video"]
        if row.get("dataset") == "cholecseg8k"
        else f"{row['video']}:{row.get('label_id', 1)}"
        if row.get("dataset") == "davis"
        else row["video"]
    )


def unit_key(row):
    return (
        row.get("dataset", "polypgen"),
        row["video"],
        row.get("label_id", 1),
        row["frame"],
    )


def sequence_means(values):
    return {key: float(np.mean(v)) for key, v in values.items() if len(v)}


def feasibility_summary(rows, conditions):
    """Expose selection coverage rather than treating skipped arms as no effect."""
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if "category_counts" not in row:
            continue
        size = conditions.get(unit_key(row), {}).get("coverage_tokens")
        stratum = "unmeasured" if size is None else "small" if size < 64 else "medium" if size < 512 else "large"
        key = json.dumps([row.get("dataset", "polypgen"), cluster(row), row.get("bank"), row.get("category"), stratum])
        counts[key][row.get("state", "unknown")] += 1
        if row.get("reason"):
            counts[key]["reason:" + row["reason"]] += 1
    return dict(counts)


def episode_frames(frames):
    """Missing annotations/sampled targets break consecutive-frame episodes."""
    if not frames:
        return set()
    first, last = min(frames), max(frames)
    scores = [frames.get(t, np.nan) for t in range(first, last + 1)]
    return {
        first + i
        for episode in failure_episodes(scores)
        for i in range(episode.start, episode.end + 1)
    }


def paired_effects(rows):
    """E2 category-vs-control contrasts, exact target/arm match before averaging."""
    analysis_variant(rows)
    lookup = {}
    for row in rows:
        if row.get("state") != "complete" or "harm" not in row:
            continue
        arm = (
            row.get("dataset", "polypgen"),
            row.get("intervention_kind", "attention_mask"),
            row["mode"],
            row["bank"],
            row.get("memory_source", "c0"),
            row.get("band", 1),
            tuple(row.get("thresholds", [0.1, 0.9])),
            row.get("analysis_stratum", "all"),
        )
        key = arm, unit_key(row), row["category"], row["seed"]
        if key in lookup:
            raise ValueError(
                f"Duplicate completed target/arm; do not concatenate repeated shards: {key}"
            )
        lookup[key] = row
    results = {}
    for (arm, target, category, seed), row in lookup.items():
        if category != "boundary" or seed != 0:
            continue
        for control in (
            "foreground",
            "near_background",
            "far_background",
            "random",
            "spatial_random",
        ):
            controls = (
                [lookup.get((arm, target, control, s)) for s in range(5)]
                if control in ("random", "spatial_random")
                else [lookup.get((arm, target, control, 0))]
            )
            if any(c is None or c["k"] != row["k"] for c in controls):
                continue
            for metric in ("dice", "boundary_f", "iou", "hd95", "low_resolution_dice"):
                values = [
                    row["harm"].get(metric),
                    *[c["harm"].get(metric) for c in controls],
                ]
                if any(v is None or not np.isfinite(v) for v in values):
                    continue
                key = (*arm, metric, control)
                item = results.setdefault(
                    key,
                    {
                        "left": defaultdict(list),
                        "right": defaultdict(list),
                        "frames": 0,
                    },
                )
                name = cluster(row)
                item["left"][name].append(values[0])
                item["right"][name].append(float(np.mean(values[1:])))
                item["frames"] += 1
    output, families = {}, defaultdict(dict)
    for key, item in results.items():
        left, right = sequence_means(item["left"]), sequence_means(item["right"])
        diffs = {name: left[name] - right[name] for name in left}
        test = paired_wilcoxon(left, right)
        name = json.dumps(key)
        output[name] = {
            "arm": list(key),
            "paired_frames": item["frames"],
            "n_sequences": len(diffs),
            "mean_excess_harm": float(np.mean(list(diffs.values()))),
            "ci95": sequence_bootstrap_interval(diffs),
            "p": test["p"],
            "minimum_effect": {"dice": 0.01, "boundary_f": 0.02}.get(key[-2]),
            "sequence_differences": diffs,
        }
        # One family for all category comparisons within dataset/operator/mode/
        # condition/taxonomy/endpoint, including both banks.
        family = (*key[:3], *key[4:-1])
        families[family][name] = test["p"]
    for family in families.values():
        for name, p in holm_adjust(family).items():
            output[name]["p_holm"] = p
    return output


def stratified_rows(rows, conditions):
    """Define episode membership once per baseline trajectory, then reuse it."""
    trajectories = defaultdict(dict)
    for row in rows:
        if row.get("state") != "complete" or not row.get("baseline"):
            continue
        group = (
            unit_key(row)[:3],
            row.get("mode"),
            row.get("memory_source", "c0"),
            row.get("intervention_kind", "attention_mask"),
            row.get("band", 1),
            tuple(row.get("thresholds", [0.1, 0.9])),
        )
        trajectories[group][row["frame"]] = row["baseline"]["dice"]
    labels = {}
    for group, frames in trajectories.items():
        ordered = sorted(frames)
        failed = episode_frames(frames)
        for i, t in enumerate(ordered):
            labels[group, t] = (
                "failure_episode"
                if t in failed
                else "success"
                if frames[t] >= 0.5
                else "transient_failure"
            )
    output = list(rows)
    for row in rows:
        if row.get("state") != "complete" or not row.get("baseline"):
            continue
        group = (
            unit_key(row)[:3],
            row.get("mode"),
            row.get("memory_source", "c0"),
            row.get("intervention_kind", "attention_mask"),
            row.get("band", 1),
            tuple(row.get("thresholds", [0.1, 0.9])),
        )
        output.append({**row, "analysis_stratum": labels[group, row["frame"]]})
        features = conditions.get(unit_key(row), {})
        if features.get("coverage_tokens") is not None:
            size = features["coverage_tokens"]
            output.append(
                {
                    **row,
                    "analysis_stratum": "size:"
                    + ("small" if size < 64 else "medium" if size < 512 else "large"),
                }
            )
    return output


def persistent_effect_rows(rows):
    """Pair every accumulated arm with the unmodified rollout on the same frame."""

    def identity(row):
        return (
            unit_key(row),
            row.get("experiment_stage", "E2"),
            row.get("rollout_operator", "attention_mask"),
            row.get("memory_source", "c0_initialized_then_predicted"),
            row.get("band", 1),
            tuple(row.get("thresholds", [0.1, 0.9])),
        )

    baseline = {}
    for row in rows:
        if row.get("category") == "baseline" and "metrics" in row:
            key = identity(row)
            if key in baseline:
                raise ValueError("Duplicate persistent baseline configuration/frame")
            baseline[key] = row
    output = []
    for row in rows:
        if (
            row.get("bank") not in {"conditioning", "recent"}
            or "metrics" not in row
            or not row.get("k")
        ):
            continue
        reference = baseline.get(identity(row))
        if reference is None:
            raise ValueError("Persistent row has no matching baseline frame")
        harm = {
            name: (
                (row["metrics"][name] - reference["metrics"][name])
                if name == "hd95"
                else (reference["metrics"][name] - row["metrics"][name])
            )
            if row["metrics"].get(name) is not None
            and reference["metrics"].get(name) is not None
            else None
            for name in ("dice", "iou", "boundary_f", "hd95", "low_resolution_dice")
        }
        output.append(
            {
                **row,
                "state": "complete" if row.get("feasible") else "skipped",
                "baseline": reference["metrics"],
                "harm": harm,
                "harm_per_token": {
                    name: v / row["k"] if v is not None else None
                    for name, v in harm.items()
                },
            }
        )
    return output


def attention_agreement(rows):
    analysis_variant(rows)
    targets = defaultdict(dict)
    for row in rows:
        if (
            row.get("state") == "complete"
            and row.get("category") in CATEGORIES
            and row.get("harm_per_token")
            and row.get("attention")
        ):
            arm = (
                row.get("dataset", "polypgen"),
                row.get("intervention_kind", "attention_mask"),
                row.get("mode", "single_step"),
                row.get("memory_source", "c0"),
                row["bank"],
                row.get("band", 1),
                tuple(row.get("thresholds", [0.1, 0.9])),
            )
            targets[(arm, unit_key(row))][row["category"]] = row
    values, coverage = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    query_values = defaultdict(lambda: defaultdict(list))
    for (arm, target), categories in targets.items():
        if set(categories) != set(CATEGORIES):
            coverage[(arm, "infeasible_category_frames")] += 1
            continue
        first = categories["boundary"]
        for query in ("foreground", "boundary", "background"):
            layers = first.get("attention", [])
            for layer_idx in [*range(len(layers)), "mean"]:
                for criterion in (
                    "attention_mass",
                    "attention_per_token",
                    "anchor_similarity",
                ):
                    saliency = []
                    for name in CATEGORIES:
                        if criterion == "anchor_similarity":
                            value = (first.get("anchor_similarity") or {}).get(name)
                        else:
                            selected = (
                                layers if layer_idx == "mean" else [layers[layer_idx]]
                            )
                            masses = [
                                layer[query][name]
                                for layer in selected
                                if layer.get(query) is not None
                            ]
                            value = float(np.mean(masses)) if masses else None
                            if criterion == "attention_per_token" and value is not None:
                                count = first["category_counts"][name]
                                value = value / count if count else None
                        saliency.append(value)
                    effect = [
                        categories[name]["harm_per_token"]["boundary_f"]
                        for name in CATEGORIES
                    ]
                    if any(v is None for v in (*saliency, *effect)):
                        continue
                    local_effect = [
                        categories[name].get("query_dice_harm", {}).get(query)
                        for name in CATEGORIES
                    ]
                    if all(v is not None and np.isfinite(v) for v in local_effect):
                        local_rho = rank_agreement(saliency, local_effect)["spearman"]
                        if np.isfinite(local_rho):
                            query_values[(arm, query, layer_idx, criterion)][cluster(first)].append(local_rho)
                    else:
                        coverage[(arm, query, "missing_query_effect")] += 1
                    rho = rank_agreement(saliency, effect)["spearman"]
                    if np.isfinite(rho):
                        values[(arm, query, layer_idx, criterion)][
                            cluster(first)
                        ].append(rho)
    output = {}
    for key, by_sequence in values.items():
        means = sequence_means(by_sequence)
        rho = float(np.mean(list(means.values())))
        output[json.dumps(key)] = {
            "rho": rho,
            "ci95": sequence_bootstrap_interval(means),
            "n_sequences": len(means),
            "n_frames": sum(map(len, by_sequence.values())),
            "disagreement_threshold": 0.3,
            "below_threshold": rho < 0.3,
        }
    return {
        "agreement": output,
        "agreement_effect_scope": "whole_object_boundary_F_harm_per_token",
        "query_specific_agreement": {
            json.dumps(key): {
                "rho": float(np.mean(list(sequence_means(by_sequence).values()))),
                "ci95": sequence_bootstrap_interval(sequence_means(by_sequence)),
                "n_sequences": len(by_sequence),
                "n_frames": sum(map(len, by_sequence.values())),
                "effect_scope": "signed_Dice_harm_in_same_query_region",
            }
            for key, by_sequence in query_values.items()
        },
        "coverage": {json.dumps(k): v for k, v in coverage.items()},
        "anchor_scope": "Official Lean max-cosine function with all prompted-foreground anchors; complete TAMP reported separately",
    }


def clustered_interaction(data, formula):
    """OLS sensitivity with sandwich covariance clustered by independent unit.

    Statsmodels supplies the estimator and finite-sample correction. This is
    separate from, not a replacement for, the registered random-intercept fit.
    Within-cluster target pairing and serial dependence may be arbitrary.
    """
    import statsmodels.formula.api as smf

    n = int(data["sequence"].nunique())
    try:
        fit = smf.ols(formula, data=data, missing="raise").fit(
            cov_type="cluster",
            cov_kwds={"groups": data["sequence"], "use_correction": True, "df_correction": True},
            use_t=True,
        )
        terms = [i for i, name in enumerate(fit.params.index) if ":" in name]
        if not terms or n <= len(terms) + 1:
            return {"state": "unavailable", "reason": "too few clusters for interaction degrees of freedom", "n_clusters": n}
        contrast = np.eye(len(fit.params))[terms]
        covariance = contrast @ fit.cov_params() @ contrast.T
        if np.linalg.matrix_rank(covariance) < len(terms):
            return {"state": "unavailable", "reason": "singular interaction covariance", "n_clusters": n}
        test = fit.wald_test(contrast, use_f=True, scalar=True)
        p = float(test.pvalue)
        if not np.isfinite(p):
            raise ValueError("Nonfinite cluster-robust interaction p-value")
        return {
            "state": "complete", "estimator": "OLS with sequence-clustered sandwich covariance",
            "n_clusters": n, "n_rows": len(data), "p_interaction": p,
            "finite_sample_correction": True,
            "few_cluster_warning": n < 30,
            "inference_scope": "sensitivity only; few-cluster inference may be unreliable",
        }
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        return {"state": "failed", "reason": str(exc), "n_clusters": n}


def interaction_models(rows, conditions):
    analysis_variant(rows)
    import pandas as pd
    import statsmodels.formula.api as smf

    records = []
    for row in rows:
        if (
            row.get("state") != "complete"
            or row.get("category") not in CATEGORIES
            or "harm" not in row
        ):
            continue
        features = conditions.get(unit_key(row), {})
        records.append(
            {
                "sequence": f"{row.get('dataset', 'polypgen')}:{cluster(row)}",
                "category": row["category"],
                "corruption": row.get("memory_source", "c0"),
                "dataset": row.get("dataset", "polypgen"),
                "bank": row["bank"],
                "operator": row.get("intervention_kind", "attention_mask"),
                "mode": row.get("mode", "single_step"),
                "band": row.get("band", 1),
                "thresholds": tuple(row.get("thresholds", [0.1, 0.9])),
                "harm": row["harm"]["boundary_f"],
                **features,
            }
        )
    data = pd.DataFrame(records)
    if data.empty:
        return {"state": "unavailable", "reason": "no interventional rows"}
    # Primary taxonomy and primary removal operator; avoid treating sensitivity
    # reruns of the same target as independent observations.
    data = data[
        (data["operator"] == "attention_mask")
        & (data["band"] == 1)
        & (data["thresholds"] == (0.1, 0.9))
    ]
    output = {}
    for (bank, mode), group in data.groupby(["bank", "mode"]):
        for condition in (
            "corruption",
            "contrast",
            "motion",
            "specular_fraction",
            "coverage_tokens",
            "class_group",
        ):
            key = f"{bank}/{mode}/{condition}"
            if condition not in group:
                output[key] = {
                    "state": "unavailable",
                    "reason": "condition measurements absent",
                }
                continue
            subset = group.dropna(subset=[condition, "harm"])
            if condition != "corruption":
                subset = subset[subset["corruption"] == "c0"]
            if subset[condition].nunique() < 2 or subset["sequence"].nunique() < 3:
                output[key] = {
                    "state": "unavailable",
                    "reason": "insufficient independent sequences or condition variation",
                }
                continue
            rhs = (
                    f"C({condition})"
                    if condition in ("corruption", "class_group")
                    else condition
                )
            robust = clustered_interaction(subset, f"harm ~ C(category) * {rhs}")
            try:
                with warnings.catch_warnings(record=True) as messages:
                    model = smf.mixedlm(
                        f"harm ~ C(category) * {rhs}", subset, groups=subset["sequence"]
                    ).fit(method="lbfgs")
                terms = [i for i, n in enumerate(model.fe_params.index) if ":" in n]
                contrast = np.zeros((len(terms), len(model.params)))
                for i, index in enumerate(terms):
                    contrast[i, index] = 1
                interaction = model.wald_test(contrast, scalar=True)
                output[key] = {
                    "state": "complete" if model.converged else "failed",
                    "converged": bool(model.converged),
                    "n_sequences": int(subset["sequence"].nunique()),
                    "n_rows": len(subset),
                    "coefficients": model.fe_params.to_dict(),
                    "p_interaction": float(interaction.pvalue),
                    "warnings": [str(w.message) for w in messages],
                    "sequence_clustered_sensitivity": robust,
                }
            except (ValueError, np.linalg.LinAlgError) as exc:
                output[key] = {"state": "failed", "reason": str(exc), "sequence_clustered_sensitivity": robust}
    valid = {
        k: v["p_interaction"] for k, v in output.items() if v.get("state") == "complete"
    }
    for key, p in holm_adjust(valid).items():
        output[key]["p_holm"] = p
    robust_valid = {
        k: v["sequence_clustered_sensitivity"]["p_interaction"]
        for k, v in output.items()
        if v.get("sequence_clustered_sensitivity", {}).get("state") == "complete"
    }
    for key, p in holm_adjust(robust_valid).items():
        output[key]["sequence_clustered_sensitivity"]["p_holm"] = p
    return output


def method_comparisons(rows):
    if not any(r.get("category") == "full" for r in rows):
        return {}
    # E2 persistent controls are not E6 baselines. Their trajectories can differ
    # by operator/taxonomy even on the same sequence and frame.
    rows = [r for r in rows if r.get("experiment_stage") not in {"E2", "E3"}]
    baseline, lookup = {}, {}
    for row in rows:
        if row.get("state") != "complete" or "metrics" not in row:
            continue
        key = unit_key(row)
        arm = row["category"]
        identity = key, arm, row.get("bank"), row.get("seed", 0)
        if identity in lookup:
            raise ValueError(
                "Duplicate method target/arm; analyze lambda trials separately"
            )
        lookup[identity] = row
        if arm == "baseline":
            baseline[key] = row
    objects = defaultdict(list)
    for key, row in baseline.items():
        objects[key[:3]].append(row)
    strata = {}
    for values in objects.values():
        ordered = sorted(values, key=lambda r: r["frame"])
        indices = episode_frames({r["frame"]: r["metrics"]["dice"] for r in ordered})
        for i, row in enumerate(ordered):
            strata[unit_key(row)] = (
                "failure_episode"
                if row["frame"] in indices
                else "success"
                if row["metrics"]["dice"] >= 0.5
                else "transient_failure"
            )
    # H4 compares the frozen method against each comparator on the baseline's
    # same failure/success frames. Keep the four preregistered primary
    # comparators distinct from descriptive ablations.
    method = {
        unit_key(r): r for r in rows if r.get("category") == "full" and "metrics" in r
    }
    paired = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
    for (key, arm, bank, seed), row in lookup.items():
        if (
            arm == "full"
            or key not in baseline
            or key not in method
            or bank is not None
        ):
            continue
        reference = baseline[key]
        for metric in ("dice", "boundary_f", "iou", "stability_error"):
            left = (
                method[key].get(metric)
                if metric == "stability_error"
                else method[key]["metrics"].get(metric)
            )
            right = (
                row.get(metric)
                if metric == "stability_error"
                else row["metrics"].get(metric)
            )
            if left is None or right is None:
                continue
            for stratum in ("all", strata[key]):
                group = (key[0], arm, stratum, metric)
                a, b = paired[group]
                a[cluster(reference)].append(left)
                b[cluster(reference)].append(right)
    output, families = {}, defaultdict(dict)
    for key, (left, right) in paired.items():
        left, right = sequence_means(left), sequence_means(right)
        difference = {s: left[s] - right[s] for s in left}
        test = paired_wilcoxon(left, right)
        name = json.dumps(key)
        output[name] = {
            "mean_gain": float(np.mean(list(difference.values()))),
            "ci95": sequence_bootstrap_interval(difference),
            "n_sequences": len(difference),
            "p": test["p"],
            "sequence_differences": difference,
        }
        output[name]["primary_comparator"] = key[1] in {
            "baseline",
            "SAM2Long",
            "Lean-SAM2",
            "efficient_saliency",
        }
        if output[name]["primary_comparator"]:
            families[(key[0], key[2], key[3])][name] = test["p"]
    for family in families.values():
        for name, p in holm_adjust(family).items():
            output[name]["p_holm"] = p
    return output


def freeze_method(trial_dirs, candidate_dir):
    """Select lambda only after the complete fixed grid finishes on dev."""
    from scripts.run_region_experiments import digest

    candidates = {}
    for path in candidate_dir.glob("lambda_*.json"):
        value = json.loads(path.read_text())
        if value.get("selection_split") != "dev" or value.get("evaluation_ready"):
            raise ValueError("Candidate config is not a dev tuning candidate")
        candidates[digest(path)] = value
    if {v["scale"] for v in candidates.values()} != {0.0, 0.25, 0.5, 1.0, 2.0}:
        raise ValueError("The full fixed lambda grid is required")
    if len({tuple(v["weights"]) for v in candidates.values()}) != 1:
        raise ValueError("Weights changed across lambda candidates")
    trials = defaultdict(list)
    receipts = []
    for directory in trial_dirs:
        setup = json.loads((directory / "setup.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        fingerprint = setup.get("method_sha256")
        if (
            status.get("state") != "complete"
            or setup.get("stage") != "E6_dev"
            or fingerprint not in candidates
        ):
            raise ValueError(
                "Lambda trial is incomplete, not dev, or from another candidate grid"
            )
        for path in directory.glob("*/rollout.jsonl"):
            trials[fingerprint].extend(
                json.loads(line) for line in path.read_text().splitlines() if line
            )
        receipts.append(
            {
                "setup_sha256": digest(directory / "setup.json"),
                "status_sha256": digest(directory / "status.json"),
            }
        )
    if set(trials) != set(candidates):
        raise ValueError("Some lambda candidates have no completed trial")
    coverage, scores = {}, {}
    required_polyp = {f"seq{i}" for i in range(2, 16) if i != 7}
    required_davis = set(
        json.loads((ROOT / "protocols/utility_retrieval_v1.json").read_text())[
            "davis_2017"
        ]["model_selection_holdout"]
    )
    for fingerprint, rows in trials.items():
        if any(not dev_row(r) for r in rows):
            raise ValueError("Test outcomes appeared in a lambda trial")
        lookup = {}
        for row in rows:
            if row["category"] not in {"baseline", "full"}:
                continue
            key = unit_key(row), row["category"]
            if key in lookup:
                raise ValueError("Duplicate dev lambda frame")
            lookup[key] = row
        targets = {key for key, arm in lookup if arm == "baseline"}
        if targets != {key for key, arm in lookup if arm == "full"}:
            raise ValueError("Missing paired method/baseline dev predictions")
        if {key[1] for key in targets if key[0] == "polypgen"} != required_polyp or {
            key[1] for key in targets if key[0] == "davis"
        } != required_davis:
            raise ValueError(
                "All 13 promptable PolypGen dev sequences and ten DAVIS holdout sequences are required"
            )
        coverage[fingerprint] = targets
        by_domain = defaultdict(lambda: defaultdict(list))
        for key in targets:
            base, method = lookup[key, "baseline"], lookup[key, "full"]
            gain = np.mean(
                [
                    method["metrics"][metric] - base["metrics"][metric]
                    for metric in ("dice", "boundary_f")
                ]
            )
            by_domain[key[0]][cluster(base)].append(float(gain))
        # Equal dataset weight; first average frames within sequence/object.
        scores[fingerprint] = float(
            np.mean(
                [
                    np.mean(list(sequence_means(values).values()))
                    for values in by_domain.values()
                ]
            )
        )
    if any(v != next(iter(coverage.values())) for v in coverage.values()):
        raise ValueError("Lambda candidates used different dev targets")
    selected = max(scores, key=lambda k: (scores[k], -candidates[k]["scale"]))
    return {
        **candidates[selected],
        "evaluation_ready": True,
        "selection_score": scores[selected],
        "selection_rule": "Max equal-domain mean of sequence-averaged Dice/boundary-F gains; tie -> smallest lambda",
        "lambda_scores": {str(candidates[k]["scale"]): v for k, v in scores.items()},
        "trial_receipts": receipts,
    }


def dev_row(row):
    dataset = row.get("dataset", "polypgen")
    if dataset == "polypgen":
        return row["video"] in {f"seq{i}" for i in range(1, 16)}
    if dataset == "davis":
        return row.get("split") in {"train", "holdout"}
    if dataset == "cholecseg8k":
        return row.get("source_video") in cholec_method_split()["dev_source_videos"]
    return False


def category_weights(rows):
    """Fixed dev-only mapping from E2 per-token harm to positive weights."""
    if analysis_variant(rows) != {"boundary_definition": "mixed_coverage", "spatial_match": "pooled_distance"}:
        raise ValueError("Sensitivity variants cannot select the registered E6 weights")
    if not rows or any(not dev_row(r) for r in rows):
        raise ValueError("All weight-selection rows must come from registered dev data")
    values = {name: defaultdict(list) for name in CATEGORIES}
    for row in rows:
        if (
            row.get("state") == "complete"
            and row.get("category") in CATEGORIES
            and row.get("intervention_kind", "attention_mask") == "attention_mask"
            and row.get("memory_source", "c0") == "c0"
            and row.get("band", 1) == 1
            and row.get("thresholds", [0.1, 0.9]) == [0.1, 0.9]
        ):
            values[row["category"]][
                f"{row.get('dataset', 'polypgen')}:{cluster(row)}"
            ].append(row["harm_per_token"]["boundary_f"])
    means = []
    for name in CATEGORIES:
        valid = sequence_means(values[name])
        if len(valid) < 3:
            raise ValueError(f"Too few independent dev sequences for {name}")
        means.append(float(np.mean(list(valid.values()))))
    scale = max(max(abs(v) for v in means), 1e-8)
    weights = np.clip(1 + np.asarray(means) / scale, 0.25, 4.0).tolist()
    return {
        "weights": weights,
        "category_order": list(CATEGORIES),
        "dev_per_token_effects": means,
        "formula": "clip(1+sequence_mean_boundary_F_harm_per_token/max_abs_effect,.25,4)",
        "selection_split": "dev",
        "frozen": True,
        "evaluation_ready": False,
    }


def query_matrix(rows):
    """Equal-sequence direct effects, retaining the four memory categories."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("state") != "complete" or row.get("category") not in CATEGORIES:
            continue
        measurements = [
            ("absolute_logit_change", row.get("query_logit_changes", {})),
            ("dice_harm", row.get("query_dice_harm", {})),
        ]
        for measurement, regions in measurements:
            for query, value in regions.items():
                if value is None:
                    continue
                key = (
                    row.get("dataset", "polypgen"),
                    row.get("intervention_kind", "attention_mask"),
                    row.get("mode", "single_step"),
                    row.get("memory_source", "c0"),
                    row["bank"],
                    row.get("band", 1),
                    tuple(row.get("thresholds", [0.1, 0.9])),
                    query,
                    row["category"],
                    measurement,
                )
                groups[key][cluster(row)].append(value)
    output = {}
    for key, values in groups.items():
        means = sequence_means(values)
        output[json.dumps(key)] = {
            "mean_effect": float(np.mean(list(means.values()))),
            "ci95": sequence_bootstrap_interval(means),
            "n_sequences": len(means),
        }
    return output


def write_tables_and_figures(result, directory, figures=False):
    """Export the same recorded estimates; never fill unrun cells with zero."""
    directory.mkdir(parents=True, exist_ok=True)
    tables = {name: result.get(name, {}) for name in ("E2", "E6", "query_matrix")}
    tables["E3_E4"] = result.get("E3_E4", {})
    # An unavailable model is a status object, not a table of estimates.
    if "state" in tables["E3_E4"]:
        tables["E3_E4"] = {"availability": tables["E3_E4"]}
    e5 = result.get("E5", {})
    tables["E5_global"] = {
        key: {**value, "effect_scope": e5.get("agreement_effect_scope")}
        for key, value in e5.get("agreement", {}).items()
    }
    tables["E5_query_specific"] = e5.get("query_specific_agreement", {})
    tables["feasibility"] = result.get("feasibility_by_cluster_bank_category_size", {})
    for name, table in tables.items():
        if not table:
            continue
        fields = [
            "comparison",
            *sorted(
                set().union(
                    *(set(v) - {"sequence_differences"} for v in table.values())
                )
            ),
        ]
        with (directory / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key, value in table.items():
                writer.writerow(
                    {
                        "comparison": key,
                        **{
                            k: json.dumps(v) if isinstance(v, (list, dict)) else v
                            for k, v in value.items()
                            if k != "sequence_differences"
                        },
                    }
                )
    if not figures:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for key, value in result["E2"].items():
        arm = json.loads(key)
        if (
            arm[1] == "attention_mask"
            and arm[2] == "single_step"
            and arm[4] == "c0"
            and arm[5] == 1
            and arm[6] == [0.1, 0.9]
            and arm[7] == "all"
            and arm[-1] == "random"
            and arm[-2] in ("dice", "boundary_f")
        ):
            groups[arm[-2]].append((f"{arm[0]} / {arm[3]}", value))
    for metric, values in groups.items():
        fig, ax = plt.subplots(figsize=(7, max(2, 0.45 * len(values))))
        for y, (label, value) in enumerate(values):
            low, high = value["ci95"]
            ax.plot([100 * low, 100 * high], [y, y], color="tab:blue")
            ax.plot(100 * value["mean_excess_harm"], y, "o", color="tab:blue")
        ax.set_yticks(range(len(values)), [v[0] for v in values])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            f"Boundary excess harm vs matched random ({metric} points; 95% sequence CI)"
        )
        fig.tight_layout()
        fig.savefig(directory / f"E2_{metric}.png", dpi=200)
        plt.close(fig)


def publish_analysis(directory, stage, files, config):
    """Track analysis choices without overwriting inference tracking.json."""
    from training.tracking import start_wandb, log_results_artifact

    tracking = directory / f"{stage}_tracking"
    tracking.mkdir(parents=True, exist_ok=True)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"region-tokens-{stage}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        {"stage": stage, **config},
        tracking,
        ["region-tokens-v1", stage],
    )
    try:
        run.summary["full_study_complete"] = False
        log_results_artifact(
            run,
            directory,
            [str(p.relative_to(directory)) for p in files]
            + [str((tracking / "tracking.json").relative_to(directory))],
        )
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--conditions", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-method", action="store_true")
    parser.add_argument("--freeze-method", action="store_true")
    parser.add_argument("--trial-dir", type=Path, action="append", default=[])
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    from scripts.run_region_experiments import validate_output

    validate_output(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.freeze_method:
        if (
            not args.trial_dir
            or not args.candidate_dir
            or args.prepare_method
            or args.input
        ):
            parser.error(
                "--freeze-method requires --trial-dir and --candidate-dir only"
            )
        write_json(args.output, freeze_method(args.trial_dir, args.candidate_dir))
        publish_analysis(
            args.output.parent,
            "E6_method_freeze",
            [args.output],
            {"trial_count": len(args.trial_dir)},
        )
        return
    if not args.input:
        parser.error("--input is required for analysis or weight preparation")
    rows = [
        json.loads(line)
        for path in args.input
        for line in path.read_text().splitlines()
        if line
    ]
    if args.prepare_method:
        base = category_weights(rows)
        from scripts.run_region_experiments import digest

        setups = []
        for path in args.input:
            source = next(
                (
                    parent / "setup.json"
                    for parent in path.parents
                    if (parent / "setup.json").is_file()
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    "Weight preparation requires inference setup provenance for each input"
                )
            setups.append(json.loads(source.read_text()))
        checkpoints = {s["checkpoint_sha256"] for s in setups}
        if len(checkpoints) != 1 or any(
            s.get("stage") not in {"E2", "E2_attention_primary"} for s in setups
        ):
            raise ValueError("Weights require E2 outcomes from one fixed checkpoint")
        base.update(
            checkpoint_sha256=next(iter(checkpoints)),
            source_sha256={str(p): digest(p) for p in args.input},
        )
        args.output.mkdir(parents=True, exist_ok=False)
        for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
            write_json(args.output / f"lambda_{scale:g}.json", {**base, "scale": scale})
        publish_analysis(
            args.output,
            "E6_candidates",
            list(args.output.glob("lambda_*.json")),
            {"checkpoint_sha256": base["checkpoint_sha256"]},
        )
        return
    conditions = {}
    for path in args.conditions:
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                key = unit_key(row)
                if key in conditions:
                    raise ValueError(f"Duplicate condition measurement: {key}")
                conditions[key] = {
                    name: row[name]
                    for name in (
                        "contrast",
                        "motion",
                        "specular_fraction",
                        "coverage_tokens",
                        "class_group",
                    )
                }
    effect_rows = rows + persistent_effect_rows(rows)
    result = {
        "analysis_variant": analysis_variant(rows),
        "review_protocol": "region_tokens_review_v1",
        "feasibility_by_cluster_bank_category_size": feasibility_summary(rows, conditions),
        "E2": paired_effects(stratified_rows(effect_rows, conditions)),
        "E3_E4": interaction_models(rows, conditions),
        "E5": attention_agreement(rows),
        "E6": method_comparisons(rows),
        "query_matrix": query_matrix(rows),
        "full_study_complete": False,
        "skipped_rows": sum(r.get("state") == "skipped" for r in rows),
        "coverage": sorted(
            {
                json.dumps(
                    [
                        r.get("dataset", "polypgen"),
                        r.get("intervention_kind", "attention_mask"),
                        r.get("mode"),
                        r.get("memory_source", "c0"),
                    ]
                )
                for r in rows
            }
        ),
        "claim": "Component analysis only; complete coverage and baseline comparisons must be verified before hypothesis claims",
    }
    write_json(args.output, clean(result))
    write_tables_and_figures(clean(result), args.output.parent / "tables", args.figures)
    publish_analysis(
        args.output.parent,
        "analysis",
        [
            args.output,
            *[p for p in (args.output.parent / "tables").glob("*") if p.is_file()],
        ],
        {"input_rows": len(rows), "input_files": [str(p) for p in args.input]},
    )


if __name__ == "__main__":
    main()
