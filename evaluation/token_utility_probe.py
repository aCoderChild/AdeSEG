"""Offline group-utility probes; scikit-learn fits, no SAM2 intervention.

One row is an entire recorded intervention (all older frames and four layers).
Neither layer records nor constituent tokens receive independent utility labels.
"""
from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust

ARMS = ("rejected", "retained", "random")
MODES = ("features", "geometry", "regional")


def block_indices(block_ids):
    """Invert Run171's row-major, disjoint 4x4-cell unfolding exactly."""
    raw = np.asarray(block_ids)
    ids = raw.astype(np.int64)
    if not np.array_equal(raw, ids):
        raise ValueError("Block IDs must be integers")
    if ids.shape != (16,) or len(np.unique(ids)) != 16 or np.any((ids < 0) | (ids >= 256)):
        raise ValueError("Expected 16 distinct blocks in a 64x64 token grid")
    offsets = (np.arange(4)[:, None] * 64 + np.arange(4)[None, :]).ravel()
    return ((ids // 16 * 4 * 64 + ids % 16 * 4)[:, None] + offsets).ravel()


def field_features(vectors, field):
    """Continuous occupancy moments; no empty-support division or threshold."""
    field = np.asarray(field, dtype=np.float64)
    covariance = ((field - field.mean())[:, None] * vectors).mean(0)
    return np.r_[field.mean(), field.std(), covariance]


def layer_features(vectors, occupancy, mixture, indices, ages, usage, latest_mean):
    """Aggregate the union removed in ONE layer, preserving its native vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 64:
        raise ValueError("Expected N x 64 native memory vectors")
    x = (np.asarray(indices) % 64 + .5) / 64
    y = (np.asarray(indices) // 64 + .5) / 64
    mean = vectors.mean(0)
    generic = np.r_[mean, vectors.std(0), mean * latest_mean,
                    np.mean(ages), np.std(ages), np.mean(usage),
                    np.linalg.norm(vectors, axis=1).mean(),
                    x.mean(), y.mean(), x.std(), y.std()]
    regional = np.r_[field_features(vectors, occupancy), field_features(vectors, mixture)]
    geometry = np.r_[field_features(vectors, x), field_features(vectors, y)]
    if regional.shape != geometry.shape or not all(np.isfinite(a).all() for a in (generic, regional, geometry)):
        raise ValueError("Invalid or unmatched probe features")
    return generic, geometry, regional


def sequence_weights(groups):
    """Equal sequence mass, normalized to the number of rows for stable alpha."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(weights) / weights.sum()


def fit_probe(x, y, groups, alpha):
    weights = sequence_weights(groups)
    model = Pipeline([("scale", StandardScaler()),
                      ("ridge", Ridge(alpha=alpha, solver="cholesky"))])
    model.fit(x, y, scale__sample_weight=weights, ridge__sample_weight=weights)
    return model


def equal_sequence_mse(y, prediction, groups):
    return float(np.mean([np.mean((y[groups == g] - prediction[groups == g]) ** 2)
                          for g in np.unique(groups)]))


def choose_alpha(x, y, groups, alphas, inner_folds):
    scores = {}
    for alpha in alphas:
        prediction = np.full(len(y), np.nan)
        for train, valid in GroupKFold(n_splits=inner_folds).split(x, y, groups):
            model = fit_probe(x[train], y[train], groups[train], alpha)
            prediction[valid] = model.predict(x[valid])
        scores[str(alpha)] = equal_sequence_mse(y, prediction, groups)
    # Prefer stronger regularization for exact ties. No outer labels enter here.
    selected = min(alphas, key=lambda a: (scores[str(a)], -a))
    return selected, scores


def nested_predictions(matrices, y, groups, *, alphas, inner_folds, on_fold=None):
    """Leave one whole sequence out, with group-isolated inner preprocessing."""
    if set(matrices) != set(MODES) or len(np.unique(groups)) <= inner_folds:
        raise ValueError("Incomplete conditions or insufficient training sequences")
    predictions = {mode: np.full(len(y), np.nan) for mode in (*MODES, "zero", "train_mean")}
    for train, test in LeaveOneGroupOut().split(matrices["features"], y, groups):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("A sequence leaked across the outer split")
        record = {"train_sequences": sorted(set(groups[train])),
                  "test_sequence": str(groups[test][0]), "conditions": {}}
        predictions["zero"][test] = 0.0
        predictions["train_mean"][test] = np.average(y[train], weights=sequence_weights(groups[train]))
        for mode, x in matrices.items():
            alpha, scores = choose_alpha(x[train], y[train], groups[train], alphas, inner_folds)
            model = fit_probe(x[train], y[train], groups[train], alpha)
            predictions[mode][test] = model.predict(x[test])
            record["conditions"][mode] = {"alpha": alpha, "inner_equal_sequence_mse": scores,
                "features": x.shape[1], "scaler_mean": model["scale"].mean_.tolist(),
                "scaler_scale": model["scale"].scale_.tolist(),
                "coefficient": model["ridge"].coef_.tolist(),
                "intercept": float(model["ridge"].intercept_)}
        if on_fold:
            on_fold(record, test, {m: p[test] for m, p in predictions.items()})
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Missing or nonfinite out-of-sequence predictions")
    return predictions


def sequence_scores(rows, y, prediction):
    """Selection summaries are offline comparisons among THREE archived arms."""
    groups = np.array([r["sequence"] for r in rows])
    scores = {}
    for name in np.unique(groups):
        ids = np.flatnonzero(groups == name)
        targets = sorted(set(rows[i]["frame"] for i in ids))
        ranks, selected, uniform, oracle = [], [], [], []
        for frame in targets:
            indices = [i for i in ids if rows[i]["frame"] == frame]
            if len(indices) != 3 or {rows[i]["arm"] for i in indices} != set(ARMS):
                raise ValueError("Each target must have exactly the three matched arms")
            u, p = y[indices], prediction[indices]
            for a, b in combinations(range(3), 2):
                if u[a] != u[b]:
                    ranks.append(.5 if p[a] == p[b] else float((p[a] < p[b]) == (u[a] < u[b])))
            # Average exact prediction ties; arm ordering cannot improve the score.
            selected.append(-float(u[p == p.min()].mean()))
            uniform.append(-float(u.mean()))
            oracle.append(-float(u.min()))
        scores[str(name)] = {"mse": float(np.mean((y[ids] - prediction[ids]) ** 2)),
            "mae": float(np.mean(np.abs(y[ids] - prediction[ids]))),
            "pair_concordance": float(np.mean(ranks)) if ranks else None,
            "non_tied_pairs": len(ranks), "targets": len(targets),
            "chosen_minus_native_dice": float(np.mean(selected)),
            "chosen_minus_uniform_arm_dice": float(np.mean(np.array(selected)-uniform)),
            "oracle_best_of_three_minus_native_dice": float(np.mean(oracle))}
    return scores


def summarize(rows, y, predictions, seed):
    per_sequence = {m: sequence_scores(rows, y, p) for m, p in predictions.items()}
    def mean_defined(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    means = {m: {key: mean_defined([r[key] for r in scores.values()])
                 for key in ("mse", "mae", "pair_concordance", "chosen_minus_native_dice",
                             "chosen_minus_uniform_arm_dice", "oracle_best_of_three_minus_native_dice")}
             for m, scores in per_sequence.items()}
    contrasts, pvalues = {}, {}
    for control in ("features", "geometry", "zero", "train_mean"):
        delta = {s: per_sequence[control][s]["mse"] - r["mse"] for s, r in per_sequence["regional"].items()}
        contrasts[control] = {"mean_mse_reduction": float(np.mean(list(delta.values()))),
            "sequence_bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=seed),
            "wins": sum(v > 0 for v in delta.values()), "sequence_deltas": delta}
        if control in ("features", "geometry"):
            pvalues[control] = paired_wilcoxon(
                {s:r["mse"] for s,r in per_sequence[control].items()},
                {s:r["mse"] for s,r in per_sequence["regional"].items()})["p"]
    gains = {s:r["chosen_minus_uniform_arm_dice"] for s,r in per_sequence["regional"].items()}
    gain_ci = sequence_bootstrap_interval(gains, seed=seed)
    checks = {
        "mse_beats_both_baselines": all(contrasts[c]["mean_mse_reduction"] > 0 for c in ("zero", "train_mean")),
        "regional_information_adds_value": all(contrasts[c]["sequence_bootstrap_95_ci"][0] > 0
            and contrasts[c]["wins"] >= 14 for c in ("features", "geometry")),
        "ranking_skill": means["regional"]["pair_concordance"] is not None
            and means["regional"]["pair_concordance"] > .5
            and all(means[c]["pair_concordance"] is not None and means["regional"]["pair_concordance"] > means[c]["pair_concordance"] for c in ("features", "geometry")),
        "offline_choice_signal": gain_ci[0] > 0 and means["regional"]["chosen_minus_native_dice"] > 0,
    }
    return {"scope": "Exploratory offline prediction of JOINT group utility; no deployment or propagation result",
        "sequences": len(per_sequence["regional"]), "rows": len(rows), "means": means,
        "per_sequence": per_sequence, "mse_contrasts": contrasts, "holm_pvalues": holm_adjust(pvalues),
        "regional_choice_vs_uniform_95_ci": gain_ci, "continuation_checks": checks,
        "continuation_passed": all(checks.values()),
        "uncertainty_limit": "Bootstrap describes sequence OOF errors without refitting nested CV; training folds overlap and this cohort was previously examined."}
