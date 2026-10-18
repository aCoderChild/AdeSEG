"""Regression tests for prospective region-study review corrections."""

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import distance_transform_edt

from modeling.region_tokens import token_categories, spatially_matched_indices
from scripts.analyze_region_experiments import (
    analysis_variant, attention_agreement, clustered_interaction,
    category_weights, feasibility_summary,
)


def test_contour_sensitivity_finds_grid_aligned_boundary_without_changing_default():
    mask = np.zeros((256, 256), np.uint8)
    mask[64:192, 64:192] = 1
    old = token_categories(mask)["categories"]
    new = token_categories(mask, boundary_definition="contour_union")["categories"]
    assert not (old == 0).any()
    assert (new == 0).any() and (new == 2).any()
    assert np.array_equal(old, token_categories(mask, boundary_definition="mixed_coverage")["categories"])
    assert not (token_categories(np.zeros_like(mask), boundary_definition="contour_union")["categories"] == 0).any()


def test_frame_distance_control_preserves_each_frames_distance_histogram():
    labels = np.full((3, 8, 8), 3)
    labels[:, 2:6, 2:6] = 1
    labels[:, 2, 2:6] = 0
    reference = np.array([18, 19, 20, 64 + 18, 128 + 18, 128 + 19])
    selected = spatially_matched_indices(labels, reference, 42, frame_stratified=True)
    assert len(np.unique(selected)) == len(reference)
    for frame in range(3):
        distance = np.floor(distance_transform_edt(~np.isin(labels[frame], [0, 1]))).astype(int).ravel()
        before = reference[reference // 64 == frame] % 64
        after = selected[selected // 64 == frame] % 64
        assert sorted(distance[before]) == sorted(distance[after])
    assert np.array_equal(selected, spatially_matched_indices(labels, reference, 42, frame_stratified=True))


def test_analysis_rejects_silent_sensitivity_pooling():
    with pytest.raises(ValueError, match="separately"):
        analysis_variant([{}, {"spatial_match": "frame_distance"}])
    with pytest.raises(ValueError, match="registered E6"):
        category_weights([{"spatial_match": "frame_distance"}])


def test_feasibility_counts_keep_skips_and_missing_size_explicit():
    row = dict(video="seq2", frame=7, bank="recent", category="foreground",
               category_counts={"boundary": 20}, state="skipped",
               reason="empty_boundary_or_insufficient_category_tokens")
    result = feasibility_summary([row], {})
    assert "unmeasured" in next(iter(result))
    assert next(iter(result.values()))["skipped"] == 1
    measured = feasibility_summary([row], {("polypgen", "seq2", 1, 7): {"coverage_tokens": 32}})
    assert '"small"' in next(iter(measured))


def test_e5_separates_local_query_effect_from_global_effect():
    names = ["boundary", "foreground", "near_background", "far_background"]
    attention = {q: dict(zip(names, [1., 2., 3., 4.])) for q in names[:1] + ["foreground", "background"]}
    rows = [dict(
        video="seq2", frame=7, state="complete", category=name, bank="recent",
        harm_per_token={"boundary_f": float(i)}, attention=[attention],
        category_counts=dict.fromkeys(names, 10),
        query_dice_harm=dict.fromkeys(["foreground", "boundary", "background"], float(-i)),
    ) for i, name in enumerate(names)]
    result = attention_agreement(rows)
    assert result["agreement"] and result["query_specific_agreement"]
    assert all(v["rho"] == pytest.approx(1) for k, v in result["agreement"].items() if "anchor_similarity" not in k)
    assert all(v["rho"] == pytest.approx(-1) for v in result["query_specific_agreement"].values())
    for row in rows:
        row.pop("query_dice_harm")
    assert not attention_agreement(rows)["query_specific_agreement"]


def test_clustered_interaction_matches_statsmodels_and_reports_cluster_count():
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(7)
    rows = []
    for seq in range(16):
        offset = rng.normal()
        for frame in range(8):
            x = rng.normal()
            shared = rng.normal()
            for category in [0, 1]:
                rows.append(dict(sequence=str(seq), category=category, x=x,
                                 harm=offset + shared + category * x * .3 + rng.normal()))
    data = pd.DataFrame(rows)
    formula = "harm ~ C(category) * x"
    result = clustered_interaction(data, formula)
    fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={
        "groups": data.sequence, "use_correction": True, "df_correction": True,
    }, use_t=True)
    contrast = np.eye(len(fit.params))[[i for i, name in enumerate(fit.params.index) if ":" in name]]
    assert result["state"] == "complete"
    assert result["n_clusters"] == 16
    assert result["p_interaction"] == pytest.approx(float(fit.wald_test(contrast, use_f=True, scalar=True).pvalue))
    tiny = data[data.sequence.isin(["0", "1"])]
    assert clustered_interaction(tiny, formula)["state"] == "unavailable"


@pytest.mark.parametrize("phase,stage,spatial", [
    ("frame_matched", "E2", "frame_distance"),
    ("corruption", "E3", "pooled_distance"),
])
def test_diagnostic_launcher_reuses_runner_and_covers_promptable_sequences(monkeypatch, phase, stage, spatial):
    import builtins
    import io
    from pathlib import Path
    import runpy
    import urllib.request

    jobs = []
    monkeypatch.setattr(builtins, "region_test_jobs", jobs, raising=False)
    monkeypatch.setenv("REGION_REVIEW_PHASE", phase)
    monkeypatch.setenv("REGION_CORRUPTION", "c2")
    source = b"import builtins, json, os\nbuiltins.region_test_jobs.append(json.loads(os.environ['REGION_JOB']))\n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(source))
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: '{"state":"complete"}')
    script = Path(__file__).resolve().parents[1] / "kaggle/region-tokens/review_validation.py"
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script))
    assert stopped.value.code == 0
    assert len(jobs) == 2
    sequences = [s for job in jobs for s in job["sequences"]]
    assert len(sequences) == len(set(sequences)) == 21
    assert set(sequences) == {f"seq{i}" for i in range(2, 24) if i != 7}
    assert all(j["stage"] == stage and j["spatial_match"] == spatial for j in jobs)
    assert all(j["memory_condition"] == ("c0" if phase == "frame_matched" else "c2") for j in jobs)
