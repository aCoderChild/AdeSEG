import numpy as np
import pytest

from modeling.region_tokens import disjoint_distance_match, region_geometry


def test_subsampled_controls_are_disjoint_and_exactly_frame_stratified():
    from modeling.region_tokens import subsampled_disjoint_match

    target = np.tile(np.array([[[1, 1, 1, 0, 0, 0]]], bool), (2, 1, 1))
    strata = np.tile(np.array([[[0, 1, 1, 0, 1, 2]]]), (2, 1, 1))
    left, right, info = subsampled_disjoint_match(
        target, ~target, strata, cap=3, seed=5
    )
    assert len(left) == len(right) == info["selected"] == 3
    assert info["common_capacity"] == 4
    assert not set(left) & set(right)
    assert target.ravel()[left].all() and (~target.ravel()[right]).all()
    for frame in range(2):
        a = strata.ravel()[left[left // 6 == frame]]
        b = strata.ravel()[right[right // 6 == frame]]
        assert sorted(a) == sorted(b)
    repeated = subsampled_disjoint_match(target, ~target, strata, cap=3, seed=5)
    assert np.array_equal(left, repeated[0])


def test_subsampled_controls_report_structural_absence():
    from modeling.region_tokens import subsampled_disjoint_match

    support = np.array([[[True, False]]])
    left, right, info = subsampled_disjoint_match(
        support, ~support, np.array([[[0, 1]]])
    )
    assert left is right is None
    assert info["selected"] == 0


def test_regional_and_null_bias_search_preserve_per_frame_norms_and_histograms():
    from modeling.region_tokens import regional_bias_candidates

    labels = {1: np.array([0, 1, 1, 2, 3, 3]), 4: np.array([0, 0, 1, 2, 2, 3])}
    real = regional_bias_candidates(labels)
    null = regional_bias_candidates(labels, seed=7)
    assert len(real) == len(null) == 8
    for (name, a), (random_name, b) in zip(real, null):
        assert name == random_name
        for frame in labels:
            assert np.allclose(a[frame].mean(), 0, atol=1e-7)
            assert np.array_equal(np.sort(a[frame]), np.sort(b[frame]))
            assert np.linalg.norm(a[frame]) == pytest.approx(np.linalg.norm(b[frame]))


def test_followup_uses_equal_search_and_selects_boundary_f_by_dice_not_by_f():
    from scripts.summarize_region_v2_local import followup_statistics

    frames, searches = [], []
    for video in ("a", "b"):
        for seed in range(3):
            for role, harm in (("target", 0.2), ("control", 0.05)):
                frames.append(
                    dict(
                        video=video,
                        frame=7,
                        condition="natural",
                        state="complete",
                        seed=seed,
                        k=64,
                        matching={"region": "boundary", "role": role},
                        harm={"dice": harm, "boundary_f": harm},
                    )
                )
        for search, seed in [("regional", None), *[("random", s) for s in range(3)]]:
            candidates = [
                {
                    "candidate": str(i),
                    "metrics": {
                        "dice": 0.6 if search == "regional" else 0.55,
                        "boundary_f": 0.3,
                    },
                    "token_l2": 1.0,
                }
                for i in range(8)
            ]
            candidates[-1]["metrics"] = {"dice": 0.4, "boundary_f": 0.99}
            searches.append(
                dict(
                    video=video,
                    frame=7,
                    condition="natural",
                    search=search,
                    seed=seed,
                    baseline={"dice": 0.5, "boundary_f": 0.5},
                    candidates=candidates,
                )
            )
    result = followup_statistics(frames, searches)
    geometry = next(
        r
        for r in result["geometry"]
        if r["metric"] == "dice"
        and r["region"] == "boundary"
        and r["condition"] == "natural"
    )
    headroom = next(
        r
        for r in result["headroom"]
        if r["metric"] == "dice" and r["condition"] == "natural"
    )
    boundary = next(
        r
        for r in result["headroom"]
        if r["metric"] == "boundary_f" and r["condition"] == "natural"
    )
    assert geometry["n"] == 2 and geometry["mean"] == pytest.approx(0.15)
    assert headroom["oracle_gain"] == pytest.approx(0.1)
    assert headroom["excess"] == pytest.approx(0.05)
    assert boundary["oracle_gain"] == pytest.approx(-0.2)
    searches[1]["candidates"][0]["token_l2"] = 2
    with pytest.raises(ValueError, match="norms"):
        followup_statistics(frames, searches)


@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_chunked_attention_observer_matches_native_and_dense(chunk):
    import torch
    from types import SimpleNamespace
    from modeling.region_tokens import capture_chunked_region_attention

    class Native(torch.nn.Module):
        def _attention(self, q, k, v, attn_mask=None):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )

    module = Native().eval()
    memory = SimpleNamespace(layers=[SimpleNamespace(cross_attn_image=module)])
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    units = [{"start": 0, "end": 4, "frame_idx": 3, "conditioning": True}]
    maps = {3: np.arange(4).reshape(2, 2)}
    baseline = module._attention(q, k, v)
    with capture_chunked_region_attention(memory, units, maps, chunk) as records:
        result = module._attention(q, k, v)
    assert torch.equal(baseline, result)
    assert "_attention" not in module.__dict__
    dense = (q @ k.transpose(-2, -1) / 8**0.5).softmax(-1).mean((0, 1, 2))
    row = records[0]
    assert row["total_mass"] == pytest.approx(1)
    assert row["reconstruction_max_abs"] < 1e-6
    assert row["pointer_mass"] == pytest.approx(float(dense[4:].sum()))
    for i, name in enumerate(
        ["boundary", "foreground", "near_background", "far_background"]
    ):
        assert row["units"][0]["regions"][name]["mass"] == pytest.approx(
            float(dense[i])
        )


def test_attention_observer_restores_after_error():
    import torch
    from types import SimpleNamespace
    from modeling.region_tokens import capture_chunked_region_attention

    module = torch.nn.Module()

    def original(*a, **k):
        return None

    module._attention = original
    memory = SimpleNamespace(layers=[SimpleNamespace(cross_attn_image=module)])
    with pytest.raises(RuntimeError):
        with capture_chunked_region_attention(memory, [], {}):
            raise RuntimeError("test")
    assert module._attention is original


def test_attention_failure_keeps_native_output_and_saves_reference_inputs(tmp_path):
    import torch
    from types import SimpleNamespace
    from modeling.region_tokens import capture_chunked_region_attention

    class Native(torch.nn.Module):
        def _attention(self, q, k, v, attn_mask=None):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v) + 0.001

    module = Native().eval()
    memory = SimpleNamespace(layers=[SimpleNamespace(cross_attn_image=module)])
    q, k, v = torch.randn(1, 1, 5, 8), torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)
    baseline = module._attention(q, k, v)
    with capture_chunked_region_attention(
        memory,
        [{"start": 0, "end": 4, "frame_idx": 3, "conditioning": False}],
        {3: np.arange(4)},
        failure_dir=tmp_path,
    ) as records:
        observed = module._attention(q, k, v)
    assert torch.equal(baseline, observed)
    assert records[0]["reconstruction_max_abs"] > 1e-4
    saved = torch.load(tmp_path / "layer0.pt", weights_only=True)
    assert torch.equal(saved["k"], k)
    assert saved["mps_max_abs"] > 1e-4


def test_attention_aggregation_is_sequence_weighted():
    from scripts.summarize_region_v2_local import regional_attention_summary

    records = []
    for video, mass in [("a", 0.1), ("a", 0.3), ("b", 0.8)]:
        records.append(
            {
                "video": video,
                "condition": "natural",
                "layers": [
                    {
                        "layer": 0,
                        "pointer_tokens": 2,
                        "units": [
                            {
                                "conditioning": False,
                                "regions": {
                                    "boundary": {"tokens": 2, "mass": mass},
                                    **{
                                        name: {"tokens": 0, "mass": 0.0}
                                        for name in (
                                            "foreground",
                                            "near_background",
                                            "far_background",
                                        )
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )
    row = regional_attention_summary(records)[0]
    assert row["n"] == 2
    assert row["mean_attention_mass"] == pytest.approx(0.5)
    assert row["mean_uniform_enrichment"] == pytest.approx(1.0)


def test_region_excess_pairs_random_within_target_before_sequence_average():
    from scripts.summarize_region_v2_local import paired_regional_removal_summary

    rows = []
    for video, frame, regional, random in [
        ("a", 1, 0.3, 0.1),
        ("a", 2, 0.5, 0.3),
        ("b", 1, 0.6, 0.1),
    ]:
        for category, value in [("boundary", regional), ("spatial_random", random)]:
            rows.append(
                dict(
                    video=video,
                    frame=frame,
                    category=category,
                    bank="recent",
                    condition="natural",
                    state="complete",
                    k=5,
                    harm={"dice": value, "boundary_f": value},
                )
            )
    result = next(
        r
        for r in paired_regional_removal_summary(rows)
        if r["metric"] == "dice"
        and r["condition"] == "natural"
        and r["category"] == "boundary"
    )
    assert result["n"] == 2 and result["targets"] == 3
    assert result["excess_harm"] == pytest.approx(0.35)
    assert 0 <= result["holm_p"] <= 1
    rows[-1]["k"] = 6
    with pytest.raises(ValueError, match="budgets differ"):
        paired_regional_removal_summary(rows)


def test_geometry_empty_and_aligned_mask():
    empty = region_geometry(np.zeros((64, 64)))
    assert (empty["distance_bin"] == -1).all()
    mask = np.zeros((64, 64))
    mask[16:48, 16:48] = 1
    geometry = region_geometry(mask)
    assert not geometry["mixed"].any()
    assert geometry["absolute_distance"][16, 20] == 0
    assert geometry["signed_distance"][32, 32] < 0
    assert geometry["signed_distance"][0, 0] > 0


def test_disjoint_match_preserves_frame_histograms():
    bins = np.tile(np.array([[0, 0, 1, 1]]), (2, 1, 1))
    target = np.zeros_like(bins, dtype=bool)
    target[:, :, [0, 2]] = True
    indices = disjoint_distance_match(target, np.ones_like(target), bins)
    assert indices is not None
    chosen = np.zeros(target.size, dtype=bool)
    chosen[indices] = True
    chosen = chosen.reshape(target.shape)
    assert not (chosen & target).any()
    assert len(np.unique(indices)) == target.sum()
    for frame in range(2):
        assert np.array_equal(
            np.sort(bins[frame][target[frame]]), np.sort(bins[frame][chosen[frame]])
        )


def test_infeasible_not_silently_relaxed():
    target = np.array([[True, True, False]])
    assert disjoint_distance_match(target, ~target, np.zeros_like(target)) is None
    with pytest.raises(ValueError):
        disjoint_distance_match(target, ~target, np.zeros((2, 2)))


def test_extra_side_stratum_prevents_cross_side_substitution():
    # Distance-only control exists, but only across the contour's side.
    target = np.array([[True, False]])
    assert disjoint_distance_match(target, ~target, np.array([[1, 1]])) is not None
    assert disjoint_distance_match(target, ~target, np.array([[3, 2]])) is None


def test_summary_pairs_targets_before_sequence_averaging():
    from scripts.summarize_region_v2_local import condition_effects

    rows = []
    for frame, boundary, controls in [(1, 0.2, [0.0, 0.2]), (2, 0.4, [0.2, 0.2])]:
        for category, values in [
            ("boundary", [boundary]),
            ("spatial_random", controls),
        ]:
            rows.extend(
                {
                    "video": "seq2",
                    "frame": frame,
                    "condition": "c0",
                    "bank": "recent",
                    "state": "complete",
                    "category": category,
                    "harm": {"dice": value},
                }
                for value in values
            )
    # (0.2 - 0.1 + 0.4 - 0.2)/2, not an average across rows.
    assert condition_effects(rows)["c0"]["seq2"] == pytest.approx(0.15)


def test_accelerator_offload_fence_before_host_snapshot(monkeypatch):
    import torch
    from scripts.run_region_v2_local import synchronize_storage

    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    synchronize_storage("cpu")
    assert calls == []
    synchronize_storage("mps")
    synchronize_storage("cuda:0")
    assert calls == ["mps", "cuda:0"]


def test_local_query_matrix_uses_sequence_not_frame_weights():
    from scripts.summarize_region_v2_local import query_region_effects

    rows = [
        {
            "state": "complete",
            "category": "boundary",
            "condition": "natural",
            "bank": "recent",
            "video": video,
            "query_dice_harm": {"boundary": value},
        }
        for video, value in [("a", 0.0), ("a", 0.2), ("b", 0.5)]
    ]
    result = query_region_effects(rows)
    assert result[0]["n"] == 2
    assert result[0]["mean_dice_harm"] == pytest.approx(0.3)
