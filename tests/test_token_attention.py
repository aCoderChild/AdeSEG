import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from modeling.token_attention import TokenAttentionObserver
from scripts.region_definition_statistics import contour_coverage
from scripts.token_attention_statistics import (
    footprint_precision,
    log_summary,
    pooled_enrichment,
    sequence_summary,
)


def test_per_token_group_mass_matches_dense_author_weights_without_changing_output():
    captures = []
    observer = TokenAttentionObserver(
        SimpleNamespace(training=False),
        lambda *data: captures.append(data),
        chunk_size=2,
    )
    observer.labels = torch.tensor([0, 1, 2, -1])
    observer.units = []
    observer.query_labels = torch.tensor([0, 0, 1, 2, 2])
    torch.manual_seed(7)
    q, k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 3)
    result = observer._observe(F.scaled_dot_product_attention, 0, q, k, v)
    assert torch.equal(result, F.scaled_dot_product_attention(q, k, v))
    weights = (q @ k.transpose(-1, -2) / np.sqrt(8)).softmax(-1)
    _, joint, counts, _ = captures[0]
    for group in range(3):
        expected = weights[:, :, observer.query_labels == group].sum((0, 1, 2)) / 10
        assert np.allclose(joint[group], expected.numpy(), atol=1e-7)
    assert np.allclose(joint.sum(1), counts / counts.sum())


def test_empty_query_groups_have_zero_mass_and_wrong_layout_fails():
    saved = []
    observer = TokenAttentionObserver(
        SimpleNamespace(training=False), lambda *data: saved.append(data)
    )
    observer.labels, observer.units = torch.tensor([0, 1, -1]), []
    observer.query_labels = torch.zeros(2, dtype=torch.long)
    q = torch.ones(1, 1, 2, 4)
    k = v = torch.ones(1, 1, 3, 4)
    observer._observe(F.scaled_dot_product_attention, 0, q, k, v)
    assert (saved[0][1][1:] == 0).all()
    observer.query_labels = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="Query labels"):
        observer._observe(F.scaled_dot_product_attention, 0, q, k, v)


def test_pooled_ratio_and_log_medians_are_not_readwise_means():
    assert pooled_enrichment(0.1 + 0.9, 1 + 1, 0.01 + 0.99, 1 + 1) == pytest.approx(1)
    assert pooled_enrichment(0, 0, 0, 0) is None
    result = log_summary([1, 4, None])
    assert result["n"] == 2
    assert result["median_log"] == pytest.approx(np.log(2))
    assert result["geometric_median"] == pytest.approx(2)
    assert result == log_summary([1, 4, None])
    assert log_summary([0])["median_log"] == "-infinity"
    assert log_summary([None])["n"] == 0


def test_footprint_precision_penalizes_thick_support_and_keeps_undefined():
    mask = np.zeros((256, 256), np.uint8)
    mask[65:191, 65:191] = 1
    broad = footprint_precision(mask, np.full((64, 64), 2))
    assert 0 < broad["including_border"]["precision"] < 0.02
    assert broad["including_border"]["support_pixels"] == 65536
    assert (
        footprint_precision(mask, np.zeros((64, 64), int))["including_border"][
            "precision"
        ]
        is None
    )


def test_saved_sequence_pooling_and_pointer_separation(tmp_path):
    labels = np.arange(4096).reshape(64, 64) % 3
    precision = footprint_precision(np.zeros((256, 256), np.uint8), labels)
    record = {
        "frame": 1,
        "conditioning": False,
        "size_bin": "Q1",
        "outcome": "overlap_present",
        "schemes": {
            "mixed_cells": {
                "self_precision": precision,
                "gt_precision": precision,
                "self_contour": contour_coverage(
                    np.zeros((256, 256), np.uint8), labels
                ),
                "gt_contour": contour_coverage(np.zeros((256, 256), np.uint8), labels),
            }
        },
    }
    (tmp_path / "assignment_frames.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "frames.jsonl").write_text(
        json.dumps({"frame": 1, "scores": {"dice": 0.75}}) + "\n"
    )
    torch.save(
        {"schemes": {"mixed_cells": {"labels": labels}}},
        tmp_path / "assignments_000000.pt",
    )
    mass = np.stack([np.full(4097, q / 4097) for q in (0.5, 0.25, 0.25)]).astype(
        np.float32
    )
    np.savez_compressed(
        tmp_path / "token_attention_000001_0.npz",
        joint_mass=mass,
        query_counts=[2, 1, 1],
        memory_frames=[0],
        pointer_count=1,
        target=1,
        layer=0,
    )
    result = sequence_summary(tmp_path)
    assert all(
        row["enrichment"] == pytest.approx(1, abs=1e-6) for row in result["rows"]
    )
    assert result["dice_strata"]["all"]["frames"] == 1
    assert result["rows"][0]["pointer_joint_mass_sum"] > 0
