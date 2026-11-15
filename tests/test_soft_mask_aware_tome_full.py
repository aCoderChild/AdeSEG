import pytest

from scripts.run_soft_mask_aware_tome_full import budget_signature, compare_budget_signatures


def _rows(merge_count=8):
    return [
        {
            "frame": target,
            "attention_reads": [
                {
                    "layer": layer,
                    "frames": [{
                        "frame": target - 1,
                        "windows": [{
                            "window": window,
                            "original_spatial_keys": 16,
                            "merges": merge_count,
                            "retained_spatial_keys": 16 - merge_count,
                        } for window in range(2)],
                    }],
                }
                for layer in range(4)
            ],
        }
        for target in (2, 3)
    ]


def test_full_run_budget_signature_requires_every_window_at_frozen_budget():
    first = budget_signature(_rows())
    second = budget_signature(list(reversed(_rows())))
    assert first == second
    assert first["target_layer_window_records"] == 16
    assert first["all_windows_used_fixed_8_of_16_merges"]
    assert compare_budget_signatures(first, second) == 16


def test_full_run_rejects_a_budget_that_differs_from_protocol():
    with pytest.raises(RuntimeError, match="outside the frozen 8/16 budget"):
        budget_signature(_rows(merge_count=7))


def test_full_run_rejects_missing_attention_layer():
    rows = _rows()
    rows[0]["attention_reads"].pop()
    with pytest.raises(RuntimeError, match="Missing attention layer"):
        budget_signature(rows)
