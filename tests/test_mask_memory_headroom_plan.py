from scripts.prepare_mask_memory_headroom import target_indices


def test_short_video_is_reported_without_unsupported_capacity_target():
    assert target_indices(7) == []


def test_duplicate_quantiles_do_not_repeat_cases():
    assert target_indices(8) == [7]
    assert target_indices(9) == [7, 8]


def test_three_fixed_timepoints_include_full_capacity_and_last_frame():
    assert target_indices(15) == [7, 10, 14]
    assert target_indices(60) == [7, 33, 59]
