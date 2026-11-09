from scripts.prepare_mask_memory_headroom import target_indices


def test_short_video_is_reported_without_unsupported_capacity_target():
    assert target_indices(8) == []


def test_duplicate_quantiles_do_not_repeat_cases():
    assert target_indices(9) == [8]
    assert target_indices(10) == [8, 9]


def test_three_fixed_timepoints_include_full_capacity_and_last_frame():
    assert target_indices(15) == [8, 11, 14]
    assert target_indices(60) == [8, 33, 59]
