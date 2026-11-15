import pytest
from modeling.token_memory_window import SpatialMemory, select_token_window

def test_budget_tracks_token_counts_not_number_of_frames():
    entries = [SpatialMemory(0, 256), SpatialMemory(1, 256), SpatialMemory(2, 256), SpatialMemory(3, 256)]
    selected, used = select_token_window(4096, entries, 5120)
    assert selected == entries and used == 5120
    larger = [SpatialMemory(0, 1024), SpatialMemory(1, 1024)]
    selected, used = select_token_window(4096, larger, 5120)
    assert selected == [larger[-1]] and used == 5120

def test_never_skips_over_a_nonfitting_entry():
    entries = [SpatialMemory(1, 100), SpatialMemory(2, 300), SpatialMemory(3, 200)]
    assert select_token_window(1000, entries, 1400) == ([entries[-1]], 1200)

def test_impossible_budgets_and_invalid_entries_fail():
    with pytest.raises(ValueError): select_token_window(4096, [], 1024)
    with pytest.raises(ValueError): select_token_window(4096, [SpatialMemory(1, 4096)], 5000)
    with pytest.raises(ValueError): select_token_window(1, [SpatialMemory(2, 1), SpatialMemory(2, 1)], 4)
    with pytest.raises(ValueError): select_token_window(1, [SpatialMemory(1, 0)], 4)

def test_protected_insurance_is_included_in_same_budget():
    entries=[SpatialMemory(1,1024),SpatialMemory(2,1024),SpatialMemory(3,4096)]
    selected,used=select_token_window(4096,entries,12288,reserved_tokens=3072)
    assert selected==entries[-2:] and used==12288
    with pytest.raises(ValueError):select_token_window(4096,entries,10000,reserved_tokens=3072)
