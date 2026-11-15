import pytest
from evaluation.causal_memory_schedule import fixed_causal_schedule


def test_fixed_schedule_includes_empty_intervals_and_every_short_return_window():
    presence = [True, True, False, False, True, True, True, True, False, True]
    result = fixed_causal_schedule(presence, 0)
    selected = {row['frame']: row for row in result['targets']}
    assert selected[2]['present'] is False and 'gt_empty_midpoint' in selected[2]['strata']
    assert selected[8]['present'] is False
    assert result['return_events'][0]['evaluated_followup'] == [4, 5, 6]
    assert result['return_events'][1]['evaluated_followup'] == [9]
    assert len(selected) == len(result['targets'])
    assert all(0 < frame < len(presence) for frame in selected)


def test_short_video_and_present_only_video_have_valid_deduplicated_targets():
    result = fixed_causal_schedule([True, True], 0)
    assert [r['frame'] for r in result['targets']] == [1]
    assert result['return_events'] == [] and result['absent_runs'] == []
    assert fixed_causal_schedule([False, False, True, True], 2)['prompt'] == 2


def test_no_post_prompt_frame_and_non_boolean_labels_fail_before_inference():
    with pytest.raises(ValueError):
        fixed_causal_schedule([True], 0)
    with pytest.raises(ValueError):
        fixed_causal_schedule([True, 0, 1], 0)
