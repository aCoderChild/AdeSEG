import pytest
from scripts.report_author_matching_strata import describe


def test_stratum_report_uses_sequence_weighting_unique_frames_and_empty_denominators():
    methods = ['native', 'candidate']
    targets = {'a': [
        {'frame': 1, 'present': True, 'strata': ['uniform_1/10', 'uniform_3/10', 'reappearance_t0']},
        {'frame': 2, 'present': True, 'strata': ['gt_present_midpoint']},
        {'frame': 3, 'present': False, 'strata': ['gt_empty_midpoint']}],
        'b': [{'frame': 1, 'present': True, 'strata': ['uniform_1/10']},
              {'frame': 2, 'present': False, 'strata': ['gt_empty_midpoint']}]}
    native = {'a': [.2, .4, 1.], 'b': [.9, 0.]}
    candidate = {'a': [.3, .5, 0.], 'b': [.5, 1.]}
    records = {}
    for sequence, entries in targets.items():
        records[sequence] = []
        for i, target in enumerate(entries):
            rows = {}
            for method in methods:
                dice = (native if method == 'native' else candidate)[sequence][i]
                rows[method] = {'present': target['present'], 'scores': {'dice': dice, 'boundary_f': dice},
                                'absent_fp': dice == 0 if not target['present'] else None}
                if method != 'native':
                    rows[method]['seconds'] = 2.0
            records[sequence].append({'frame': target['frame'], 'rows': rows})
    plan = {'protocol': {'conditions': ['candidate'], 'sequences': ['a', 'b']},
            'sequences': {key: {'schedule': {'targets': values}} for key, values in targets.items()}}
    result = describe(records, plan)
    assert result['all_present']['native']['equal_sequence_present_dice'] == pytest.approx(.6)
    assert result['all_present']['candidate']['equal_sequence_present_dice'] == pytest.approx(.45)
    assert result['all_present']['candidate']['equal_sequence_delta_native'] == pytest.approx(-.15)
    assert result['all_present']['candidate']['present_frame_wins_ties_losses'] == [2, 0, 1]
    assert result['uniform']['candidate']['targets'] == 2  # two uniform tags do not duplicate a frame
    empty = result['all_empty']['candidate']
    assert empty['equal_sequence_present_dice'] is None
    assert empty['false_alarms'] == 1 and empty['gt_empty_targets'] == 2
    assert empty['pooled_false_alarm_rate'] == .5
    assert result['all_present']['native']['median_replay_check_scoring_seconds'] is None
