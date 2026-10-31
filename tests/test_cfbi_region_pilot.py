import pytest
from scripts.run_cfbi_region_pilot import training_targets


def test_training_targets_use_only_fixed_sequence_positions():
    assert training_targets(3, 64) == [4, 23, 43, 63]
    assert training_targets(0, 5) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        training_targets(0, 4)


def test_report_lists_each_condition_and_sequence():
    from scripts.run_cfbi_region_pilot import report
    value={'present_dice':.7,'iou':.6,'boundary_f':.5,'gt_empty_false_alarm_count':2}
    sequences={'seq2':value,'seq3':value}
    text=report({'native':sequences,'conditions':{m:sequences for m in ('regional','shuffled','generic')}},['seq2','seq3'])
    assert '| regional | 0.700000 | 0.600000 | 0.500000 | 4 |' in text
    assert '| seq3 | 0.700000 | 0.700000 | 0.700000 | 0.700000 |' in text



def test_baseline_coverage_rejects_missing_or_invalid_frames():
    from scripts.run_cfbi_region_pilot import verify_baseline_coverage
    rows=[{'frame':1,'present':True},{'frame':2,'present':False}]
    verify_baseline_coverage(rows,0,3)
    with pytest.raises(RuntimeError,match='cover'):
        verify_baseline_coverage(rows[:1],0,3)
    with pytest.raises(RuntimeError,match='presence'):
        verify_baseline_coverage([{'frame':1,'present':1}],0,2)


def test_all_nonzero_smoke_sources_are_frozen():
    from scripts.run_cfbi_region_pilot import SOURCES
    assert {'modeling/cfbi_region_gate.py', 'modeling/cfbi_training.py',
            'protocols/cfbi_region_gate_development_v1.json'} <= set(SOURCES)
