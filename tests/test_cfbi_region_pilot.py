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
