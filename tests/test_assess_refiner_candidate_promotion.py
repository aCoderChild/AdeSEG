import pytest

from scripts.assess_refiner_candidate_promotion import screen


def row(seq, present, base, cand, ctrl, bp=0, cp=0, op=0):
    return {'sequence': seq, 'labeled_present': present,
            'baseline': {'dice': base, 'predicted_pixels': bp},
            'candidate': {'dice': cand, 'predicted_pixels': cp},
            'control': {'dice': ctrl, 'predicted_pixels': op}}


def test_requires_both_classes_and_improvement_without_empty_degradation():
    positive = row('v1', True, .5, .8, .7)
    negative = row('v1', False, 0, 0, 0, bp=10, cp=9, op=8)
    assert not screen([positive])['eligible']
    assert screen([positive, negative])['eligible']
    assert not screen([positive, row('v1', False, 0, 0, 0, bp=10, cp=11)])['eligible']
    assert not screen([row('v1', True, .5, .6, .7), negative])['eligible']


def test_present_dice_is_averaged_equally_by_video():
    metrics = [row('v1', True, .9, .8, .7), row('v1', True, .9, .8, .7),
               row('v2', True, .1, .4, .2), row('v2', False, 0, 0, 0)]
    result = screen(metrics)
    assert result['mean_per_video_present_dice']['baseline'] == .5
    assert result['mean_per_video_present_dice']['candidate'] == pytest.approx(.6)
    assert result['eligible']


def test_empty_video_degradation_cannot_be_hidden_by_another_video():
    metrics = [row('v1', True, .5, .8, .7),
               row('v1', False, 0, 0, 0, bp=0, cp=1),
               row('v2', False, 0, 0, 0, bp=100, cp=0)]
    result = screen(metrics)
    assert result['empty_predicted_pixels']['candidate'] < result['empty_predicted_pixels']['baseline']
    assert not result['checks']['no_worse_empty_false_alarm_rate_per_video']
    assert not result['eligible']
