from copy import deepcopy
import pytest
from scripts.run_native_value_consolidation_full import compare,SEQUENCES


def summaries():
    return {a:{s:{'present_dice':v,'iou':v,'boundary_f':v,
                  'gt_empty_false_alarm_count':1,'gt_empty_targets':2,
                  'return_count':2,'undetected_returns':1} for s in SEQUENCES}
            for a,v in [('native',0.7),('joint',0.71),('value',0.72)]}


def test_full_screen_requires_all_sequences_and_robust_control_gain():
    data=summaries();result=compare(data)
    assert result['screen']['passed']
    assert result['contrasts']['value_minus_native']['mean_delta']==pytest.approx(0.02)
    missing=deepcopy(data);missing['value'].pop('seq23')
    with pytest.raises(ValueError,match='21 eligible'):compare(missing)
    concentrated=deepcopy(data)
    for s in SEQUENCES:concentrated['value'][s]['present_dice']=0.7
    concentrated['value']['seq8']['present_dice']=1.0
    assert not compare(concentrated)['screen']['passed']
    assert not compare(concentrated)['screen']['gain_without_seq8_positive']
