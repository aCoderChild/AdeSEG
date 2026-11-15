import numpy as np
import pytest
from evaluation.prompt_geometry_audit import evaluate_prompt_geometry


def test_prompt_geometry_pixel_center_and_empty_case():
    truth=np.zeros((4,5),bool);truth[1:3,1:3]=True;pred=truth.copy()
    r=evaluate_prompt_geometry([1,1,3,3],pred,truth)
    assert r['prompt_mask_dice']==1 and r['box_truth_coverage']==1 and r['box_precision_pixels']==1 and r['box_pixels']==4
    e=evaluate_prompt_geometry([1,1,3,3],pred,np.zeros_like(truth))
    assert e['prompt_mask_dice'] is None and e['empty_prompt_false_alarm'] is True
    with pytest.raises(ValueError):evaluate_prompt_geometry([-1,0,2,2],pred,truth)
