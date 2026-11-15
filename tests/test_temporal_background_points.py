import numpy as np
from modeling.temporal_background_points import points_in_mask

def test_points_come_from_automatic_mask_and_random_is_matched():
    mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
    scores=np.arange(64).reshape(8,8)
    a=points_in_mask(scores,mask,7);b=points_in_mask(-scores,mask,7)
    assert a['feature_points']==[[5.5,5.5],[2.5,2.5]]
    assert a['random_points']==b['random_points']
    assert all(mask[int(y),int(x)] for x,y in a['feature_points']+a['random_points'])

def test_empty_or_single_cell_is_unavailable():
    assert points_in_mask(np.zeros((2,2)),np.zeros((8,8),bool),1) is None
