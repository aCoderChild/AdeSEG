import numpy as np
import pytest
from modeling.continuous_mask_prompt import mask_and_score_prompts

def test_prompt_is_continuous_deterministic_and_same_distribution_control():
 mask=np.zeros((256,256),bool);mask[50:180,30:200]=True
 score=np.arange(256*256,dtype=np.float32).reshape(256,256)
 a=mask_and_score_prompts(mask,score,12);b=mask_and_score_prompts(mask,score,12)
 assert all(np.array_equal(a[k],b[k]) and a[k].shape==(1,256,256) for k in a)
 assert set(np.unique(a['mask_only']))=={-4.,4.}
 delta=(a['score']-a['mask_only']).ravel();shuffled=(a['shuffled']-a['mask_only']).ravel()
 assert np.allclose(np.sort(delta),np.sort(shuffled),atol=1e-6)

def test_degenerate_maps_fail_closed():
 with pytest.raises(ValueError):mask_and_score_prompts(np.ones((4,4)),np.zeros((256,256)),0)
