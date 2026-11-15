import numpy as np
import pytest
from modeling.samurai_auto_prompt import prepare_prompt


def test_seq23_prompt_crops_without_losing_foreground():
    mask=np.zeros((720,1280),dtype=np.uint8);mask[5,500]=255
    result,viewport=prepare_prompt(mask,'seq23',mask.shape)
    assert result.shape==(720,834) and result[5,54] and result.sum()==1
    assert viewport==(446,0,1280,720)
    mask[5,10]=255
    with pytest.raises(ValueError):prepare_prompt(mask,'seq23',mask.shape)


def test_prompt_rejects_mismatched_geometry_and_nonbinary_values():
    with pytest.raises(ValueError):prepare_prompt(np.zeros((3,4)),'seq11',(4,3))
    with pytest.raises(ValueError):prepare_prompt(np.full((3,4),42),'seq11',(3,4))
    result,viewport=prepare_prompt(np.ones((3,4),dtype=np.uint8),'seq11',(3,4))
    assert result.dtype==bool and result.all() and viewport==(0,0,4,3)
