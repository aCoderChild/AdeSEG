import pytest
import torch

from scripts.capture_raw_multimask_mixed_v1 import move


def test_move_preserves_nested_decoder_input_structure():
    x={'backbone_features':torch.randn(1,2,3,3),'point_inputs':None,
       'high_res_features':[torch.ones(1,1,2,2)],'multimask_output':True,
       'nested':(1,'a')}
    y=move(x,'cpu')
    assert torch.equal(x['backbone_features'],y['backbone_features'])
    assert torch.equal(x['high_res_features'][0],y['high_res_features'][0])
    assert y['point_inputs'] is None and y['multimask_output'] is True and y['nested']==(1,'a')


def test_move_rejects_unexpected_decoder_value():
    with pytest.raises(ValueError,match='Unexpected'):
        move({'unsupported':object()},'cpu')
