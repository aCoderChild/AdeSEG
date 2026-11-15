import pytest
from scripts.audit_unarchived_prompt_gate_v1 import verify_trace


def example():
    frames=[{'image_sha256':'a'},{'image_sha256':'b'},{'image_sha256':'c'}]
    attempts=[{'frame':0,'image_sha256':'a','box':None,'confidence':None},
              {'frame':1,'image_sha256':'b','box':[1,1,5,4],'confidence':.4}]
    prompt={'first_detection_index':1,'box':[1,1,5,4],'confidence':.4}
    return attempts,frames,prompt


def test_valid_reference_trace():
    attempts,frames,prompt=example()
    assert verify_trace(attempts,frames,prompt,(6,5))==1


def test_rejects_prior_detection_and_wrong_image():
    attempts,frames,prompt=example();attempts[0]['box']=[0,0,1,1];attempts[0]['confidence']=.3
    with pytest.raises(ValueError,match='earlier detection'):
        verify_trace(attempts,frames,prompt,(6,5))
    attempts,frames,prompt=example();attempts[0]['image_sha256']='x'
    with pytest.raises(ValueError,match='input/order'):
        verify_trace(attempts,frames,prompt,(6,5))


def test_rejects_terminal_mismatch_and_out_of_bounds():
    attempts,frames,prompt=example();prompt['box']=[1,1,5,3]
    with pytest.raises(ValueError,match='terminal prompt'):
        verify_trace(attempts,frames,prompt,(6,5))
    attempts,frames,prompt=example();attempts[-1]['box']=prompt['box']=[1,1,9,4]
    with pytest.raises(ValueError,match='terminal prompt'):
        verify_trace(attempts,frames,prompt,(6,5))
