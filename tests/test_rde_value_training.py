import torch
import pytest
from scripts.rde_value_training import aligned_teacher
from scripts.run_token_conditioned_key_adapter import _tree_digest


def fixture():
    a=torch.randn(1,64,2,2).bfloat16();b=torch.randn_like(a)
    score=torch.tensor([[1.]])
    bank={'cond_frame_outputs':{0:{'maskmem_features':a}},'non_cond_frame_outputs':{1:{'maskmem_features':b,'object_score_logits':score}}}
    spans=[{'frame':0,'start':0,'stop':4},{'frame':1,'start':4,'stop':8}]
    entry={'frame':1,'object_id':1,'native_memory_sha256':_tree_digest(b),'native_score_sha256':_tree_digest(score),'value':b.clone()+1}
    return bank,spans,{1:entry}


def test_teacher_alignment_preserves_prompt_and_source_order():
    bank,spans,cache=fixture();before=_tree_digest(bank)
    t=aligned_teacher(bank,spans,cache,2,1)
    assert torch.equal(t[:4],bank['cond_frame_outputs'][0]['maskmem_features'].flatten(2).permute(2,0,1))
    assert torch.equal(t[4:],cache[1]['value'].flatten(2).permute(2,0,1))
    assert _tree_digest(bank)==before


@pytest.mark.parametrize('fault',['future','missing','owner','score','frame','gradient','object_id'])
def test_wrong_teacher_provenance_fails(fault):
    bank,spans,cache=fixture();target=2
    if fault=='future':target=1
    elif fault=='missing':cache={}
    elif fault=='owner':bank['non_cond_frame_outputs'][1]['maskmem_features']=torch.zeros_like(cache[1]['value'])
    elif fault=='score':bank['non_cond_frame_outputs'][1]['object_score_logits']=torch.tensor([[-1.]])
    elif fault=='frame':cache[1]['frame']=7
    elif fault=='gradient':cache[1]['value']=cache[1]['value'].float().requires_grad_()
    elif fault=='object_id':cache[1]['object_id']=2
    with pytest.raises(RuntimeError):aligned_teacher(bank,spans,cache,target,1)
