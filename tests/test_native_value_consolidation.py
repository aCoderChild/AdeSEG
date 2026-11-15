import torch
from modeling.native_value_consolidation import consolidate_values


def test_value_consolidation_keeps_keys_conditioning_pointers_and_inputs():
    torch.manual_seed(8)
    k=torch.randn(1,2,34,8);v=torch.randn_like(k)
    spans=[{'frame':0,'t_pos':0,'start':0,'stop':16},
           {'frame':1,'t_pos':1,'start':16,'stop':32}]
    oldk,oldv=k.clone(),v.clone()
    a,b,audit=consolidate_values(k,v,spans,pointers=2)
    assert torch.equal(a,k)
    assert torch.equal(b[...,:16,:],v[...,:16,:])
    assert torch.equal(b[...,-2:,:],v[...,-2:,:])
    assert not torch.equal(b[...,16:32,:],v[...,16:32,:])
    assert torch.equal(k,oldk) and torch.equal(v,oldv)
    assert a.shape==k.shape and b.shape==v.shape and audit['groups']==1
    jointk,jointv,_=consolidate_values(k,v,spans,pointers=2,mode='joint')
    assert torch.equal(jointv,b) and not torch.equal(jointk,k)


def test_noop_returns_exact_original_tensors():
    k=torch.randn(1,1,16,8);v=torch.randn_like(k)
    a,b,_=consolidate_values(k,v,[],pointers=0,mode='noop')
    assert a is k and b is v


def test_gate_allows_propagated_query_difference_but_rejects_memory_change():
    from copy import deepcopy
    import pytest
    from scripts.run_native_value_consolidation_gate import validate_native_read_inputs
    audits={m:[{'frames':[(0,0)],'token_count':16,'pointer_keys':0,
                'qkv_input_sha256':['same' if i==0 else m,'k','v']}
               for i in range(4)] for m in ('noop','value','joint')}
    validate_native_read_inputs(audits)
    bad=deepcopy(audits);bad['value'][2]['qkv_input_sha256'][2]='changed'
    with pytest.raises(RuntimeError,match='memory K/V inputs differ'):
        validate_native_read_inputs(bad)
