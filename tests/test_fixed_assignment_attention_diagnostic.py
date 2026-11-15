import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from modeling.fixed_assignment_attention_diagnostic import (
    expand_history_cluster_means,
    measure_fixed_assignment_read,
    FixedAssignmentSelectivityObserver,
)


def fixture(noncond, map_tokens=16):
    torch.manual_seed(118)
    spans = [{"frame": 0, "t_pos": 0, "start": 0, "stop": map_tokens}]
    for i in range(noncond):
        spans.append({"frame": i+1, "t_pos": 11-noncond+i,
                      "start": (i+1)*map_tokens, "stop": (i+2)*map_tokens})
    count = (noncond+1)*map_tokens
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, count+2, 8)
    value = torch.randn_like(key)
    roles = torch.full((1, count, 3), 1/3)
    return query, key, value, roles, spans


@pytest.mark.parametrize("allocation", ["uniform", "recent_dense"])
@pytest.mark.parametrize("noncond,map_tokens", [(7,16),(10,64)])
def test_fixed_assignment_identity_and_pointer_input_preservation(allocation, noncond, map_tokens):
    q,k,v,roles,spans = fixture(noncond,map_tokens)
    originals = [x.clone() for x in (q,k,v,roles)]
    ke,ve,km,vm,bias,audit = expand_history_cluster_means(
        k,v,roles,spans,pointers=2,allocation=allocation,native_map_tokens=map_tokens,
    )
    assert audit["spatial_keys_after"] == 7*map_tokens
    assert torch.equal(ke[..., :map_tokens, :], k[..., :map_tokens, :])
    assert torch.equal(ve[..., :map_tokens, :], v[..., :map_tokens, :])
    for changed,original in ((ke,k),(ve,v),(km,k),(vm,v)):
        assert torch.equal(changed[..., -2:, :],original[..., -2:, :])
    dense_key_only = F.scaled_dot_product_attention(q,ke,v)
    compressed = F.scaled_dot_product_attention(q,km,vm,attn_mask=bias)
    torch.testing.assert_close(compressed,dense_key_only,rtol=5e-6,atol=1e-6)
    result = measure_fixed_assignment_read(
        q,k,v,roles,spans,pointers=2,allocation=allocation,native_map_tokens=map_tokens,
    )
    assert result["joint_vs_key_only_identity_residual"]["max_absolute_difference"] < 1e-6
    assert result["counterfactuals"]["value_mean_only_expanded"]["relative_squared_error"] > 0
    assert result["interaction_l2"] > 0
    assert all(torch.equal(old,new) for old,new in zip(originals,(q,k,v,roles)))


def test_no_compression_counterfactuals_are_exact_native():
    q,k,v,roles,spans = fixture(6)
    result = measure_fixed_assignment_read(q,k,v,roles,spans,pointers=2,native_map_tokens=16)
    assert result["memory_audit"]["spatial_keys_after"] == 7*16
    assert all(row["relative_squared_error"] == 0
               for row in result["counterfactuals"].values())
    assert result["interaction_l2"] == 0


@pytest.mark.parametrize("zero_values", [False,True])
def test_observer_returns_exact_dense_reference_and_rejects_custom_scale(zero_values):
    model = SimpleNamespace(num_maskmem=7,
        _prepare_memory_conditioned_features=lambda **kwargs: None)
    observer = FixedAssignmentSelectivityObserver(model)
    observer.original_sdpa = F.scaled_dot_product_attention
    observer.active, observer.frame = (0,2), 1
    observer.spans = [{"frame":0,"t_pos":0,"start":0,"stop":4096}]
    observer.memberships = torch.full((1,4096,3),1/3)
    torch.manual_seed(119)
    q=torch.randn(1,1,5,8);k=torch.randn(1,1,4098,8);v=torch.randn_like(k)
    if zero_values:
        v.zero_()
    expected=F.scaled_dot_product_attention(q,k,v)
    assert torch.equal(observer._sdpa(q,k,v),expected)
    assert observer.records[0]["object_id"]==1
    assert all(x["max_absolute_difference"]==0 for x in
               observer.records[0]["measurements"]["counterfactuals"].values())
    with pytest.raises(RuntimeError, match="Unsupported memory SDPA"):
        observer._sdpa(q,k,v,scale=2.0)
