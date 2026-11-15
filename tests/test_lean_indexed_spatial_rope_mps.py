import pytest
import torch
import sys
from pathlib import Path
from modeling.lean_indexed_spatial_rope_mps import decode_mps_positions
from modeling.lean_memory_position_lookup import MemoryPositionLookup


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS required')
def test_mps_lookup_matches_exact_cpu_reference_and_rejects_corruption():
    torch.manual_seed(43)
    yy=torch.randn(64,32,device='mps');xx=torch.randn(64,32,device='mps')
    grid=torch.cat((yy[:,None,:].expand(64,64,32),xx[None,:,:].expand(64,64,32)),dim=-1).reshape(4096,64)
    offsets=torch.randn(7,64,device='mps')
    selected=torch.tensor([0,3,9,321,4095,56],device='mps')
    ts=torch.tensor([0,4,1,6,5,2],device='mps')
    spatial=grid[selected]+offsets[ts]
    ptr=torch.full((2,64),9999.,device='mps')
    memory=torch.cat((spatial,ptr))[:,None,:]
    actual=decode_mps_positions(memory,2,grid,offsets,chunk=2)
    assert torch.equal(actual,selected)
    cpu=MemoryPositionLookup(grid.cpu(),offsets.cpu()).decode(memory.cpu(),2)
    assert torch.equal(actual.cpu(),cpu)
    bad=memory.clone();bad[0,0,0]+=0.125
    with pytest.raises(ValueError,match='exact'):
        decode_mps_positions(bad,2,grid,offsets,chunk=2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS required')
def test_real_lean_query_and_indexed_key_rotary_against_cpu_reference():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/Lean-SAM2'))
    from sam2.modeling.memory_attention import _apply_rotary_repeat_2d
    torch.manual_seed(63)
    q=torch.randn(1,2,4096,16)
    key=torch.randn(1,2,19,16)
    angle=torch.randn(4096,8)
    frequencies=torch.polar(torch.ones_like(angle),angle)
    selected=torch.tensor([0,25,144,4095,301,25,4,12,66,998,11,9,8,66,77,234,165,304,88])
    q_cpu=_apply_rotary_repeat_2d(q,frequencies,4096)
    q_mps=_apply_rotary_repeat_2d(q.to('mps'),frequencies.to('mps'),4096).cpu()
    torch.testing.assert_close(q_mps,q_cpu,rtol=1e-6,atol=1e-6)
    def key_rotate(k,fc):
        kc=torch.view_as_complex(k.float().reshape(1,2,19,8,2))
        return torch.view_as_real(kc*fc.view(1,1,19,8)).reshape(1,2,19,16)
    k_cpu=key_rotate(key,frequencies[selected])
    k_mps=key_rotate(key.to('mps'),frequencies.to('mps')[selected.to('mps')]).cpu()
    torch.testing.assert_close(k_mps,k_cpu,rtol=1e-6,atol=1e-6)
