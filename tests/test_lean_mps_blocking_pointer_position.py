import sys
from pathlib import Path
import pytest
import torch
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_scoped_workaround_replaces_only_expected_method_and_restores():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/Lean-SAM2'))
    from sam2.modeling.sam2_base import SAM2Base
    model=SAM2Base.__new__(SAM2Base)
    torch.nn.Module.__init__(model)
    model.register_parameter('_device_probe',torch.nn.Parameter(torch.zeros(1,device='mps')))
    original=model._prepare_memory_conditioned_features
    with pytest.raises(RuntimeError,match='injected'):
        with blocking_mps_pointer_position(model):
            assert model._prepare_memory_conditioned_features.__func__ is not original.__func__
            with pytest.raises(RuntimeError,match='existing'):
                with blocking_mps_pointer_position(model):
                    pass
            raise RuntimeError('injected')
    assert model._prepare_memory_conditioned_features.__func__ is original.__func__
    assert '_prepare_memory_conditioned_features' not in vars(model)
