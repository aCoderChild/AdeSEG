from types import MethodType, SimpleNamespace
import json

import pytest
import torch

from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_verified_runtime_lease import _FACTORY_ISSUER, _RuntimeLease
from modeling.lean_verified_window_runtime import _mps_hardware_inventory


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.memory_attention = SimpleNamespace(forward=lambda: None)
        self.token_window_runtime = SimpleNamespace(budget=16384, insurance_reserve=3072)
        self.mem_keep_ratio, self.num_maskmem = .25, 7
        self.ema_history_replace_old_tail_in_attn = False
        self.ema_history_resident_prune = False

    @property
    def device(self):
        return torch.device('mps')

    def _prepare_memory_conditioned_features(self): pass
    def _cache_tarr_affinity_from_new_memory(self): pass
    def _dump_retained_tokens_record(self): pass


def make_lease():
    model = Predictor().eval()
    refiner = torch.nn.Linear(1, 1).eval()
    return model, refiner, _RuntimeLease(model, refiner, model.token_window_runtime,
                                          module_tensor_fingerprints(model),
                                          module_tensor_fingerprints(refiner),
                                          _issuer=_FACTORY_ISSUER)


def test_lease_rejects_wrong_identity_hook_policy_and_expiration():
    model, refiner, lease = make_lease()
    lease.validate_live(model, refiner, tensors=True)
    with pytest.raises(RuntimeError, match='another'):
        lease.validate_live(Predictor().eval(), refiner, tensors=False)
    model.token_window_runtime.budget = 1
    with pytest.raises(RuntimeError, match='policy'):
        lease.validate_live(model, refiner, tensors=False)
    model.token_window_runtime.budget = 16384
    model._prepare_memory_conditioned_features = MethodType(lambda self: None, model)
    with pytest.raises(RuntimeError, match='hook'):
        lease.validate_live(model, refiner, tensors=False)
    lease.close()
    with pytest.raises(RuntimeError, match='inactive'):
        lease.validate_live(model, refiner, tensors=False)
    with pytest.raises(RuntimeError, match='observations'):
        lease.observations()


def test_lease_rejects_changed_refiner_tensor_and_nonfactory_issuer():
    model, refiner, lease = make_lease()
    with torch.no_grad():
        refiner.weight.add_(1)
    with pytest.raises(RuntimeError, match='tensor state'):
        lease.validate_live(model, refiner, tensors=True)
    with pytest.raises(RuntimeError, match='factory'):
        _RuntimeLease(model, refiner, model.token_window_runtime, {}, {}, _issuer=object())


def test_gpu_inventory_omits_display_identifiers(monkeypatch):
    import modeling.lean_verified_window_runtime as runtime
    payload = {'SPDisplaysDataType': [{'_name': 'Apple GPU', 'sppci_device_type': 'spdisplays_gpu',
                                       'sppci_cores': '10', 'display_serial': 'private'}]}
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda *_args, **_kwargs: json.dumps(payload))
    assert _mps_hardware_inventory() == [{'name': 'Apple GPU', 'cores': '10'}]
