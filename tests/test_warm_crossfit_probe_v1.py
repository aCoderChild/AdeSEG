import json

import pytest
import torch

from scripts import warm_crossfit_probe_v1 as probe
from scripts.source_group_refiner_shared_v2 import initial_hash


class ZeroResidual(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        torch.nn.init.zeros_(self.head[-1].weight); torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, early, deep, logits):
        return logits + self.head(torch.ones_like(logits))


def test_targeted_probe_checks_identity_gradient_and_view_without_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ARMS', ('mask_only',))
    monkeypatch.setattr(probe, 'model_for', lambda _: ZeroResidual())
    monkeypatch.setattr(probe, 'inputs', lambda row, device: ((torch.zeros(1), torch.zeros(1),
        torch.zeros(1, 1, 2, 2)), (720, 834) if row['sequence'] == 'seq23' else (4, 5)))
    rows = {('seq6', 24): {'sequence': 'seq6'}, ('seq23', 1): {'sequence': 'seq23'}}
    preflight = {'initial_state_sha256': {'mask_only': initial_hash(ZeroResidual())}}
    out = tmp_path/'probe'; guard = {'open_attempts': 0}
    probe.probe(preflight, 'plan', rows, out, guard, device='cpu')
    receipt = json.loads((out/'completion.json').read_text())
    assert receipt['passed'] and len(receipt['checks']) == 2 and guard['open_attempts'] == 0
    monkeypatch.setattr(probe, 'inputs', lambda *a: ((torch.zeros(1), torch.zeros(1),
        torch.full((1, 1, 2, 2), float('nan'))), (4, 5)))
    with pytest.raises(ValueError, match='finite no-op'):
        probe.probe(preflight, 'plan', rows, tmp_path/'bad', guard, device='cpu')


def test_probe_rejects_zero_residual_head_gradient(tmp_path, monkeypatch):
    class Disconnected(ZeroResidual):
        def forward(self, early, deep, logits):
            return logits + 0 * self.head(torch.ones_like(logits))
    monkeypatch.setattr(probe, 'ARMS', ('mask_only',))
    monkeypatch.setattr(probe, 'model_for', lambda _: Disconnected())
    monkeypatch.setattr(probe, 'inputs', lambda row, device: ((torch.zeros(1), torch.zeros(1),
        torch.zeros(1, 1, 2, 2)), (720, 834) if row['sequence'] == 'seq23' else (4, 5)))
    rows = {('seq6', 24): {'sequence': 'seq6'}, ('seq23', 1): {'sequence': 'seq23'}}
    preflight = {'initial_state_sha256': {'mask_only': initial_hash(Disconnected())}}
    with pytest.raises(ValueError, match='gradient is missing or zero'):
        probe.probe(preflight, 'plan', rows, tmp_path/'bad', {'open_attempts': 0}, device='cpu')
