import ast
import json
from pathlib import Path

import pytest
import torch

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest
from scripts.run_source_group_boundary_controls_v1 import make_phases
from scripts.source_group_boundary_controls_shared_v1 import ARMS, LOSS, dependencies, initial_hash, model_for
from scripts.train_source_group_boundary_controls_v1 import PARITY_KEYS, build_plan, matched_loss
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss

ROOT = Path(__file__).resolve().parents[1]


def test_four_arms_reproduce_prior_parameter_initialization():
    assert ARMS == ('region_boundary', 'global_boundary', 'descriptor_roll_boundary', 'spatial_shuffle_boundary')
    hashes = []
    for arm in ARMS:
        torch.manual_seed(20261115)
        model = model_for(arm)
        hashes.append(initial_hash(model))
        assert sum(p.numel() for p in model.parameters()) == 48005
    torch.manual_seed(20261115)
    old = ForegroundConditionedRefiner(gate_mode='region', contrast_head=True)
    assert len(set(hashes)) == 1 and hashes[0] == initial_hash(old)


@pytest.mark.parametrize(('arm', 'gate_mode'), [
    ('region_boundary', 'region'), ('global_boundary', 'global'),
    ('descriptor_roll_boundary', 'shuffled'),
])
def test_reference_and_gate_controls_exactly_reproduce_existing_forward(arm, gate_mode):
    torch.manual_seed(41)
    model = model_for(arm).train()
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    old = ForegroundConditionedRefiner(gate_mode=gate_mode, contrast_head=True).train()
    old.load_state_dict(model.state_dict())
    args = (torch.randn(1, 32, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 1, 8, 8))
    torch.testing.assert_close(model(*args), old(*args), rtol=0, atol=0)


def test_controls_dispatch_same_loss_and_different_gate_information():
    pred = torch.randn(1, 1, 32, 32, requires_grad=True)
    truth = (torch.rand_like(pred) > 0.5).float()
    for arm in ARMS:
        assert torch.equal(matched_loss(arm, pred, truth), cmsa_boundary_loss(pred, truth))
    assert len(set(LOSS.values())) == 1
    assert model_for('global_boundary').gate_mode == 'global'
    assert model_for('descriptor_roll_boundary').gate_mode == 'shuffled'
    assert model_for('spatial_shuffle_boundary').control == 'spatial_shuffle'


@pytest.mark.parametrize('arm', ARMS)
def test_control_checkpoint_roundtrip_and_reject_wrong_control(arm):
    torch.manual_seed(22)
    model = model_for(arm).train()
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    args = (torch.randn(1, 32, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 1, 8, 8))
    expected = model(*args).detach()
    payload = {'format': 'foreground_conditioned_refiner_v1', 'research_only': True,
               'training_steps': 1, 'state_dict': model.state_dict(),
               'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64,
               'train_manifest_sha256': 'c'*64, 'author_sha256': author_sha256(),
               'conditioned_source_sha256': digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
               'base_source_sha256': digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
               'gate_mode': model.gate_mode, 'contrast_head': True, 'arm': 'multilevel',
               'control': model.control}
    fitted = model_for(arm)
    fitted.load_trained_state(payload, backbone_sha256='a'*64,
                              feature_pipeline_sha256='b'*64, train_manifest_sha256='c'*64)
    fitted.eval()
    with torch.no_grad():
        torch.testing.assert_close(fitted(*args), expected, rtol=0, atol=0)
    altered = dict(payload, control='bad')
    with pytest.raises(ValueError, match='control'):
        model_for(arm).load_trained_state(altered, backbone_sha256='a'*64,
                                         feature_pipeline_sha256='b'*64, train_manifest_sha256='c'*64)


def test_plan_roundtrip_and_phase_gates():
    samples = [{'split': 'train', 'sequence': s, 'frame': f} for s, f in PARITY_KEYS]
    manifest = {'feature_pipeline_sha256': 'b'*64,
                'feature_pipeline': {'runtime_environment': {'device': 'mps'}}, 'samples': samples}
    plan = build_plan(manifest, 'a'*64)
    assert json.loads(json.dumps(plan)) == plan
    assert plan['arms'] == list(ARMS) and plan['loss'] == LOSS
    phases = [name for name, _ in make_phases('/tmp/nonexistent')]
    assert phases == ['manifest', 'plan', 'parity', *(f'train_{x}' for x in ARMS), 'verify', 'score']


def test_every_static_project_import_is_source_bound():
    sources = dependencies()
    pending = [Path(k) for k in sources if k.endswith('.py')]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for node in ast.walk(ast.parse((ROOT/current).read_text())):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split('.')[0] not in {'scripts', 'modeling', 'datasets'}:
                    continue
                path = Path(*name.split('.')).with_suffix('.py')
                if (ROOT/path).is_file():
                    assert str(path) in sources, f'Unbound local dependency: {path}'
                    pending.append(path)
