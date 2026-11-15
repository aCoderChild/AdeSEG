import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from scripts.run_source_group_boundary_study_v1 import make_phases
from scripts.score_source_group_boundary_v1 import boundary_five_pixel, metric, summary
from scripts.source_group_boundary_shared_v1 import ARMS, LOSS, dependencies, initial_hash, model_for
from scripts.train_region_mixed_loss_v1 import pixel_loss
from scripts.train_source_group_boundary_v1 import PARITY_KEYS, build_plan, matched_loss
from scripts.memory_token_io import digest
import scripts.score_source_group_boundary_v1 as scorer
import scripts.verify_source_group_boundary_v1 as verifier


def test_six_arm_factorial_and_identical_initialization():
    assert set(ARMS) == {f'{a}_{b}' for a in ('region', 'no_contrast', 'mask_only') for b in ('bce', 'boundary')}
    identities = []
    for arm in ARMS:
        torch.manual_seed(20261115)
        model = model_for(arm)
        identities.append(initial_hash(model))
        assert sum(p.numel() for p in model.parameters()) == 48005
        assert model._mask_only is arm.startswith('mask_only_')
        assert model.contrast_head is (not arm.startswith('no_contrast_'))
    assert len(set(identities)) == 1


def test_matched_loss_dispatch_and_gradient():
    p = torch.zeros(1, 1, 32, 32, requires_grad=True)
    y = torch.zeros_like(p)
    y[..., 8:22, 7:24] = 1
    for mechanism in ('region', 'no_contrast', 'mask_only'):
        assert torch.equal(matched_loss(mechanism + '_bce', p, y), pixel_loss(p, y, 'bce'))
        assert torch.equal(matched_loss(mechanism + '_boundary', p, y), cmsa_boundary_loss(p, y))
        grad = torch.autograd.grad(matched_loss(mechanism + '_boundary', p, y), p)[0]
        assert torch.isfinite(grad).all()
    assert len(set(LOSS.values())) == 2


def test_parity_plan_and_source_binding_roundtrip():
    samples = [{'split': 'train', 'sequence': s, 'frame': f} for s, f in PARITY_KEYS]
    manifest = {'feature_pipeline_sha256': 'b' * 64,
                'feature_pipeline': {'runtime_environment': {'device': 'mps'}}, 'samples': samples}
    plan = build_plan(manifest, 'a' * 64)
    assert json.loads(json.dumps(plan)) == plan
    assert plan['arms'] == list(ARMS)
    assert plan['loss'] == LOSS
    assert plan['parity_keys'] == [['seq2', 4], ['seq4', 18], ['seq4', 24]]
    sources = dependencies()
    assert 'modeling/cmsa_boundary_training_loss.py' in sources
    assert 'scripts/run_source_group_boundary_study_v1.py' in sources
    assert 'protocols/CMSA_BOUNDARY_LOSS_PILOT_V1.md' in sources


def test_orchestration_runs_each_factorial_arm_before_replay_and_score(tmp_path):
    phases = make_phases(Path(tmp_path))
    names = [name for name, _ in phases]
    assert names[:3] == ['manifest', 'plan', 'parity']
    assert names[3:9] == [f'train_{arm}' for arm in ARMS]
    assert names[-2:] == ['verify', 'score']
    assert all('--arm' in command and command[command.index('--arm')+1] == arm
               for (_, command), arm in zip(phases[3:9], ARMS, strict=True))


def test_statically_imported_project_modules_are_source_bound():
    root = Path(__file__).resolve().parents[1]
    sources = dependencies()
    pending = [Path(k) for k in sources if k.endswith('.py')]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        tree = ast.parse((root/current).read_text())
        for node in ast.walk(tree):
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
                if (root/path).is_file():
                    assert str(path) in sources, f'Unbound local dependency: {path}'
                    pending.append(path)


def test_boundary_tolerance_and_empty_masks():
    a = np.zeros((40, 40), dtype=bool)
    b = np.zeros_like(a)
    assert boundary_five_pixel(a, b) == 1.0
    a[20, 15] = True
    assert boundary_five_pixel(a, b) == 0.0
    b[20, 20] = True
    assert boundary_five_pixel(a, b) == 1.0
    b[20, 20] = False
    b[20, 21] = True
    assert boundary_five_pixel(a, b) == 0.0
    assert metric(a, b)['zero_overlap'] is True
    rows = [{'present': True, 'model': metric(a, b)},
            {'present': False, 'model': metric(a, np.zeros_like(a))}]
    result = summary(rows, 'model')
    assert result['present_zero_overlap_frames'] == 1
    assert result['mean_present_boundary_f_5px'] == 0.0
    assert result['empty_false_alarms'] == 1


@pytest.mark.parametrize('arm', ARMS)
def test_trained_checkpoint_save_load_preserves_outputs(tmp_path, arm):
    from modeling.lean_channel_attention import author_sha256
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(123)
    model = model_for(arm).train()
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    args = (torch.randn(1, 32, 32, 32), torch.randn(1, 256, 8, 8), torch.randn(1, 1, 32, 32))
    expected = model(*args).detach()
    payload = {'format': 'foreground_conditioned_refiner_v1', 'research_only': True,
               'training_steps': 1, 'state_dict': model.state_dict(),
               'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64,
               'train_manifest_sha256': 'c'*64, 'author_sha256': author_sha256(),
               'conditioned_source_sha256': digest(root/'modeling/foreground_conditioned_refiner.py'),
               'base_source_sha256': digest(root/'modeling/multilevel_contrast_refiner.py'),
               'gate_mode': model.gate_mode, 'contrast_head': model.contrast_head,
               'arm': 'mask_only' if arm.startswith('mask_only_') else 'multilevel'}
    path = tmp_path/'checkpoint.pt'
    torch.save(payload, path)
    fitted = model_for(arm)
    fitted.load_trained_state(torch.load(path, map_location='cpu', weights_only=True),
                              backbone_sha256='a'*64, feature_pipeline_sha256='b'*64,
                              train_manifest_sha256='c'*64)
    fitted.eval()
    with torch.no_grad():
        actual = fitted(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _fake_complete_arm(tmp_path, monkeypatch):
    arm = 'region_boundary'
    folder = tmp_path/arm
    folder.mkdir()
    (folder/'predictions').mkdir()
    file = folder/'predictions/mask.png'
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(file)
    rows = [{'sequence': 'seq9', 'frame': i, 'shape': [2, 2],
             'baseline': str(file), 'refined': str(file),
             'baseline_sha256': digest(file), 'refined_sha256': digest(file)} for i in range(50)]
    manifest = {'checkpoint_sha256': 'c'*64, 'feature_pipeline_sha256': 'f'*64,
                'samples': [{'split': 'validation', 'sequence': 'seq9', 'frame': i} for i in range(50)]}
    order = [[['seq2', i] for i in range(164)] for _ in range(10)]
    plan = {'initial_state_sha256': 'i'*64, 'environment': {'device': 'test'}, 'training_order': order}
    plan_path = tmp_path/'plan.json'; plan_path.write_text('{}')
    parity_path = tmp_path/'parity.json'; parity_path.write_text('{}')
    payload = {'research_only': True, 'training_steps': 1640, 'study_arm': arm,
               'train_manifest_sha256': 'm'*64, 'backbone_sha256': manifest['checkpoint_sha256'],
               'feature_pipeline_sha256': manifest['feature_pipeline_sha256'], 'plan_sha256': digest(plan_path),
               'parity_sha256': digest(parity_path), 'initial_state_sha256': plan['initial_state_sha256'],
               'dependency_sha256': dependencies(), 'environment': plan['environment'], 'seed': 20261115,
               'epochs': 10, 'parameter_count': 48005, 'loss': LOSS[arm],
               'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
               'gate_mode': 'region', 'contrast_head': True, 'arm': 'multilevel'}
    ck = folder/'checkpoint.pt'; torch.save(payload, ck)
    frozen = folder/'frozen_predictions.json'
    frozen.write_text(json.dumps({'checkpoint_sha256': digest(ck), 'rows': rows}))
    history = [{'epoch': i+1, 'samples': 164, 'sample_order': order[i], 'mean_training_loss': 1.0}
               for i in range(10)]
    completion = {'passed': True, 'arm': arm, 'manifest_sha256': 'm'*64,
                  'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path),
                  'checkpoint_sha256': digest(ck), 'frozen_predictions_sha256': digest(frozen),
                  'initial_state_sha256': plan['initial_state_sha256'], 'training_order': order,
                  'label_guard': {'blocked_open_attempts': 0, 'train_paths': 0, 'blocked_paths': 0},
                  'history': history}
    (folder/'completion.json').write_text(json.dumps(completion))
    monkeypatch.setattr(verifier, 'mask_path_sets', lambda _: (set(), set()))
    return arm, folder, manifest, plan_path, parity_path, plan, payload, completion, frozen, ck


def test_verifier_rejects_wrong_objective_arm_and_incomplete_history(tmp_path, monkeypatch):
    arm, folder, manifest, plan_path, parity_path, plan, payload, done, frozen, ck = _fake_complete_arm(tmp_path, monkeypatch)
    def check():
        return verifier.check_frozen_arm(folder, arm, manifest, 'm'*64, plan_path, parity_path, plan)
    assert len(check()['mapping']) == 50
    for field, wrong in [('loss', LOSS['region_bce']), ('study_arm', 'region_bce')]:
        bad = dict(payload); bad[field] = wrong
        torch.save(bad, ck)
        frozen_data = json.loads(frozen.read_text()); frozen_data['checkpoint_sha256'] = digest(ck)
        frozen.write_text(json.dumps(frozen_data))
        completion = dict(done); completion['checkpoint_sha256'] = digest(ck); completion['frozen_predictions_sha256'] = digest(frozen)
        (folder/'completion.json').write_text(json.dumps(completion))
        with pytest.raises(ValueError, match='checkpoint identity'):
            check()
    torch.save(payload, ck)
    frozen_data = json.loads(frozen.read_text()); frozen_data['checkpoint_sha256'] = digest(ck)
    frozen.write_text(json.dumps(frozen_data))
    done['checkpoint_sha256'] = digest(ck); done['frozen_predictions_sha256'] = digest(frozen)
    done['history'] = done['history'][:-1]
    (folder/'completion.json').write_text(json.dumps(done))
    with pytest.raises(ValueError, match='incomplete training'):
        check()


def test_scoring_refuses_missing_or_failed_replay_before_any_label(tmp_path, monkeypatch):
    verification = tmp_path/'verification'; verification.mkdir()
    monkeypatch.setattr(scorer, 'load_prepared', lambda _: ({}, 'm'))
    monkeypatch.setattr(scorer, 'require_plan', lambda *_: {'environment': {}})
    monkeypatch.setattr(scorer, 'require_parity', lambda *_: None)
    monkeypatch.setattr(scorer, 'mask_path_sets', lambda _: (set(), set()))
    opened = []
    monkeypatch.setattr(scorer, 'ensure_target', lambda *_: opened.append(True))
    argv = ['score']
    for key, value in [('manifest', 'm'), ('plan', 'p'), ('parity', 'q'),
                       ('verification', str(verification)), ('run-root', 'r'), ('output', 'out')]:
        argv.extend(['--'+key, value])
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(FileNotFoundError):
        scorer.main()
    evidence = verification/'verification.json'; evidence.write_text(json.dumps({'passed': False}))
    (verification/'completion.json').write_text(json.dumps({'passed': True, 'verification_sha256': digest(evidence)}))
    with pytest.raises(ValueError, match='replay verification'):
        scorer.main()
    assert not opened
