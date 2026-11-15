import torch
import ast
import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from modeling.training_foreground_preservation_v1 import positive_preservation_penalty
from scripts.source_group_preservation_shared_v1 import ARMS, LOSS, dependencies, mechanism_for, model_for, training_width
from scripts.train_source_group_preservation_v1 import matched_loss
from scripts.run_source_group_preservation_study_v1 import FROZEN_MANIFEST, make_phases
from scripts.memory_token_io import digest
import scripts.score_source_group_preservation_v1 as scorer
import scripts.verify_source_group_preservation_v1 as verifier
import scripts.train_source_group_preservation_v1 as trainer


def test_training_width_schedule_is_deterministic_matched_and_cycles():
    a, b = 'region_aug_base', 'mask_only_aug_preserve'
    for epoch in range(10):
        assert training_width(a, 'seq13', 108, epoch) == training_width(b, 'seq13', 108, epoch)
        assert training_width('region_clean_preserve', 'seq13', 108, epoch) == 1
    assert set(training_width(a, 'seq13', 108, e) for e in range(3)) == {1, 33, 65}


def test_all_arms_same_parameter_identity_and_control_flags():
    import hashlib
    tensors = []
    for arm in ARMS:
        torch.manual_seed(20261115)
        model = model_for(arm)
        assert sum(p.numel() for p in model.parameters()) == 48005
        assert model._mask_only == (mechanism_for(arm) == 'mask_only')
        assert model.contrast_head == (mechanism_for(arm) != 'no_contrast')
        tensors.append(hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest())
    assert len(set(tensors)) == 1


def test_only_preservation_arms_add_one_sided_penalty():
    prediction = torch.full((1, 1, 16, 16), -2.)
    teacher = torch.full_like(prediction, 3.)
    target = torch.ones_like(prediction)
    base = cmsa_boundary_loss(prediction, target)
    penalty = positive_preservation_penalty(prediction, teacher, target)
    for arm in ARMS:
        expected = base + penalty if arm.endswith('_preserve') else base
        torch.testing.assert_close(matched_loss(arm, prediction, target, teacher), expected)


def test_launcher_reuses_frozen_manifest_and_scores_last(tmp_path):
    phases = make_phases(tmp_path)
    assert [name for name, _ in phases] == ['plan', 'parity', *['train_'+a for a in ARMS], 'verify', 'score']
    for _, cmd in phases:
        assert str(FROZEN_MANIFEST) in cmd


def test_plan_binds_exact_training_coverage_rows_and_hashes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    samples = [{'split': 'train', 'sequence': 'seq2', 'frame': i, 'label_sha256': f'label{i}',
                'snapshot_sha256': f'snapshot{i}'} for i in range(164)]
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'rows': [{'frame': i, 'mask_sha256': f'prediction{i}'} for i in range(164)]}))
    manifest = {'samples': samples, 'sequences': {'seq2': {'capture_receipt_path': str(receipt)}},
                'feature_pipeline_sha256': 'f'*64, 'feature_pipeline': {'runtime_environment': {}}}
    error = {'passed': True, 'manifest_sha256': 'm'*64, 'label_guard': {'blocked_open_attempts': 0},
             'source_sha256': digest(root/'scripts/audit_source_group_training_errors_v1.py'),
             'rows': [{'sequence': 'seq2', 'frame': i, 'label_sha256': f'label{i}',
                       'prediction_sha256': f'prediction{i}'} for i in range(164)]}
    expanded = {'passed': True, 'manifest_sha256': 'm'*64, 'label_guard': {'blocked_open_attempts': 0},
                'source_sha256': digest(root/'scripts/audit_training_logit_expansion_v1.py'),
                'expansion_sha256': digest(root/'modeling/training_foreground_preservation_v1.py'),
                'rows': [{'sequence': 'seq2', 'frame': i, 'label_sha256': f'label{i}',
                          'snapshot_sha256': f'snapshot{i}'} for i in range(164)]}
    a, b = tmp_path/'a.json', tmp_path/'b.json'; a.write_text(json.dumps(error)); b.write_text(json.dumps(expanded))
    monkeypatch.setattr(trainer, 'TRAIN_AUDIT', a); monkeypatch.setattr(trainer, 'EXPANSION_AUDIT', b)
    assert trainer.build_plan(manifest, 'm'*64)['training_coverage_audit_sha256'] == digest(a)
    expanded['rows'][0]['snapshot_sha256'] = 'wrong'; b.write_text(json.dumps(expanded))
    with pytest.raises(ValueError, match='sample hash'):
        trainer.build_plan(manifest, 'm'*64)
    expanded['rows'][0]['snapshot_sha256'] = 'snapshot0'; expanded['rows'][0]['frame'] = 1
    b.write_text(json.dumps(expanded))
    with pytest.raises(ValueError, match='sample identities'):
        trainer.build_plan(manifest, 'm'*64)


def test_all_statically_imported_project_modules_are_bound():
    root = Path(__file__).resolve().parents[1]
    sources = dependencies()
    pending = [Path(k) for k in sources if k.endswith('.py')]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for node in ast.walk(ast.parse((root/current).read_text())):
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


@pytest.mark.parametrize('arm', ARMS)
def test_checkpoint_reload_reproduces_unaugmented_inference(tmp_path, arm):
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
               'arm': 'mask_only' if mechanism_for(arm) == 'mask_only' else 'multilevel'}
    path = tmp_path/'checkpoint.pt'; torch.save(payload, path)
    fitted = model_for(arm)
    fitted.load_trained_state(torch.load(path, map_location='cpu', weights_only=True),
                             backbone_sha256='a'*64, feature_pipeline_sha256='b'*64,
                             train_manifest_sha256='c'*64)
    fitted.eval()
    with torch.no_grad():
        torch.testing.assert_close(fitted(*args), expected, rtol=0, atol=0)


def _fake_complete_arm(tmp_path, monkeypatch):
    arm = 'region_aug_preserve'; folder = tmp_path/arm; folder.mkdir()
    file = folder/'mask.png'; Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(file)
    rows = [{'sequence': 'seq9', 'frame': i, 'shape': [2, 2], 'baseline': str(file), 'refined': str(file),
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
               'optimizer': 'Adam', 'learning_rate': .0003, 'weight_decay': .0001,
               'gate_mode': 'region', 'contrast_head': True, 'arm': 'multilevel'}
    ck = folder/'checkpoint.pt'; torch.save(payload, ck)
    frozen = folder/'frozen_predictions.json'; frozen.write_text(json.dumps({'checkpoint_sha256': digest(ck), 'rows': rows}))
    history = [{'epoch': i+1, 'samples': 164, 'sample_order': order[i],
                'widths': [training_width(arm, s, f, i) for s, f in order[i]], 'mean_training_loss': 1.0}
               for i in range(10)]
    completion = {'passed': True, 'arm': arm, 'manifest_sha256': 'm'*64,
                  'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path),
                  'checkpoint_sha256': digest(ck), 'frozen_predictions_sha256': digest(frozen),
                  'initial_state_sha256': plan['initial_state_sha256'], 'training_order': order,
                  'label_guard': {'blocked_open_attempts': 0, 'train_paths': 0, 'blocked_paths': 0}, 'history': history}
    (folder/'completion.json').write_text(json.dumps(completion))
    monkeypatch.setattr(verifier, 'mask_path_sets', lambda _: (set(), set()))
    return arm, folder, manifest, plan_path, parity_path, plan, completion, payload, frozen, ck


def test_verifier_rejects_tampered_augmentation_schedule_and_missing_history(tmp_path, monkeypatch):
    arm, folder, manifest, plan_path, parity_path, plan, done, _, _, _ = _fake_complete_arm(tmp_path, monkeypatch)
    def check():
        return verifier.check_frozen_arm(folder, arm, manifest, 'm'*64, plan_path, parity_path, plan)
    assert len(check()['mapping']) == 50
    done['history'][0]['widths'][0] = 2
    (folder/'completion.json').write_text(json.dumps(done))
    with pytest.raises(ValueError, match='incomplete training'):
        check()
    done['history'][0]['widths'][0] = training_width(arm, 'seq2', 0, 0)
    done['history'] = done['history'][:-1]
    (folder/'completion.json').write_text(json.dumps(done))
    with pytest.raises(ValueError, match='incomplete training'):
        check()


def test_verifier_rejects_rehashed_wrong_checkpoint_arm_and_objective(tmp_path, monkeypatch):
    arm, folder, manifest, plan_path, parity_path, plan, done, payload, frozen, ck = _fake_complete_arm(tmp_path, monkeypatch)
    def check():
        return verifier.check_frozen_arm(folder, arm, manifest, 'm'*64, plan_path, parity_path, plan)
    assert len(check()['mapping']) == 50
    for field, wrong in [('loss', LOSS['region_aug_base']), ('study_arm', 'region_aug_base'),
                         ('contrast_head', False), ('train_manifest_sha256', 'x'*64)]:
        bad = dict(payload); bad[field] = wrong
        torch.save(bad, ck)
        frozen_data = json.loads(frozen.read_text()); frozen_data['checkpoint_sha256'] = digest(ck)
        frozen.write_text(json.dumps(frozen_data))
        completion = dict(done); completion['checkpoint_sha256'] = digest(ck); completion['frozen_predictions_sha256'] = digest(frozen)
        (folder/'completion.json').write_text(json.dumps(completion))
        with pytest.raises(ValueError, match='checkpoint identity'):
            check()


def test_scorer_refuses_missing_or_failed_replay_before_target_access(tmp_path, monkeypatch):
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
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(FileNotFoundError):
        scorer.main()
    evidence = verification/'verification.json'; evidence.write_text(json.dumps({'passed': False}))
    (verification/'completion.json').write_text(json.dumps({'passed': True, 'verification_sha256': digest(evidence)}))
    with pytest.raises(ValueError, match='replay verification'):
        scorer.main()
    assert not opened
