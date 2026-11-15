import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.channel_value_fixed_read_v1 import model_for
from scripts.memory_token_io import digest
from scripts.mechanism_crossfit_control_core_v1 import ARMS
from scripts.run_mechanism_crossfit_study_v1 import check_probe, phases, preserve_partial
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.warm_crossfit_shared_fold_v1 import seed_arm
from scripts import run_mechanism_crossfit_fold_v1 as worker
from scripts import score_mechanism_crossfit_v1 as scorer


def test_phases_train_all_before_predict_and_verify_before_score(tmp_path):
    names = [x[0] for x in phases(tmp_path, tmp_path/'plan', 0)]
    assert names == (['probe'] + [f'train_f{x}' for x in range(1, 6)] +
                     [f'predict_f{x}' for x in range(1, 6)] + ['verify', 'score', 'audit'])
    artifact = tmp_path/'train/fold1/group.json'; artifact.parent.mkdir(parents=True)
    (artifact.parent/'partial.txt').write_text('partial'); (tmp_path/'train_f1.log').write_text('log')
    preserve_partial(tmp_path, 'train_f1', artifact)
    partial = list((tmp_path/'partial_attempts').iterdir())
    assert len(partial) == 1 and (partial[0]/'output/partial.txt').read_text() == 'partial'
    assert not artifact.parent.exists()


def test_training_gate_rejects_changed_or_failed_probe(tmp_path, monkeypatch):
    from scripts import run_mechanism_crossfit_study_v1 as launch
    folder = tmp_path/'probe'; folder.mkdir()
    checks = []
    for seq in ('seq6', 'seq23'):
        for arm in ARMS:
            checkpoint = folder/f'{seq}_{arm}.pt'; checkpoint.write_bytes(b'checkpoint')
            checks.append({'sequence': seq, 'frame': 0, 'arm': arm,
                'video_shape': [720, 834] if seq == 'seq23' else [16, 16],
                'checkpoint_path': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'head_bias_gradient_abs': 1.0, 'gradient_l2': 1.0})
    receipt = folder/'completion.json'
    record = {'passed': True, 'plan_sha256': 'plan', 'warm_preflight_sha256': 'warm', 'device': 'mps',
              'environment': {'test': 1}, 'label_guard': {'open_attempts': 0}, 'checks': checks}
    receipt.write_text(json.dumps(record))
    monkeypatch.setattr(launch, 'phase_artifact', lambda root, phase, path: digest(path))
    samples = {('seq6', 0): {}, ('seq23', 0): {}}
    check_probe(tmp_path, 'plan', 'warm', {'environment': {'test': 1}}, samples)
    Path(checks[0]['checkpoint_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='probe evidence'):
        check_probe(tmp_path, 'plan', 'warm', {'environment': {'test': 1}}, samples)


def test_grouped_fold_worker_trains_and_predicts_only_heldout(tmp_path, monkeypatch):
    seed_arm('cpu'); initial = initial_hash(model_for('foreground_background_asinh'))
    plan = {'initial_state_sha256': initial, 'source_sha256': {}}
    warm = {'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64,
            'aggregate_sha256': 'c'*64, 'environment': {'test': True}}
    rows = {('seq2', 0): {'sequence': 'seq2', 'origin': 'historical', 'label_sha256': 'fake'},
            ('seq3', 0): {'sequence': 'seq3', 'origin': 'historical', 'label_sha256': 'fake'}}
    index = {'source_frames': {'seq2': [{'source_frame_index': 2}], 'seq3': [{'source_frame_index': 3}]}}
    values = (torch.randn(1, 32, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 1, 8, 8))
    shape = (16, 16)
    class Cache:
        def __init__(self, *args): pass
        def get(self, item, device): return values, shape
        def verify_boundary(self, **kwargs): pass
        def stats(self): return {'final_sha256_verified': True}
    monkeypatch.setattr(worker, 'VerifiedCaptureCache', Cache)
    monkeypatch.setattr(worker, 'fold_definition', lambda *args: ({'seq2'}, set(), {'seq3'}, [[('seq2', 0)]]))
    monkeypatch.setattr(worker, 'training_label_hashes', lambda *args: {})
    monkeypatch.setattr(worker, 'load_manifest', lambda *args: {})
    monkeypatch.setattr(worker, 'checked_train_target', lambda *args: torch.ones(1, 1, *shape))
    monkeypatch.setattr(worker, 'rgb_input', lambda *args: torch.rand(1, 3, *shape))
    monkeypatch.setattr(worker, 'sources', lambda: {})
    monkeypatch.setattr(worker, 'inputs', lambda *args: (values, shape))
    monkeypatch.setattr(worker, 'phase_artifact', lambda root, phase, path: digest(path))
    train = tmp_path/'train/fold1'; plan_sha = 'd'*64; warm_sha = 'e'*64
    worker.train(plan, plan_sha, warm, warm_sha, index, rows, 1, train,
                 {'blocked_open_attempts': 0}, device='cpu', cache_mib=0)
    group = json.loads((train/'group.json').read_text())
    assert group['training_steps'] == 1 and set(group['arms']) == set(ARMS)
    pred = tmp_path/'predict/fold1'
    worker.predict(plan, plan_sha, warm, warm_sha, index, rows, 1, train, pred,
                   {'open_attempts': 0}, device='cpu')
    result = json.loads((pred/'completion.json').read_text())
    assert [(x['sequence'], x['frame']) for x in result['rows']] == [('seq3', 0)]
    assert set(result['rows'][0]['masks']) == {'baseline', *ARMS}
    assert all(digest(v['path']) == v['sha256'] for v in result['rows'][0]['masks'].values())


def test_primary_and_independent_metrics_agree_for_synthetic_frozen_outputs(tmp_path, monkeypatch):
    target = np.zeros((16, 16), np.uint8); target[4:8, 4:8] = 1
    label = tmp_path/'label.png'; Image.fromarray(target).save(label)
    samples = {('seq3', 0): {'label': str(label), 'origin': 'historical', 'label_sha256': digest(label)}}
    masks = {}
    for arm in ('baseline', 'asinh_reference', *ARMS):
        output = (target * 255).astype(np.uint8)
        if arm == 'early_only': output[0, 0] = 255
        path = tmp_path/f'{arm}.png'; Image.fromarray(output).save(path)
        masks[arm] = {'path': str(path), 'sha256': digest(path)}
    monkeypatch.setattr(scorer, 'label_hashes', lambda index, rows: {('seq3', 0): digest(label)})
    monkeypatch.setattr(scorer, 'fold_split', lambda n: (set(), set(), {'seq3'} if n == 1 else set()))
    result = scorer.compute({}, samples, {('seq3', 0): {'masks': masks}})
    other = scorer.compute({}, samples, {('seq3', 0): {'masks': masks}}, independent=True)
    errors = []; scorer.compare(result, other, [], errors)
    assert not errors
    assert result['paired_vs_baseline']['early_only']['pooled']['worsened'] == 1
    assert result['paired_vs_asinh_reference']['early_only']['pooled']['worsened'] == 1
    assert result['no_initialization_dice'] is None


def test_audit_rejects_missing_score_before_label_guard_release(tmp_path, monkeypatch):
    import sys
    root = tmp_path/'batch'; (root/'verify').mkdir(parents=True)
    verification = root/'verify/verification.json'
    proof = {'plan_sha256': 'plan'}
    verification.write_text(json.dumps({'passed': True, 'label_guard': {'open_attempts': 0},
        'source_sha256': digest(scorer.__file__), **proof}))
    plan = tmp_path/'plan'; plan.mkdir()
    guard = {'open_attempts': 0, 'active': True}
    monkeypatch.setattr(scorer, 'BASE', tmp_path)
    monkeypatch.setattr(scorer, 'install_phase_mask_guard', lambda _: guard)
    monkeypatch.setattr(scorer, 'study_inputs', lambda _: ({}, 'plan', {}, 'warm', {}, {}))
    monkeypatch.setattr(scorer, 'validate_all', lambda *args, **kwargs: ({}, proof))
    monkeypatch.setattr(scorer, 'phase_artifact', lambda root, phase, path: digest(path))
    reached = []
    monkeypatch.setattr(scorer, 'compute', lambda *args, **kwargs: reached.append(True))
    monkeypatch.setattr(sys, 'argv', ['score', '--mode', 'audit', '--plan', str(plan),
        '--batch-root', str(root), '--output', str(root/'audit')])
    with pytest.raises(FileNotFoundError): scorer.main()
    assert guard['active'] and not reached


def test_direct_training_worker_cannot_skip_bound_probe(tmp_path, monkeypatch):
    import sys
    from scripts import run_mechanism_crossfit_study_v1 as launch
    root = tmp_path/'batch'; root.mkdir(); plan_path = tmp_path/'plan'; plan_path.mkdir()
    monkeypatch.setattr(worker, 'new_output', lambda path: Path(path).resolve())
    monkeypatch.setattr(worker, 'install_fold_training_guard', lambda data, fold: {'blocked_open_attempts': 0})
    monkeypatch.setattr(worker, 'require_environment_flags', lambda: None)
    monkeypatch.setattr(worker.torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(worker.torch, 'use_deterministic_algorithms', lambda _: None)
    monkeypatch.setattr(worker, 'study_inputs', lambda _: ({'source_sha256': {}}, 'sha',
        {'environment': {}}, 'warm', {}, {}))
    monkeypatch.setattr(worker, 'environment_identity', lambda: {})
    monkeypatch.setattr(launch, 'validate_resume', lambda *args: (root, 'manifest', 1))
    monkeypatch.setattr(launch, 'phase_artifact', lambda *args: (_ for _ in ()).throw(ValueError('missing probe')))
    called = []; monkeypatch.setattr(worker, 'train', lambda *args, **kwargs: called.append(True))
    monkeypatch.setattr(sys, 'argv', ['worker', '--mode', 'train', '--plan', str(plan_path), '--fold', '1',
                                      '--output', str(root/'train/fold1')])
    with pytest.raises(ValueError, match='missing probe'): worker.main()
    assert not called


@pytest.mark.parametrize('tamper', ['verification', 'report', 'score_guard', 'audit_guard'])
def test_reference_study_rejects_unbound_proof_or_report(tmp_path, monkeypatch, tamper):
    from scripts import preflight_mechanism_crossfit_v1 as control_preflight
    from scripts import score_warm_crossfit_v1 as warm_scorer
    root = tmp_path/'reference'; (root/'verify').mkdir(parents=True); (root/'score').mkdir(); (root/'audit').mkdir()
    verify = root/'verify/verification.json'; verify.write_text('{}')
    report = root/'score/report.md'; report.write_text('original')
    score = {'passed': True, 'verification_sha256': digest(verify), 'report_sha256': digest(report),
             'prelabel_guard': {'open_attempts': 0}}
    if tamper == 'verification': score['verification_sha256'] = 'incorrect'
    if tamper == 'score_guard': score['prelabel_guard']['open_attempts'] = 1
    score_file = root/'score/results.json'; score_file.write_text(json.dumps(score))
    audit = {'passed': True, 'score_sha256': digest(score_file), 'error_count': 0,
             'prelabel_guard': {'open_attempts': 0}}
    if tamper == 'audit_guard': audit['prelabel_guard']['open_attempts'] = 1
    (root/'audit/audit.json').write_text(json.dumps(audit))
    if tamper == 'report': report.write_text('modified')
    monkeypatch.setattr(control_preflight, 'BASE', tmp_path)
    monkeypatch.setattr(control_preflight, 'phase_artifact', lambda root, name, path: digest(path))
    monkeypatch.setattr(warm_scorer, 'validate_all', lambda *args, **kwargs:
        ({}, {i: {} for i in range(357)}, {}, {'manifest_sha256': 'manifest'}))
    with pytest.raises(ValueError, match='score or audit differs'):
        control_preflight.reference_study(root, tmp_path/'warm', {}, 'warm')
