import json
import random

import numpy as np
import pytest
import torch

from scripts import warm_crossfit_fold_training_v1 as single
from scripts import warm_crossfit_shared_fold_v1 as shared
from scripts.memory_token_io import digest
from scripts.source_group_refiner_shared_v2 import initial_hash


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))

    def forward(self, early, deep, logits):
        noise = torch.rand(()) + random.random() + float(np.random.rand())
        return logits + self.weight * (early + noise * deep)


def test_shared_updates_equal_independent_workers_with_isolated_rng(tmp_path, monkeypatch):
    arms = ('mask_only', 'global', 'global_contrast', 'foreground_background_no_contrast',
            'foreground_background', 'foreground_background_asinh')
    monkeypatch.setattr(shared, 'ARMS', arms)
    rows = {}
    for frame in range(3):
        feature = tmp_path / f'f{frame}'; snapshot = tmp_path / f's{frame}'
        feature.write_bytes(bytes([frame + 1])); snapshot.write_bytes(bytes([frame + 11]))
        rows[('seq8', frame)] = {'sequence': 'seq8', 'frame': frame, 'origin': 'historical',
            'label_sha256': 'synthetic', 'feature': str(feature), 'feature_sha256': digest(feature),
            'snapshot': str(snapshot), 'snapshot_sha256': digest(snapshot)}
    order = [[('seq8', x) for x in (0, 2, 1, 0)], [('seq8', x) for x in (1, 0, 2, 1)]]
    for module in (single, shared):
        monkeypatch.setattr(module, 'samples_for', lambda p: ({'source_frames': {}}, rows))
        monkeypatch.setattr(module, 'fold_definition', lambda p, f, r: ({'seq8'}, set(), set(), order))
        monkeypatch.setattr(module, 'training_label_hashes', lambda *a: {})
        monkeypatch.setattr(module, 'model_for', lambda arm: StochasticModel())
        monkeypatch.setattr(module, 'cpu_inputs', lambda item: ((torch.tensor([float(item['frame'] + 1)]),
            torch.tensor([0.2]), torch.tensor([0.1])), (1, 1)))
        monkeypatch.setattr(module, 'checked_train_target', lambda *a: torch.tensor([0.7]))
        monkeypatch.setattr(module, 'to_video', lambda prediction, shape: prediction)
        monkeypatch.setattr(module, 'cmsa_boundary_loss', lambda p, t: (p - t).square().mean())
        monkeypatch.setattr(module, 'save_payload', lambda model, **kwargs: {'weight': model.weight.detach().clone()})
    shared.seed_arm('cpu')
    start = initial_hash(StochasticModel())
    plan = {'initial_state_sha256': {a: start for a in arms}, 'environment': {}}
    for arm in arms:
        single.train(plan, 'plan', 1, arm, tmp_path / ('single_' + arm),
                     {'blocked_open_attempts': 0}, device='cpu', cache_mib=0)
    shared.train_shared(plan, 'plan', 1, tmp_path / 'shared', {'blocked_open_attempts': 0},
                        device='cpu', cache_mib=0)
    group = json.loads((tmp_path / 'shared/group.json').read_text())
    assert group['peak_process_rss_bytes'] > 0
    for arm in arms:
        alone = tmp_path / ('single_' + arm)
        together = tmp_path / 'shared' / arm
        assert torch.equal(torch.load(alone / 'checkpoint.pt', weights_only=True)['weight'],
                           torch.load(together / 'checkpoint.pt', weights_only=True)['weight'])
        assert json.loads((alone / 'completion.json').read_text())['history'] == \
               json.loads((together / 'completion.json').read_text())['history']
        assert group['arms'][arm]['sha256'] == digest(together / 'completion.json')
    assert group['cache']['misses'] == sum(map(len, order))


def test_shared_rejects_same_model_for_multiple_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'ARMS', ('a', 'b'))
    monkeypatch.setattr(shared, 'samples_for', lambda p: ({'source_frames': {}}, {}))
    monkeypatch.setattr(shared, 'fold_definition', lambda *a: (set(), set(), set(), []))
    monkeypatch.setattr(shared, 'training_label_hashes', lambda *a: {})
    model = StochasticModel()
    monkeypatch.setattr(shared, 'model_for', lambda arm: model)
    initial = initial_hash(model)
    with pytest.raises(ValueError, match='Arm parameters are shared'):
        shared.train_shared({'initial_state_sha256': {'a': initial, 'b': initial}}, 'p', 1,
                            tmp_path / 'out', {'blocked_open_attempts': 0}, device='cpu', cache_mib=0)


def test_shared_study_order_and_registration_resume_preserves_models(tmp_path):
    from scripts import run_warm_crossfit_shared_v1 as launch
    tasks = launch.phases(tmp_path, tmp_path / 'plan', cache_mib=17)
    names = [x[0] for x in tasks]
    assert len(names) == 69
    assert names[0] == 'probe'
    assert all(x.startswith('train_shared_f') for x in names[1:6])
    assert all(x.startswith('train_f') for x in names[6:36])
    assert all(x.startswith('predict_f') for x in names[36:66])
    assert names[-3:] == ['verify', 'score', 'audit']
    phase, _, artifact = tasks[6]
    artifact.parent.mkdir(parents=True)
    artifact.write_text('frozen trained arm')
    log = tmp_path / (phase + '.log'); log.write_text('partial registration')
    launch.preserve_partial(tmp_path, phase, artifact)
    assert artifact.read_text() == 'frozen trained arm'
    assert not log.exists()
    assert len(list((tmp_path / 'partial_attempts').glob('*/phase.log'))) == 1
    phase, _, artifact = tasks[1]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    (artifact.parent / 'partial.pt').write_text('partial fold')
    launch.preserve_partial(tmp_path, phase, artifact)
    assert not artifact.parent.exists()
    assert len(list((tmp_path / 'partial_attempts').glob('*/output/partial.pt'))) == 1


def test_shared_registration_rejects_swapped_checkpoint(tmp_path, monkeypatch):
    from scripts import register_warm_crossfit_arm_v1 as reg
    arms = ('a', 'b')
    monkeypatch.setattr(reg, 'ARMS', arms)
    groupdir = tmp_path / 'train/fold1'; groupdir.mkdir(parents=True)
    entries = {}
    for arm in arms:
        folder = groupdir / arm; folder.mkdir()
        checkpoint = folder / 'checkpoint.pt'; checkpoint.write_bytes(arm.encode())
        rec = {'passed': True, 'fold': 1, 'arm': arm, 'preflight_sha256': 'p',
               'checkpoint_sha256': digest(checkpoint),
               'training_mode': 'shared_inputs_separate_models_optimizers_and_rng'}
        completion = folder / 'completion.json'; completion.write_text(json.dumps(rec))
        entries[arm] = {'path': str(completion), 'sha256': digest(completion),
                        'checkpoint_sha256': digest(checkpoint)}
    group = groupdir / 'group.json'
    group.write_text(json.dumps({'passed': True, 'fold': 1, 'preflight_sha256': 'p', 'arms': entries}))
    monkeypatch.setattr(reg, 'phase_artifact', lambda root, name, path: digest(path))
    assert reg.verify_arm(tmp_path, 'p', 1, 'a') == entries['a']['sha256']
    (groupdir / 'a/checkpoint.pt').write_bytes(b'b')
    with pytest.raises(ValueError, match='Shared arm artifacts differ'):
        reg.verify_arm(tmp_path, 'p', 1, 'a')


def test_actual_six_arm_models_do_not_mutate_shared_inputs():
    from scripts.channel_value_fixed_read_v1 import model_for
    arms = ('mask_only', 'global', 'global_contrast', 'foreground_background_no_contrast',
            'foreground_background', 'foreground_background_asinh')
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        values = (torch.randn(1, 32, 4, 4), torch.randn(1, 256, 2, 2), torch.randn(1, 1, 4, 4))
        original = tuple(x.clone() for x in values)
        for arm in arms:
            model = model_for(arm).train()
            model(*values).sum().backward()
            assert all(torch.equal(x, y) for x, y in zip(values, original))
    finally:
        torch.set_num_threads(old_threads)


def test_shared_all_fold_score_verification_and_group_tamper(tmp_path, monkeypatch):
    from PIL import Image
    from scripts import run_warm_crossfit_shared_v1 as launch
    from scripts import score_warm_crossfit_v1 as scorer
    from scripts import register_warm_crossfit_arm_v1 as reg
    arms = ('mask_only',)
    for module in (launch, scorer, reg): monkeypatch.setattr(module, 'ARMS', arms)
    monkeypatch.setattr(scorer, 'BASE', tmp_path)
    monkeypatch.setattr(scorer, 'sources', lambda: {})
    monkeypatch.setattr(scorer, 'trained_model', lambda *a, **k: None)
    fold_sequences = {1: ['seq6', 'seq23'], 2: ['seq2'], 3: ['seq8'], 4: ['seq12'], 5: ['seq18']}
    samples = {(seq, 0): {'sequence': seq, 'frame': 0} for seqs in fold_sequences.values() for seq in seqs}
    index = {'source_frames': {seq: [{'source_frame_index': 5}] for seq, _ in samples}}
    monkeypatch.setattr(scorer, 'samples_for', lambda _: (index, samples))
    monkeypatch.setattr(scorer, 'fold_definition', lambda plan, fold, rows:
                        (*scorer.fold_split(fold), [[['irrelevant', 1]]]))
    planpath = tmp_path / 'preflight'; planpath.mkdir()
    plan = {'environment': {}, 'initial_state_sha256': {'mask_only': 'initial'}}
    tasks = launch.phases(tmp_path, planpath, cache_mib=0)
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'preflight_path': str(planpath),
        'preflight_sha256': 'p', 'source_sha256': {}, 'cache_mib': 0,
        'trainer': 'shared_six_arm_v1', 'phases': [x[0] for x in tasks]}))
    checks = []
    (tmp_path / 'probe').mkdir()
    for seq in ('seq6', 'seq23'):
        file = tmp_path / 'probe' / f'{seq}_probe.pt'; file.write_bytes(b'probe')
        checks.append({'arm': 'mask_only', 'sequence': seq, 'frame': 0,
            'video_shape': [720, 834] if seq == 'seq23' else [4, 4], 'head_bias_gradient_abs': 1.0,
            'initial_state_sha256': 'initial', 'checkpoint_path': str(file),
            'checkpoint_sha256': digest(file), 'optimizer_updated': True, 'gradient_l2': 1.0,
            'synthetic_loss': 1.0})
    for fold in range(1, 6):
        groupdir = tmp_path / 'train' / f'fold{fold}'; armfolder = groupdir / 'mask_only'
        armfolder.mkdir(parents=True)
        checkpoint = armfolder / 'checkpoint.pt'; torch.save({'training_steps': 1}, checkpoint)
        completion = armfolder / 'completion.json'
        completion.write_text(json.dumps({'passed': True, 'fold': fold, 'arm': 'mask_only',
            'preflight_sha256': 'p', 'checkpoint_sha256': digest(checkpoint), 'training_steps': 1,
            'initial_state_sha256': 'initial', 'device': 'mps', 'environment': {},
            'training_mode': 'shared_inputs_separate_models_optimizers_and_rng',
            'label_guard': {'blocked_open_attempts': 0}, 'cache': {'limit_bytes': 0,
            'peak_bytes': 0, 'hits': 0, 'misses': 1, 'unique_reads': 1, 'final_sha256_verified': True}}))
        (groupdir / 'group.json').write_text(json.dumps({'passed': True, 'fold': fold,
            'preflight_sha256': 'p', 'arms': {'mask_only': {'path': str(completion),
            'sha256': digest(completion), 'checkpoint_sha256': digest(checkpoint)}}}))
    phase_rows = []
    for name, _, artifact in tasks[:-3]:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if name == 'probe':
            artifact.write_text(json.dumps({'passed': True, 'kind': 'shared_six_arm_probe_v1',
                'preflight_sha256': 'p', 'device': 'mps',
                'environment': {}, 'label_guard': {'open_attempts': 0}, 'checks': checks}))
        elif name.startswith('predict_'):
            fold = int(name.split('_')[1][1:]); train = tmp_path / 'train' / f'fold{fold}' / 'mask_only'
            pred_rows = []
            for seq in fold_sequences[fold]:
                source = [720, 1280] if seq == 'seq23' else [4, 4]
                row = {'sequence': seq, 'frame': 0, 'source_frame_index': 5,
                    'source_shape': source, 'video_shape': [720, 834] if seq == 'seq23' else source}
                for kind in ('baseline', 'refined'):
                    file = artifact.parent / f'{seq}_{kind}.png'
                    Image.fromarray(np.zeros(source, dtype=np.uint8)).save(file)
                    row[kind + '_path'] = str(file); row[kind + '_sha256'] = digest(file)
                pred_rows.append(row)
            artifact.write_text(json.dumps({'passed': True, 'fold': fold, 'arm': 'mask_only',
                'preflight_sha256': 'p', 'checkpoint_sha256': digest(train / 'checkpoint.pt'),
                'training_completion_sha256': digest(train / 'completion.json'), 'device': 'mps',
                'environment': {}, 'label_guard': {'open_attempts': 0}, 'rows': pred_rows}))
        log = tmp_path / (name + '.log'); log.write_text('done')
        phase_rows.append({'phase': name, 'passed': True, 'artifact_path': str(artifact),
            'artifact_sha256': digest(artifact), 'log_sha256': digest(log)})
    (tmp_path / 'phases.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in phase_rows))
    _, _, records, proof = scorer.validate_all(tmp_path, planpath, plan, 'p')
    assert len(records) == 5 and proof['patient_test_folds']['seq23'] == 1
    (tmp_path / 'train/fold1/mask_only/checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Shared arm artifacts differ'):
        scorer.validate_all(tmp_path, planpath, plan, 'p')


def test_shared_probe_exercises_update_rng_and_checkpoint(tmp_path, monkeypatch):
    from scripts import warm_crossfit_shared_probe_v1 as probe
    arms = ('a', 'b')
    monkeypatch.setattr(probe, 'ARMS', arms)
    monkeypatch.setattr(shared, 'ARMS', arms)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
            torch.nn.init.zeros_(self.head[-1].weight); torch.nn.init.zeros_(self.head[-1].bias)
        def forward(self, early, deep, logits):
            return logits + self.head(early)

    monkeypatch.setattr(shared, 'model_for', lambda arm: Tiny())
    monkeypatch.setattr(shared, 'to_video', lambda prediction, shape: prediction)
    monkeypatch.setattr(shared, 'cmsa_boundary_loss', lambda p, t: (p - t).square().mean())
    monkeypatch.setattr(probe, 'to_video', lambda prediction, shape: prediction)
    monkeypatch.setattr(probe, 'inputs', lambda row, device: ((torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)), row['shape']))
    monkeypatch.setattr(probe, 'save_payload', lambda model, **kwargs: {'weights': model.state_dict()})
    def load(payload, **kwargs):
        model = Tiny(); model.load_state_dict(payload['weights']); return model
    monkeypatch.setattr(probe, 'trained_model', load)
    start = initial_hash(Tiny())
    plan = {'initial_state_sha256': {a: start for a in arms}, 'environment': {}}
    rows = {('seq6', 1): {'shape': (4, 4)}, ('seq23', 2): {'shape': (720, 834)}}
    probe.probe(plan, 'p', rows, tmp_path / 'probe', {'open_attempts': 0}, device='cpu')
    receipt = json.loads((tmp_path / 'probe/completion.json').read_text())
    assert receipt['kind'] == 'shared_six_arm_probe_v1'
    assert len(receipt['checks']) == 4
    assert all(x['gradient_l2'] > 0 and x['optimizer_updated'] for x in receipt['checks'])
    assert all(digest(x['checkpoint_path']) == x['checkpoint_sha256'] for x in receipt['checks'])
    def altered(payload, **kwargs):
        model = load(payload, **kwargs)
        with torch.no_grad(): model.head[-1].bias.add_(1)
        return model
    monkeypatch.setattr(probe, 'trained_model', altered)
    with pytest.raises(ValueError, match='checkpoint output differs'):
        probe.probe(plan, 'p', rows, tmp_path / 'altered', {'open_attempts': 0}, device='cpu')
