"""Read-only capture/path registry tests; no masks or GPU are opened."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_cache_release_crossfit_registry_3fold_v1 as registry
from scripts import verify_cache_release_crossfit_registry_3fold_v1 as verifier
from scripts.memory_token_io import digest


def inputs(tmp_path):
    old = []; fresh = []; videos = {}; source = {}; planned_old = []; planned_fresh = []
    for sequence, count, rows, plan_rows in (('seq2', 214, old, planned_old),
                                             ('seq11', 143, fresh, planned_fresh)):
        videos[sequence] = []; source[sequence] = []
        for frame in range(count):
            path = tmp_path / f'{sequence}-{frame}.png'
            videos[sequence].append(SimpleNamespace(frame_index=frame + 100, mask_path=path))
            source[sequence].append({'source_frame_index': frame + 100})
            plan_rows.append({'sequence': sequence, 'frame': frame})
            item = {'sequence': sequence, 'frame': frame, 'source_frame_index': frame + 100}
            if sequence == 'seq2':
                item.update(feature=f'/feature-{frame}', snapshot=f'/snapshot-{frame}', label=str(path))
            else:
                item.update(feature_file=f'/feature-{frame}', snapshot_file=f'/snapshot-{frame}')
            rows.append(item)
    index = {'existing_reads': planned_old, 'fresh_reads': planned_fresh, 'source_frames': source}
    return index, {'samples': old}, {'observations': fresh}, videos


def test_registry_maps_all_357_without_opening_mask_paths(tmp_path):
    index, old, handoff, videos = inputs(tmp_path)
    assert not Path(videos['seq2'][0].mask_path).exists()
    rows = registry.attach_rows(index, old, handoff, videos)
    assert len(rows) == 357
    assert rows[('seq2', 0)]['origin'] == 'historical'
    assert rows[('seq11', 0)]['origin'] == 'warm'
    assert rows[('seq11', 0)]['label'] == str(videos['seq11'][0].mask_path)


def test_registry_rejects_duplicate_wrong_label_and_source_index(tmp_path):
    index, old, handoff, videos = inputs(tmp_path)
    duplicate = copy.deepcopy(handoff)
    duplicate['observations'][-1] = copy.deepcopy(duplicate['observations'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        registry.attach_rows(index, old, duplicate, videos)
    bad_old = copy.deepcopy(old); bad_old['samples'][0]['label'] = str(tmp_path / 'wrong.png')
    with pytest.raises(ValueError, match='Label path'):
        registry.attach_rows(index, bad_old, handoff, videos)
    bad_fresh = copy.deepcopy(handoff); bad_fresh['observations'][0]['source_frame_index'] = -1
    with pytest.raises(ValueError, match='source frame'):
        registry.attach_rows(index, old, bad_fresh, videos)


def test_build_rejects_authorizing_preflight_and_swapped_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'BASE', tmp_path)
    data = tmp_path / 'dataset.jsonl'; data.write_text('manifest metadata only')
    monkeypatch.setattr(registry, 'DATA', data)
    source_map = {'fake_source': 'a' * 64}
    monkeypatch.setattr(registry, 'preflight_sources', lambda: source_map)
    monkeypatch.setattr(registry, 'sources', lambda: source_map)
    handoff = {'short_gate_sha256': 'short', 'seq6_proof_sha256': 'six', 'memory_gate_sha256': 'memory'}
    index = {'existing_manifest_sha256': 'old', 'feature_pipeline_sha256': 'pipeline',
             'checkpoint_sha256': 'backbone', 'data_manifest_sha256': digest(data)}
    old = {'feature_pipeline': {'runtime_environment': {'device': 'mps'}}}
    monkeypatch.setattr(registry, 'verified_inputs', lambda _: (index, old, handoff, 'handoff', 'cert', 'warm'))
    monkeypatch.setattr(registry, 'load_manifest', lambda *args: {})
    monkeypatch.setattr(registry, 'attach_rows', lambda *args: {('seq2', 0): {'sequence': 'seq2', 'frame': 0}})
    monkeypatch.setattr(registry, 'fold_split', lambda _: (frozenset({'seq2'}), frozenset(), frozenset()))
    monkeypatch.setattr(registry, 'patient_balanced_schedule', lambda *args: [])
    folds = [{'fold': i, 'train': ['seq2'], 'validation': [], 'test': [], 'training_order': []}
             for i in range(1, 4)]
    plan = {'passed': True, 'training_authorized': False, 'format': 'cache_release_warm_crossfit_training_preflight_3fold_v1',
            'label_guard': {'open_attempts': 0}, 'source_sha256': source_map, 'index_plan_sha256': registry.PLAN_SHA,
            'seed': registry.SEED, 'epochs': registry.EPOCHS,
            'updates_per_patient_per_epoch': registry.UPDATES_PER_PATIENT, 'arms': list(registry.ARMS),
            'handoff_path': '/mixed', 'handoff_sha256': 'handoff', 'certificate_sha256': 'cert',
            'warm_gate_sha256': 'warm', 'short_gate_sha256': 'short', 'seq6_proof_sha256': 'six',
            'memory_gate_sha256': 'memory', 'old_manifest_sha256': 'old', 'feature_pipeline_sha256': 'pipeline',
            'backbone_sha256': 'backbone', 'folds': folds,
            'initial_state_sha256': {arm: 'f' * 64 for arm in registry.ARMS},
            'optimizer': {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001},
            'loss': 'corrected CMSA boundary weighted BCE and IoU plus soft Dice', 'environment': {'device': 'mps'}}
    folder = tmp_path / 'preflight'; folder.mkdir()
    def save(value):
        path = folder / 'plan.json'; path.write_text(json.dumps(value))
        (folder / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': False,
            'label_guard': {'open_attempts': 0}, 'plan_sha256': digest(path)}))
    save(plan)
    assert registry.build(folder)['training_authorized'] is False
    save(plan | {'training_authorized': True})
    with pytest.raises(ValueError, match='preflight or source'):
        registry.build(folder)
    save(plan | {'handoff_sha256': 'swapped'})
    with pytest.raises(ValueError, match='inputs changed'):
        registry.build(folder)
    save(plan | {'environment': {'device': 'cpu'}})
    with pytest.raises(ValueError, match='environment metadata'):
        registry.build(folder)
    save(plan | {'optimizer': {'name': 'SGD'}})
    with pytest.raises(ValueError, match='optimizer'):
        registry.build(folder)


def test_cache_release_registry_verifier_reconstructs_exact_new_format(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'BASE', tmp_path)
    monkeypatch.setattr(verifier, 'sources', lambda: {'v': 'a' * 64})
    original = {'passed': True, 'training_authorized': False, 'format': 'cache_release_crossfit_registry_3fold_v1',
                'preflight_path': '/preflight', 'preflight_sha256': 'pre', 'preflight_completion_sha256': 'done',
                'handoff_sha256': 'handoff', 'seq6_proof_sha256': 's6', 'memory_gate_sha256': 'memory'}
    monkeypatch.setattr(verifier.registry, 'build', lambda _: copy.deepcopy(original))
    folder = tmp_path / 'registry'; folder.mkdir()
    def save(value):
        path = folder / 'registry.json'; path.write_text(json.dumps(value | {'label_guard': {'open_attempts': 0}}))
        (folder / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': False,
            'registry_sha256': digest(path), 'label_guard': {'open_attempts': 0}}))
    save(original)
    assert verifier.verify(folder)['format'] == 'cache_release_registry_verification_3fold_v1'
    save(original | {'memory_gate_sha256': 'tampered'})
    with pytest.raises(ValueError, match='reconstruction'):
        verifier.verify(folder)
    save(original | {'format': 'bounded_crossfit_registry_v1'})
    with pytest.raises(ValueError, match='registry or non-training completion'):
        verifier.verify(folder)
