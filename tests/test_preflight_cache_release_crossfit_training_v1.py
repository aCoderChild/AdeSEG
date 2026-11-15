"""Synthetic mixed-handoff tests: no GPU or target-label access."""
import copy
import json

import pytest

from scripts import preflight_cache_release_crossfit_training_v1 as preflight
from scripts.memory_token_io import digest


def make_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, 'BASE', tmp_path)
    monkeypatch.setattr(preflight, 'FRESH', ('seq6', 'seq11', 'seq20'))
    certificate = tmp_path / 'certificate.json'; certificate.write_text('{}')
    short_gate = tmp_path / 'short_gate.json'; short_gate.write_text('{}')
    memory_gate = tmp_path / 'memory_gate.json'; memory_gate.write_text('{}')
    warm_gate = tmp_path / 'warm_gate.json'; warm_gate.write_text('{}')
    seq6 = tmp_path / 'seq6.json'; seq6.write_text('{}')
    seq11 = tmp_path / 'seq11.json'; seq11.write_text('{}')
    eager_audit = tmp_path / 'seq20.json'; eager_audit.write_text('{}')
    original = {'passed': True, 'format': 'cache_release_crossfit_handoff_v1', 'training_authorized': False,
        'warm_capture_certificate_path': str(certificate), 'warm_capture_certificate_sha256': digest(certificate),
        'short_gate_path': str(short_gate), 'memory_gate_path': str(memory_gate), 'seq6_proof_path': str(seq6),
        'audits': {'seq6': {'path': str(warm_gate)}, 'seq11': {'path': str(seq11)},
                   'seq20': {'path': str(eager_audit)}}, 'observations': []}
    folder = tmp_path / 'handoff'; folder.mkdir()

    def save(value, authorized=False):
        path = folder / 'handoff.json'
        path.write_text(json.dumps(value | {'label_guard': {'open_attempts': 0}}))
        (folder / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': authorized,
            'handoff_sha256': digest(path), 'label_guard': {'open_attempts': 0}}))

    monkeypatch.setattr(preflight, 'validate_plan', lambda: {'existing_manifest_sha256': 'historical'})
    monkeypatch.setattr(preflight, 'rebuild_handoff', lambda *args: copy.deepcopy(original))
    monkeypatch.setattr(preflight, 'verify_certificate', lambda _: ({'warm_gate_path': str(warm_gate),
        'warm_gate_sha256': digest(warm_gate)}, digest(certificate)))
    monkeypatch.setattr(preflight, 'reconstruct_gate', lambda _: ({}, digest(warm_gate)))
    monkeypatch.setattr(preflight, 'load_prepared', lambda _: ({}, 'historical'))
    monkeypatch.setattr(preflight, 'structural_check', lambda *args: None)
    save(original)
    return folder, original, save


def test_handoff_reconstruction_rejects_tampered_rows_even_with_new_completion(tmp_path, monkeypatch):
    folder, original, save = make_fixture(tmp_path, monkeypatch)
    assert preflight.verified_inputs(folder)[3] == digest(folder / 'handoff.json')
    changed = copy.deepcopy(original)
    changed['observations'] = [{'sequence': 'seq11', 'feature_file': '/substituted'}]
    save(changed)
    with pytest.raises(ValueError, match='independent reconstruction'):
        preflight.verified_inputs(folder)


def test_handoff_is_explicitly_non_authorizing_and_requires_all_audits(tmp_path, monkeypatch):
    folder, original, save = make_fixture(tmp_path, monkeypatch)
    save(original, authorized=True)
    with pytest.raises(ValueError, match='non-training completion'):
        preflight.verified_inputs(folder)
    changed = copy.deepcopy(original); changed['audits'].pop('seq20')
    save(changed)
    with pytest.raises(ValueError, match='audit coverage'):
        preflight.verified_inputs(folder)


def test_projection_requires_seq11_sidecar_and_never_changes_handoff(monkeypatch):
    h = 'a' * 64
    plan = {'existing_manifest_sha256': h, 'feature_pipeline_sha256': h}
    certificate = {'warm_plan_sha256': h}
    row = {'sequence': 'seq11', 'reference_kind': 'matched_cache_release_reference',
           'seq6_proof_path': '/seq6', 'seq6_proof_sha256': h,
           'memory_gate_path': '/memory', 'memory_gate_sha256': h}
    for mode in ('reference', 'capture'):
        row[f'{mode}_sidecar_path'] = '/sidecar'; row[f'{mode}_sidecar_sha256'] = h
        for kind in ('setup', 'frame'):
            row[f'{mode}_{kind}_trace_path'] = '/trace'; row[f'{mode}_{kind}_trace_sha256'] = h
    handoff = {'index_plan_sha256': preflight.PLAN_SHA, 'existing_manifest_sha256': h,
        'existing_feature_pipeline_sha256': h, 'seq6_proof_path': '/seq6', 'seq6_proof_sha256': h,
        'memory_gate_path': '/memory', 'memory_gate_sha256': h, 'observations': [row]}
    before = copy.deepcopy(handoff)
    seen = {}
    def check(_plan, _old, aggregate, **kwargs):
        seen.update(aggregate)
        return {'checked': True}
    monkeypatch.setattr(preflight, 'validate_warm_index_structure', check)
    result = preflight.structural_check(plan, {}, handoff, certificate, '/cert', h, '/gate', h)
    assert result == {'checked': True} and handoff == before
    assert seen['observations'][0]['reference_kind'] == 'matched_warm_reference'
    handoff['observations'][0]['seq6_proof_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='proof, sidecar or trace'):
        preflight.structural_check(plan, {}, handoff, certificate, '/cert', h, '/gate', h)


def test_real_214_plus_143_structure_rejects_changed_bounded_view():
    h = 'a' * 64
    old = [{'sequence': 'seq2', 'frame': i, 'source_frame_index': i + 500, 'image_sha256': h,
            'feature': f'f{i}', 'feature_sha256': h, 'snapshot': f's{i}', 'snapshot_sha256': h}
           for i in range(214)]
    old_plan = [{'sequence': r['sequence'], 'frame': r['frame'], 'image_sha256': h,
                 'feature_path': r['feature'], 'feature_sha256': h,
                 'snapshot_path': r['snapshot'], 'snapshot_sha256': h} for r in old]
    seq6 = [{'frame': 0, 'source_frame_index': 11, 'image_sha256': h}]
    seq11 = [{'frame': i, 'source_frame_index': 1000 + i, 'image_sha256': h} for i in range(142)]
    view, roi = {'kind': 'full_source_frame'}, [0, 0, 100, 100]
    fresh = []
    for sequence, frames in (('seq6', seq6), ('seq11', seq11)):
        for frame in frames:
            row = {'sequence': sequence, **frame, 'label_metadata_locator': frame['frame'],
                   'view': view, 'roi_xyxy': roi,
                   'reference_kind': 'historical_warm_reference' if sequence == 'seq6' else 'matched_cache_release_reference'}
            for name in ('label_metadata', 'reference_receipt', 'capture_receipt', 'sequence_audit'):
                row[name + '_path'] = '/gate' if sequence == 'seq6' and name == 'sequence_audit' else '/' + name
                row[name + '_sha256'] = h
            for name in ('feature', 'snapshot'):
                row[name + '_file'] = '/' + name
                row[name + '_sha256'] = h
            if sequence == 'seq11':
                row.update(seq6_proof_path='/proof6', seq6_proof_sha256=h,
                           memory_gate_path='/memory', memory_gate_sha256=h)
                for mode in ('reference', 'capture'):
                    row[f'{mode}_sidecar_path'] = '/sidecar'; row[f'{mode}_sidecar_sha256'] = h
                    for kind in ('setup', 'frame'):
                        row[f'{mode}_{kind}_trace_path'] = '/trace'; row[f'{mode}_{kind}_trace_sha256'] = h
            fresh.append(row)
    plan = {'existing_manifest_sha256': h, 'feature_pipeline_sha256': h,
            'existing_reads': old_plan,
            'fresh_reads': [{'sequence': s, **f} for s, rows in (('seq6', seq6), ('seq11', seq11)) for f in rows],
            'source_frames': {'seq2': [{'frame': r['frame'], 'source_frame_index': r['source_frame_index'],
                                        'image_sha256': h} for r in old], 'seq6': seq6, 'seq11': seq11},
            'prompts': {s: {'view': view, 'prompt_provenance': {'roi_xyxy': roi}} for s in ('seq6', 'seq11')}}
    handoff = {'index_plan_sha256': preflight.PLAN_SHA, 'existing_manifest_sha256': h,
               'existing_feature_pipeline_sha256': h, 'seq6_proof_path': '/proof6', 'seq6_proof_sha256': h,
               'memory_gate_path': '/memory', 'memory_gate_sha256': h,
               'observations': fresh}
    old_manifest = {'samples': old, 'feature_pipeline_sha256': h}
    args = (plan, old_manifest, handoff, {'warm_plan_sha256': h}, '/certificate', h, '/gate', h)
    result = preflight.structural_check(*args)
    assert len(result['existing_keys']) == 214 and len(result['fresh_keys']) == 143
    handoff['observations'][1]['roi_xyxy'] = [1, 0, 100, 100]
    with pytest.raises(ValueError, match='view'):
        preflight.structural_check(*args)


def test_actual_eager_trainer_rejects_mixed_preflight(tmp_path, monkeypatch):
    from scripts import warm_crossfit_fold_training_v1 as eager

    monkeypatch.setattr(eager, 'BASE', tmp_path)
    monkeypatch.setattr(eager, 'sources', lambda: {'frozen_eager_source.py': 'e' * 64})
    folder = tmp_path / 'mixed'; folder.mkdir()
    plan = {'passed': True, 'training_authorized': False,
            'format': 'cache_release_warm_crossfit_training_preflight_v1',
            'source_sha256': {'frozen_eager_source.py': 'e' * 64, 'bounded_source.py': 'b' * 64},
            'index_plan_sha256': eager.PLAN_SHA, 'seed': eager.SEED, 'epochs': eager.EPOCHS,
            'updates_per_patient_per_epoch': eager.UPDATES_PER_PATIENT, 'arms': list(eager.ARMS),
            'initial_state_sha256': {arm: 'f' * 64 for arm in eager.ARMS},
            'optimizer': {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001},
            'loss': 'corrected CMSA boundary weighted BCE and IoU plus soft Dice'}
    path = folder / 'plan.json'; path.write_text(json.dumps(plan))
    (folder / 'completion.json').write_text(json.dumps({'passed': True, 'plan_sha256': digest(path),
                                                       'label_guard': {'open_attempts': 0}}))
    with pytest.raises(ValueError, match='preflight or sources changed'):
        eager.load_preflight(folder)


def test_eager_sampler_rejects_mixed_schema_even_if_source_maps_match(tmp_path, monkeypatch):
    from scripts import warm_crossfit_fold_training_v1 as eager

    monkeypatch.setattr(eager, 'BASE', tmp_path)
    source_map = {'same.py': 'a' * 64}
    monkeypatch.setattr(eager, 'sources', lambda: source_map)
    existing = tmp_path / 'historical'; existing.mkdir()
    manifest = existing / 'manifest.json'; manifest.write_text('{}')
    monkeypatch.setattr(eager, 'EXISTING', existing)
    index = {'checkpoint_sha256': 'checkpoint', 'existing_manifest_sha256': digest(manifest)}
    monkeypatch.setattr(eager, 'validate_plan', lambda: index)
    folder = tmp_path / 'mixed'; folder.mkdir()
    plan = {'passed': True, 'training_authorized': False,
            'format': 'cache_release_warm_crossfit_training_preflight_v1', 'source_sha256': source_map,
            'index_plan_sha256': eager.PLAN_SHA, 'seed': eager.SEED, 'epochs': eager.EPOCHS,
            'updates_per_patient_per_epoch': eager.UPDATES_PER_PATIENT, 'arms': list(eager.ARMS),
            'initial_state_sha256': {arm: 'f' * 64 for arm in eager.ARMS},
            'optimizer': {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001},
            'loss': 'corrected CMSA boundary weighted BCE and IoU plus soft Dice',
            'backbone_sha256': 'checkpoint', 'old_manifest_sha256': digest(manifest)}
    path = folder / 'plan.json'; path.write_text(json.dumps(plan))
    (folder / 'completion.json').write_text(json.dumps({'passed': True, 'plan_sha256': digest(path),
                                                       'label_guard': {'open_attempts': 0}}))
    loaded, _ = eager.load_preflight(folder)
    # Even if source equality were to arise in a future refactor, the eager
    # sampler cannot build rows without the separate eager aggregate contract.
    with pytest.raises(KeyError, match='aggregate_path'):
        eager.samples_for(loaded)
