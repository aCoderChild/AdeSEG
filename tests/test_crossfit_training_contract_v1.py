import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.crossfit_training_contract_v1 import (
    aligned_target, checked_aligned_training_target, fold_split, mask_path_sets,
    patient_balanced_schedule, restore_view_mask, training_label_hashes,
    validate_warm_index_structure,
)
from scripts.memory_token_io import digest
from scripts.preflight_exploratory_crossfit_capture_v2 import FOLDS, NOINIT


def samples():
    # Different frame counts ensure patient balancing is exercised.
    return [{'sequence': s, 'frame': i}
            for group in FOLDS for s in group if s not in NOINIT
            for i in range(2 if s == 'seq10' else 5)]


def write_manifest(tmp_path):
    rows = []
    for i in range(1, 24):
        path = tmp_path / f'seq{i}.png'
        path.write_bytes(b'guard-only')
        rows.append({'video_id': f'seq{i}', 'mask': str(path)})
    manifest = tmp_path / 'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    return manifest, rows


def test_folds_cover_23_patients_exactly_once_and_do_not_overlap():
    tests = []
    for fold in range(1, 6):
        train, validation, test = fold_split(fold)
        assert not (train & validation or train & test or validation & test)
        assert len(train | validation | test) == 23
        tests.extend(test)
    assert set(tests) == {f'seq{i}' for i in range(1, 24)}
    assert len(tests) == 23
    for bad in (0, 6, True, '1'):
        with pytest.raises(ValueError):
            fold_split(bad)


def test_schedule_is_reproducible_equal_by_patient_and_excludes_validation_test():
    rows = samples()
    train, validation, test = fold_split(2)
    first = patient_balanced_schedule(rows, 2, seed=418, epochs=3, updates_per_patient=7)
    assert first == patient_balanced_schedule(rows, 2, seed=418, epochs=3, updates_per_patient=7)
    assert first != patient_balanced_schedule(rows, 2, seed=419, epochs=3, updates_per_patient=7)
    for order in first:
        assert Counter(s for s, _ in order) == Counter({s: 7 for s in train if s not in NOINIT})
        assert not {s for s, _ in order} & (validation | test | set(NOINIT))
        assert all((s, f) in {(r['sequence'], r['frame']) for r in rows} for s, f in order)


def test_schedule_rejects_duplicate_missing_unknown_and_no_initialization():
    rows = samples()
    for bad in (rows + [rows[0]], rows + [{'sequence': 'seq42', 'frame': 0}],
                rows + [{'sequence': 'seq1', 'frame': 0}],
                [x for x in rows if x['sequence'] != 'seq8']):
        with pytest.raises(ValueError):
            patient_balanced_schedule(bad, 1, seed=0, epochs=1)


def test_schedule_continues_through_all_long_patient_reads_across_epochs():
    rows = [row for row in samples() if row['sequence'] != 'seq8']
    rows.extend({'sequence': 'seq8', 'frame': frame} for frame in range(25))
    orders = patient_balanced_schedule(rows, 1, seed=12, epochs=3, updates_per_patient=12)
    seq8 = [frame for epoch in orders for sequence, frame in epoch if sequence == 'seq8']
    assert len(seq8) == 36
    assert set(seq8) == set(range(25))


def test_fold_guard_blocks_validation_test_and_noinit_in_fresh_process(tmp_path):
    manifest, rows = write_manifest(tmp_path)
    train, validation, test = fold_split(1)
    allowed, blocked = mask_path_sets(manifest, train)
    assert len(allowed) + len(blocked) == 23
    code = '''\nimport sys\nfrom scripts.crossfit_training_contract_v1 import install_fold_training_guard\nstate=install_fold_training_guard(sys.argv[1],1)\nwith open(sys.argv[2],"rb") as f: assert f.read()==b"guard-only"\nfor p in sys.argv[3:]:\n    try:\n        open(p,"rb")\n        raise AssertionError("protected mask opened")\n    except PermissionError:\n        pass\nassert state["blocked_open_attempts"]==3\n'''
    # Fold 1: seq8 trains; seq2 validates; seq6 tests; seq7 is train-group but no-init.
    paths = [rows[i - 1]['mask'] for i in (8, 2, 6, 7)]
    result = subprocess.run([sys.executable, '-c', code, str(manifest), *paths],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'seq8' in train and 'seq2' in validation and 'seq6' in test


def test_guard_rejects_cross_group_alias_and_missing_population(tmp_path):
    manifest, rows = write_manifest(tmp_path)
    rows[1]['mask'] = rows[7]['mask']  # seq2 validation aliases seq8 training.
    manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    with pytest.raises(ValueError, match='guard sets'):
        mask_path_sets(manifest, fold_split(1)[0])
    manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows[:-1]))
    with pytest.raises(ValueError, match='guard sets'):
        mask_path_sets(manifest, fold_split(1)[0])


def test_seq23_target_crops_exact_fixed_viewport_without_resizing(tmp_path):
    pixels = np.zeros((720, 1280), dtype=np.uint8)
    pixels[10:20, 30:40] = 1  # Outside the frozen view.
    pixels[40:45, 500:505] = 1
    path = tmp_path / 'mask.png'
    Image.fromarray(pixels).save(path)
    tensor = aligned_target(path, 'seq23', (720, 834), (720, 1280))
    assert tensor.shape == (1, 1, 720, 834)
    assert tensor.sum().item() == 25
    assert torch.all(tensor[0, 0, 40:45, 54:59] == 1)
    with pytest.raises(ValueError, match='geometry'):
        aligned_target(path, 'seq23', (720, 1280), (720, 1280))
    with pytest.raises(ValueError, match='source geometry'):
        aligned_target(path, 'seq23', (720, 834), (720, 1279))


def test_non_seq23_target_requires_full_frame_geometry(tmp_path):
    path = tmp_path / 'mask.png'
    Image.fromarray(np.ones((9, 12), dtype=np.uint8)).save(path)
    assert aligned_target(path, 'seq20', (9, 12), (9, 12)).sum().item() == 108
    with pytest.raises(ValueError, match='capture-view'):
        aligned_target(path, 'seq20', (9, 11), (9, 12))


def test_training_hash_projection_enforces_fold_and_distinct_ordinal(tmp_path):
    metadata = tmp_path / 'metadata.json'
    metadata.write_text('{\n"label_sha256": "' + 'a' * 64 + '",\n'
                        '"label_sha256": "' + 'b' * 64 + '"\n}\n')
    source_frames = {'seq8': [
        {'frame': 0, 'source_frame_index': 7, 'image_sha256': 'image0'},
        {'frame': 1, 'source_frame_index': 12, 'image_sha256': 'image1'},
    ]}
    observation = {'sequence': 'seq8', 'frame': 1, 'source_frame_index': 12,
                   'image_sha256': 'image1', 'label_metadata_locator': 1,
                   'label_metadata_path': str(metadata), 'label_metadata_sha256': digest(metadata)}
    assert training_label_hashes(1, [observation], source_frames) == {('seq8', 1): 'b' * 64}
    for override in ({'sequence': 'seq6'}, {'label_metadata_locator': 12},
                     {'source_frame_index': 1}, {'image_sha256': 'wrong'}):
        with pytest.raises((PermissionError, ValueError)):
            training_label_hashes(1, [{**observation, **override}], source_frames)
    with pytest.raises(ValueError, match='Duplicate'):
        training_label_hashes(1, [observation, observation], source_frames)


def test_checked_training_target_rejects_wrong_label_bytes(tmp_path):
    path = tmp_path / 'mask.png'
    Image.fromarray(np.ones((2, 3), dtype=np.uint8)).save(path)
    assert checked_aligned_training_target(path, digest(path), 'seq8', (2, 3), (2, 3)).sum() == 6
    with pytest.raises(ValueError, match='before decoding'):
        checked_aligned_training_target(path, '0' * 64, 'seq8', (2, 3), (2, 3))


def test_seq23_restores_predictions_to_original_coordinates():
    view = np.zeros((720, 834), dtype=np.uint8)
    view[40:45, 54:59] = 1
    result = restore_view_mask(view, 'seq23', (720, 1280))
    assert result.shape == (720, 1280)
    assert result.sum() == 25 and result[40:45, 500:505].all()
    assert not result[:, :446].any()
    with pytest.raises(ValueError, match='geometry'):
        restore_view_mask(view, 'seq23', (720, 1279))
    with pytest.raises(ValueError, match='binary'):
        restore_view_mask(np.full((2, 3), 0.5), 'seq8', (2, 3))


def test_warm_index_binds_all_frozen_samples_gate_and_view():
    h = 'a' * 64
    old = [{'sequence': 'seq2', 'frame': i, 'source_frame_index': i + 500, 'image_sha256': h,
            'feature': f'f{i}', 'feature_sha256': h,
            'snapshot': f's{i}', 'snapshot_sha256': h} for i in range(214)]
    old_plan = [{'sequence': r['sequence'], 'frame': r['frame'], 'image_sha256': h,
                 'feature_path': r['feature'], 'feature_sha256': h,
                 'snapshot_path': r['snapshot'], 'snapshot_sha256': h} for r in old]
    frames = [{'frame': i, 'source_frame_index': i + 1000, 'image_sha256': h} for i in range(143)]
    fresh_plan = [{'sequence': 'seq6', **f} for f in frames]
    view = {'kind': 'full_source_frame'}; roi = [0, 0, 100, 100]
    fresh = []
    for frame in frames:
        row = {'sequence': 'seq6', **frame, 'label_metadata_locator': frame['frame'],
               'view': view, 'roi_xyxy': roi, 'reference_kind': 'historical_warm_reference',
               'warm_gate_path': '/gate', 'warm_gate_sha256': h,
               'certificate_path': '/certificate', 'certificate_sha256': h}
        for name in ('label_metadata', 'reference_receipt', 'capture_receipt', 'sequence_audit'):
            row[name + '_path'] = '/' + name
            row[name + '_sha256'] = h
        for name in ('feature', 'snapshot'):
            row[name + '_file'] = '/' + name
            row[name + '_sha256'] = h
        row['sequence_audit_path'] = '/gate'
        fresh.append(row)
    plan = {'existing_manifest_sha256': h, 'feature_pipeline_sha256': h,
            'existing_reads': old_plan, 'fresh_reads': fresh_plan,
            'source_frames': {'seq6': frames, 'seq2': [
                {'frame': r['frame'], 'source_frame_index': r['source_frame_index'], 'image_sha256': h}
                for r in old]},
            'prompts': {'seq6': {'view': view, 'prompt_provenance': {'roi_xyxy': roi}}}}
    manifest = {'samples': old, 'feature_pipeline_sha256': h}
    aggregate = {'passed': True, 'plan_sha256': h, 'index_plan_sha256': h,
                 'warm_preflight_sha256': h, 'warm_gate_sha256': h, 'warm_gate_path': '/gate',
                 'certificate_path': '/certificate', 'certificate_sha256': h,
                 'existing_manifest_sha256': h, 'existing_feature_pipeline_sha256': h,
                 'observations': fresh}
    def check(value):
        return validate_warm_index_structure(plan, manifest, value, index_sha256=h,
            warm_preflight_sha256=h, warm_gate_path='/gate', warm_gate_sha256=h,
            certificate_path='/certificate', certificate_sha256=h)
    assert len(check(aggregate)['existing_keys']) == 214
    assert len(check(aggregate)['fresh_keys']) == 143
    import copy
    for field, value in (('warm_gate_sha256', 'b' * 64), ('label_metadata_locator', 1234),
                         ('source_frame_index', 0), ('roi_xyxy', [1, 0, 100, 100]),
                         ('certificate_sha256', 'b' * 64), ('reference_kind', 'matched_warm_reference'),
                         ('sequence_audit_path', '/other')):
        changed = copy.deepcopy(aggregate)
        changed['observations'][0][field] = value
        with pytest.raises(ValueError):
            check(changed)
    changed = copy.deepcopy(aggregate)
    changed['observations'][1] = copy.deepcopy(changed['observations'][0])
    with pytest.raises(ValueError, match='identities'):
        check(changed)
