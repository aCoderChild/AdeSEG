import json

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import warm_crossfit_video_contract_v1 as contract
from scripts import warm_crossfit_fold_training_v1 as worker
from scripts.crossfit_training_contract_v1 import fold_split
from scripts.memory_token_io import digest


def test_heldout_owner_and_no_initialization():
    for i in range(1, 24):
        seq = f'seq{i}'; fold = contract.heldout_fold(seq)
        train, val, test = fold_split(fold)
        assert seq in test and seq not in train and seq not in val
    assert contract.NO_INITIALIZATION == {'seq1', 'seq7'}
    with pytest.raises(ValueError): contract.heldout_fold('seq24')


def test_wrong_fold_checkpoint_is_rejected(tmp_path, monkeypatch):
    seq = 'seq23'; fold = contract.heldout_fold(seq); arm = contract.PROPOSED_ARM
    definitions = []
    for n in range(1, 6):
        train, val, test = fold_split(n)
        definitions.append({'fold': n, 'train': sorted(train), 'validation': sorted(val), 'test': sorted(test)})
    plan = {'folds': definitions}; folder = tmp_path / 'train' / f'fold{fold}' / arm; folder.mkdir(parents=True)
    checkpoint = folder / 'checkpoint.pt'; torch.save({'training_steps': 1, 'crossfit_fold': fold}, checkpoint)
    rec = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': 'plan',
        'checkpoint_sha256': digest(checkpoint), 'training_steps': 1, 'label_guard': {'blocked_open_attempts': 0}}
    receipt = folder / 'completion.json'; receipt.write_text(json.dumps(rec))
    monkeypatch.setattr(contract, 'phase_artifact', lambda batch, phase, path: digest(path))
    monkeypatch.setattr(worker, 'trained_model', lambda payload, **kwargs: object())
    monkeypatch.setattr(worker, 'load_preflight', lambda path: (plan, 'plan'))
    _, proof = contract.load_heldout_checkpoint(tmp_path, 'folder', plan, 'plan', seq)
    assert proof['fold'] == fold
    rec['fold'] = 2 if fold != 2 else 1; receipt.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='checkpoint receipt differs'):
        contract.load_heldout_checkpoint(tmp_path, 'folder', plan, 'plan', seq)
    with pytest.raises(ValueError, match='No initialization'):
        contract.load_heldout_checkpoint(tmp_path, 'folder', plan, 'plan', 'seq1')
    with pytest.raises(ValueError, match='preflight differs'):
        contract.load_heldout_checkpoint(tmp_path, 'folder', {'folds': []}, 'plan', seq)


def test_reference_rejects_tampered_or_nonbinary_mask(tmp_path):
    path = tmp_path / '000000.png'; Image.fromarray(np.zeros((4, 5), np.uint8)).save(path)
    ref = {'passed': True, 'mode': 'reference', 'sequence': 'seq3', 'prompt_frame': 0,
        'conditioning_sha256': 'prompt', 'identity': {'model_state_sha256': 'model'},
        'label_guard': {'open_attempts': 0}, 'rows': [{'frame': 0, 'mask_sha256': digest(path)}],
        'positions': [], 'selection': [], 'anchor_events': [], 'token_window_events': []}
    receipt = tmp_path / 'receipt.json'; receipt.write_text(json.dumps(ref))
    def check():
        return contract.validate_reference('seq3', receipt, digest(receipt), expected_frames=1,
            expected_start=0, expected_mask_sha='prompt', expected_state_sha='model', expected_shapes=[(5, 4)])
    check()
    Image.fromarray(np.full((4, 5), 127, np.uint8)).save(path)
    with pytest.raises(ValueError, match='mask differs'): check()
    ref['rows'][0]['mask_sha256'] = digest(path); receipt.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match='encoding differs'): check()


def test_manifest_object_must_match_bound_file(tmp_path, monkeypatch):
    folder = tmp_path / 'existing'; folder.mkdir()
    (folder / 'manifest.json').write_text(json.dumps({'sequences': {'seq3': {}}}))
    monkeypatch.setattr(contract, 'EXISTING', folder)
    expected = digest(folder / 'manifest.json')
    with pytest.raises(ValueError, match='manifest object differs'):
        contract.old_sequence_inputs('seq3', {'sequences': {'seq3': {'injected': True}}}, expected)


def test_fresh_seq23_viewport_reference_and_outside_pixels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_crossfit_warm_video_v2 as video
    folder = tmp_path / 'ref'; folder.mkdir()
    imagepath = tmp_path / 'frame.jpg'; Image.fromarray(np.zeros((720, 1280, 3), np.uint8)).save(imagepath)
    mask = np.zeros((720, 834), bool); mask[2:5, 2:5] = True
    conditioning = folder / 'conditioning_mask.png'; Image.fromarray(mask.astype(np.uint8)*255).save(conditioning)
    saved = folder / '000000.png'; Image.fromarray(np.zeros((720, 1280), np.uint8)).save(saved)
    record = {'passed': True, 'sequence': 'seq23', 'mode': 'reference', 'prompt_frame': 0,
        'conditioning_sha256': digest(conditioning), 'model_state_sha256': 'model',
        'label_guard': {'open_attempts': 0}, 'rows': [{'frame': 0, 'mask_sha256': digest(saved)}],
        'positions': [], 'selection': [], 'anchor_events': [], 'token_window_events': []}
    path = folder / 'receipt.json'; path.write_text(json.dumps(record))
    row = {'sequence': 'seq23', 'roi_xyxy': [446, 0, 1280, 720], 'reference_kind': 'matched_warm_reference',
           'reference_receipt_path': str(path), 'reference_receipt_sha256': digest(path)}
    monkeypatch.setattr(contract, 'phase_artifact', lambda root, name, path: digest(path))
    frames = [SimpleNamespace(image_path=imagepath)]
    monkeypatch.setattr(video, 'video_inputs', lambda *args: (frames, [{'image_sha256': digest(imagepath)}],
        0, [], mask, [0, 0, 10, 10], [[1, 1], [2, 2]], [446, 0, 1280, 720]))
    warm = {'old_identity': {'model_state_sha256': 'model'}}
    result = contract.fresh_sequence_inputs('seq23', {'observations': [row]}, {}, {}, warm)
    assert result['roi_xyxy'] == [446, 0, 1280, 720] and result['image_sha256'] == [digest(imagepath)]
    bad = np.zeros((720, 1280), np.uint8); bad[5, 0] = 255; Image.fromarray(bad).save(saved)
    record['rows'][0]['mask_sha256'] = digest(saved); path.write_text(json.dumps(record))
    row['reference_receipt_sha256'] = digest(path)
    with pytest.raises(ValueError, match='escaped viewport'):
        contract.fresh_sequence_inputs('seq23', {'observations': [row]}, {}, {}, warm)


def test_video_runner_rejects_unreviewed_full_execution(tmp_path, monkeypatch):
    from scripts.run_warm_crossfit_video_v1 import run
    from scripts import crossfit_observer_capture_shared_v1 as shared
    monkeypatch.setattr(shared, 'phase_artifact', lambda root, phase, path: 'hash')
    with pytest.raises(ValueError, match='requires a phase-bound smoke audit'):
        run(tmp_path / 'certificate/certificate.json', 'hash', 'seq23', 'B',
            tmp_path / 'full/seq23/B', {'open_attempts': 0}, limit=0)


def test_video_study_freezes_all_patients_before_scoring(tmp_path):
    from scripts.run_warm_crossfit_video_study_v1 import phases
    tasks = phases(tmp_path, tmp_path / 'crossfit', tmp_path / 'preflight')
    names = [x[0] for x in tasks]
    assert len(names) == 74 and names[0] == 'certificate'
    assert names[7] == 'smoke_audit'
    assert len([x for x in names if x.startswith('full_')]) == 63
    assert names[-3:] == ['verify', 'score', 'audit']
    assert all('seq1_' not in x and 'seq7_' not in x for x in names if x.startswith('full_'))


def test_warm_mask_mismatch_retains_diagnostic(tmp_path):
    from scripts.run_warm_crossfit_video_v1 import compare_baseline_mask
    refdir = tmp_path / 'reference'; refdir.mkdir()
    ref = refdir / 'receipt.json'; ref.write_text('{}')
    mask = refdir / '000000.png'; Image.fromarray(np.zeros((3, 3), np.uint8)).save(mask)
    record = {'_receipt_path': str(ref), 'rows': [{'frame': 0, 'mask_sha256': digest(mask)}]}
    observed = np.zeros((3, 3), bool); observed[0, 0] = True
    diagnostic = tmp_path / 'diagnostic.json'
    with pytest.raises(ValueError, match='diagnostic preserved'):
        compare_baseline_mask(observed, record, 0, diagnostic)
    saved = json.loads(diagnostic.read_text())
    assert saved['differing_pixels'] == 1 and 'not cross-device parity' in saved['scope']


def test_seq6_legacy_reference_uses_phase_bound_later_conditioning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_crossfit_warm_video_v2 as video
    framefile = tmp_path / 'frame.jpg'; Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(framefile)
    saved = np.zeros((4, 5), np.uint8); saved[1, 1] = 255
    refdir = tmp_path / 'ref'; refdir.mkdir()
    refmask = refdir / '000000.png'; Image.fromarray(saved).save(refmask)
    reference = {'passed': True, 'sequence': 'seq6', 'mode': 'reference', 'prompt_frame': 0,
        'prompt_sha256': 'prompt', 'identity': {'model_state_sha256': 'model'},
        'rows': [{'frame': 0, 'mask_sha256': digest(refmask)}], 'positions': [], 'selection': [],
        'anchor_events': [], 'token_window_events': []}
    ref = refdir / 'receipt.json'; ref.write_text(json.dumps(reference))
    gatedir = tmp_path / 'batch/seq6/capture'; gatedir.mkdir(parents=True)
    conditioning = gatedir / 'conditioning_mask.png'; Image.fromarray(saved).save(conditioning)
    gate_record = {'passed': True, 'sequence': 'seq6', 'mode': 'capture',
        'conditioning_sha256': digest(conditioning), 'label_guard': {'open_attempts': 0},
        **{key: reference[key] for key in ('rows', 'positions', 'selection', 'anchor_events', 'token_window_events')}}
    gatepath = gatedir / 'receipt.json'; gatepath.write_text(json.dumps(gate_record))
    row = {'sequence': 'seq6', 'roi_xyxy': [0, 0, 5, 4], 'reference_kind': 'historical_warm_reference',
        'reference_receipt_path': str(ref), 'reference_receipt_sha256': digest(ref)}
    cert = {'seq6_historical_reference': {'reference_path': str(ref), 'reference_sha256': digest(ref)},
            'seq6_gate_capture': {'receipt_path': str(gatepath), 'receipt_sha256': digest(gatepath)}}
    warm = {'old_identity': {'model_state_sha256': 'model'},
            'fresh_prompts': {'seq6': {'prompt_metadata_sha256': 'prompt'}}}
    monkeypatch.setattr(contract, 'phase_artifact', lambda root, name, path: digest(path))
    monkeypatch.setattr(video, 'video_inputs', lambda *args: ([SimpleNamespace(image_path=framefile)],
        [{'image_sha256': digest(framefile)}], 0, [], saved > 0, [0, 0, 3, 3], [[1, 1], [2, 2]], [0, 0, 5, 4]))
    result = contract.fresh_sequence_inputs('seq6', {'observations': [row]}, cert, {}, warm)
    assert result['reference_sha256'] == digest(ref)
    gate_record['label_guard']['open_attempts'] = 1; gatepath.write_text(json.dumps(gate_record))
    cert['seq6_gate_capture']['receipt_sha256'] = digest(gatepath)
    with pytest.raises(ValueError, match='historical trajectory or gate differs'):
        contract.fresh_sequence_inputs('seq6', {'observations': [row]}, cert, {}, warm)
