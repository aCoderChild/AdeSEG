"""Synthetic freeze/phase and fail-closed held-out tests; no GPU or labels."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import predict_bounded_crossfit_fold_v1 as prediction
from scripts import run_bounded_crossfit_shared_study_v1 as supervisor
from scripts import score_bounded_crossfit_v1 as scoring
from scripts.memory_token_io import digest


def test_prediction_and_scoring_are_disabled_before_input_or_label_access(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, 'ENABLE_REVIEWED_PREDICTION', False)
    monkeypatch.setattr(scoring, 'ENABLE_REVIEWED_SCORING', False)
    with pytest.raises(ValueError, match='awaits independent review'):
        prediction.predict(None, 1, 'mask_only', {}, {}, tmp_path / 'prediction', {'open_attempts': 0})
    monkeypatch.setattr('sys.argv', ['score', '--mode', 'verify', '--batch-root', str(tmp_path),
                                   '--output', str(tmp_path / 'verify')])
    with pytest.raises(ValueError, match='awaits independent review'):
        scoring.main()
    assert not (tmp_path / 'prediction').exists() and not (tmp_path / 'verify').exists()


def test_all_30_predictions_must_precede_score_release(tmp_path):
    tasks = supervisor.phases(tmp_path, tmp_path / 'registry', 512)
    prefix = [{'phase': name} for name, _, _ in tasks if name not in ('verify', 'score', 'audit')]
    scoring.check_phase_order(prefix, tasks)
    scoring.check_phase_order(prefix + [{'phase': 'verify'}], tasks, completed=('verify',))
    missing = [row for row in prefix if row['phase'] != 'predict_f3_mask_only']
    with pytest.raises(ValueError, match='prediction phases required'):
        scoring.check_phase_order(missing, tasks)
    reordered = list(prefix); reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
    with pytest.raises(ValueError, match='prediction phases required'):
        scoring.check_phase_order(reordered, tasks)
    with pytest.raises(ValueError, match='prediction phases required'):
        scoring.check_phase_order(prefix + [{'phase': 'score'}], tasks, completed=('verify',))


def test_prediction_rejects_group_with_different_authorization_before_loading_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, 'BASE', tmp_path)
    folder = tmp_path / 'batch' / 'fold1'; folder.mkdir(parents=True)
    path = folder / 'group.json'
    path.write_text(json.dumps({'passed': True, 'fold': 1, 'authorization': {'handoff': 'other'},
                                'label_guard': {'blocked_open_attempts': 0},
                                'arms': {arm: {} for arm in prediction.ARMS}}))
    monkeypatch.setattr(prediction, 'phase_artifact', lambda *args: digest(path))
    with pytest.raises(ValueError, match='training group differs'):
        prediction.checked_fold(folder.parent, 1, 'mask_only', {'handoff': 'expected'}, {'samples': []})


def test_prediction_requires_final_cache_verification_before_loading_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, 'BASE', tmp_path)
    batch = tmp_path / 'batch'; folder = batch / 'fold1'; arm_dir = folder / 'mask_only'; arm_dir.mkdir(parents=True)
    (batch / 'run_manifest.json').write_text(json.dumps({'cache_mib': 1}))
    checkpoint = arm_dir / 'checkpoint.pt'; checkpoint.write_bytes(b'placeholder')
    guard = {'blocked_open_attempts': 0}; auth = {'handoff': 'expected'}
    cache = {'limit_bytes': 1024**2, 'peak_bytes': 100, 'hits': 0, 'misses': 1,
             'unique_reads': 1, 'final_sha256_verified': False}
    record = {'passed': True, 'fold': 1, 'arm': 'mask_only', 'authorization': auth, 'device': 'mps',
              'environment': {}, 'initial_state_sha256': 'init', 'training_steps': 1,
              'label_guard': guard, 'cache': cache, 'checkpoint_sha256': digest(checkpoint)}
    path = arm_dir / 'completion.json'; group_path = folder / 'group.json'
    def save():
        path.write_text(json.dumps(record))
        arms = {arm: {} for arm in prediction.ARMS}
        arms['mask_only'] = {'path': str(path), 'sha256': digest(path), 'checkpoint_sha256': digest(checkpoint)}
        group_path.write_text(json.dumps({'passed': True, 'fold': 1, 'authorization': auth,
                                         'label_guard': guard, 'cache': cache, 'arms': arms}))
    save()
    monkeypatch.setattr(prediction, 'phase_artifact', lambda *args: digest(group_path))
    monkeypatch.setattr(prediction, 'fold_definition', lambda *args: (set(), set(), set(), [[['seq2', 0]]]))
    registry = {'samples': [{'sequence': 'seq2', 'frame': i} for i in range(357)],
                'environment': {}, 'initial_state_sha256': {'mask_only': 'init'}}
    with pytest.raises(ValueError, match='training receipt or checkpoint'):
        prediction.checked_fold(batch, 1, 'mask_only', auth, registry)
    cache['final_sha256_verified'] = True; save()
    monkeypatch.setattr(prediction.torch, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('load reached')))
    with pytest.raises(RuntimeError, match='load reached'):
        prediction.checked_fold(batch, 1, 'mask_only', auth, registry)


@pytest.fixture
def complete_frozen_batch(tmp_path, monkeypatch):
    """Synthetic 357-read/30-prediction batch with tiny fake PNG decodes."""
    monkeypatch.setattr(scoring, 'BASE', tmp_path)
    monkeypatch.setattr(scoring.worker, 'sources', lambda: {})
    monkeypatch.setattr(scoring, 'phase_artifact', lambda _root, _name, path: digest(path))
    monkeypatch.setattr(scoring.verifier, 'verify', lambda _: {'marker': 'verified'})
    authorization = {'marker': 'authorized'}
    monkeypatch.setattr(scoring.worker, 'authorize', lambda *args: authorization)
    sequences = [f'seq{i}' for i in range(1, 24) if i not in (1, 7)]
    samples = []; source = {}
    for seq in sequences:
        count = 337 if seq == 'seq2' else 1
        source[seq] = [{'source_frame_index': i + 100} for i in range(count)]
        samples.extend({'sequence': seq, 'frame': i} for i in range(count))
    assert len(samples) == 357
    rows = {(item['sequence'], item['frame']): item for item in samples}
    registry_folder = tmp_path / 'registry'; registry_folder.mkdir()
    registry = {'samples': samples, 'environment': {'device': 'mps'}}
    (registry_folder / 'registry.json').write_text(json.dumps(registry))
    (registry_folder / 'completion.json').write_text('{}')
    index = {'source_frames': source}
    monkeypatch.setattr(scoring, 'validate_plan', lambda: index)
    def split(_registry, fold, _rows):
        test = {f'seq{i}' for i in range(1, 24) if (i - 1) % 5 + 1 == fold}
        return set(), set(), test, []
    monkeypatch.setattr(scoring, 'fold_definition', split)
    def checked(_root, fold, _arm, _authorization, _registry):
        return None, rows, split(None, fold, None)[2], f'group{fold}', f'train{fold}', f'checkpoint{fold}'
    monkeypatch.setattr(scoring, 'checked_fold', checked)
    class FakeImage:
        format = 'PNG'
        def __init__(self, path): self.path = path
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def convert(self, _mode):
            return np.zeros((720, 1280) if 'seq23_' in str(self.path) else (3, 4), dtype=np.uint8)
    monkeypatch.setattr(scoring.Image, 'open', lambda path: FakeImage(path))
    root = tmp_path / 'batch'; root.mkdir()
    tasks = supervisor.phases(root, registry_folder, 0)
    manifest = {'format': 'bounded_shared_crossfit_training_study_v1', 'registry_path': str(registry_folder),
                'registry_sha256': digest(registry_folder / 'registry.json'),
                'registry_completion_sha256': digest(registry_folder / 'completion.json'),
                'source_sha256': {}, 'cache_mib': 0, 'max_process_rss_bytes': 6 * 1024**3,
                'phases': [name for name, _, _ in tasks]}
    snapshot = root / 'source_snapshot'; snapshot.mkdir()
    (snapshot / 'snapshot_manifest.json').write_text(json.dumps({'source_sha256': {}}))
    manifest['source_snapshot_sha256'] = digest(snapshot / 'snapshot_manifest.json')
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    for name, _, artifact in tasks:
        if name in ('verify', 'score', 'audit'):
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if name == 'verification':
            value = {'marker': 'verified', 'label_guard': {'open_attempts': 0}}
        elif name == 'authorization':
            value = authorization | {'label_guard': {'open_attempts': 0}}
        elif name == 'probe':
            value = {'max_process_rss_bytes': 6 * 1024**3}
        elif name.startswith('predict_'):
            _, fold_text, *arm_parts = name.split('_'); fold = int(fold_text[1:]); arm = '_'.join(arm_parts)
            test = split(None, fold, None)[2]; records = []
            for (seq, frame) in sorted(rows):
                if seq not in test:
                    continue
                shape = [720, 1280] if seq == 'seq23' else [3, 4]
                row = {'sequence': seq, 'frame': frame, 'source_frame_index': frame + 100,
                       'video_shape': [720, 834] if seq == 'seq23' else shape, 'source_shape': shape}
                for kind in ('baseline', 'refined'):
                    path = artifact.parent / f'{seq}_{frame:06d}_{kind}.png'
                    path.write_bytes(f'{seq}:{frame}:{kind}'.encode())
                    row[kind + '_path'] = str(path); row[kind + '_sha256'] = digest(path)
                records.append(row)
            value = {'passed': True, 'format': 'bounded_crossfit_prediction_v1', 'fold': fold, 'arm': arm,
                     'authorization': authorization, 'group_sha256': f'group{fold}',
                     'training_completion_sha256': f'train{fold}', 'checkpoint_sha256': f'checkpoint{fold}',
                     'device': 'mps', 'environment': registry['environment'], 'label_guard': {'open_attempts': 0},
                     'rows': records}
        else:
            value = {'passed': True}
        artifact.write_text(json.dumps(value))
    ledger = [{'phase': name} for name, _, _ in tasks if name not in ('verify', 'score', 'audit')]
    (root / 'phases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger))
    return root, tasks, ledger


def test_full_freeze_validation_rejects_missing_duplicate_wrong_fold_and_source(complete_frozen_batch):
    root, tasks, _ = complete_frozen_batch
    _, samples, records, _ = scoring.validate_all(root)
    assert len(samples) == 357 and len(records) == 30
    name, _, path = next(task for task in tasks if task[0] == 'predict_f2_mask_only')
    original = json.loads(path.read_text())
    assert len(original['rows']) > 1
    for mutate in ('missing', 'duplicate', 'wrong_fold', 'source'):
        value = json.loads(json.dumps(original))
        if mutate == 'missing': value['rows'].pop()
        if mutate == 'duplicate': value['rows'][-1] = value['rows'][0]
        if mutate == 'wrong_fold': value['rows'][0]['sequence'] = 'seq3'
        if mutate == 'source': value['rows'][0]['source_frame_index'] = -1
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='coverage|source frame'):
            scoring.validate_all(root)
    path.write_text(json.dumps(original))


def test_full_freeze_validation_rejects_bad_baseline_and_seq23_geometry(complete_frozen_batch):
    root, tasks, _ = complete_frozen_batch
    _, _, path = next(task for task in tasks if task[0] == 'predict_f2_mask_only')
    value = json.loads(path.read_text()); row = value['rows'][0]
    image = Path(row['baseline_path']); image.write_bytes(b'different baseline')
    row['baseline_sha256'] = digest(image); path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Baseline differs'):
        scoring.validate_all(root)
    image.write_bytes(f"{row['sequence']}:{row['frame']}:baseline".encode())
    row['baseline_sha256'] = digest(image); path.write_text(json.dumps(value))
    seq23_fold = (23 - 1) % 5 + 1
    _, _, path23 = next(task for task in tasks if task[0] == f'predict_f{seq23_fold}_mask_only')
    value23 = json.loads(path23.read_text())
    next(row for row in value23['rows'] if row['sequence'] == 'seq23')['video_shape'] = [720, 1280]
    path23.write_text(json.dumps(value23))
    with pytest.raises(ValueError, match='Seq23 view'):
        scoring.validate_all(root)


def test_score_never_releases_labels_when_prediction_phase_missing(complete_frozen_batch, monkeypatch):
    root, _, ledger = complete_frozen_batch
    ledger.pop()
    (root / 'phases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger))
    monkeypatch.setattr(scoring, 'ENABLE_REVIEWED_SCORING', True)
    monkeypatch.setattr(scoring, 'assert_supervised', lambda *args: None)
    monkeypatch.setattr(scoring, 'install_phase_mask_guard', lambda _: {'active': True, 'open_attempts': 0})
    monkeypatch.setattr(scoring, 'compute', lambda *args, **kwargs: pytest.fail('Target scoring was reached'))
    monkeypatch.setattr('sys.argv', ['score', '--mode', 'score', '--batch-root', str(root),
                                   '--output', str(root / 'score')])
    with pytest.raises(ValueError, match='prediction phases required'):
        scoring.main()
    assert not (root / 'score').exists()
