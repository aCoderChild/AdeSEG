"""Contract checks for the separate, disabled three-fold fixed-read path."""
import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import bounded_crossfit_shared_worker_3fold_v1 as worker
from scripts import bounded_crossfit_fold_3fold_v1 as fold_module
from scripts import run_bounded_crossfit_shared_study_3fold_v1 as study
from scripts import score_bounded_crossfit_3fold_v1 as scoring
from scripts import score_bounded_crossfit_metrics_3fold_v1 as metrics
from scripts.crossfit_training_contract_3fold_v1 import fold_split, patient_balanced_schedule
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, SEED, EPOCHS, UPDATES_PER_PATIENT


def test_threefold_plan_has_three_training_and_eighteen_prediction_phases(tmp_path):
    phases = study.phases(tmp_path, tmp_path / 'registry', 512)
    names = [row[0] for row in phases]
    assert [n for n in names if n.startswith('train_')] == ['train_f1', 'train_f2', 'train_f3']
    assert len([n for n in names if n.startswith('predict_')]) == 3 * len(ARMS) == 18
    assert names[-3:] == ['verify', 'score', 'audit']
    prefix = [{'phase': n} for n in names[:-3]]
    scoring.check_phase_order(prefix, phases)
    with pytest.raises(ValueError, match='prediction phases'):
        scoring.check_phase_order([r for r in prefix if r['phase'] != 'predict_f3_mask_only'], phases)
    with pytest.raises(ValueError, match='prediction phases'):
        scoring.check_phase_order(prefix + [{'phase': 'score'}], phases, completed=('verify',))


def test_fold_definition_rejects_changed_patient_or_schedule():
    samples = [{'sequence': f'seq{i}', 'frame': 0} for i in range(1, 24) if i not in (1, 7)]
    rows = {(r['sequence'], r['frame']): r for r in samples}
    folds = []
    for fold in range(1, 4):
        train, validation, test = fold_split(fold)
        order = patient_balanced_schedule(samples, fold, SEED, EPOCHS, UPDATES_PER_PATIENT)
        folds.append({'fold': fold, 'train': sorted(train), 'validation': sorted(validation),
                      'test': sorted(test), 'training_order': order})
        assert not validation and not train & test
    registry = {'folds': folds}
    fold_module.fold_definition(registry, 1, rows)
    altered = copy.deepcopy(registry)
    altered['folds'][0]['train'].append(altered['folds'][0]['test'][0])
    with pytest.raises(ValueError, match='Fold identities'):
        fold_module.fold_definition(altered, 1, rows)
    with pytest.raises(ValueError, match='Invalid fold'):
        fold_module.fold_definition(registry, 4, rows)


def test_threefold_metric_routes_each_patient_to_its_heldout_fold(tmp_path, monkeypatch):
    names = ('seq2', 'seq6', 'seq5')
    assert len({next(i for i in range(1, 4) if seq in fold_split(i)[2]) for seq in names}) == 3
    samples, records = {}, {}
    for seq in names:
        label = tmp_path / f'{seq}_label.png'
        prediction = tmp_path / f'{seq}_prediction.png'
        Image.fromarray(np.ones((5, 6), dtype=np.uint8)).save(label)
        Image.fromarray(np.ones((5, 6), dtype=np.uint8) * 255).save(prediction)
        key = (seq, 0)
        samples[key] = {'label': str(label)}
        owner = next(i for i in range(1, 4) if seq in fold_split(i)[2])
        record = {'baseline_path': str(prediction), 'baseline_sha256': digest(prediction),
                  'refined_path': str(prediction), 'refined_sha256': digest(prediction)}
        for arm in ARMS:
            records.setdefault((owner, arm), {'mapping': {}})['mapping'][key] = record
    monkeypatch.setattr(metrics, 'label_hashes', lambda index, samples: {k: digest(v['label']) for k, v in samples.items()})
    monkeypatch.setattr(metrics, 'primary_metric', lambda mask, truth: {
        'present': True, 'dice': 1., 'iou': 1., 'boundary_f_5px': 1., 'false_alarm': False})
    result = metrics.compute({}, samples, records)
    for seq in names:
        assert result['rows'][seq][0]['fold'] == next(i for i in range(1, 4) if seq in fold_split(i)[2])
    assert result['summaries']['baseline']['patients_with_present_reads'] == 3


def test_training_probe_prediction_and_score_can_be_disabled(monkeypatch):
    from scripts import probe_bounded_crossfit_shared_3fold_v1 as probe
    from scripts import predict_bounded_crossfit_fold_3fold_v1 as prediction
    monkeypatch.setattr(study, 'ENABLE_REVIEWED_STUDY', False)
    monkeypatch.setattr(worker, 'ENABLE_REVIEWED_TRAINING', False)
    monkeypatch.setattr(probe, 'ENABLE_REVIEWED_PROBE', False)
    monkeypatch.setattr(prediction, 'ENABLE_REVIEWED_PREDICTION', False)
    monkeypatch.setattr(scoring, 'ENABLE_REVIEWED_SCORING', False)
    assert not study.ENABLE_REVIEWED_STUDY
    assert not worker.ENABLE_REVIEWED_TRAINING
    assert not probe.ENABLE_REVIEWED_PROBE
    assert not prediction.ENABLE_REVIEWED_PREDICTION
    assert not scoring.ENABLE_REVIEWED_SCORING


def test_threefold_protocol_is_in_frozen_source_identity():
    sources = worker.sources()
    assert 'protocols/BOUNDED_CROSSFIT_FIXED_READ_3FOLD_V1.md' in sources
