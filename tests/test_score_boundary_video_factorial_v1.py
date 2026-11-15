import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import score_boundary_video_factorial_v1 as scorer
from scripts.run_boundary_video_factorial_v1 import DETECTOR_SHA256
from scripts.memory_token_io import digest


def test_map_requires_exact_nonaliased_twelve_directories(tmp_path):
    data = {}
    for sequence in scorer.SEQUENCES:
        data[sequence] = {}
        for arm in scorer.ARMS:
            folder = tmp_path / sequence / arm
            folder.mkdir(parents=True)
            data[sequence][arm] = str(folder)
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(data))
    assert len(scorer.mapped_runs(path, tmp_path)) == 3
    data['seq20']['B'] = data['seq20']['A']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='distinct'):
        scorer.mapped_runs(path, tmp_path)
    path.write_text('{"seq11":{},"seq11":{}}')
    with pytest.raises(ValueError, match='Duplicate'):
        scorer.mapped_runs(path, tmp_path)


def test_metrics_and_summary_empty_and_present():
    truth = np.zeros((7, 7), dtype=bool)
    truth[1:3, 1:3] = True
    exact = scorer.metric(truth, truth)
    assert exact['dice'] == 1 and exact['iou'] == 1 and exact['precision'] == 1 and exact['recall'] == 1
    empty = scorer.metric(np.zeros_like(truth), np.zeros_like(truth))
    assert empty['dice'] == 1 and empty['false_alarm'] is False
    false = scorer.metric(truth, np.zeros_like(truth))
    assert false['false_alarm'] is True and false['predicted_pixels'] == 4
    missing = scorer.metric(np.zeros_like(truth), truth)
    assert missing['dice'] == 0 and missing['precision'] is None
    rows = [{'post_prompt': True, 'A': exact}, {'post_prompt': True, 'A': false},
            {'post_prompt': False, 'A': missing}]
    summary = scorer.summary(rows, 'A')
    assert summary['present_frames'] == 1 and summary['empty_frames'] == 1
    assert summary['false_alarms'] == 1 and summary['mean_present_dice'] == 1
    assert scorer.summary(rows, 'A', False)['zero_overlap_present_frames'] == 1


def test_paired_differences_exclude_empty_and_initialization():
    rows = [
        {'post_prompt': True, 'A': {'present': True, 'dice': 0.5}, 'B': {'present': True, 'dice': 0.75}},
        {'post_prompt': True, 'A': {'present': True, 'dice': 0.8}, 'B': {'present': True, 'dice': 0.7}},
        {'post_prompt': True, 'A': {'present': False, 'dice': 1.0}, 'B': {'present': False, 'dice': 0.0}},
        {'post_prompt': False, 'A': {'present': True, 'dice': 0.0}, 'B': {'present': True, 'dice': 1.0}},
    ]
    s = scorer.paired_summary(rows, 'B', 'A')
    assert s['present_frames'] == 2 and s['improved'] == 1 and s['worsened'] == 1
    assert s['mean_present_dice_difference'] == pytest.approx(0.075)


def test_scoring_waits_for_all_runs_before_loading_labels(monkeypatch, tmp_path):
    root = tmp_path / 'experiment'
    root.mkdir()
    runmap = tmp_path / 'map.json'
    runmap.write_text('{}')
    monkeypatch.setattr(scorer, 'mapped_runs', lambda *_: {'unverified': True})
    monkeypatch.setattr(scorer, 'load_manifest', lambda *_: {})
    def fail_verification(*_args):
        raise ValueError('unfrozen runs')
    def forbidden_label(*_args):
        raise AssertionError('label opened before freeze')
    monkeypatch.setattr(scorer, 'check_frozen_runs', fail_verification)
    monkeypatch.setattr(scorer, 'load_label_mask', forbidden_label)
    monkeypatch.setattr(scorer.sys, 'argv', ['score', '--experiment-root', str(root),
                                              '--run-map', str(runmap), '--output', str(root / 'out')])
    with pytest.raises(ValueError, match='unfrozen'):
        scorer.main()


def test_frozen_run_rejects_partial_and_changed_prompt(tmp_path):
    image = tmp_path / 'image.png'
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image)
    runner = tmp_path / 'runner.py'
    runner.write_text('runner')
    frame = SimpleNamespace(image_path=image, frame_index=0)
    baseline = tmp_path / 'lean_mps_baseline_seq11_full_attempt01'
    baseline.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(baseline / '000000.png')
    old = {'passed': True, 'full_video': True, 'sequence': 'seq11', 'prompt_frame': 0,
           'prompt': {'prompt_metadata_sha256': 'p', 'prompt_mask_sha256': 'm'},
           'identity': {'checkpoint_sha256': 'c', 'detector_sha256': DETECTOR_SHA256},
           'records': [{'frame': 0, 'source_frame_index': 0, 'image_sha256': digest(image),
                        'mask_sha256': digest(baseline / '000000.png')}]}
    (baseline / 'receipt.json').write_text(json.dumps(old))
    folder = tmp_path / 'arm'
    folder.mkdir()
    receipt = {'passed': True, 'full_video': False, 'limit': 1, 'sequence': 'seq11',
               'arm': 'A', 'runner_source_sha256': digest(runner), 'prompt_frame': 0,
               'prompt_sha256': 'p', 'prompt_mask_sha256': 'm', 'checkpoint_sha256': 'c',
               'detector_sha256': DETECTOR_SHA256}
    (folder / 'completion.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity, scope'):
        scorer.check_frozen_runs({'seq11': {'A': folder}}, {'seq11': [frame]}, tmp_path, runner)
    receipt.update(full_video=True, limit=0, prompt_sha256='wrong')
    (folder / 'completion.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity, scope'):
        scorer.check_frozen_runs({'seq11': {'A': folder}}, {'seq11': [frame]}, tmp_path, runner)


def test_runtime_fingerprint_and_telemetry_tampering_rejected(tmp_path):
    verification = tmp_path / 'verification'
    verification.mkdir()
    files = ('memory_events.json', 'positions.json', 'selections.json', 'telemetry.json')
    for name in files:
        (tmp_path / name).write_text('[]')
    for name in ('model', 'refiner'):
        (verification / f'{name}_tensor_fingerprints.json').write_text('{}')
    runtime = {'passed': True, 'candidate_checkpoint_sha256': 'candidate', 'device': 'mps float32',
               'fallback': 'disabled', 'fast_math': 'disabled', 'memory_budget': 16384,
               'insurance_reserve': 3072, 'baseline_replays': 2,
               'source_sha256': {'scripts/score_boundary_video_factorial_v1.py': digest(scorer.RUNNER.parent / 'score_boundary_video_factorial_v1.py')},
               **{f'{name}_tensor_fingerprints_sha256': digest(verification / f'{name}_tensor_fingerprints.json') for name in ('model', 'refiner')}}
    (verification / 'runtime_verification.json').write_text(json.dumps(runtime))
    receipt = {'candidate_checkpoint_sha256': 'candidate',
               'runtime_verification_sha256': digest(verification / 'runtime_verification.json'),
               **{name.replace('.json', '_sha256'): digest(tmp_path / name) for name in files}}
    scorer._check_events(tmp_path, receipt)
    (tmp_path / 'telemetry.json').write_text('[{"tampered": true}]')
    with pytest.raises(ValueError, match='telemetry'):
        scorer._check_events(tmp_path, receipt)
    (tmp_path / 'telemetry.json').write_text('[]')
    (verification / 'model_tensor_fingerprints.json').write_text('{"tampered": true}')
    with pytest.raises(ValueError, match='fingerprints'):
        scorer._check_events(tmp_path, receipt)


def test_conditioning_must_match_all_compact_state_fields():
    keys = ('mask_sha256', 'pred_masks_sha256', 'maskmem_features_sha256',
            'obj_ptr_sha256', 'object_score_logits_sha256')
    baseline = {key: 'fixed' for key in keys}
    scorer.check_conditioning(dict(baseline), baseline)
    for key in keys:
        altered = dict(baseline)
        altered[key] = 'changed'
        with pytest.raises(ValueError, match='Conditioning'):
            scorer.check_conditioning(altered, baseline)


def test_telemetry_coverage_and_values_checked_before_scoring(tmp_path):
    logit = {'min': -1024.0, 'max': -1024.0, 'positive_pixels': 0,
             'numel': 4, 'sha256': 'a' * 64}
    frame = {'frame': 1, 'conditioning': False, 'object_score_logit': -2.0,
             'native_object_present': False, 'native_exact_suppressed': True,
             'native': logit, 'refined': logit, 'output': logit, 'memory_high_res': logit}
    (tmp_path / 'telemetry.json').write_text(json.dumps([frame]))
    receipt = {'prompt_frame': 0, 'prediction_wall_seconds': 2.0}
    stats, timing = scorer.check_telemetry(tmp_path, receipt, 2)
    assert stats['native_absence_frames'] == 1 and stats['native_exact_suppressed_frames'] == 1
    assert timing['prediction_wall_seconds'] == 2.0
    with pytest.raises(ValueError, match='incomplete'):
        scorer.check_telemetry(tmp_path, receipt, 3)
    frame['object_score_logit'] = float('nan')
    (tmp_path / 'telemetry.json').write_text(json.dumps([frame]))
    with pytest.raises(ValueError, match='suppression'):
        scorer.check_telemetry(tmp_path, receipt, 2)
