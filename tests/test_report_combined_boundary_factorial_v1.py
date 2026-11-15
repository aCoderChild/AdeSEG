import copy
import json
import hashlib

import pytest

from scripts.report_combined_boundary_factorial_v1 import ARMS, EXPECTED, IDENTITY_KEYS, check_cross_study_identity, checked_study, paired_analysis, reaggregate


def row(frame, present, dice, pixels):
    m = dict(present=present, dice=dice, recall=dice if present else None,
             boundary_f_5px=dice, predicted_pixels=pixels,
             false_alarm=None if present else bool(pixels))
    return dict(frame=frame, post_prompt=True, **{arm: dict(m) for arm in ARMS})


def study():
    rows = [row(1, True, .5, 2), row(2, True, 0, 0), row(3, False, 1, 7)]
    summary = {'frames': 3, 'present_frames': 2, 'mean_present_dice': .25,
               'zero_overlap_present_frames': 1, 'empty_frames': 1, 'false_alarms': 1,
               'mean_empty_predicted_pixels': 7, 'mean_present_recall': .25,
               'mean_present_boundary_f_5px': .25}
    summ = {'seq1': {arm: {'post_prompt': copy.deepcopy(summary)} for arm in ARMS}}
    s = {'rows': {'seq1': rows}, 'summaries': summ}
    return s, {'summaries': copy.deepcopy(summ)}, {}


def test_reaggregate_checks_both_summaries_and_zero_overlap():
    s = study()
    videos, result = reaggregate([s], expected={'seq1'})
    assert videos['seq1']['A']['zero_overlap_present_frames'] == 1
    assert result['B']['equal_patient_mean_dice'] == .25
    assert result['D']['false_alarms'] == 1
    s[1]['summaries']['seq1']['A']['post_prompt']['false_alarms'] = 0
    with pytest.raises(ValueError, match='false_alarms'):
        reaggregate([s], expected={'seq1'})


def test_reject_duplicate_and_inconsistent_labels():
    s = study()
    with pytest.raises(ValueError, match='duplicate sequence'):
        reaggregate([s, s], expected={'seq1'})
    s[0]['rows']['seq1'][0]['B']['present'] = False
    with pytest.raises(ValueError, match='presence differs'):
        reaggregate([s], expected={'seq1'})


def test_frozen_score_must_match_independent_audit(tmp_path):
    s = study(); folder = tmp_path/'run'; (folder/'scores').mkdir(parents=True)
    score = {'passed': True, **s[0]}; path = folder/'scores/results.json'
    path.write_text(json.dumps(score))
    aud = tmp_path/'run_independent_audit_attempt01'; aud.mkdir()
    receipt = {'passed': True, 'error_count': 0, 'score_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **s[1]}
    (aud/'audit.json').write_text(json.dumps(receipt))
    checked_study(tmp_path, 'run')
    path.write_text(json.dumps(score) + ' ')
    with pytest.raises(ValueError, match='different score'):
        checked_study(tmp_path, 'run')


def test_cross_study_identity_binds_every_completion(tmp_path):
    s = study(); common = {key: 'a'*64 for key in IDENTITY_KEYS}
    for arm in ARMS:
        folder = tmp_path/'run'/'seq1'/arm; folder.mkdir(parents=True)
        (folder/'completion.json').write_text(json.dumps(dict(passed=True, sequence='seq1', arm=arm, full_video=True, limit=0, **common)))
    identity, receipts = check_cross_study_identity(tmp_path, ('run',), [s])
    assert identity == common and len(receipts) == 4
    path = tmp_path/'run'/'seq1'/'B'/'completion.json'
    changed = json.loads(path.read_text()); changed['candidate_checkpoint_sha256'] = 'b'*64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='identity differs'):
        check_cross_study_identity(tmp_path, ('run',), [s])


def test_patient_bootstrap_samples_patients_and_repeats_exactly():
    per = {seq: {a: {'mean_present_dice': .5 if a == 'A' else .6} for a in ARMS} for seq in EXPECTED}
    result = paired_analysis(per, replicates=100, seed=7)
    assert result == paired_analysis(per, replicates=100, seed=7)
    assert result['D']['mean_delta'] == pytest.approx(.1)
    assert result['D']['percentile_95_ci'] == pytest.approx([.1, .1])
    assert len(result['D']['subgroups']['priority_three']['sequences']) == 3
    assert len(result['D']['subgroups']['additional_seven']['sequences']) == 7
