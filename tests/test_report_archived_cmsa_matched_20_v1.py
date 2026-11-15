import copy
import pytest

from scripts.report_archived_cmsa_matched_20_v1 import SEQUENCES, compare


def fixture():
    hashes = {k: k for k in ('cmsa', 'verification', 'reconciliation', 'additional', 'priority', 'factorial')}
    metric = {'present': True, 'dice': .5, 'false_alarm': None}
    direct = {'frame': 2, 'source_frame_index': 6, 'post_prompt': True,
              'initialization_frame': False, **{k: dict(metric) for k in ('native','samurai','sam2long','lean','A','D')}}
    cmsa = dict(direct, cmsa=dict(metric))
    summary = {'present_frames': 1, 'empty_frames': 0, 'mean_present_dice': .5, 'false_alarms': 0}
    table = {s: {'native': summary, 'samurai': summary, 'sam2long': summary,
                 'lean': summary if s == 'seq11' else None,
                 'refiner': summary if s == 'seq11' else None} for s in SEQUENCES}
    data = {'cmsa': {'passed': True, 'verification_sha256': 'verification',
                     'rows': {s: [copy.deepcopy(cmsa)] for s in SEQUENCES},
                     'summaries': {s: dict(summary) for s in SEQUENCES}},
            'cmsa_audit': {'passed': True, 'error_count': 0, 'score_sha256': 'cmsa', 'verification_sha256': 'verification'},
            'verification': {'passed': True, 'reconciliation_sha256': 'reconciliation', 'prior_sha256': 'factorial'},
            'reconciliation': {'passed': True},
            'merged': {'passed': True, 'table': table, 'inputs_sha256': {'additional_reconciliation': 'reconciliation',
                       'priority_factorial_score': 'factorial', 'additional_score': 'additional', 'priority_four_score': 'priority'}},
            'additional': {'passed': True, 'rows': {s: [copy.deepcopy(direct)] for s in SEQUENCES}},
            'priority': {'passed': True, 'rows': {s: [copy.deepcopy(direct)] for s in ('seq11','seq20')}},
            'factorial': {'passed': True, 'rows': {s: [copy.deepcopy(direct)] for s in ('seq11','seq20')}}}
    return data, hashes


def test_matched_cohorts_and_frame_identity():
    data, hashes = fixture()
    result = compare(data, hashes)
    assert len(result['all_20']['sequences']) == 20
    assert result['all_six']['sequences'] == ['seq11']
    assert result['all_20']['metrics']['cmsa']['present_frames'] == 20
    data['cmsa']['rows']['seq20'][0]['source_frame_index'] = 7
    with pytest.raises(ValueError, match='frame identity'):
        compare(data, hashes)


def test_presence_and_proof_mismatch_fail():
    data, hashes = fixture()
    data['cmsa']['rows']['seq11'][0]['cmsa']['present'] = False
    with pytest.raises(ValueError, match='label presence'):
        compare(data, hashes)
    data, hashes = fixture()
    hashes['additional'] = 'tampered'
    with pytest.raises(ValueError, match='Upstream score identity'):
        compare(data, hashes)
    data, hashes = fixture()
    data['merged']['table']['seq11']['refiner'] = dict(data['merged']['table']['seq11']['refiner'], mean_present_dice=.8)
    with pytest.raises(ValueError, match='score mean mismatch'):
        compare(data, hashes)
