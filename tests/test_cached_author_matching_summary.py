import json
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.memory_token_io import digest
from scripts import summarize_cached_author_matching as analysis


@pytest.mark.parametrize('defect', ['presence', 'nan', 'range', 'absent_fp', 'integrity', 'time'])
def test_rejects_malformed_treatment_records(defect):
    target = {'frame': 3, 'present': True}
    value = {'present': True, 'scores': {'dice': .6, 'iou': .5, 'boundary_f': .7},
             'absent_fp': None, 'seconds': 1.}
    row = {'frame': 3, 'native_parity': True, 'bank_unchanged': True,
           'native_restored': True, 'rows': {'native': copy.deepcopy(value),
                                           'atc_regional': copy.deepcopy(value)}}
    treatment = row['rows']['atc_regional']
    if defect == 'presence':
        treatment['present'] = False
    elif defect == 'nan':
        treatment['scores']['dice'] = float('nan')
    elif defect == 'range':
        treatment['scores']['boundary_f'] = 1.1
    elif defect == 'absent_fp':
        target['present'] = False
        for result in row['rows'].values():
            result['present'] = False
            result['absent_fp'] = False
        treatment['absent_fp'] = None
    elif defect == 'integrity':
        row['native_parity'] = 'false'
    elif defect == 'time':
        treatment['seconds'] = float('inf')
    with pytest.raises(ValueError):
        analysis.validate_target_record(row, target, ['native', 'atc_regional'])


def test_screen_cannot_trade_false_alarms_or_missed_returns_for_dice(tmp_path, monkeypatch):
    protocol = json.loads((Path(__file__).resolve().parents[1] / 'protocols/author_matching_causal_cohort_v2.json').read_text())
    methods = ['native'] + protocol['conditions']
    annotation = tmp_path / 'mask.bin'
    annotation.write_bytes(b'frozen synthetic annotation')
    fake_frames = [SimpleNamespace(mask_path=annotation)] * 4
    videos = {name: fake_frames for name in protocol['sequences']}
    import datasets.video
    monkeypatch.setattr(datasets.video, 'load_manifest', lambda path, split: videos if split == 'dev' else {})
    monkeypatch.setattr(analysis, 'validate_output', Path)
    results, specifications = {}, {}
    base_dice = dict(native=.5, tome_regional=.51, tome_global=.52, atc_regional=.6, atc_global=.55)
    for sequence in protocol['sequences']:
        targets = []
        for frame, present in [(1, True), (2, False), (3, True)]:
            rows = {}
            for method in methods:
                missed_return = method == 'atc_regional' and sequence == 'seq20' and frame == 3
                dice = 0. if missed_return else base_dice[method]
                rows[method] = {'present': present, 'scores': {'dice': dice, 'iou': dice/(2-dice), 'boundary_f': dice},
                                'absent_fp': bool(method == 'atc_regional' and sequence == 'seq20') if not present else None}
            targets.append({'frame': frame, 'rows': rows, 'native_parity': True,
                            'bank_unchanged': True, 'native_restored': True})
        results[sequence] = targets
        specifications[sequence] = {'annotation_sha256': {'0': digest(annotation)},
            'schedule': {'targets': [{'frame': 1, 'present': True}, {'frame': 2, 'present': False},
                                      {'frame': 3, 'present': True}],
                         'return_events': [{'return_frame': 3, 'evaluated_followup': [3]}]}}
    plan = {'protocol': protocol, 'sequences': specifications,
            'counts': {'sequences': 21, 'targets': 63, 'present_targets': 42,
                       'gt_empty_targets': 21, 'reappearance_events': 21}}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'results.json').write_text(json.dumps(results))
    (folder / 'completion.json').write_text(json.dumps({'passed': True, 'scope': 'broader_causal_diagnostic',
        'plan_sha256': digest(plan_path), 'results_sha256': digest(folder / 'results.json')}))
    analysis.summarize(folder, plan_path)
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['decisions']['atc_global']['passes_diagnostic_screen']
    rejected = summary['decisions']['atc_regional']
    assert not rejected['passes_diagnostic_screen']
    assert 'false_alarm_safety' in rejected['failed_criteria']
    assert 'return_window_safety' in rejected['failed_criteria']
    assert summary['aggregate']['native']['gt_empty_targets'] == 21
    assert summary['aggregate']['atc_regional']['returns_with_overlap'] == 20
    assert summary['aggregate']['native']['returns_with_overlap'] == 21
    assert len(summary['contrasts']) == 5
    # Exercise the real summarize entry point, not just the record validator.
    treatment = results['seq2'][0]['rows']['atc_regional']
    for key, malformed, message in [('present', False, 'strata'),
                                     ('scores', {**treatment['scores'], 'dice': float('nan')}, 'Invalid present-target')]:
        original = treatment[key]
        treatment[key] = malformed
        (folder / 'results.json').write_text(json.dumps(results))
        completion = json.loads((folder / 'completion.json').read_text())
        completion['results_sha256'] = digest(folder / 'results.json')
        (folder / 'completion.json').write_text(json.dumps(completion))
        with pytest.raises(ValueError, match=message):
            analysis.summarize(folder, plan_path)
        treatment[key] = original
    (folder / 'results.json').write_text(json.dumps(results))
    completion['results_sha256'] = digest(folder / 'results.json')
    (folder / 'completion.json').write_text(json.dumps(completion))
    annotation.write_bytes(b'changed annotation')
    with pytest.raises(ValueError, match='annotations changed'):
        analysis.summarize(folder, plan_path)
