import json
import sys

import pytest

from scripts import merge_lean_auto_multiscale_cohort as merge
from scripts import report_lean_auto_multiscale as report
from scripts.memory_token_io import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_merges_audited_partitions_and_rejects_changed_record(tmp_path, monkeypatch):
    names = [f'seq{i}' for i in range(2, 24) if i != 7]
    history = tmp_path / 'history'
    comparison = {'target_frames': 21, 'empty_frames': 0, 'native_dice': 0.6,
                  'lean_dice': 0.7, 'native_false_alarms': 0, 'lean_false_alarms': 0,
                  'sequences': [{'sequence': n, 'native_dice': 0.6, 'lean_dice': 0.7} for n in names]}
    write(history / 'comparison.json', comparison)
    write(history / 'independent_mask_audit.json', {'passed': True, 'comparison_sha256': digest(history / 'comparison.json')})
    parts = [tmp_path / 'a', tmp_path / 'b']
    row = {'frames': 1, 'present': 1, 'empty': 0, 'all_present_dice': 0.8,
           'all_false_alarms': 0, 'first_detection_index': 0, 'first_detection_labeled_present': True,
           'present_before_first_detection': 0, 'point_status': 'available', 'lean_mask_status': 'automatic_mask',
           'memory_events': 1, 'selection_events': 1, 'historical_present_dice': 0.8,
           'historical_false_alarms': 0, 'historical_present': 1, 'historical_empty': 0}
    for part, subset in zip(parts, (names[:10], names[10:])):
        write(part / 'setup.json', {'sequences': subset, 'signature': {'runner': 'same'}})
        for name in subset:
            run = part / name
            write(run / 'setup.json', {'signature': {'runner': 'same'}})
            write(run / 'records.json', [name])
            record_hash = digest(run / 'records.json')
            write(run / 'completion.json', {'passed': True, 'records_sha256': record_hash})
            write(run / 'independent_audit.json', {'passed': True, 'records_sha256': record_hash, 'sequences': {name: row}})
        write(part / 'results.json', {'sequences': {name: row for name in subset}})
        write(part / 'completion.json', {'passed': True, 'results_sha256': digest(part / 'results.json')})
    monkeypatch.setattr(merge, 'validate_output', lambda path: path)
    out = tmp_path / 'merged'
    monkeypatch.setattr(sys, 'argv', ['merge', '--part', str(parts[0]), '--part', str(parts[1]),
                                   '--historical', str(history), '--output', str(out)])
    merge.main()
    result = json.loads((out / 'results.json').read_text())
    assert result['all_present_frames'] == 21
    assert result['automatic_combined_historical_dice'] == pytest.approx(0.8)
    monkeypatch.setattr(report, 'validate_output', lambda path: path)
    monkeypatch.setattr(sys, 'argv', ['report', '--merged', str(out), '--historical', str(history)])
    report.main()
    assert 'Ground-truth mask' in (out / 'report.md').read_text()
    write(parts[0] / names[0] / 'records.json', ['changed'])
    monkeypatch.setattr(sys, 'argv', ['merge', '--part', str(parts[0]), '--part', str(parts[1]),
                                   '--historical', str(history), '--output', str(tmp_path / 'rejected')])
    with pytest.raises(ValueError, match='provenance mismatch'):
        merge.main()
