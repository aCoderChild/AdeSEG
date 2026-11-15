"""Check disjoint cohort accounting and receipt verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_merge_accepts_complete_disjoint_partitions_and_checks_receipts(tmp_path):
    names = [f'seq{i}' for i in range(2, 24) if i != 7]
    common = {'device': 'cpu fp32', 'runner_sha256': 'runner', 'checkpoint_sha256': 'checkpoint',
              'cmsa_setup_sha256': 'setup', 'cmsa_frames_sha256': 'frames'}
    parts = []
    for number, subset in enumerate((names[:10], names[10:])):
        part = tmp_path / f'part{number}'
        part.mkdir()
        parts.append(part)
        save(part / 'setup.json', {**common, 'sequences': subset})
        values = {}
        for name in subset:
            run = part / name
            run.mkdir()
            save(run / 'records.json', [{'index': 0}])
            save(run / 'completion.json', {'passed': True, 'records_sha256': sha(run / 'records.json')})
            save(run / 'setup.json', {'sam2_checkpoint_sha256': 'checkpoint', 'runner_sha256': 'runner',
                 'cmsa_setup_sha256': 'setup', 'cmsa_frames_sha256': 'frames', 'device': 'cpu fp32'})
            row = {'cmsa_present_dice': 0.8, 'sasvi_present_dice': 0.6, 'empty': 1,
                   'cmsa_false_alarms': 1, 'sasvi_false_alarms': 0}
            values[name] = row
            save(run / 'audit_results.json', {**row, 'frames': []})
        save(part / 'results.json', {'per_sequence': values})
        save(part / 'completion.json', {'passed': True, 'results_sha256': sha(part / 'results.json')})
    command = [sys.executable, str(ROOT / 'scripts/merge_sasvi_cohort_partitions.py'),
               '--part', str(parts[0]), '--part', str(parts[1])]
    output = tmp_path / 'merged'
    subprocess.run([*command, '--output', str(output)], check=True, capture_output=True)
    result = json.loads((output / 'results.json').read_text())
    assert result['sequences'] == 21
    assert abs(result['sasvi_equal_sequence_dice'] - 0.6) < 1e-12
    assert result['empty_targets'] == 21
    (parts[0] / names[0] / 'records.json').write_text('changed')
    failed = subprocess.run([*command, '--output', str(tmp_path / 'rejected')], capture_output=True)
    assert failed.returncode != 0
