#!/usr/bin/env python3
"""Execute and audit disjoint automatic Lean cohort sequences on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'detector', 'output', 'gate-a', 'gate-b', 'memory-gate-a', 'memory-gate-b', 'historical'):
        p.add_argument('--' + flag, required=True, type=Path)
    p.add_argument('--sequence', action='append', required=True)
    a = p.parse_args()
    if len(set(a.sequence)) != len(a.sequence):
        raise ValueError('Duplicate sequence')
    gate_data = []
    gate_identity = []
    for gate in (a.gate_a, a.gate_b):
        meta = json.loads((gate / 'setup.json').read_text())
        receipt = json.loads((gate / 'completion.json').read_text())
        audit = json.loads((gate / 'independent_audit.json').read_text())
        if not receipt['passed'] or not audit['passed'] or digest(gate / 'records.json') != receipt['records_sha256'] or audit['records_sha256'] != receipt['records_sha256']:
            raise ValueError('Automatic gate incomplete or unaudited')
        gate_data.append(json.loads((gate / 'records.json').read_text()))
        gate_identity.append(meta['signature'])
    if gate_data[0] != gate_data[1] or gate_identity[0] != gate_identity[1]:
        raise ValueError('Automatic gate replay mismatch')
    gate_by_sequence = {row['sequence']: row for row in gate_data[0]}
    if set(gate_by_sequence) != {'seq3', 'seq20'}:
        raise ValueError('Required gate sequences missing')
    runner = ROOT / 'scripts/run_lean_auto_multiscale.py'
    audit_script = ROOT / 'scripts/audit_lean_auto_multiscale.py'
    if gate_identity[0]['runner_sha256'] != digest(runner):
        raise ValueError('Runner differs from gate')
    output = validate_output(a.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    setup = {'sequences': a.sequence, 'signature': gate_identity[0], 'auditor_sha256': digest(audit_script),
             'gate_records_sha256': [digest(g / 'records.json') for g in (a.gate_a, a.gate_b)],
             'historical_comparison_sha256': digest(a.historical / 'comparison.json')}
    (output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    completed = {}
    for name in a.sequence:
        if digest(runner) != setup['signature']['runner_sha256'] or digest(audit_script) != setup['auditor_sha256']:
            raise RuntimeError('Runner or auditor changed during cohort')
        run = output / name
        cmd = [sys.executable, str(runner), '--mode', 'cohort', '--checkpoint', str(a.checkpoint), '--detector', str(a.detector),
               '--memory-gate-a', str(a.memory_gate_a), '--memory-gate-b', str(a.memory_gate_b), '--gate-a', str(a.gate_a), '--gate-b', str(a.gate_b),
               '--output', str(run), '--sequence', name]
        print(f'Starting {name} ({len(completed)}/{len(a.sequence)} complete)', flush=True)
        with (output / 'logs' / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                result = subprocess.run([sys.executable, str(audit_script), '--run', str(run), '--historical', str(a.historical)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (output / 'progress.json').write_text(json.dumps({'completed': list(completed), 'failed': name, 'exit_code': result.returncode}, indent=2) + '\n')
            raise RuntimeError(f'{name} failed; inspect its log')
        record = json.loads((run / 'records.json').read_text())[0]
        if name in gate_by_sequence and record != gate_by_sequence[name]:
            raise ValueError(f'Full-cohort replay differs from gate: {name}')
        audit = json.loads((run / 'independent_audit.json').read_text())
        completed[name] = audit['sequences'][name]
        (output / 'progress.json').write_text(json.dumps({'completed': list(completed), 'per_sequence': completed}, indent=2) + '\n')
        print(f"Completed {name}: all-frame present Dice={completed[name]['all_present_dice']:.5f}", flush=True)
    (output / 'results.json').write_text(json.dumps({'sequences': completed}, indent=2) + '\n')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(output / 'results.json')}, indent=2) + '\n')
    print('Partition complete', flush=True)


if __name__ == '__main__':
    main()
