#!/usr/bin/env python3
"""Run fixed-100% Lean retention diagnostics in isolated video processes."""
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
    for name in ('checkpoint','merged','historical','gate-a','gate-b','dense-gate-a','dense-gate-b','verification','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--sequence', action='append', required=True)
    a = p.parse_args()
    if len(set(a.sequence)) != len(a.sequence): raise ValueError('Duplicate sequence')
    verification = json.loads(a.verification.read_text())
    if (not verification['passed'] or verification['source_sha256'] != digest(ROOT / 'scripts/verify_lean_dense_engineering.py')
        or verification['noop_records_sha256'] != digest(a.gate_a / 'records.json')
        or verification['dense_records_sha256'] != digest(a.dense_gate_a / 'records.json')):
        raise ValueError('Engineering verification is invalid')
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    logs = out / 'logs'; logs.mkdir()
    runner = ROOT / 'scripts/run_lean_dense_memory_diagnostic.py'
    identity = {'runner_sha256': digest(runner), 'verification_sha256': digest(a.verification), 'sequences': a.sequence}
    (out / 'setup.json').write_text(json.dumps(identity, indent=2) + '\n')
    complete = []
    for name in a.sequence:
        if digest(runner) != identity['runner_sha256'] or digest(a.verification) != identity['verification_sha256']:
            raise ValueError('Runner or engineering receipt changed')
        print(f'Starting {name}: {len(complete)}/{len(a.sequence)} complete', flush=True)
        args = [sys.executable, str(runner), '--mode', 'ablation', '--sequence', name,
                '--checkpoint', str(a.checkpoint), '--merged', str(a.merged), '--historical', str(a.historical),
                '--gate-a', str(a.gate_a), '--gate-b', str(a.gate_b),
                '--dense-gate-a', str(a.dense_gate_a), '--dense-gate-b', str(a.dense_gate_b), '--output', str(out/name)]
        with (logs/f'{name}.log').open('w') as stream:
            result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            (out/'progress.json').write_text(json.dumps({'completed': complete, 'failed': name}, indent=2)+'\n')
            raise RuntimeError(f'{name} failed; inspect log')
        receipt = json.loads((out/name/'completion.json').read_text())
        if not receipt['passed'] or digest(out/name/'records.json') != receipt['records_sha256']:
            raise ValueError(f'{name} incomplete')
        complete.append(name)
        (out/'progress.json').write_text(json.dumps({'completed': complete}, indent=2)+'\n')
        print(f'Completed {name}', flush=True)
    (out/'completion.json').write_text(json.dumps({'passed': True, 'sequences': complete}, indent=2)+'\n')


if __name__ == '__main__': main()
