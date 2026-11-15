#!/usr/bin/env python3
"""Run the verified CPU RoPE correction in isolated video processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'merged', 'historical', 'replay-a', 'replay-b',
                 'fixed-a', 'fixed-b', 'verification', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sequence', action='append', required=True)
    args = parser.parse_args()
    cohort = json.loads((ROOT / 'protocols/lean_cpu_rope_repeat_fix_v1.json').read_text())['cohort']
    if len(set(args.sequence)) != len(args.sequence) or not set(args.sequence) <= set(cohort):
        raise ValueError('Duplicate or out-of-scope sequence')
    verification = json.loads(args.verification.read_text())
    runner = ROOT / 'scripts/run_lean_rope_cpu_fix.py'
    if (not verification['passed']
        or verification['source_sha256'] != digest(ROOT / 'scripts/verify_lean_rope_fix_gates.py')
        or verification['identity']['runner_sha256'] != digest(runner)
        or verification['replay_records_sha256'] != digest(args.replay_a / 'records.json')
        or verification['fixed_records_sha256'] != digest(args.fixed_a / 'records.json')):
        raise ValueError('Engineering verification is invalid')
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    logs = out / 'logs'
    logs.mkdir()
    identity = {'runner_sha256': digest(runner), 'verification_sha256': digest(args.verification),
                'sequences': args.sequence}
    (out / 'setup.json').write_text(json.dumps(identity, indent=2) + '\n')
    completed = []
    for name in args.sequence:
        if digest(runner) != identity['runner_sha256'] or digest(args.verification) != identity['verification_sha256']:
            raise ValueError('Runner or engineering verification changed')
        print(f'Starting {name}: {len(completed)}/{len(args.sequence)} complete', flush=True)
        cmd = [sys.executable, str(runner), '--mode', 'fixed', '--sequence', name]
        for arg in ('checkpoint', 'merged', 'historical', 'replay_a', 'replay_b', 'fixed_a', 'fixed_b'):
            cmd.extend(['--' + arg.replace('_', '-'), str(getattr(args, arg))])
        cmd.extend(['--output', str(out / name)])
        with (logs / f'{name}.log').open('w') as stream:
            result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            (out / 'progress.json').write_text(json.dumps({'completed': completed, 'failed': name}, indent=2) + '\n')
            raise RuntimeError(f'{name} failed; inspect log')
        receipt = json.loads((out / name / 'completion.json').read_text())
        if not receipt['passed'] or digest(out / name / 'records.json') != receipt['records_sha256']:
            raise ValueError(f'{name} incomplete')
        completed.append(name)
        (out / 'progress.json').write_text(json.dumps({'completed': completed}, indent=2) + '\n')
        print(f'Completed {name}', flush=True)
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'sequences': completed}, indent=2) + '\n')


if __name__ == '__main__':
    main()
