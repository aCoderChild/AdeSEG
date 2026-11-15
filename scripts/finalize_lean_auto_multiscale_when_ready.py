#!/usr/bin/env python3
"""Finish the combined report after both independently audited partitions end."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', type=Path, action='append', required=True)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--status', type=Path, required=True)
    a = p.parse_args()
    if len(a.part) != 2:
        raise ValueError('Expected two partitions')
    status = validate_output(a.status)
    while True:
        for part in a.part:
            progress = part / 'progress.json'
            if progress.exists():
                try:
                    data = json.loads(progress.read_text())
                except json.JSONDecodeError:
                    continue  # The partition may be writing its progress receipt.
                if data.get('failed'):
                    status.write_text(json.dumps({'state': 'failed', 'partition': str(part), 'sequence': data['failed']}, indent=2) + '\n')
                    return 1
        if all((part / 'completion.json').exists() for part in a.part):
            break
        time.sleep(20)
    command = [sys.executable, str(ROOT / 'scripts/merge_lean_auto_multiscale_cohort.py')]
    for part in a.part:
        command += ['--part', str(part)]
    command += ['--historical', str(a.historical), '--output', str(a.output)]
    try:
        merge = subprocess.run(command, check=True, capture_output=True, text=True)
        report = subprocess.run([sys.executable, str(ROOT / 'scripts/report_lean_auto_multiscale.py'),
                                 '--merged', str(a.output), '--historical', str(a.historical)],
                                check=True, capture_output=True, text=True)
        status.write_text(json.dumps({'state': 'complete', 'merged_output': str(a.output),
                                      'report': report.stdout.strip(), 'summary': json.loads(merge.stdout)}, indent=2) + '\n')
        progress = status.parent / 'progress.md'
        with progress.open('a') as stream:
            stream.write(f'\nBoth partitions completed and passed their audits. Final results: `{a.output}/results.json`; report: `{a.output}/report.md`.\n')
    except Exception as exc:
        status.write_text(json.dumps({'state': 'finalization_failed', 'error': str(exc),
                                     'stderr': getattr(exc, 'stderr', None)}, indent=2) + '\n')
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
