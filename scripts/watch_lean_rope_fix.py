#!/usr/bin/env python3
"""Wait for both corrected Lean partitions and run the independent audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', type=Path, action='append', required=True)
    for name in ('merged', 'historical', 'verification', 'fixed-gate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    while True:
        for part in a.part:
            progress = part/'progress.json'
            if progress.exists() and json.loads(progress.read_text()).get('failed'):
                raise RuntimeError(f'Partition failed: {part}')
        if all((part/'completion.json').exists() for part in a.part): break
        time.sleep(20)
    cmd = [sys.executable, str(ROOT/'scripts/audit_lean_rope_fix.py')]
    for part in a.part: cmd.extend(['--part', str(part)])
    for name in ('merged', 'historical', 'verification', 'fixed_gate', 'output'):
        cmd.extend(['--'+name.replace('_','-'), str(getattr(a,name))])
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__': main()
