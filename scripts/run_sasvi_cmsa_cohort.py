#!/usr/bin/env python3
"""Run and independently audit the frozen SASVi/CMSA adaptation by sequence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cmsa-run', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gate-a', required=True, type=Path)
    p.add_argument('--gate-b', required=True, type=Path)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--sequence', action='append', help='Optional disjoint cohort partition')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    runner = ROOT / 'scripts/run_sasvi_cmsa_polypgen.py'
    auditor = ROOT / 'scripts/audit_sasvi_cmsa_polypgen.py'
    gate_setup = [json.loads((gate / 'setup.json').read_text()) for gate in (args.gate_a, args.gate_b)]
    gate_records = [json.loads((gate / 'records.json').read_text()) for gate in (args.gate_a, args.gate_b)]
    if gate_setup[0] != gate_setup[1] or gate_setup[0]['runner_sha256'] != digest(runner):
        raise ValueError('Gate setup mismatch')
    if gate_setup[0]['device'] != f'{args.device} fp32' or gate_setup[0]['frames'] < 21 or gate_setup[0]['sam2_checkpoint_sha256'] != digest(args.checkpoint):
        raise ValueError('Insufficient gate or incorrect device/checkpoint')
    if (args.gate_a / 'prompt_events.json').read_text() != (args.gate_b / 'prompt_events.json').read_text():
        raise ValueError('Gate point prompts differ')
    gate_max_pixel_fraction = 0.
    for x, y in zip(*gate_records):
        if (x['index'], x['image_sha256'], x['cmsa_mask_sha256']) != (y['index'], y['image_sha256'], y['cmsa_mask_sha256']):
            raise ValueError('Gate input mismatch')
        name, i = gate_setup[0]['sequence'], x['index']
        path_a = args.gate_a / 'predictions' / name / f'{i:06d}.png'
        path_b = args.gate_b / 'predictions' / name / f'{i:06d}.png'
        if digest(path_a) != x['mask_sha256'] or digest(path_b) != y['mask_sha256']:
            raise ValueError('Gate mask hash mismatch')
        a = np.asarray(Image.open(path_a)) > 0
        b = np.asarray(Image.open(path_b)) > 0
        gate_max_pixel_fraction = max(gate_max_pixel_fraction, float(np.mean(a != b)))
    if len(gate_records[0]) != gate_setup[0]['frames'] or len(gate_records[1]) != gate_setup[0]['frames'] or gate_max_pixel_fraction > 1e-5:
        raise ValueError('Gate masks disagree beyond 0.001% of pixels')
    available = [f'seq{i}' for i in range(2, 24) if i != 7]
    names = args.sequence or available
    if len(set(names)) != len(names) or not set(names).issubset(available):
        raise ValueError('Invalid or duplicate sequence')
    setup = {'sequences': names, 'device': f'{args.device} fp32' + (', MPS fallback enabled' if args.device == 'mps' else ''),
             'runner_sha256': digest(runner), 'auditor_sha256': digest(auditor),
             'gate_a_records_sha256': digest(args.gate_a / 'records.json'), 'gate_b_records_sha256': digest(args.gate_b / 'records.json'),
             'gate_max_pixel_fraction': gate_max_pixel_fraction,
             'checkpoint_sha256': digest(args.checkpoint), 'cmsa_setup_sha256': digest(args.cmsa_run / 'setup.json'),
             'cmsa_frames_sha256': digest(args.cmsa_run / 'frames.json')}
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1')
    completed = {}
    for name in names:
        if digest(runner) != setup['runner_sha256'] or digest(auditor) != setup['auditor_sha256']:
            raise RuntimeError('Cohort runner or auditor changed')
        run = args.output / name
        command = [sys.executable, str(runner), '--cmsa-run', str(args.cmsa_run), '--checkpoint', str(args.checkpoint),
                   '--output', str(run), '--sequence', name, '--device', args.device]
        print(f'Starting {name} ({len(completed)}/{len(names)} complete)', flush=True)
        with (args.output / 'logs' / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                result = subprocess.run([sys.executable, str(auditor), '--run', str(run), '--cmsa-run', str(args.cmsa_run)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (args.output / 'progress.json').write_text(json.dumps({'completed': list(completed), 'failed': name,
                                                                    'exit_code': result.returncode}, indent=2) + '\n')
            raise RuntimeError(f'{name} failed; see log')
        audit = json.loads((run / 'audit_results.json').read_text())
        completed[name] = {k: v for k, v in audit.items() if k != 'frames'}
        (args.output / 'progress.json').write_text(json.dumps({'completed': list(completed), 'per_sequence': completed}, indent=2) + '\n')
        print(f'Completed {name}: present Dice CMSA={audit["cmsa_present_dice"]:.5f}, SASVi={audit["sasvi_present_dice"]:.5f}', flush=True)
    result = {'sequences': len(completed), 'cmsa_equal_sequence_dice': float(np.mean([r['cmsa_present_dice'] for r in completed.values()])),
              'sasvi_equal_sequence_dice': float(np.mean([r['sasvi_present_dice'] for r in completed.values()])),
              'empty_targets': sum(r['empty'] for r in completed.values()),
              'cmsa_false_alarms': sum(r['cmsa_false_alarms'] for r in completed.values()),
              'sasvi_false_alarms': sum(r['sasvi_false_alarms'] for r in completed.values()),
              'per_sequence': completed}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(args.output / 'results.json')}, indent=2) + '\n')
    print('Cohort complete', flush=True)


if __name__ == '__main__':
    main()
