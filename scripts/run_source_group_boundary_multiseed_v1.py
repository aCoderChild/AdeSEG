#!/usr/bin/env python3
"""Run the frozen source-group fixed-read study after all captures are audited.

Default is a CPU-only dry run; --execute writes a new directory on the user's
Drive. This launcher does not auto-resume partially trained arms. It keeps all
partial artifacts and stops on the first failed phase or low starting space.
Space is monitored during phases, but an active phase is not interrupted. The
capture lock prevents concurrent invocations of the project capture launcher;
it does not prevent other, unrelated GPU processes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.prepare_source_group_refiner_manifest_v1 import SEQUENCES, resolve_capture_map
from scripts.resume_source_group_capture_v1 import (
    BASE, CHECKPOINT, FREEZE, GATE, OUT, PRE, REF, discover_completed,
    environment_setup, exclusive_lock,
)
from scripts.source_group_boundary_multiseed_shared_v1 import ARMS, SEEDS, dependencies

GUARD_BYTES = 22 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def checked_space(path, guard=GUARD_BYTES):
    free = shutil.disk_usage(path).free
    if free < guard:
        raise RuntimeError(f'Free disk space {free} bytes is below {guard}; no phase started')
    return free


def make_capture_map(completed):
    """Use only the revalidated audit's explicit paths, with deterministic choice."""
    if set(completed) != set(SEQUENCES) or any(not completed[s] for s in SEQUENCES):
        raise ValueError('All nine audited sequence captures are required')
    result = {}
    for seq in SEQUENCES:
        choice = sorted(completed[seq], key=lambda x: x['audit_path'])[0]
        audit_path = Path(choice['audit_path']).resolve()
        if digest(audit_path) != choice['audit_sha256']:
            raise ValueError(f'{seq}: independently checked audit changed')
        audit = read(audit_path)
        if not audit.get('passed') or audit.get('sequence') != seq:
            raise ValueError(f'{seq}: audit is incomplete')
        result[seq] = {'reference': str(Path(audit['reference_receipt_path']).resolve().parent),
                       'capture': str(Path(audit['capture_receipt_path']).resolve().parent),
                       'audit': str(audit_path.parent)}
    resolve_capture_map(result)
    return result


def make_phases(run):
    """Freeze both seed cohorts before scoring either one."""
    run = Path(run)
    py = sys.executable
    manifest = run/'manifest'
    script = lambda name: str(ROOT/'scripts'/name)
    phases = [
        ('manifest', None, [py, script('prepare_source_group_refiner_manifest_v1.py'),
         '--preflight', str(PRE), '--source-freeze', str(FREEZE), '--short-gate', str(GATE),
         '--reference-capture', str(REF), '--checkpoint', str(CHECKPOINT),
         '--capture-map', str(run/'capture_map.json'), '--output', str(manifest)]),
    ]
    for seed in SEEDS:
        base = run/str(seed)
        plan, parity, runs = base/'plan/plan.json', base/'parity/parity.json', base/'runs'
        common = ['--manifest', str(manifest), '--plan', str(plan), '--parity', str(parity)]
        phases.extend([
            (f'{seed}_plan', seed, [py, script('train_source_group_boundary_multiseed_v1.py'), '--mode', 'plan',
             '--manifest', str(manifest), '--output', str(base/'plan')]),
            (f'{seed}_parity', seed, [py, script('train_source_group_boundary_multiseed_v1.py'), '--mode', 'parity',
             '--manifest', str(manifest), '--plan', str(plan), '--output', str(base/'parity')]),
        ])
        for arm in ARMS:
            phases.append((f'{seed}_train_{arm}', seed,
                [py, script('train_source_group_boundary_multiseed_v1.py'), '--mode', 'train',
                 *common, '--arm', arm, '--output', str(runs/arm)]))
        phases.append((f'{seed}_verify', seed, [py, script('verify_source_group_boundary_multiseed_v1.py'),
                      *common, '--run-root', str(runs), '--output', str(base/'verification')]))
    phases.append(('joint_gate', None, [py, script('freeze_source_group_boundary_multiseed_v1.py'),
                                      '--run', str(run)]))
    for seed in SEEDS:
        base = run/str(seed)
        phases.append((f'{seed}_score', seed, [py, script('score_source_group_boundary_multiseed_v1.py'),
            '--manifest', str(manifest), '--plan', str(base/'plan/plan.json'),
            '--parity', str(base/'parity/parity.json'), '--verification', str(base/'verification'),
            '--run-root', str(base/'runs'), '--joint-gate', str(run/'joint_gate/joint.json'),
            '--output', str(base/'scores')]))
    for seed in SEEDS:
        base = run/str(seed)
        phases.append((f'{seed}_independent_audit', seed, [py, script('audit_source_group_boundary_multiseed_v1.py'),
                      '--study', str(run), '--seed', str(seed), '--output', str(base/'independent_audit')]))
    phases.append(('report', None, [py, script('report_source_group_boundary_multiseed_v1.py'),
                                  '--run', str(run), '--output', str(run/'summary')]))
    return phases


def monitor_space(run, stop, ready, errors):
    try:
        with (run/'free_space.jsonl').open('x') as out:
            while True:
                free = shutil.disk_usage(run).free
                out.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                      'free_bytes': free, 'below_guard': free < GUARD_BYTES})+'\n')
                out.flush()
                ready.set()
                if stop.wait(5):
                    break
    except Exception as error:
        errors.append(repr(error))
        ready.set()


def execute(run, mapping, capture_evidence):
    run = Path(run)
    launcher_sha = digest(Path(__file__))
    discovery_sha = digest(ROOT/'scripts/resume_source_group_capture_v1.py')
    source_hashes = dependencies()
    map_path = run/'capture_map.json'
    with map_path.open('x') as out:
        json.dump(mapping, out, indent=2)
        out.write('\n')
    record = {'launcher_sha256': launcher_sha, 'discovery_sha256': discovery_sha,
              'dependency_sha256': source_hashes,
              'capture_map_sha256': digest(map_path), 'capture_evidence': capture_evidence,
              'preflight_sha256': digest(PRE/'manifest.json'),
              'source_freeze_sha256': digest(FREEZE/'source_freeze.json'),
              'short_gate_sha256': digest(GATE), 'checkpoint_sha256': digest(CHECKPOINT),
              'min_free_bytes': GUARD_BYTES, 'seeds': list(SEEDS),
              'fixed_child_environment': {'PYTHONHASHSEED': '0', 'PYTORCH_ENABLE_MPS_FALLBACK': '0',
                                          'PYTORCH_MPS_FAST_MATH': '0'},
              'scope': 'Post-hoc exploratory additional seeds; both are replay-verified before scoring.'}
    record_path = run/'run_manifest.json'
    with record_path.open('x') as out:
        json.dump(record, out, indent=2)
        out.write('\n')
    record_sha = digest(record_path)
    env = os.environ.copy()
    env.pop('ADESEG_BOUNDARY_MULTISEED', None)
    env.update({'PYTORCH_ENABLE_MPS_FALLBACK': '0', 'PYTORCH_MPS_FAST_MATH': '0', 'PYTHONHASHSEED': '0'})
    for seed in SEEDS:
        (run/str(seed)/'runs').mkdir(parents=True)
    stop, ready, errors = threading.Event(), threading.Event(), []
    monitor = threading.Thread(target=monitor_space, args=(run, stop, ready, errors), daemon=True)
    monitor.start()
    if not ready.wait(10) or errors:
        stop.set()
        monitor.join(timeout=5)
        raise RuntimeError('Free-space monitor failed to start')
    try:
        for name, seed, command in make_phases(run):
            if (digest(Path(__file__)) != launcher_sha or digest(record_path) != record_sha
                    or digest(map_path) != record['capture_map_sha256'] or dependencies() != source_hashes
                    or digest(ROOT/'scripts/resume_source_group_capture_v1.py') != discovery_sha):
                raise RuntimeError('Launcher, plan, capture map or source dependencies changed')
            if errors:
                raise RuntimeError('Free-space monitor failed; no next phase started')
            free = checked_space(run)
            event = {'phase': name, 'seed': seed, 'status': 'started', 'free_bytes': free,
                     'child_environment': {**record['fixed_child_environment'],
                         'ADESEG_BOUNDARY_MULTISEED': str(seed) if seed is not None else None},
                     'launcher_sha256': launcher_sha, 'run_manifest_sha256': record_sha}
            with (run/'events.jsonl').open('a') as out:
                out.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
            with (run/f'{name}.log').open('x') as log:
                phase_env = dict(env)
                if seed is not None:
                    phase_env['ADESEG_BOUNDARY_MULTISEED'] = str(seed)
                result = subprocess.run(command, cwd=ROOT, env=phase_env, stdout=log, stderr=subprocess.STDOUT)
            event['status'] = ('failed' if result.returncode else
                               'monitor_failed_after_phase' if errors else 'passed')
            event['returncode'] = result.returncode
            event['free_bytes'] = shutil.disk_usage(run).free
            with (run/'events.jsonl').open('a') as out:
                out.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {run/name}.log')
            if errors:
                raise RuntimeError(f'{name} exited successfully, but space monitoring failed; '
                                   'phase outputs may exist and the study was stopped')
    finally:
        stop.set()
        monitor.join()
    if errors:
        raise RuntimeError('Space monitoring failed after the final phase; phase outputs may exist')
    print(f'Completed exploratory fixed-read study: {run}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    # Hold the capture launcher lock so captures cannot change while we map or train.
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        pre, config, freeze_sha = environment_setup()
        if config['min_free_bytes'] != GUARD_BYTES:
            raise ValueError('Capture guard changed')
        completed = discover_completed(OUT, pre, freeze_sha)
        pending = [seq for seq in SEQUENCES if seq not in completed]
        print(json.dumps({'completed': sorted(completed), 'pending': pending,
                          'free_bytes': shutil.disk_usage(OUT).free, 'guard_bytes': GUARD_BYTES,
                          'execute': args.execute}), flush=True)
        if pending:
            raise RuntimeError('Cannot start study until every audited capture is complete')
        mapping = make_capture_map(completed)
        checked_space(OUT)
        if not args.execute:
            return
        run = Path(tempfile.mkdtemp(prefix='source_group_boundary_multiseed_v1_', dir=validate_output(OUT)))
        execute(run, mapping, completed)


if __name__ == '__main__':
    main()
