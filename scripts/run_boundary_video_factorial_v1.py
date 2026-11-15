#!/usr/bin/env python3
"""Run the user-requested exploratory A/B/C/D priority-video factorial.

The separate short-video gate is required by the inference runner. All twelve
predictions finish and freeze before the separate scorer can open labels. This
is an explicit exploratory exception to the previous promotion criterion, not
a locked or independent evaluation. The 22 GiB reserve is checked before
each phase; monitoring does not interrupt a phase already writing.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import BASE, OUT, REF, exclusive_lock

SEQUENCES = ('seq20', 'seq23', 'seq11')
ARMS = ('A', 'B', 'C', 'D')
GUARD = 22 * 1024**3
RUNNER = ROOT / 'scripts/run_boundary_refiner_priority_v1.py'
SCORER = ROOT / 'scripts/score_boundary_video_factorial_v1.py'
STUDY = OUT / 'source_group_boundary_factorial_v1_bvqxd475'
AUDIT = OUT / 'source_group_boundary_factorial_v1_bvqxd475_independent_audit_attempt01/audit.json'
PRE = BASE / 'lean_channel_attention_training_pilot_v1/split_preflight_attempt03'
CHECKPOINT = BASE / 'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
DETECTOR = ROOT / 'checkpoints/polypgen_yolov8n.pt'
BASELINE = OUT / 'lean_window_replay_mps01'
SEQ23 = BASE / 'lean_token_window_candidate_v1/seq23_attempt01'
SEQ23_BACKUP = BASE / 'lean_seq23_no_inset_v1/full_attempt02'
DETECTOR_SHA256 = '858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e'


def free_bytes(path):
    free = shutil.disk_usage(path).free
    if free < GUARD:
        raise RuntimeError(f'Free disk {free} bytes is below {GUARD}; phase not started')
    return free


def phase_environment():
    env = os.environ.copy()
    env.update(PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0', PYTHONHASHSEED='0')
    return env


def verify_gate_audit(gate, audit):
    short = read_json(gate)
    checked = read_json(audit)
    if (not checked.get('passed') or checked.get('gate_sha256') != digest(gate)
            or checked.get('audit_source_sha256') != digest(ROOT / 'scripts/audit_boundary_video_short_v1.py')
            or checked.get('candidate_checkpoint_sha256') != digest(STUDY / 'runs/region_boundary/checkpoint.pt')
            or digest(DETECTOR) != DETECTOR_SHA256):
        raise ValueError('Independent short-gate audit or detector differs')
    for seq in ('seq4', 'seq23'):
        for arm in ('A', 'A_repeat', 'B', 'C', 'D'):
            run = short['runs'][seq][arm]
            actual = digest(Path(run['folder']) / 'completion.json')
            completion = read_json(Path(run['folder']) / 'completion.json')
            if (checked['sequences'][seq][arm]['completion_sha256'] != actual
                    or run['completion_sha256'] != actual or completion.get('detector_sha256') != DETECTOR_SHA256):
                raise ValueError('Short-gate audit does not bind this detector and run')
    return checked


def commands(run, gate):
    """Generate full-video arms in A-then-B/C/D order and one final scorer."""
    run = Path(run)
    py = sys.executable
    common = ['--study', str(STUDY), '--audit', str(AUDIT), '--checkpoint', str(CHECKPOINT),
              '--detector', str(DETECTOR), '--split-preflight', str(PRE),
              '--baseline-reference', str(BASELINE), '--seq4-capture', str(REF),
              '--seq23-reference', str(SEQ23), '--seq23-backup', str(SEQ23_BACKUP),
              '--engineering-gate', str(gate)]
    result, mapping = [], {}
    for seq in SEQUENCES:
        mapping[seq] = {}
        for arm in ARMS:
            folder = run / seq / arm
            mapping[seq][arm] = str(folder)
            command = [py, str(RUNNER), *common, '--sequence', seq, '--arm', arm, '--output', str(folder)]
            if arm != 'A':
                command.extend(('--matched-a', str(run / seq / 'A')))
            result.append((f'{seq}_{arm}', command))
    result.append(('score', [py, str(SCORER), '--experiment-root', str(BASE),
                             '--run-map', str(run / 'run_map.json'), '--output', str(run / 'scores')]))
    return result, mapping


def monitor_space(folder, stop, ready, errors):
    try:
        with (folder / 'free_space.jsonl').open('x') as out:
            while True:
                free = shutil.disk_usage(folder).free
                out.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                      'free_bytes': free, 'below_guard': free < GUARD}) + '\n')
                out.flush()
                ready.set()
                if stop.wait(5):
                    break
    except Exception as exc:
        errors.append(repr(exc))
        ready.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    gate = args.gate.resolve(strict=True)
    gate_audit = gate.parent.parent / 'independent_audit_attempt01/audit.json'
    # The immutable source-bound verifier is also invoked by every child.
    from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate
    verify_full_gate(gate, study=STUDY, audit=AUDIT, checkpoint=CHECKPOINT,
                     split_preflight=PRE, baseline_reference=BASELINE, seq4_capture=REF)
    verify_gate_audit(gate, gate_audit)
    if not args.execute:
        phases, _ = commands(OUT / 'DRY_RUN', gate)
        print(json.dumps({'ready': True, 'free_bytes': free_bytes(OUT),
                          'phases': [name for name, _ in phases], 'gate_sha256': digest(gate)}, indent=2))
        return
    with exclusive_lock(OUT / '.source_group_capture_resume_v1.lock'):
        # Gate and available storage may change while acquiring the shared lock.
        verify_full_gate(gate, study=STUDY, audit=AUDIT, checkpoint=CHECKPOINT,
                         split_preflight=PRE, baseline_reference=BASELINE, seq4_capture=REF)
        verify_gate_audit(gate, gate_audit)
        free_bytes(OUT)
        folder = Path(tempfile.mkdtemp(prefix='boundary_video_factorial_v1_', dir=OUT))
        phases, mapping = commands(folder, gate)
        manifest = folder / 'run_manifest.json'
        manifest.write_text(json.dumps({'scope': 'User-requested exploratory exception; not an independent test.',
            'launcher_sha256': digest(Path(__file__)), 'scorer_sha256': digest(SCORER),
            'runner_sha256': digest(RUNNER), 'gate_path': str(gate), 'gate_sha256': digest(gate),
            'independent_gate_audit_path': str(gate_audit), 'independent_gate_audit_sha256': digest(gate_audit),
            'sequences': list(SEQUENCES), 'arms': list(ARMS), 'run_map': mapping,
            'space_guard_bytes': GUARD, 'mps_fallback': '0', 'mps_fast_math': '0',
            'python_hash_seed': '0',
            'short_gate_hash_seed_note': 'PYTHONHASHSEED=0 was observed in the short-gate launcher command, but was not recorded in the short-gate runtime receipt.'}, indent=2) + '\n')
        manifest_hash = digest(manifest)
        (folder / 'run_map.json').write_text(json.dumps(mapping, indent=2) + '\n')
        map_hash = digest(folder / 'run_map.json')
        env = phase_environment()
        stop, ready, errors = threading.Event(), threading.Event(), []
        monitor = threading.Thread(target=monitor_space, args=(folder, stop, ready, errors), daemon=True)
        monitor.start(); ready.wait(10)
        try:
            if not ready.is_set() or errors:
                raise RuntimeError(f'Storage monitor failed: {errors}')
            for name, command in phases:
                if (digest(manifest) != manifest_hash or digest(folder / 'run_map.json') != map_hash
                        or digest(Path(__file__)) != read_json(manifest)['launcher_sha256']
                        or digest(SCORER) != read_json(manifest)['scorer_sha256']
                        or digest(RUNNER) != read_json(manifest)['runner_sha256']
                        or digest(gate) != read_json(manifest)['gate_sha256']
                        or digest(gate_audit) != read_json(manifest)['independent_gate_audit_sha256']):
                    raise RuntimeError('Frozen launch inputs changed')
                if errors:
                    raise RuntimeError(f'Storage monitor failed: {errors}')
                before = free_bytes(folder)
                print(json.dumps({'phase': name, 'status': 'starting', 'output': str(folder), 'free_bytes': before}), flush=True)
                start = time.monotonic()
                with (folder / f'{name}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if errors:
                    raise RuntimeError(f'Storage monitor failed: {errors}')
                with (folder / 'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase': name, 'passed': True, 'wall_seconds': time.monotonic()-start,
                                          'free_bytes_before': before, 'log_sha256': digest(folder / f'{name}.log')}) + '\n')
                print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
        finally:
            stop.set(); monitor.join(10)
            if monitor.is_alive() or errors:
                raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed': True, 'output': str(folder)}), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    main()
