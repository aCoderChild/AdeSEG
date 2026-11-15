#!/usr/bin/env python3
"""Run the two-patient exploratory boundary-refiner A/B/C/D extension.

The additional patients and checkpoint were selected after previous development
results were inspected. This is not a blinded test. Both the original and the
extension engineering gates, plus their independent artifact audits, must pass
before any full run starts. All 8 runs finish before scoring opens labels.
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
from scripts.run_boundary_video_factorial_v1 import (AUDIT, BASELINE, CHECKPOINT,
    DETECTOR, DETECTOR_SHA256, PRE, STUDY, verify_gate_audit)

SEQUENCES = ('seq6', 'seq10')
ARMS = ('A', 'B', 'C', 'D')
GUARD = 22 * 1024**3
RUNNER = ROOT / 'scripts/run_remaining_boundary_refiner_v1.py'
SCORER = ROOT / 'scripts/score_remaining_boundary_factorial_v1.py'


def study_sources():
    """Bind the implementation of both primary and independent metrics."""
    return {'launcher': Path(__file__), 'scorer': SCORER, 'runner': RUNNER,
            'preflight_source': ROOT/'scripts/preflight_remaining_boundary_video_v1.py',
            'extension_gate_source': ROOT/'scripts/verify_remaining_boundary_video_short_gate_v1.py',
            'extension_auditor_source': ROOT/'scripts/audit_remaining_boundary_video_short_v1.py',
            'independent_auditor': ROOT/'scripts/audit_remaining_boundary_factorial_v1.py',
            'primary_metric_helper': ROOT/'scripts/score_boundary_video_factorial_v1.py',
            'primary_boundary_helper': ROOT/'scripts/score_source_group_boundary_v1.py',
            'independent_metric_helper': ROOT/'scripts/audit_boundary_video_factorial_v1.py',
            'mask_loader': ROOT/'datasets/io.py', 'video_loader': ROOT/'datasets/video.py'}


def read(path):
    return json.loads(Path(path).read_text())


def input_digest(name, path):
    """The preflight is an output directory; all other inputs are files."""
    return digest(path/'manifest.json') if name == 'preflight' else digest(path)


def free_bytes(path):
    free = shutil.disk_usage(path).free
    if free < GUARD:
        raise RuntimeError(f'Free disk {free} bytes is below {GUARD}; phase not started')
    return free


def phase_environment():
    env = os.environ.copy()
    env.update(PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0', PYTHONHASHSEED='0')
    return env


def validate_input_roots(paths, root):
    root = Path(root).resolve(strict=True)
    if not all(Path(path).resolve(strict=True).is_relative_to(root) for path in paths):
        raise ValueError('Preflight or gate lies outside the experiment root')


def verify_original_gate(gate, gate_audit):
    from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate
    verify_full_gate(gate, study=STUDY, audit=AUDIT, checkpoint=CHECKPOINT,
                     split_preflight=PRE, baseline_reference=BASELINE, seq4_capture=REF)
    verify_gate_audit(gate, gate_audit)
    if digest(DETECTOR) != DETECTOR_SHA256:
        raise ValueError('Original detector differs')


def verify_extended_gate(preflight, gate, gate_audit, original_gate):
    """Recheck both the extended gate and its separate artifact audit."""
    # Imported lazily: the extended runner/gate are authored independently.
    from scripts.verify_remaining_boundary_video_short_gate_v1 import verify_extended_full_gate
    receipt = verify_extended_full_gate(gate, extended_preflight=preflight, engineering_gate=original_gate,
                                       study=STUDY, audit=AUDIT, checkpoint=CHECKPOINT, detector=DETECTOR,
                                       split_preflight=PRE, baseline_reference=BASELINE, seq4_capture=REF)
    audit = read(gate_audit)
    if (audit.get('passed') is not True or audit.get('extension_gate_sha256') != digest(gate)
            or audit.get('extended_preflight_sha256') != input_digest('preflight', preflight)
            or audit.get('engineering_gate_sha256') != digest(original_gate)
            or audit.get('runner_sha256') != digest(RUNNER) or audit.get('runs_verified') != 10
            or audit.get('candidate_checkpoint_sha256') != digest(STUDY/'runs/region_boundary/checkpoint.pt')
            or audit.get('auditor_sha256') != digest(ROOT/'scripts/audit_remaining_boundary_video_short_v1.py')):
        raise ValueError('Independent extension short-gate audit differs')
    for seq in ('seq6','seq10'):
        for arm in ('A','A_repeat','B','C','D'):
            source=receipt['runs'][seq][arm]
            actual=digest(Path(source['folder'])/'completion.json')
            if (source['completion_sha256'] != actual or
                    audit['sequences'][seq][arm]['completion_sha256'] != actual):
                raise ValueError('Independent extension audit does not bind every short run')


def commands(run, preflight, original_gate, extension_gate, extension_audit):
    """Generate A before B/C/D per video; no scorer until all 8 complete."""
    run = Path(run)
    common = ['--study', str(STUDY), '--audit', str(AUDIT), '--checkpoint', str(CHECKPOINT),
              '--detector', str(DETECTOR), '--split-preflight', str(PRE),
              '--baseline-reference', str(BASELINE), '--seq4-capture', str(REF),
              '--engineering-gate', str(original_gate), '--extended-preflight', str(preflight),
              '--extension-gate', str(extension_gate), '--extension-audit', str(extension_audit)]
    phases, mapping = [], {}
    for seq in SEQUENCES:
        mapping[seq] = {}
        for arm in ARMS:
            folder = run/seq/arm
            mapping[seq][arm] = str(folder)
            cmd = [sys.executable, str(RUNNER), *common, '--sequence', seq, '--arm', arm, '--output', str(folder)]
            if arm != 'A':
                cmd.extend(('--matched-a', str(run/seq/'A')))
            phases.append((f'{seq}_{arm}', cmd))
    phases.append(('score', [sys.executable, str(SCORER), '--experiment-root', str(BASE),
                             '--run-map', str(run/'run_map.json'), '--output', str(run/'scores')]))
    return phases, mapping


def monitor_space(folder, stop, ready, errors):
    try:
        with (folder/'free_space.jsonl').open('x') as out:
            while True:
                free = shutil.disk_usage(folder).free
                out.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                      'free_bytes': free, 'below_guard': free < GUARD})+'\n')
                out.flush(); ready.set()
                if stop.wait(5): break
    except Exception as exc:
        errors.append(repr(exc)); ready.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'gate', 'gate-audit', 'extension-gate', 'extension-audit'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    preflight, gate, gate_audit, extension_gate, extension_audit = (
        getattr(a, name.replace('-', '_')).resolve(strict=True)
        for name in ('preflight', 'gate', 'gate-audit', 'extension-gate', 'extension-audit'))
    validate_input_roots((preflight,gate,gate_audit,extension_gate,extension_audit),BASE)
    def check():
        from scripts.preflight_remaining_boundary_video_v1 import load_extended_preflight
        load_extended_preflight(preflight, CHECKPOINT, DETECTOR)
        verify_original_gate(gate, gate_audit)
        verify_extended_gate(preflight, extension_gate, extension_audit, gate)
    check()
    if not a.execute:
        phases, _ = commands(OUT/'DRY_RUN', preflight, gate, extension_gate, extension_audit)
        print(json.dumps({'ready': True, 'free_bytes': free_bytes(OUT), 'phases': [x[0] for x in phases]})); return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        check(); free_bytes(OUT)
        folder = Path(tempfile.mkdtemp(prefix='remaining_boundary_factorial_v1_', dir=OUT))
        phases, mapping = commands(folder, preflight, gate, extension_gate, extension_audit)
        manifest = folder/'run_manifest.json'
        source_paths = study_sources()
        inputs = {'preflight': preflight, 'gate': gate, 'gate_audit': gate_audit,
                  'extension_gate': extension_gate, 'extension_audit': extension_audit}
        manifest.write_text(json.dumps({'scope': 'Exploratory; user-reported distinct patients, previously inspected; not blinded.',
          'sequences': list(SEQUENCES), 'arms': list(ARMS), 'run_map': mapping,
          'source_sha256': {k: digest(v) for k,v in source_paths.items()},
          'inputs': {k: {'path': str(v), 'sha256': input_digest(k,v)} for k,v in inputs.items()},
          'preflight_completion_sha256': digest(preflight/'completion.json'),
          'checkpoint_sha256': digest(CHECKPOINT), 'detector_sha256': digest(DETECTOR),
          'space_guard_bytes': GUARD, 'mps_fallback': '0', 'mps_fast_math': '0', 'python_hash_seed': '0'}, indent=2)+'\n')
        (folder/'run_map.json').write_text(json.dumps(mapping, indent=2)+'\n')
        frozen_manifest, frozen_map = digest(manifest), digest(folder/'run_map.json')
        stop, ready, errors = threading.Event(), threading.Event(), []
        monitor = threading.Thread(target=monitor_space,args=(folder,stop,ready,errors),daemon=True)
        monitor.start(); ready.wait(10)
        try:
            if not ready.is_set() or errors: raise RuntimeError(f'Storage monitor failed: {errors}')
            for name, cmd in phases:
                if (digest(manifest) != frozen_manifest or digest(folder/'run_map.json') != frozen_map
                        or any(digest(path) != read(manifest)['source_sha256'][key] for key,path in source_paths.items())
                        or digest(preflight/'completion.json') != read(manifest)['preflight_completion_sha256']
                        or any(input_digest(key,path) != read(manifest)['inputs'][key]['sha256'] for key,path in inputs.items())):
                    raise RuntimeError('Frozen launch inputs changed')
                if errors: raise RuntimeError(f'Storage monitor failed: {errors}')
                free = free_bytes(folder)
                print(json.dumps({'phase':name,'status':'starting','output':str(folder),'free_bytes':free}),flush=True)
                start = time.monotonic()
                with (folder/f'{name}.log').open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,env=phase_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
                if errors: raise RuntimeError(f'Storage monitor failed: {errors}')
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'passed':True,'wall_seconds':time.monotonic()-start,
                                          'free_bytes_before':free,'log_sha256':digest(folder/f'{name}.log')})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set(); monitor.join(10)
            if monitor.is_alive() or errors: raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed':True,'output':str(folder)}),flush=True)


if __name__ == '__main__': main()
