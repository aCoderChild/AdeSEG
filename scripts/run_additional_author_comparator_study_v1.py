#!/usr/bin/env python3
"""Run source-bound author-comparator gates, videos, then separate scoring.

The patients are previously inspected; this is an exploratory extension,
not independent validation. Labels are read only after all selected videos finish.
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
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.resume_source_group_capture_v1 import BASE, OUT, exclusive_lock
from scripts.run_additional_author_comparator_v1 import (CHECKPOINT_SHA, DATA, DETECTOR_SHA, METHODS,
    REPRESENTATIVE, SEQUENCES, TRAIN_PREFLIGHT, input_data, sources)

CHECKPOINT = BASE / 'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
DETECTOR = ROOT / 'checkpoints/polypgen_yolov8n.pt'
EXTENDED = OUT / 'extended_boundary_video_preflight_attempt01'
REMAINING = OUT / 'remaining_boundary_video_preflight_v1_attempt04'
RUNNER = ROOT / 'scripts/run_additional_author_comparator_v1.py'
VERIFIER = ROOT / 'scripts/verify_additional_author_comparator_gate_v1.py'
SCORER = ROOT / 'scripts/score_additional_author_comparators_v1.py'
AUDITOR = ROOT / 'scripts/audit_additional_author_comparators_v1.py'
GUARD = 22 * 1024**3
REPRESENTATIVES = ('seq5', 'seq6', 'seq8', 'seq12', 'seq15', 'seq17')


def source_hashes():
    return {'launcher': digest(Path(__file__)), 'scorer': digest(SCORER),
            'auditor': digest(AUDITOR),
            'prior_extended_auditor': digest(ROOT / 'scripts/audit_extended_boundary_factorial_v1.py'),
            'prior_remaining_auditor': digest(ROOT / 'scripts/audit_remaining_boundary_factorial_v1.py'),
            'metric': digest(ROOT / 'scripts/score_boundary_video_factorial_v1.py'),
            'boundary_metric': digest(ROOT / 'scripts/score_source_group_boundary_v1.py'),
            'independent_metric': digest(ROOT / 'scripts/audit_boundary_video_factorial_v1.py'),
            'mask_loader': digest(ROOT / 'datasets/io.py'),
            'mask_guard_helper': digest(ROOT / 'scripts/source_group_training_label_guard_v1.py'),
            'runner_sources': sources()}


def environment():
    result = os.environ.copy()
    result.update(PYTHONHASHSEED='0', PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0')
    return result


def available(path):
    free = shutil.disk_usage(path).free
    if free < GUARD:
        raise RuntimeError(f'Free disk {free} is below {GUARD}; phase not started')
    return free


def commands(folder, selection=SEQUENCES):
    folder = Path(folder)
    selection = tuple(selection)
    if selection not in (SEQUENCES, ('seq10',)):
        raise ValueError('Unexpected comparator study scope')
    py = sys.executable
    common = ['--checkpoint', str(CHECKPOINT), '--detector', str(DETECTOR),
              '--extended-preflight', str(EXTENDED), '--remaining-preflight', str(REMAINING)]
    phases, mapping = [], {seq: {} for seq in selection}
    representatives = tuple(s for s in REPRESENTATIVES if s in {REPRESENTATIVE[t] for t in selection})
    for method in METHODS:
        for seq in representatives:
            root = folder/'short'/method/seq
            for device, name in (('cpu', 'cpu'), ('mps', 'mps_a'), ('mps', 'mps_b')):
                cmd = [py, str(RUNNER), '--method', method, '--sequence', seq, '--device', device,
                       '--limit', '5', *common, '--output', str(root/name)]
                phases.append((f'short_{method}_{seq}_{name}', cmd))
            cmd = [py, str(VERIFIER), '--cpu', str(root/'cpu'), '--mps-a', str(root/'mps_a'),
                   '--mps-b', str(root/'mps_b'), *common, '--output', str(root/'gate')]
            phases.append((f'gate_{method}_{seq}', cmd))
    for seq in selection:
        for method in METHODS:
            output = folder/'full'/seq/method
            gate = folder/'short'/method/REPRESENTATIVE[seq]/'gate/gate.json'
            mapping[seq][method] = str(output)
            phases.append((f'full_{seq}_{method}', [py, str(RUNNER), '--method', method,
                '--sequence', seq, '--device', 'mps', '--gate', str(gate), *common, '--output', str(output)]))
    phases.append(('verify', [py, str(SCORER), '--mode', 'verify', '--run-map', str(folder/'run_map.json'),
                              '--output', str(folder/'verification')]))
    phases.append(('score', [py, str(SCORER), '--mode', 'score', '--run-map', str(folder/'run_map.json'),
                             '--verification', str(folder/'verification/verification.json'),
                             '--output', str(folder/'scores')]))
    phases.append(('audit', [py, str(AUDITOR), '--run-map', str(folder/'run_map.json'),
                             '--verification', str(folder/'verification/verification.json'),
                             '--scores', str(folder/'scores'), '--output', str(folder/'independent_audit')]))
    return phases, mapping


def monitor(folder, stop, ready, errors):
    try:
        with (folder/'free_space.jsonl').open('x') as out:
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--scope', choices=('all', 'seq10'), default='all')
    a = p.parse_args()
    guard = install_mask_open_guard(DATA)
    selection = SEQUENCES if a.scope == 'all' else ('seq10',)
    def check():
        if digest(CHECKPOINT) != CHECKPOINT_SHA or digest(DETECTOR) != DETECTOR_SHA:
            raise ValueError('Frozen checkpoint/detector differs')
        for seq in selection:
            input_data(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
            input_data(REPRESENTATIVE[seq], EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        source_hashes()
        if guard['open_attempts']:
            raise RuntimeError('Launcher attempted to open a target label')
    check()
    if not a.execute:
        phases, _ = commands(OUT/'DRY_RUN', selection)
        print(json.dumps({'ready': True, 'free_bytes': available(OUT),
                          'phases': [name for name, _ in phases]}))
        return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        check()
        available(OUT)
        folder = Path(tempfile.mkdtemp(prefix='additional_author_comparator_v1_', dir=OUT))
        phases, mapping = commands(folder, selection)
        manifest = folder/'run_manifest.json'
        manifest.write_text(json.dumps({'scope': 'Exploratory, previously inspected videos.',
            'sequences': list(selection), 'methods': list(METHODS),
            'representatives': [s for s in REPRESENTATIVES if s in {REPRESENTATIVE[t] for t in selection}],
            'representative_for': {s: REPRESENTATIVE[s] for s in selection},
            'run_map': mapping, 'phases': [name for name, _ in phases], 'sources': source_hashes(),
            'checkpoint_sha256': digest(CHECKPOINT), 'detector_sha256': digest(DETECTOR),
            'extended_preflight_sha256': digest(EXTENDED/'manifest.json'),
            'remaining_preflight_sha256': digest(REMAINING/'manifest.json'),
            'train_preflight_sha256': digest(TRAIN_PREFLIGHT/'manifest.json'),
            'launcher_label_guard': dict(guard),
            'python_hash_seed': '0', 'mps_fallback': '0', 'mps_fast_math': '0',
            'space_guard_bytes': GUARD}, indent=2) + '\n')
        (folder/'run_map.json').write_text(json.dumps(mapping, indent=2) + '\n')
        frozen_manifest, frozen_map = digest(manifest), digest(folder/'run_map.json')
        stop, ready, errors = threading.Event(), threading.Event(), []
        thread = threading.Thread(target=monitor, args=(folder, stop, ready, errors), daemon=True)
        thread.start()
        ready.wait(10)
        try:
            if not ready.is_set() or errors:
                raise RuntimeError(f'Storage monitor failed: {errors}')
            for name, cmd in phases:
                saved = json.loads(manifest.read_text())
                if (digest(manifest) != frozen_manifest or digest(folder/'run_map.json') != frozen_map
                        or source_hashes() != saved['sources'] or digest(CHECKPOINT) != saved['checkpoint_sha256']
                        or digest(DETECTOR) != saved['detector_sha256']
                        or digest(EXTENDED/'manifest.json') != saved['extended_preflight_sha256']
                        or digest(REMAINING/'manifest.json') != saved['remaining_preflight_sha256']
                        or digest(TRAIN_PREFLIGHT/'manifest.json') != saved['train_preflight_sha256']
                        or guard['open_attempts'] or errors):
                    raise RuntimeError('Frozen inputs or monitor changed')
                free = available(folder)
                print(json.dumps({'phase': name, 'status': 'starting', 'output': str(folder), 'free_bytes': free}), flush=True)
                start = time.monotonic()
                with (folder/f'{name}.log').open('x') as log:
                    subprocess.run(cmd, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
                if errors:
                    raise RuntimeError(f'Storage monitor failed: {errors}')
                receipt_sha = None
                if name.startswith('full_'):
                    _, seq, method = name.split('_', 2)
                    receipt_sha = digest(Path(mapping[seq][method])/'receipt.json')
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase': name, 'passed': True, 'wall_seconds': time.monotonic()-start,
                                          'free_bytes_before': free, 'log_sha256': digest(folder/f'{name}.log'),
                                          'receipt_sha256': receipt_sha}) + '\n')
                print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
        finally:
            stop.set()
            thread.join(10)
            if thread.is_alive() or errors:
                raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed': True, 'output': str(folder)}), flush=True)


if __name__ == '__main__':
    main()
