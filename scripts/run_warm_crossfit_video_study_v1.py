#!/usr/bin/env python3
"""Resumable source-bound warm video study; dry run by default.

Full-video branches require the phase-bound, reconstructed MPS smoke audit;
the scorer verifies every completed branch before opening evaluation labels.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_video_v1 import sources
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_warm_crossfit_study_v1 import headroom, preserve_partial, run_phase
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import load_preflight, require
from scripts.warm_crossfit_video_contract_v1 import NO_INITIALIZATION

SEQ = tuple(f'seq{i}' for i in range(1, 24) if f'seq{i}' not in NO_INITIALIZATION)
ARMS = ('B', 'C', 'D')
CERT_TOKEN = 'VIDEO_CERTIFICATE_SHA256'


def phases(root, crossfit, preflight):
    root = Path(root); py = sys.executable; cert = root / 'certificate/certificate.json'
    tasks = [('certificate', [py, str(ROOT / 'scripts/preflight_warm_crossfit_video_v1.py'),
        '--crossfit-root', str(crossfit), '--preflight', str(preflight), '--output', str(cert.parent)], cert)]
    for seq in ('seq3', 'seq23'):
        for arm in ARMS:
            out = root / 'smoke' / seq / arm
            tasks.append((f'smoke_{seq}_{arm}', [py, str(ROOT / 'scripts/run_warm_crossfit_video_v1.py'),
                '--certificate', str(cert), '--certificate-sha256', CERT_TOKEN, '--sequence', seq,
                '--arm', arm, '--limit', '4', '--output', str(out)], out / 'receipt.json'))
    out = root / 'smoke_audit'
    tasks.append(('smoke_audit', [py, str(ROOT / 'scripts/audit_warm_crossfit_video_smoke_v1.py'),
        '--root', str(root), '--certificate', str(cert), '--certificate-sha256', CERT_TOKEN,
        '--output', str(out)], out / 'audit.json'))
    for seq in SEQ:
        for arm in ARMS:
            out = root / 'full' / seq / arm
            tasks.append((f'full_{seq}_{arm}', [py, str(ROOT / 'scripts/run_warm_crossfit_video_v1.py'),
                '--certificate', str(cert), '--certificate-sha256', CERT_TOKEN, '--sequence', seq,
                '--arm', arm, '--limit', '0', '--smoke-audit', str(root / 'smoke_audit/audit.json'),
                '--output', str(out)], out / 'receipt.json'))
    for mode in ('verify', 'score', 'audit'):
        out = root / mode
        file = out / ('verification.json' if mode == 'verify' else 'results.json' if mode == 'score' else 'audit.json')
        tasks.append((mode, [py, str(ROOT / 'scripts/score_warm_crossfit_video_v1.py'), '--mode', mode,
                             '--root', str(root), '--output', str(out)], file))
    return tasks


def validate_resume(root, crossfit, preflight, plan_sha, source_map):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(OUT.resolve(strict=True)), 'Batch outside Drive')
    manifest_path = root / 'run_manifest.json'; manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text()); tasks = phases(root, crossfit, preflight)
    require(digest(manifest_path) == manifest_sha and manifest.get('crossfit_root') == str(crossfit) and
            manifest.get('preflight_path') == str(preflight) and manifest.get('preflight_sha256') == plan_sha and
            manifest.get('source_sha256') == source_map and manifest.get('phases') == [x[0] for x in tasks],
            'Video study manifest differs')
    log = root / 'phases.jsonl'
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    require(len(rows) < len(tasks) and [x['phase'] for x in rows] == [x[0] for x in tasks[:len(rows)]],
            'Video phase prefix differs')
    for row, (_, _, artifact) in zip(rows, tasks, strict=False):
        require(phase_artifact(root, row['phase'], artifact) == row['artifact_sha256'], 'Video phase changed')
    return root, manifest_sha, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit-root', type=Path, required=True); parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--resume', type=Path); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); guard = install_mask_open_guard(DATA)
    crossfit = args.crossfit_root.resolve(strict=True); preflight = args.preflight.resolve(strict=True)
    _, plan_sha = load_preflight(preflight); source_map = sources(); headroom(OUT)
    if args.resume: validate_resume(args.resume, crossfit, preflight, plan_sha, source_map)
    require(guard['open_attempts'] == 0, 'Annotation opened during launcher preflight')
    if not args.execute:
        print(json.dumps({'passed': True, 'scope': 'Dry run only; no MPS or annotation reads.',
            'full_execution_ready': False, 'phases': [x[0] for x in phases(OUT / 'DRY_RUN', crossfit, preflight)]}))
        return
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT / '.source_group_capture_resume_v1.lock'):
        require(load_preflight(preflight)[1] == plan_sha and sources() == source_map, 'Preflight changed')
        if args.resume:
            root, manifest_sha, completed = validate_resume(args.resume, crossfit, preflight, plan_sha, source_map)
            name, _, artifact = phases(root, crossfit, preflight)[completed]; preserve_partial(root, name, artifact)
        else:
            root = Path(tempfile.mkdtemp(prefix='warm_crossfit_video_v1_', dir=OUT)); completed = 0
            manifest = root / 'run_manifest.json'
            manifest.write_text(json.dumps({'scope': 'Exploratory warm cross-fit video; inspected patients.',
                'crossfit_root': str(crossfit), 'preflight_path': str(preflight), 'preflight_sha256': plan_sha,
                'source_sha256': source_map, 'phases': [x[0] for x in phases(root, crossfit, preflight)]}, indent=2) + '\n')
            manifest_sha = digest(manifest)
        tasks = phases(root, crossfit, preflight)
        for name, command, artifact in tasks[completed:]:
            require(digest(root / 'run_manifest.json') == manifest_sha and sources() == source_map and
                    load_preflight(preflight)[1] == plan_sha and guard['open_attempts'] == 0,
                    'Video manifest or label guard changed')
            headroom(root); started = time.monotonic()
            if name != 'certificate':
                cert = root / 'certificate/certificate.json'; cert_sha = phase_artifact(root, 'certificate', cert)
                command = [cert_sha if x == CERT_TOKEN else x for x in command]
            print(json.dumps({'phase': name, 'status': 'starting', 'output': str(root)}), flush=True)
            run_phase(command, root / (name + '.log'), root)
            require(artifact.is_file() and sources() == source_map, 'Video phase artifact differs')
            with (root / 'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                    'artifact_sha256': digest(artifact), 'log_sha256': digest(root / (name + '.log')),
                    'wall_seconds': time.monotonic() - started}) + '\n')
            print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
    print(json.dumps({'passed': True, 'output': str(root)}))


if __name__ == '__main__': main()
