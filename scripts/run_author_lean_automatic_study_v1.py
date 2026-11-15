#!/usr/bin/env python3
"""Run the frozen MPS Lean author-system comparison, then score and audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import OUT, exclusive_lock
from scripts.run_additional_author_comparator_study_v2 import (CHECKPOINT, DETECTOR, EXTENDED, REMAINING,
    available, environment, monitor, run_phase)
from scripts.run_author_lean_automatic_v1 import DATA, SEQUENCES, identity, inputs
from scripts.source_group_label_guard_v1 import install_mask_open_guard

RUNNER = ROOT/'scripts/run_author_lean_automatic_v1.py'
VERIFIER = ROOT/'scripts/verify_author_lean_automatic_gate_v1.py'
SCORER = ROOT/'scripts/score_author_lean_automatic_v1.py'
AUDITOR = ROOT/'scripts/audit_author_lean_automatic_v1.py'
SOURCES = (RUNNER, VERIFIER, SCORER, AUDITOR, Path(__file__),
           ROOT/'scripts/run_additional_author_comparator_study_v2.py',
           ROOT/'scripts/resume_source_group_capture_v1.py', ROOT/'scripts/score_boundary_video_factorial_v1.py',
           ROOT/'scripts/score_source_group_boundary_v1.py', ROOT/'scripts/audit_boundary_video_factorial_v1.py',
           ROOT/'scripts/source_group_training_label_guard_v1.py', ROOT/'datasets/io.py')


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}


def commands(folder):
    folder = Path(folder); py = sys.executable
    common = ['--checkpoint', str(CHECKPOINT), '--detector', str(DETECTOR),
              '--extended-preflight', str(EXTENDED), '--remaining-preflight', str(REMAINING)]
    phases = []
    for seq in ('seq6', 'seq23'):
        phases.append((f'smoke_{seq}', [py, str(RUNNER), '--sequence', seq, '--limit', '5',
                       *common, '--output', str(folder/'smoke'/seq)]))
    phases.append(('gate', [py, str(VERIFIER), '--seq6', str(folder/'smoke/seq6'),
                   '--seq23', str(folder/'smoke/seq23'), *common, '--output', str(folder/'gate')]))
    for seq in SEQUENCES:
        phases.append((f'full_{seq}', [py, str(RUNNER), '--sequence', seq, '--gate',
                       str(folder/'gate/gate.json'), *common, '--output', str(folder/'full'/seq)]))
    phases.extend([('verify', [py, str(SCORER), '--mode', 'verify', '--run', str(folder)]),
                   ('score', [py, str(SCORER), '--mode', 'score', '--run', str(folder)]),
                   ('audit', [py, str(AUDITOR), '--run', str(folder)])])
    return phases


def artifact(folder, name):
    if name.startswith('smoke_'): return folder/'smoke'/name.split('_', 1)[1]/'receipt.json'
    if name.startswith('full_'): return folder/'full'/name.split('_', 1)[1]/'receipt.json'
    return folder / {'gate': 'gate/gate.json', 'verify': 'verification/verification.json',
                     'score': 'scores/results.json', 'audit': 'independent_audit/audit.json'}[name]


def validate_resume(folder):
    folder = Path(folder).resolve(strict=True)
    if not folder.is_relative_to(OUT.resolve(strict=True)): raise ValueError('Batch outside experiment root')
    phases = commands(folder); path = folder/'run_manifest.json'; sha = digest(path)
    data = json.loads(path.read_text())
    if (digest(path) != sha or data.get('sequences') != list(SEQUENCES) or
            data.get('phases') != [name for name, _ in phases] or data.get('sources') != source_hashes() or
            data.get('identity') != identity('seq6', CHECKPOINT, DETECTOR) or
            data.get('label_guard', {}).get('open_attempts') != 0):
        raise ValueError('Frozen batch inputs differ')
    rows = [json.loads(s) for s in (folder/'phases.jsonl').read_text().splitlines()] if (folder/'phases.jsonl').exists() else []
    if len(rows) >= len(phases) or [r.get('phase') for r in rows] != [x[0] for x in phases[:len(rows)]]:
        raise ValueError('Completed phases are not an incomplete prefix')
    for row in rows:
        name = row['phase']
        if (row.get('passed') is not True or row.get('artifact_sha256') != digest(artifact(folder, name)) or
                row.get('log_sha256') != digest(folder/(name+'.log'))):
            raise ValueError('Frozen phase artifact or log changed')
    if len(rows) >= 3:
        from scripts.verify_author_lean_automatic_gate_v1 import verify_gate
        verify_gate(folder/'gate/gate.json', identity('seq6', CHECKPOINT, DETECTOR),
                    CHECKPOINT, DETECTOR, EXTENDED, REMAINING)
    return folder, phases, sha, len(rows)


def preserve_partial(folder, name):
    target = artifact(folder, name).parent; log = folder/(name+'.log')
    if target.exists() or log.exists():
        dest = folder/'partial_attempts'/f'{name}_{uuid.uuid4().hex}'; dest.mkdir(parents=True)
        if target.exists(): target.rename(dest/'output')
        if log.exists(): log.rename(dest/'phase.log')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', type=Path); a = p.parse_args()
    if any(os.environ.get(k) != '0' for k in ('PYTHONHASHSEED', 'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH')):
        raise ValueError('Required MPS environment differs')
    guard = install_mask_open_guard(DATA)
    if a.resume: _, _, _, done = validate_resume(a.resume)
    else: done = 0
    for seq in SEQUENCES: inputs(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
    source_hashes(); identity('seq6', CHECKPOINT, DETECTOR)
    if guard['open_attempts']: raise ValueError('Label open during preflight')
    if not a.execute:
        print(json.dumps({'preflight_passed': True, 'completed_phases': done,
                          'free_bytes': available(OUT), 'target_label_opens': 0})); return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        available(OUT)
        if a.resume:
            folder, phases, manifest_sha, done = validate_resume(a.resume)
            preserve_partial(folder, phases[done][0])
        else:
            folder = Path(tempfile.mkdtemp(prefix='author_lean_automatic_mps_v1_', dir=OUT))
            phases = commands(folder)
            manifest = {'scope': 'Previously inspected cases; compatibility-corrected author Lean.',
                        'sequences': list(SEQUENCES), 'phases': [name for name, _ in phases],
                        'sources': source_hashes(), 'identity': identity('seq6', CHECKPOINT, DETECTOR),
                        'label_guard': dict(guard)}
            (folder/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            manifest_sha = digest(folder/'run_manifest.json')
        stop, ready, errors = threading.Event(), threading.Event(), []
        thread = threading.Thread(target=monitor, args=(folder, stop, ready, errors,
            folder/f'free_space_{uuid.uuid4().hex}.jsonl'), daemon=True)
        thread.start(); ready.wait(10)
        try:
            if not ready.is_set() or errors: raise RuntimeError('Storage monitor failed')
            for name, cmd in phases[done:]:
                saved = json.loads((folder/'run_manifest.json').read_text())
                if (digest(folder/'run_manifest.json') != manifest_sha or source_hashes() != saved['sources'] or
                        identity('seq6', CHECKPOINT, DETECTOR) != saved['identity'] or guard['open_attempts'] or errors):
                    raise RuntimeError('Frozen source, model or guard changed')
                free = available(folder); start = time.monotonic()
                print(json.dumps({'phase': name, 'status': 'starting', 'output': str(folder), 'free_bytes': free}), flush=True)
                run_phase(cmd, folder/(name+'.log'), folder, errors)
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase': name, 'passed': True, 'wall_seconds': time.monotonic()-start,
                        'artifact_sha256': digest(artifact(folder, name)),
                        'log_sha256': digest(folder/(name+'.log'))})+'\n')
                print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
        finally:
            stop.set(); thread.join(10)
            if thread.is_alive() or errors: raise RuntimeError('Storage monitor did not close cleanly')
    print(json.dumps({'passed': True, 'output': str(folder)}), flush=True)


if __name__ == '__main__': main()
