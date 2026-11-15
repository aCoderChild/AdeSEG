#!/usr/bin/env python3
"""Run only the two direct-prompt author-Lean smokes and their gate.

The resulting batch uses the existing launch manifest and phase format, so the
reviewed full-study launcher can later resume it. This script never starts a
full-video, scoring or audit phase.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import OUT, exclusive_lock
from scripts.run_additional_author_comparator_study_v2 import (CHECKPOINT, DETECTOR, EXTENDED, REMAINING,
    available, monitor, environment)
from scripts.run_author_lean_automatic_study_v1 import artifact, commands, source_hashes
from scripts.run_author_lean_automatic_v1 import DATA, SEQUENCES, identity, inputs
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def short_phases(folder):
    planned = commands(folder)
    if [x[0] for x in planned[:3]] != ['smoke_seq6', 'smoke_seq23', 'gate']:
        raise ValueError('Author-Lean short phase order differs')
    return planned[:3]


MIN_SHORT_FREE = 26 * 1024**3


def short_available(folder):
    free = available(folder)
    if free < MIN_SHORT_FREE:
        raise RuntimeError('Short smoke 26-GiB reserve lost; partial outputs retained')
    return free


def run_short_phase(cmd, log, folder, errors):
    short_available(folder)
    with log.open('x') as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                try:
                    short_available(folder)
                    if errors: raise RuntimeError('Storage monitor failed')
                except Exception:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise
        if code or errors: raise RuntimeError(f'Short phase failed; inspect {log}')
        short_available(folder)


def execute_phases(folder, validate, errors):
    for name, cmd in short_phases(folder):
        validate()
        free = short_available(folder); started = time.monotonic()
        print(json.dumps({'phase': name, 'status': 'starting', 'output': str(folder), 'free_bytes': free}), flush=True)
        run_short_phase(cmd, folder/(name+'.log'), folder, errors)
        validate()
        with (folder/'phases.jsonl').open('a') as log:
            log.write(json.dumps({'phase': name, 'passed': True, 'wall_seconds': time.monotonic()-started,
                'artifact_sha256': digest(artifact(folder, name)), 'log_sha256': digest(folder/(name+'.log'))})+'\n')
        print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)


def write_handoff(folder, own_sha, manifest_sha):
    # Archive the exact short launcher: the original full launcher does not verify it.
    snapshot = folder/'short_only_launcher.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    if digest(snapshot) != own_sha: raise RuntimeError('Short launcher snapshot changed')
    handoff = {'scope': 'Two short smokes and gate only; no full videos or label reads.',
        'launcher_snapshot': str(snapshot), 'launcher_sha256': own_sha,
        'manifest_sha256': manifest_sha, 'phases_sha256': digest(folder/'phases.jsonl'),
        'gate_sha256': digest(artifact(folder, 'gate')),
        'full_resume_requires_independent_handoff_validation': True}
    (folder/'short_handoff.json').write_text(json.dumps(handoff, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if any(os.environ.get(k) != '0' for k in ('PYTHONHASHSEED', 'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH')):
        raise ValueError('Required MPS environment differs')
    guard = install_mask_open_guard(DATA); own_sha = digest(Path(__file__)); sources = source_hashes()
    for seq in SEQUENCES: inputs(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
    expected_identity = identity('seq6', CHECKPOINT, DETECTOR); short_available(OUT)
    if guard['open_attempts']: raise ValueError('Label opened during short preflight')
    if not args.execute:
        print(json.dumps({'passed': True, 'scope': 'Dry run only.',
                          'phases': [name for name, _ in short_phases(OUT/'DRY_RUN')]})); return
    import torch
    if not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        short_available(OUT)
        if source_hashes() != sources or digest(Path(__file__)) != own_sha or \
                identity('seq6', CHECKPOINT, DETECTOR) != expected_identity:
            raise ValueError('Author-Lean preflight changed')
        folder = Path(tempfile.mkdtemp(prefix='author_lean_automatic_mps_v1_', dir=OUT))
        manifest = {'scope': 'Previously inspected cases; compatibility-corrected author Lean.',
            'sequences': list(SEQUENCES), 'phases': [name for name, _ in commands(folder)],
            'sources': sources, 'identity': expected_identity, 'label_guard': dict(guard),
            'short_only_launcher_sha256': own_sha}
        path = folder/'run_manifest.json'; path.write_text(json.dumps(manifest, indent=2)+'\n'); manifest_sha = digest(path)
        stop, ready, errors = threading.Event(), threading.Event(), []
        thread = threading.Thread(target=monitor, args=(folder, stop, ready, errors,
            folder/f'free_space_{uuid.uuid4().hex}.jsonl'), daemon=True)
        thread.start(); ready.wait(10)
        try:
            if not ready.is_set() or errors: raise RuntimeError('Storage monitor failed')
            def validate():
                if digest(path) != manifest_sha or source_hashes() != sources or digest(Path(__file__)) != own_sha or \
                        identity('seq6', CHECKPOINT, DETECTOR) != expected_identity or guard['open_attempts'] or errors:
                    raise RuntimeError('Short-study sources or guard changed')
            execute_phases(folder, validate, errors)
            write_handoff(folder, own_sha, manifest_sha)
        finally:
            stop.set(); thread.join(10)
            if thread.is_alive() or errors: raise RuntimeError('Storage monitor did not close cleanly')
    print(json.dumps({'passed': True, 'scope': 'Two direct-prompt MPS smokes plus gate; no full videos or labels.',
                      'output': str(folder), 'label_open_attempts': guard['open_attempts']}))


if __name__ == '__main__': main()
