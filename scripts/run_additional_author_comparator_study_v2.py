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
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.resume_source_group_capture_v1 import BASE, OUT, exclusive_lock
from scripts.run_additional_author_comparator_v2 import (CHECKPOINT_SHA, DATA, DETECTOR_SHA, METHODS,
    REPRESENTATIVE, SEQUENCES, TRAIN_PREFLIGHT, input_data, sources)

CHECKPOINT = BASE / 'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
DETECTOR = ROOT / 'checkpoints/polypgen_yolov8n.pt'
EXTENDED = OUT / 'extended_boundary_video_preflight_attempt01'
REMAINING = OUT / 'remaining_boundary_video_preflight_v1_attempt04'
RUNNER = ROOT / 'scripts/run_additional_author_comparator_v2.py'
VERIFIER = ROOT / 'scripts/verify_additional_author_comparator_gate_v2.py'
SCORER = ROOT / 'scripts/score_additional_author_comparators_v2.py'
AUDITOR = ROOT / 'scripts/audit_additional_author_comparators_v2.py'
GUARD = 22 * 1024**3


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
            'lock_helper': digest(ROOT / 'scripts/resume_source_group_capture_v1.py'),
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
    for method in METHODS:
        seq=REPRESENTATIVE[method];root=folder/'short'/method/seq
        cmd=[py,str(RUNNER),'--method',method,'--sequence',seq,'--device','mps',
             '--limit','5',*common,'--output',str(root/'smoke')]
        phases.append((f'short_{method}_{seq}_mps',cmd))
        cmd=[py,str(VERIFIER),'--smoke',str(root/'smoke'),*common,'--output',str(root/'gate')]
        phases.append((f'gate_{method}_{seq}',cmd))
    for seq in selection:
        for method in METHODS:
            output = folder/'full'/seq/method
            gate = folder/'short'/method/REPRESENTATIVE[method]/'gate/gate.json'
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


def monitor(folder, stop, ready, errors, monitor_path):
    try:
        with monitor_path.open('x') as out:
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


def run_phase(cmd, log, folder, errors):
    """Stop an active child if the 22-GiB reserve or monitor is lost."""
    with log.open('x') as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code=process.wait(timeout=2);break
            except subprocess.TimeoutExpired:
                if errors or shutil.disk_usage(folder).free < GUARD:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError('Storage reserve lost during phase; partial outputs retained')
        if code or errors or shutil.disk_usage(folder).free < GUARD:
            raise RuntimeError(f'Phase failed or storage reserve lost; inspect {log}')


def artifact(folder,name):
    if name in ('verify','score','audit'):
        return folder/({'verify':'verification/verification.json','score':'scores/results.json',
                        'audit':'independent_audit/audit.json'}[name])
    parts=name.split('_')
    if parts[0]=='short':return folder/'short'/parts[1]/parts[2]/'smoke/receipt.json'
    if parts[0]=='gate':return folder/'short'/parts[1]/parts[2]/'gate/gate.json'
    if parts[0]=='full':return folder/'full'/parts[1]/parts[2]/'receipt.json'
    raise ValueError('Unknown phase artifact')


def validate_resume(folder):
    folder=Path(folder).resolve(strict=True)
    if not folder.is_relative_to(OUT.resolve(strict=True)):
        raise ValueError('Resume directory outside experiment folder')
    manifest=folder/'run_manifest.json';saved=json.loads(manifest.read_text());selection=tuple(saved.get('sequences',[]))
    phases,mapping=commands(folder,selection);map_path=folder/'run_map.json'
    if (saved.get('sources')!=source_hashes() or saved.get('run_map')!=mapping or
            json.loads(map_path.read_text())!=mapping or saved.get('methods')!=list(METHODS) or
            saved.get('representative_for_method')!=REPRESENTATIVE or
            saved.get('phases')!=[name for name,_ in phases] or
            saved.get('checkpoint_sha256')!=digest(CHECKPOINT) or saved.get('detector_sha256')!=digest(DETECTOR) or
            saved.get('extended_preflight_sha256')!=digest(EXTENDED/'manifest.json') or
            saved.get('remaining_preflight_sha256')!=digest(REMAINING/'manifest.json') or
            saved.get('train_preflight_sha256')!=digest(TRAIN_PREFLIGHT/'manifest.json') or
            saved.get('launcher_label_guard',{}).get('open_attempts')!=0 or
            saved.get('python_hash_seed')!='0' or saved.get('mps_fallback')!='0' or saved.get('mps_fast_math')!='0'):
        raise ValueError('Frozen comparator launch inputs differ')
    phase_file=folder/'phases.jsonl'
    rows=[json.loads(x) for x in phase_file.read_text().splitlines() if x.strip()] if phase_file.exists() else []
    if len(rows)>=len(phases) or [x.get('phase') for x in rows]!=[name for name,_ in phases[:len(rows)]]:
        raise ValueError('Completed phases are not an incomplete planned prefix')
    for row in rows:
        name=row['phase'];file=artifact(folder,name)
        if (row.get('passed') is not True or row.get('log_sha256')!=digest(folder/(name+'.log')) or
                row.get('artifact_sha256')!=digest(file)):
            raise ValueError('Completed comparator artifact or log changed')
        if name.startswith('full_') and row.get('receipt_sha256')!=digest(file):
            raise ValueError('Frozen comparator full receipt differs')
    # Reconstruct every completed representative smoke audit before reuse.
    from scripts.run_additional_author_comparator_v2 import identity,verify_gate
    for method in METHODS:
        seq=REPRESENTATIVE[method];gate_name=f'gate_{method}_{seq}'
        if gate_name in {x['phase'] for x in rows}:
            _,_,_,prompt=input_data(seq,EXTENDED,REMAINING,CHECKPOINT,DETECTOR)
            expected=identity(method,seq,CHECKPOINT,DETECTOR,EXTENDED,REMAINING,'mps')
            verify_gate(artifact(folder,gate_name),expected,prompt,CHECKPOINT,DETECTOR,EXTENDED,REMAINING)
    return folder,selection,phases,mapping,digest(manifest),digest(map_path),len(rows)


def preserve_partial(folder,name):
    target=artifact(folder,name).parent;log=folder/(name+'.log')
    if not target.exists() and not log.exists():return None
    dest=folder/'partial_attempts'/f'{name}_{uuid.uuid4().hex}';dest.mkdir(parents=True,exist_ok=False)
    if target.exists():target.rename(dest/'output')
    if log.exists():log.rename(dest/'phase.log')
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--scope', choices=('all', 'seq10'), default='all')
    p.add_argument('--resume', type=Path, help='Resume a source-bound incomplete comparator batch')
    a = p.parse_args()
    guard = install_mask_open_guard(DATA)
    selection = SEQUENCES if a.scope == 'all' else ('seq10',)
    completed=0
    if a.resume:
        _,selection,_,_,_,_,completed=validate_resume(a.resume)
    def check():
        if digest(CHECKPOINT) != CHECKPOINT_SHA or digest(DETECTOR) != DETECTOR_SHA:
            raise ValueError('Frozen checkpoint/detector differs')
        for seq in set(selection)|set(REPRESENTATIVE.values()):
            input_data(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        source_hashes()
        if guard['open_attempts']:
            raise RuntimeError('Launcher attempted to open a target label')
    check()
    if not a.execute:
        phases, _ = commands(OUT/'DRY_RUN', selection)
        print(json.dumps({'preflight_passed': True, 'scope': 'No GPU inference or target labels opened.',
                          'free_bytes': available(OUT),'completed_phases':completed,
                          'remaining_phases': [name for name, _ in phases[completed:]]}))
        return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        check()
        available(OUT)
        if a.resume:
            folder,selection,phases,mapping,frozen_manifest,frozen_map,completed=validate_resume(a.resume)
            preserve_partial(folder,phases[completed][0])
        else:
            folder = Path(tempfile.mkdtemp(prefix='additional_author_comparator_v2_', dir=OUT))
            phases, mapping = commands(folder, selection);completed=0
            manifest = folder/'run_manifest.json'
            manifest.write_text(json.dumps({'scope': 'Exploratory, previously inspected videos.',
            'sequences': list(selection), 'methods': list(METHODS),
            'representative_for_method': REPRESENTATIVE,
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
        manifest=folder/'run_manifest.json'
        stop, ready, errors = threading.Event(), threading.Event(), []
        monitor_path=folder/(f'free_space_resume_{uuid.uuid4().hex}.jsonl' if a.resume else 'free_space.jsonl')
        thread = threading.Thread(target=monitor, args=(folder, stop, ready, errors, monitor_path), daemon=True)
        thread.start()
        ready.wait(10)
        try:
            if not ready.is_set() or errors:
                raise RuntimeError(f'Storage monitor failed: {errors}')
            for name, cmd in phases[completed:]:
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
                run_phase(cmd,folder/f'{name}.log',folder,errors)
                if errors:
                    raise RuntimeError(f'Storage monitor failed: {errors}')
                receipt_sha = None
                if name.startswith('full_'):
                    _, seq, method = name.split('_', 2)
                    receipt_sha = digest(Path(mapping[seq][method])/'receipt.json')
                artifact_sha=digest(artifact(folder,name))
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase': name, 'passed': True, 'wall_seconds': time.monotonic()-start,
                                          'free_bytes_before': free, 'log_sha256': digest(folder/f'{name}.log'),
                                          'receipt_sha256': receipt_sha,'artifact_sha256':artifact_sha}) + '\n')
                print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
        finally:
            stop.set()
            thread.join(10)
            if thread.is_alive() or errors:
                raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed': True, 'output': str(folder)}), flush=True)


if __name__ == '__main__':
    main()
