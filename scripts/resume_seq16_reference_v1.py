#!/usr/bin/env python3
"""Prepare a new seq16 capture from a verified completed reference.

The old failed capture remains untouched. Every observer read is rerun from
the beginning. The copied reference phase is explicitly marked as reuse, not
reported as a new inference run. This script is not authorized by a dry run.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_crossfit_warm_video_v2 import checked_run
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import (BASE, DATA, GUARD, MARGIN, OUT, PER_READ, PLAN_SHA,
    phase_artifact, require, validate_plan)
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_crossfit_warm_video_study_v2 import (FULL_AUDITOR, RUNNER, certified_seq6, checked_audits,
    headroom_with_written, remaining_count, run_phase, source_hashes, tensor_bytes)
from scripts.run_crossfit_warm_video_v2 import WARM_SHA, warm_plan
from scripts.source_group_label_guard_v1 import install_mask_open_guard

SEQUENCE = 'seq16'
ORIGINAL_MANIFEST_SHA = '93669f09222e8d1f24c03171bfab79c217758923326fc844956eed3d891b3120'
ORIGINAL_REFERENCE_SHA = 'fd744d71735be85a5d71c0a837ee1bb747fe0e260fe4f38cf17a462847bd8a67'


def source_identity():
    entries = ('scripts/resume_seq16_reference_v1.py', 'scripts/verify_seq16_reference_reuse_v1.py')
    names = set(local_import_closure(ROOT, entries)) | set(entries) | {'scripts/local_import_closure_v1.py'}
    return {name: digest(ROOT/name) for name in sorted(names)}


def file_manifest(folder):
    folder = Path(folder).resolve(strict=True); result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'Reference must not contain symlinks')
        if path.is_file(): result[str(path.relative_to(folder))] = digest(path)
    require(result and 'receipt.json' in result, 'Reference file coverage differs')
    return result


def validate_original(original, certificate_path, plan, certificate, certificate_sha):
    original = Path(original).resolve(strict=True)
    require(original.is_relative_to(OUT.resolve(strict=True)), 'Original batch outside Drive')
    manifest_file = original/'run_manifest.json'; require(digest(manifest_file) == ORIGINAL_MANIFEST_SHA,
        'Original seq16 manifest differs')
    manifest = json.loads(manifest_file.read_text())
    require(digest(manifest_file) == ORIGINAL_MANIFEST_SHA and manifest.get('stage') == 'full' and
        manifest.get('plan_sha256') == PLAN_SHA and manifest.get('warm_preflight_sha256') == WARM_SHA and
        manifest.get('certificate_path') == str(Path(certificate_path).resolve()) and
        manifest.get('certificate_sha256') == certificate_sha and manifest.get('source_sha256') == source_hashes() and
        manifest.get('label_guard', {}).get('open_attempts') == 0, 'Original seq16 provenance differs')
    rows = [json.loads(x) for x in (original/'phases.jsonl').read_text().splitlines() if x.strip()]
    require([x.get('phase') for x in rows] == ['seq16_reference'], 'Original phase history differs')
    ref = original/SEQUENCE/'reference'; receipt = ref/'receipt.json'
    require(phase_artifact(original, 'seq16_reference', receipt) == ORIGINAL_REFERENCE_SHA,
            'Original reference phase differs')
    require(not (original/SEQUENCE/'capture/receipt.json').exists() and
            not (original/SEQUENCE/'audit/audit.json').exists(), 'Original capture was not partial')
    warm = warm_plan()
    rec, sha = checked_run(original, SEQUENCE, 'reference', plan, warm,
                           certificate['warm_gate_sha256'], certificate_sha)
    require(sha == ORIGINAL_REFERENCE_SHA and rec.get('observations') == [], 'Reference reconstruction differs')
    files = file_manifest(ref)
    require(digest(manifest_file) == ORIGINAL_MANIFEST_SHA and
            phase_artifact(original, 'seq16_reference', receipt) == ORIGINAL_REFERENCE_SHA,
            'Original reference changed during reconstruction')
    return {'original_batch': str(original), 'original_manifest_sha256': ORIGINAL_MANIFEST_SHA,
            'original_reference_receipt_sha256': ORIGINAL_REFERENCE_SHA, 'reference_files': files,
            'reference_bytes': sum((ref/name).stat().st_size for name in files)}


def copy_verified_reference(original, destination, proof):
    source = Path(original)/SEQUENCE/'reference'; destination = Path(destination)
    require(file_manifest(source) == proof['reference_files'], 'Original reference changed before copy')
    shutil.copytree(source, destination, symlinks=False)
    require(file_manifest(source) == file_manifest(destination) == proof['reference_files'],
            'Original or copied reference changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True); parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--prior-audit', type=Path, action='append', default=[]); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); guard = install_mask_open_guard(DATA)
    plan = validate_plan(); certificate, cert_sha = verify_certificate(args.certificate)
    original = args.original.resolve(strict=True); frozen_sources = source_hashes(); own_sources = source_identity()
    proof = validate_original(original, args.certificate, plan, certificate, cert_sha)
    accepted = checked_audits(args.prior_audit, plan, frozen_sources, args.certificate)
    accepted['seq6'] = certified_seq6(certificate, plan)
    require(SEQUENCE not in accepted, 'Seq16 was already accepted')
    count = sum(x['sequence'] == SEQUENCE for x in plan['fresh_reads'])
    require(count == 12 and guard['open_attempts'] == 0, 'Seq16 grid or label guard differs')
    free = shutil.disk_usage(BASE).free; need = GUARD + MARGIN + count*PER_READ + proof['reference_bytes']
    require(free >= need, f'Insufficient safe headroom for copy and capture: {free} < {need}')
    if not args.execute:
        print(json.dumps({'ready_for_requested_stage': True, 'scope': 'CPU-only review; no inference or labels.',
            'reference_reuse': proof, 'remaining_reads': remaining_count(plan, accepted),
            'free_bytes': free, 'minimum_bytes': need, 'label_open_attempts': 0})); return
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        require(validate_plan() == plan and verify_certificate(args.certificate) == (certificate, cert_sha) and
                source_hashes() == frozen_sources and source_identity() == own_sources and
                validate_original(original, args.certificate, plan, certificate, cert_sha) == proof,
                'Seq16 reuse inputs changed before launch')
        require(shutil.disk_usage(BASE).free >= need, 'Headroom changed before reference reuse')
        root = Path(tempfile.mkdtemp(prefix='seq16_reference_reuse_v1_', dir=OUT))
        manifest = root/'run_manifest.json'
        manifest.write_text(json.dumps({'stage': 'full', 'scope': 'Seq16 reference copied and verified; observer rerun.',
            'plan_sha256': PLAN_SHA, 'warm_preflight_sha256': WARM_SHA, 'source_sha256': frozen_sources,
            'reuse_source_sha256': own_sources, 'reference_reuse': proof,
            'certificate_path': str(args.certificate.resolve()), 'certificate_sha256': cert_sha,
            'phases': ['seq16_reference', 'seq16_capture', 'seq16_audit'], 'label_guard': dict(guard)}, indent=2) + '\n')
        manifest_sha = digest(manifest); ref = root/SEQUENCE/'reference'; ref.parent.mkdir(parents=True)
        copy_verified_reference(original, ref, proof)
        log = root/'seq16_reference.log'; log.write_text(json.dumps({'reference_reused': True, **proof}, indent=2) + '\n')
        with (root/'phases.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': 'seq16_reference', 'passed': True, 'reference_reused': True,
                'artifact_path': str(ref/'receipt.json'), 'artifact_sha256': digest(ref/'receipt.json'),
                'log_sha256': digest(log)}) + '\n')
        require(phase_artifact(root, 'seq16_reference', ref/'receipt.json') == ORIGINAL_REFERENCE_SHA,
                'Copied reference phase differs')
        capture = root/SEQUENCE/'capture'; audit = root/SEQUENCE/'audit'; budget = count*PER_READ
        tasks = [('seq16_capture', [sys.executable, str(RUNNER), '--sequence', SEQUENCE, '--mode', 'capture',
            '--certificate', str(args.certificate), '--reference', str(ref), '--output', str(capture)], capture),
            ('seq16_audit', [sys.executable, str(FULL_AUDITOR), '--sequence', SEQUENCE, '--root', str(root),
            '--certificate', str(args.certificate), '--output', str(audit)], None)]
        for name, command, folder in tasks:
            require(digest(manifest) == manifest_sha and source_hashes() == frozen_sources and
                source_identity() == own_sources and verify_certificate(args.certificate) == (certificate, cert_sha) and
                file_manifest(original/SEQUENCE/'reference') == proof['reference_files'] and guard['open_attempts'] == 0,
                'Seq16 reuse binding changed')
            written = tensor_bytes(capture) if name == 'seq16_audit' else 0
            space = headroom_with_written(BASE, count, written, budget); started = time.monotonic()
            print(json.dumps({'phase': name, 'status': 'starting', 'output': str(root), **space}), flush=True)
            run_phase(command, root/(name+'.log'), folder, count, deduct_output=name == 'seq16_capture',
                      written=written, sequence_budget=budget)
            artifact = capture/'receipt.json' if folder else audit/'audit.json'
            require(artifact.is_file() and digest(manifest) == manifest_sha and source_identity() == own_sources and
                    source_hashes() == frozen_sources and validate_plan() == plan and
                    verify_certificate(args.certificate) == (certificate, cert_sha), 'Seq16 reuse phase changed')
            with (root/'phases.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                    'artifact_sha256': digest(artifact), 'log_sha256': digest(root/(name+'.log')),
                    'wall_seconds': time.monotonic()-started}) + '\n')
        # A separate CPU process reconstructs both the original reference and
        # the completed new audit before this batch may be handed off.
        reuse_verification = root/'reuse_verification'
        verify_command = [sys.executable, str(ROOT/'scripts/verify_seq16_reference_reuse_v1.py'),
                          '--batch', str(root), '--output', str(reuse_verification)]
        headroom_with_written(BASE, count, tensor_bytes(capture), budget)
        run_phase(verify_command, root/'reuse_verification.log', None, count,
                  written=tensor_bytes(capture), sequence_budget=budget)
        verified_path = reuse_verification/'verification.json'
        verified_sha = digest(verified_path); verified = json.loads(verified_path.read_text())
        require(digest(verified_path) == verified_sha and verified.get('passed') is True and
                verified.get('batch') == str(root) and verified.get('batch_manifest_sha256') == manifest_sha and
                verified.get('label_guard', {}).get('open_attempts') == 0,
                'Independent seq16 reuse verification differs')
        combined = checked_audits([*args.prior_audit, audit/'audit.json'], plan, frozen_sources, args.certificate)
        require(SEQUENCE in combined and guard['open_attempts'] == 0 and
                file_manifest(original/SEQUENCE/'reference') == file_manifest(ref) == proof['reference_files'],
                'Completed seq16 reuse audit differs')
        combined['seq6'] = certified_seq6(certificate, plan)
        print(json.dumps({'passed': True, 'output': str(root), 'audit_path': str(audit/'audit.json'),
            'reuse_verification_path': str(verified_path), 'reuse_verification_sha256': verified_sha,
            'reference_reused': True, 'remaining_reads': remaining_count(plan, combined),
            'label_open_attempts': 0}))


if __name__ == '__main__':
    main()
