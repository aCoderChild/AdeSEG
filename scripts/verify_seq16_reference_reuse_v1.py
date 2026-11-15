#!/usr/bin/env python3
"""Reconstruct a seq16 reuse batch, without target labels or model inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_crossfit_warm_video_v2 import audit as rebuild_audit
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, OUT, PLAN_SHA, phase_artifact, require, validate_plan
from scripts.memory_token_io import digest
from scripts.resume_seq16_reference_v1 import (ORIGINAL_MANIFEST_SHA, ORIGINAL_REFERENCE_SHA, SEQUENCE,
    file_manifest, source_identity, validate_original)
from scripts.run_crossfit_warm_video_study_v2 import source_hashes
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def verify(batch):
    batch = Path(batch).resolve(strict=True); require(batch.is_relative_to(OUT.resolve(strict=True)), 'Batch outside Drive')
    manifest_path = batch/'run_manifest.json'; manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(digest(manifest_path) == manifest_sha and manifest.get('stage') == 'full' and
            manifest.get('plan_sha256') == PLAN_SHA and manifest.get('source_sha256') == source_hashes() and
            manifest.get('reuse_source_sha256') == source_identity() and
            manifest.get('label_guard', {}).get('open_attempts') == 0, 'Reuse manifest differs')
    certificate, cert_sha = verify_certificate(manifest['certificate_path'])
    require(cert_sha == manifest['certificate_sha256'], 'Reuse certificate differs')
    proof = manifest['reference_reuse']; original = Path(proof['original_batch'])
    require(proof.get('original_manifest_sha256') == ORIGINAL_MANIFEST_SHA and
            proof.get('original_reference_receipt_sha256') == ORIGINAL_REFERENCE_SHA and
            validate_original(original, manifest['certificate_path'], validate_plan(), certificate, cert_sha) == proof,
            'Original seq16 reference proof differs')
    copied = batch/SEQUENCE/'reference'
    require(file_manifest(copied) == proof['reference_files'] and
            phase_artifact(batch, 'seq16_reference', copied/'receipt.json') == ORIGINAL_REFERENCE_SHA,
            'Copied reference differs from original')
    phases = [json.loads(line) for line in (batch/'phases.jsonl').read_text().splitlines() if line.strip()]
    require([row['phase'] for row in phases] == ['seq16_reference', 'seq16_capture', 'seq16_audit'] and
            phases[0].get('reference_reused') is True, 'Reuse phase history differs')
    phase_artifact(batch, 'seq16_capture', batch/SEQUENCE/'capture/receipt.json')
    audit_path = batch/SEQUENCE/'audit/audit.json'
    audit_sha = phase_artifact(batch, 'seq16_audit', audit_path); saved = json.loads(audit_path.read_text())
    rebuilt = rebuild_audit(batch, SEQUENCE, manifest['certificate_path'])
    require(digest(audit_path) == audit_sha and saved.get('label_guard', {}).get('open_attempts') == 0 and
            {k: v for k, v in saved.items() if k != 'label_guard'} == rebuilt and
            file_manifest(copied) == proof['reference_files'] and digest(manifest_path) == manifest_sha,
            'Completed seq16 audit differs')
    return {'passed': True, 'scope': 'Reused seq16 reference and rerun observer, no labels or efficacy.',
        'batch': str(batch), 'batch_manifest_sha256': manifest_sha, 'original_batch': str(original),
        'original_manifest_sha256': ORIGINAL_MANIFEST_SHA, 'reference_receipt_sha256': ORIGINAL_REFERENCE_SHA,
        'capture_audit_sha256': audit_sha, 'certificate_sha256': cert_sha,
        'reuse_sources': source_identity(), 'auditor_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA); result = verify(args.batch)
    require(guard['open_attempts'] == 0, 'Annotation opened during verification')
    result['label_guard'] = dict(guard); out.mkdir(parents=True)
    (out/'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(out)}))


if __name__ == '__main__':
    main()
