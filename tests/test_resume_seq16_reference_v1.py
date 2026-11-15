import json
from pathlib import Path

import pytest

from scripts import resume_seq16_reference_v1 as reuse
from scripts import verify_seq16_reference_reuse_v1 as verifier
from scripts.memory_token_io import digest


def test_reference_copy_is_exact_and_original_is_not_modified(tmp_path):
    source = tmp_path/'old/seq16/reference'; source.mkdir(parents=True)
    (source/'receipt.json').write_text('{"passed": true}')
    (source/'000002.png').write_bytes(b'original frame')
    original = reuse.file_manifest(source)
    destination = tmp_path/'new/seq16/reference'; destination.parent.mkdir(parents=True)
    reuse.copy_verified_reference(tmp_path/'old', destination, {'reference_files': original})
    assert reuse.file_manifest(source) == reuse.file_manifest(destination) == original
    (destination/'000002.png').write_bytes(b'tampered copy')
    assert reuse.file_manifest(source) == original
    assert reuse.file_manifest(destination) != original


def test_reference_rejects_symlink_or_changed_original(tmp_path):
    source = tmp_path/'old/seq16/reference'; source.mkdir(parents=True)
    (source/'receipt.json').write_text('{}'); proof = {'reference_files': reuse.file_manifest(source)}
    (source/'receipt.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed before copy'):
        reuse.copy_verified_reference(tmp_path/'old', tmp_path/'new', proof)
    (source/'alias').symlink_to(source/'receipt.json')
    with pytest.raises(ValueError, match='symlinks'): reuse.file_manifest(source)


def test_original_reference_requires_bound_manifest_and_incomplete_capture(tmp_path, monkeypatch):
    original = tmp_path/'old'; ref = original/'seq16/reference'; ref.mkdir(parents=True)
    (ref/'receipt.json').write_text('{}'); manifest = original/'run_manifest.json'
    cert = tmp_path/'certificate.json'; cert.write_text('{}')
    manifest.write_text(json.dumps({'stage': 'full', 'plan_sha256': reuse.PLAN_SHA,
        'warm_preflight_sha256': reuse.WARM_SHA, 'certificate_path': str(cert.resolve()),
        'certificate_sha256': 'certsha', 'source_sha256': {}, 'label_guard': {'open_attempts': 0}}))
    (original/'phases.jsonl').write_text(json.dumps({'phase': 'seq16_reference'}) + '\n')
    monkeypatch.setattr(reuse, 'OUT', tmp_path)
    monkeypatch.setattr(reuse, 'ORIGINAL_MANIFEST_SHA', digest(manifest))
    monkeypatch.setattr(reuse, 'ORIGINAL_REFERENCE_SHA', digest(ref/'receipt.json'))
    monkeypatch.setattr(reuse, 'source_hashes', lambda: {})
    monkeypatch.setattr(reuse, 'phase_artifact', lambda root, phase, path: digest(path))
    monkeypatch.setattr(reuse, 'warm_plan', lambda: {})
    monkeypatch.setattr(reuse, 'checked_run', lambda *args: ({'observations': []}, digest(ref/'receipt.json')))
    proof = reuse.validate_original(original, cert, {}, {'warm_gate_sha256': 'gate'}, 'certsha')
    assert proof['reference_files']['receipt.json'] == digest(ref/'receipt.json')
    capture = original/'seq16/capture'; capture.mkdir(); (capture/'receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='not partial'):
        reuse.validate_original(original, cert, {}, {'warm_gate_sha256': 'gate'}, 'certsha')


def test_verify_rejects_changed_copy_before_accepting_audit(tmp_path, monkeypatch):
    batch = tmp_path/'new'; ref = batch/'seq16/reference'; ref.mkdir(parents=True)
    (ref/'receipt.json').write_text('{}'); (ref/'frame.png').write_bytes(b'correct')
    files = reuse.file_manifest(ref)
    capture = batch/'seq16/capture'; audit = batch/'seq16/audit'; capture.mkdir(); audit.mkdir()
    (capture/'receipt.json').write_text('{}'); (audit/'audit.json').write_text(json.dumps(
        {'passed': True, 'label_guard': {'open_attempts': 0}}))
    proof = {'original_batch': str(tmp_path/'old'), 'original_manifest_sha256': reuse.ORIGINAL_MANIFEST_SHA,
             'original_reference_receipt_sha256': reuse.ORIGINAL_REFERENCE_SHA, 'reference_files': files}
    (batch/'run_manifest.json').write_text(json.dumps({'stage': 'full', 'plan_sha256': verifier.PLAN_SHA,
        'source_sha256': {}, 'reuse_source_sha256': {}, 'label_guard': {'open_attempts': 0},
        'certificate_path': 'certificate', 'certificate_sha256': 'certsha', 'reference_reuse': proof}))
    (batch/'phases.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'phase': 'seq16_reference', 'reference_reused': True}, {'phase': 'seq16_capture'}, {'phase': 'seq16_audit'}]))
    monkeypatch.setattr(verifier, 'OUT', tmp_path)
    monkeypatch.setattr(verifier, 'source_hashes', lambda: {})
    monkeypatch.setattr(verifier, 'source_identity', lambda: {})
    monkeypatch.setattr(verifier, 'verify_certificate', lambda _: ({}, 'certsha'))
    monkeypatch.setattr(verifier, 'validate_plan', lambda: {})
    monkeypatch.setattr(verifier, 'validate_original', lambda *args: proof)
    monkeypatch.setattr(verifier, 'phase_artifact', lambda root, name, path:
        reuse.ORIGINAL_REFERENCE_SHA if name == 'seq16_reference' else digest(path))
    monkeypatch.setattr(verifier, 'rebuild_audit', lambda *args: {'passed': True})
    assert verifier.verify(batch)['passed'] is True
    (ref/'frame.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Copied reference differs'): verifier.verify(batch)
