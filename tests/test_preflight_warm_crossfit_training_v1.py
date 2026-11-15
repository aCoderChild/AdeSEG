"""Synthetic row-provenance checks; no patient labels or GPU are used."""
import json

import pytest

from scripts import preflight_warm_crossfit_training_v1 as preflight
from scripts.memory_token_io import digest


def test_aggregate_must_match_bound_receipt_and_label_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, 'BASE', tmp_path)
    gate = tmp_path / 'gate.json'; gate.write_text('{}')
    certificate_path = tmp_path / 'certificate.json'; certificate_path.write_text('{}')
    reference = tmp_path / 'reference.json'; reference.write_text('{}')
    capture = tmp_path / 'capture.json'
    observation = {'frame': 24, 'feature_file': '/feature', 'feature_sha256': 'f' * 64,
                   'snapshot_file': '/snapshot', 'snapshot_sha256': 's' * 64}
    capture.write_text(json.dumps({'observations': [observation]}))
    metadata = {'path': '/archived-metadata', 'sha256': 'a' * 64}
    row = {'sequence': 'seq6', **observation, 'capture_receipt_path': str(capture),
           'capture_receipt_sha256': digest(capture), 'reference_receipt_path': str(reference),
           'reference_receipt_sha256': digest(reference), 'sequence_audit_path': str(gate),
           'sequence_audit_sha256': digest(gate), 'label_metadata_path': metadata['path'],
           'label_metadata_sha256': metadata['sha256']}
    certificate = {'warm_gate_path': str(gate), 'warm_gate_sha256': digest(gate),
        'warm_plan_sha256': 'w' * 64, 'source_sha256': {},
        'seq6_gate_capture': {'receipt_path': str(capture), 'receipt_sha256': digest(capture)},
        'seq6_historical_reference': {'reference_path': str(reference), 'reference_sha256': digest(reference)}}
    aggregate = {'prompt_policy': 'warm_box_points_then_commit_archived_mask',
                 'certificate_path': str(certificate_path), 'certificate_sha256': digest(certificate_path),
                 'warm_gate_sha256': digest(gate), 'sources': {},
                 'audits': {'seq6': {'path': str(gate), 'sha256': digest(gate)}}, 'observations': [row]}
    folder = tmp_path / 'aggregate'; folder.mkdir()

    def write_aggregate():
        path = folder / 'captures.json'; path.write_text(json.dumps(aggregate))
        (folder / 'completion.json').write_text(json.dumps({'passed': True, 'captures_sha256': digest(path),
                                                          'label_guard': {'open_attempts': 0}}))

    monkeypatch.setattr(preflight, 'validate_plan', lambda: {'existing_manifest_sha256': 'old',
                                                            'fresh_reads': [{'sequence': 'seq6'}]})
    monkeypatch.setattr(preflight, 'load_prepared', lambda _: ({}, 'old'))
    monkeypatch.setattr(preflight, 'verify_certificate', lambda _: (certificate, digest(certificate_path)))
    monkeypatch.setattr(preflight, 'reconstruct_gate', lambda _: ({}, digest(gate)))
    monkeypatch.setattr(preflight, 'validate_warm_index_structure', lambda *a, **k: None)
    monkeypatch.setattr(preflight, 'checked_audits', lambda *a: {})
    monkeypatch.setattr(preflight, 'certified_seq6', lambda *a: tmp_path)
    monkeypatch.setattr(preflight, 'archived_frame_metadata', lambda *a: metadata)
    write_aggregate()
    assert preflight.verified_aggregate(folder)[3] == digest(folder / 'captures.json')
    row['feature_file'] = '/different-but-same-shape'
    write_aggregate()
    with pytest.raises(ValueError, match='aggregate row differs'):
        preflight.verified_aggregate(folder)
    row['feature_file'] = '/feature'
    row['label_metadata_sha256'] = 'b' * 64
    write_aggregate()
    with pytest.raises(ValueError, match='aggregate row differs'):
        preflight.verified_aggregate(folder)


def test_source_closure_detects_transitive_and_local_changes(tmp_path, monkeypatch):
    local = tmp_path / 'local.py'; local.write_text('original')
    dep = tmp_path / 'dep.py'; dep.write_text('original')
    warm = tmp_path / 'warm.py'; warm.write_text('original')
    monkeypatch.setattr(preflight, 'ROOT', tmp_path)
    monkeypatch.setattr(preflight, 'SOURCES', ('local.py',))
    monkeypatch.setattr(preflight, 'channel_dependencies', lambda: {'current': {'dep.py': digest(dep)}, 'prior': {}})
    monkeypatch.setattr(preflight, 'warm_capture_sources', lambda: {'warm.py': digest(warm)})
    frozen = preflight.sources()
    local.write_text('changed')
    assert preflight.sources() != frozen
    local.write_text('original'); dep.write_text('changed')
    assert preflight.sources() != frozen
