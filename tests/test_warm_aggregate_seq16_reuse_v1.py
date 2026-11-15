import json
from pathlib import Path

import pytest

from scripts import verify_warm_aggregate_seq16_reuse_v1 as handoff
from scripts.memory_token_io import digest


def test_handoff_binds_seq16_audit_reference_and_captured_reads(tmp_path, monkeypatch):
    existing = tmp_path/'existing'; existing.mkdir(); (existing/'manifest.json').write_text('{}')
    batch = tmp_path/'batch'; ref = batch/'seq16/reference'; cap = batch/'seq16/capture'; audit = batch/'seq16/audit'
    for path in (ref, cap, audit): path.mkdir(parents=True)
    (ref/'receipt.json').write_text('{}'); (audit/'audit.json').write_text('{}')
    rows = []; observations = []
    for frame in range(12):
        item = {'frame': frame, 'feature_file': f'feature{frame}', 'feature_sha256': 'f'*64,
                'snapshot_file': f'snapshot{frame}', 'snapshot_sha256': 's'*64}
        observations.append(item)
        rows.append(dict(item, sequence='seq16', reference_kind='matched_warm_reference',
            reference_receipt_path=str(ref/'receipt.json'), reference_receipt_sha256=digest(ref/'receipt.json'),
            capture_receipt_path=str(cap/'receipt.json'), capture_receipt_sha256='',
            sequence_audit_path=str(audit/'audit.json'), sequence_audit_sha256=digest(audit/'audit.json')))
    (cap/'receipt.json').write_text(json.dumps({'observations': observations}))
    for row in rows: row['capture_receipt_sha256'] = digest(cap/'receipt.json')
    reuse = {'passed': True, 'batch': str(batch), 'capture_audit_sha256': digest(audit/'audit.json'),
             'reference_receipt_sha256': digest(ref/'receipt.json'), 'certificate_sha256': 'cert',
             'label_guard': {'open_attempts': 0}}
    reuse_file = tmp_path/'reuse.json'; reuse_file.write_text(json.dumps(reuse))
    aggregate_dir = tmp_path/'aggregate'; aggregate_dir.mkdir()
    aggregate = {'passed': True, 'plan_sha256': handoff.PLAN_SHA, 'index_plan_sha256': handoff.PLAN_SHA,
        'warm_preflight_sha256': handoff.WARM_SHA, 'sources': {}, 'certificate_path': 'certificate',
        'certificate_sha256': 'cert', 'warm_gate_sha256': 'gate', 'warm_gate_path': 'gate-path',
        'existing_manifest_sha256': digest(existing/'manifest.json'), 'audits': {'seq16':
            {'path': str(audit/'audit.json'), 'sha256': digest(audit/'audit.json')}}, 'observations': rows}
    path = aggregate_dir/'captures.json'; path.write_text(json.dumps(aggregate))
    (aggregate_dir/'completion.json').write_text(json.dumps({'passed': True, 'captures_sha256': digest(path),
                                                           'label_guard': {'open_attempts': 0}}))
    monkeypatch.setattr(handoff, 'BASE', tmp_path); monkeypatch.setattr(handoff, 'EXISTING', existing)
    monkeypatch.setattr(handoff, 'source_hashes', lambda: {})
    monkeypatch.setattr(handoff, 'verify_certificate', lambda _: ({'warm_gate_sha256': 'gate'}, 'cert'))
    monkeypatch.setattr(handoff, 'validate_plan', lambda: {'existing_manifest_sha256': digest(existing/'manifest.json'),
        'fresh_reads': [{'sequence': 'seq16', 'frame': i} for i in range(12)]})
    monkeypatch.setattr(handoff, 'validate_warm_index_structure', lambda *args, **kwargs: None)
    monkeypatch.setattr(handoff, 'verify_reuse', lambda _: {k: v for k, v in reuse.items() if k != 'label_guard'})
    monkeypatch.setattr(handoff, 'phase_artifact', lambda root, name, p: digest(p))
    assert handoff.verify(aggregate_dir, reuse_file)['passed'] is True
    rows[0]['feature_sha256'] = '0'*64; path.write_text(json.dumps(aggregate))
    (aggregate_dir/'completion.json').write_text(json.dumps({'passed': True, 'captures_sha256': digest(path),
                                                           'label_guard': {'open_attempts': 0}}))
    with pytest.raises(ValueError, match='observation lineage'):
        handoff.verify(aggregate_dir, reuse_file)
