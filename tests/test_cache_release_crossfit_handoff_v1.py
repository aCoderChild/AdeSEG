"""Synthetic mixed-capture proof and grid checks, no MPS or target labels."""
import copy
import json
from pathlib import Path

import pytest

from scripts import verify_cache_release_crossfit_handoff_v1 as handoff
from scripts.memory_token_io import digest


def test_cache_binding_requires_both_sidecars_and_all_four_traces(tmp_path, monkeypatch):
    root = tmp_path / 'batch'; seq = 'seq11'; saved = {}
    for mode in ('reference', 'capture'):
        folder = root / seq / mode; folder.mkdir(parents=True)
        receipt_path = folder / 'receipt.json'
        for kind in ('setup', 'frame'):
            trace = root / seq / f'{mode}_{kind}_cache.jsonl'; trace.write_text(mode + kind)
        receipt = {f'cache_{kind}_sha256': digest(root / seq / f'{mode}_{kind}_cache.jsonl')
                   for kind in ('setup', 'frame')}
        receipt_path.write_text(json.dumps(receipt))
        side = folder / 'bounded_receipt.json'
        side.write_text(json.dumps({'passed': True, 'original_receipt_sha256': digest(receipt_path)}))
        saved[f'{mode}_sha256'] = digest(receipt_path); saved[f'{mode}_bounded_sha256'] = digest(side)
    monkeypatch.setattr(handoff, 'phase_artifact', lambda _root, _name, path: digest(path))
    result = handoff.cache_bindings(root, seq, saved)
    assert result['reference_sidecar_sha256'] == saved['reference_bounded_sha256']
    assert sum('_trace_sha256' in key for key in result) == 4
    (root / seq / 'capture_frame_cache.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='trace differs'):
        handoff.cache_bindings(root, seq, saved)


def test_release_audit_requires_manifest_identity_and_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, 'OUT', tmp_path)
    root = tmp_path / 'batch'; folder = root / 'seq11' / 'audit'; folder.mkdir(parents=True)
    for name in ('certificate', 'short', 'memory', 'seq6'):
        (tmp_path / name).write_text(name)
    cert, short, memory, seq6 = [tmp_path / name for name in ('certificate', 'short', 'memory', 'seq6')]
    saved = {'passed': True, 'sequence': 'seq11', 'certificate_sha256': 'c', 'short_gate_sha256': 's',
             'memory_gate_sha256': 'm', 'seq6_proof_sha256': 'six', 'label_guard': {'open_attempts': 0}}
    path = folder / 'audit.json'; path.write_text(json.dumps(saved))
    manifest = {'sequence': 'seq11', 'certificate_path': str(cert), 'short_gate_path': str(short),
                'memory_gate_path': str(memory), 'memory_gate_sha256': 'm',
                'seq6_proof_path': str(seq6), 'seq6_proof_sha256': 'six'}
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(handoff, 'phase_artifact', lambda _root, _name, file: digest(file))
    monkeypatch.setattr(handoff, 'audit_cache_release_v2', lambda *args: {k: v for k, v in saved.items() if k != 'label_guard'})
    assert handoff.verify_release_audit(path, 'seq11', cert, short, memory, seq6, 'c', 's', 'm', 'six')[2] == digest(path)
    changed = copy.deepcopy(manifest); changed['memory_gate_sha256'] = 'different'
    (root / 'run_manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='diagnostic'):
        handoff.verify_release_audit(path, 'seq11', cert, short, memory, seq6, 'c', 's', 'm', 'six')
    changed = copy.deepcopy(manifest); changed['seq6_proof_path'] = str(tmp_path / 'other')
    (root / 'run_manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='compatibility proof'):
        handoff.verify_release_audit(path, 'seq11', cert, short, memory, seq6, 'c', 's', 'm', 'six')
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(handoff, 'audit_cache_release_v2', lambda *args: {'passed': True})
    with pytest.raises(ValueError, match='independent reconstruction'):
        handoff.verify_release_audit(path, 'seq11', cert, short, memory, seq6, 'c', 's', 'm', 'six')


def test_eager_audit_is_reconstructed_not_only_receipt_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, 'OUT', tmp_path)
    root = tmp_path / 'batch'; folder = root / 'seq10' / 'audit'; folder.mkdir(parents=True)
    certificate = tmp_path / 'certificate.json'; certificate.write_text('{}')
    path = folder / 'audit.json'
    saved = {'passed': True, 'sequence': 'seq10', 'target_frames': [1, 2],
             'label_guard': {'open_attempts': 0}}
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(handoff, 'phase_artifact', lambda _root, _name, file: digest(file))
    monkeypatch.setattr(handoff, 'audit_eager_v2', lambda *args: {k: v for k, v in saved.items() if k != 'label_guard'})
    assert handoff.verify_eager_audit(root, 'seq10', certificate) == digest(path)
    monkeypatch.setattr(handoff, 'audit_eager_v2', lambda *args: {'passed': True, 'sequence': 'seq10', 'target_frames': [1]})
    with pytest.raises(ValueError, match='independent reconstruction'):
        handoff.verify_eager_audit(root, 'seq10', certificate)


def test_cache_release_observation_preserves_roi_and_all_proof_bindings(tmp_path):
    feature = tmp_path / 'feature.pt'; feature.write_bytes(b'feature')
    snapshot = tmp_path / 'snapshot.pt'; snapshot.write_bytes(b'snapshot')
    item = {'frame': 0, 'feature_file': str(feature), 'feature_sha256': digest(feature),
            'snapshot_file': str(snapshot), 'snapshot_sha256': digest(snapshot)}
    plan = {'source_frames': {'seq11': [{'source_frame_index': 12, 'image_sha256': 'image'}]},
            'prompts': {'seq11': {'view': {'kind': 'full'}, 'prompt_provenance': {'roi_xyxy': [0, 0, 20, 20]}}}}
    binding = {'reference_receipt_path': '/ref', 'reference_receipt_sha256': 'r',
               'capture_receipt_path': '/cap', 'capture_receipt_sha256': 'c',
               'reference_sidecar_path': '/rs', 'reference_sidecar_sha256': 'rs',
               'capture_sidecar_path': '/cs', 'capture_sidecar_sha256': 'cs',
               'reference_setup_trace_path': '/rst', 'reference_setup_trace_sha256': 'rst',
               'reference_frame_trace_path': '/rft', 'reference_frame_trace_sha256': 'rft',
               'capture_setup_trace_path': '/cst', 'capture_setup_trace_sha256': 'cst',
               'capture_frame_trace_path': '/cft', 'capture_frame_trace_sha256': 'cft'}
    row = handoff.make_observation(plan, 'seq11', item, {'path': '/metadata', 'sha256': 'meta'}, '/audit', 'audit',
        binding, kind='matched_cache_release_reference', seq6_path='/seq6', seq6_sha='six',
        memory_path='/memory', memory_sha='memory')
    assert row['roi_xyxy'] == [0, 0, 20, 20] and row['source_frame_index'] == 12
    assert row['capture_sidecar_sha256'] == 'cs' and row['reference_frame_trace_sha256'] == 'rft'
    assert row['seq6_proof_sha256'] == 'six' and row['memory_gate_sha256'] == 'memory'
    feature.write_bytes(b'changed')
    with pytest.raises(ValueError, match='observation changed'):
        handoff.make_observation(plan, 'seq11', item, {'path': '/metadata', 'sha256': 'meta'}, '/audit', 'audit',
            binding, kind='matched_cache_release_reference', seq6_path='/seq6', seq6_sha='six',
            memory_path='/memory', memory_sha='memory')


def test_full_143_read_partition_requires_all_distinct_producers(tmp_path, monkeypatch):
    for name in ('certificate', 'short', 'memory', 'seq6-proof', 'warm-gate'):
        (tmp_path / name).write_text(name)
    cert, short, memory, seq6proof, warmgate = [tmp_path / name for name in
                                                 ('certificate', 'short', 'memory', 'seq6-proof', 'warm-gate')]
    plan = {'existing_manifest_sha256': 'old', 'feature_pipeline_sha256': 'pipeline',
            'source_frames': {}, 'prompts': {}, 'fresh_reads': []}
    eager = {}; cache_paths = []; cache_roots = {}; seq6_capture = seq6_reference = None
    for sequence in handoff.FRESH:
        count = 11 if sequence == 'seq10' else 12
        root = tmp_path / f'batch-{sequence}'; root.mkdir()
        ref = root / sequence / 'reference' / 'receipt.json'; ref.parent.mkdir(parents=True)
        ref.write_text('{}')
        cap = root / sequence / 'capture' / 'receipt.json'; cap.parent.mkdir(parents=True)
        observations = []
        for i in range(count):
            feature = cap.parent / f'f{i}'; feature.write_bytes(f'{sequence}-{i}-feature'.encode())
            snapshot = cap.parent / f's{i}'; snapshot.write_bytes(f'{sequence}-{i}-snapshot'.encode())
            observations.append({'frame': i, 'feature_file': str(feature), 'feature_sha256': digest(feature),
                                 'snapshot_file': str(snapshot), 'snapshot_sha256': digest(snapshot)})
        cap.write_text(json.dumps({'observations': observations}))
        plan['source_frames'][sequence] = [{'source_frame_index': 100 + i, 'image_sha256': f'{sequence}-{i}'}
                                           for i in range(count)]
        plan['prompts'][sequence] = {'view': {'kind': 'full'}, 'prompt_provenance': {'roi_xyxy': [0, 0, 20, 20]}}
        plan['fresh_reads'].extend({'sequence': sequence, 'frame': i} for i in range(count))
        if sequence == 'seq6':
            seq6_capture, seq6_reference = cap, ref
        else:
            audit = root / sequence / 'audit' / 'audit.json'; audit.parent.mkdir(parents=True); audit.write_text('{}')
            if sequence in handoff.EAGER: eager[sequence] = root
            else: cache_paths.append(audit); cache_roots[sequence] = root
    certificate = {'seq6_gate_capture': {'receipt_path': str(seq6_capture), 'receipt_sha256': digest(seq6_capture)},
        'seq6_historical_reference': {'reference_path': str(seq6_reference), 'reference_sha256': digest(seq6_reference)},
        'warm_gate_path': str(warmgate), 'warm_gate_sha256': digest(warmgate)}
    monkeypatch.setattr(handoff, 'validate_plan', lambda: plan)
    monkeypatch.setattr(handoff, 'verify_certificate', lambda _: (certificate, 'cert-sha'))
    monkeypatch.setattr(handoff, 'verify_short_gate', lambda _: ({'certificate_sha256': 'cert-sha'}, 'short-sha'))
    monkeypatch.setattr(handoff, 'verify_memory_gate', lambda _: ({'certificate_sha256': 'cert-sha'}, 'memory-sha'))
    monkeypatch.setattr(handoff, 'verify_seq6_proof', lambda *args: ({'memory_gate_sha256': 'memory-sha'}, 'six-sha'))
    monkeypatch.setattr(handoff, 'certified_seq6', lambda *args: None)
    monkeypatch.setattr(handoff, 'checked_audits', lambda *args: eager)
    monkeypatch.setattr(handoff, 'verify_eager_audit', lambda *args: 'eager-proof')
    monkeypatch.setattr(handoff, 'source_hashes', lambda: {})
    monkeypatch.setattr(handoff, 'sources', lambda: {})
    monkeypatch.setattr(handoff, 'sequence_inputs', lambda *args: None)
    monkeypatch.setattr(handoff, 'archived_frame_metadata', lambda *args: {'path': '/meta', 'sha256': 'meta'})
    monkeypatch.setattr(handoff, 'phase_artifact', lambda _root, _name, path: digest(path))
    monkeypatch.setattr(handoff, 'verify_release_audit', lambda path, seq, *args:
                        (cache_roots[seq], {}, digest(path), f'manifest-{seq}'))
    def fake_cache_bindings(root, sequence, _saved):
        ref = root / sequence / 'reference/receipt.json'; cap = root / sequence / 'capture/receipt.json'
        result = {'reference_receipt_path': str(ref), 'reference_receipt_sha256': digest(ref),
                  'capture_receipt_path': str(cap), 'capture_receipt_sha256': digest(cap)}
        for mode in ('reference', 'capture'):
            result[mode + '_sidecar_path'] = '/sidecar'; result[mode + '_sidecar_sha256'] = 'sidecar'
            for kind in ('setup', 'frame'):
                result[f'{mode}_{kind}_trace_path'] = '/trace'; result[f'{mode}_{kind}_trace_sha256'] = 'trace'
        return result
    monkeypatch.setattr(handoff, 'cache_bindings', fake_cache_bindings)
    result = handoff.build(cert, short, memory, seq6proof, [], cache_paths)
    assert result['training_authorized'] is False and len(result['observations']) == 143
    assert len([row for row in result['observations'] if row['reference_kind'] == 'matched_cache_release_reference']) == 84
    assert set(result['cache_release_producers']) == handoff.CACHE_RELEASE
    with pytest.raises(ValueError, match='audit count'):
        handoff.build(cert, short, memory, seq6proof, [], cache_paths[:-1])
    with pytest.raises(ValueError, match='uniqueness'):
        handoff.build(cert, short, memory, seq6proof, [], cache_paths[:-1] + [cache_paths[0]])
    feature = Path(result['observations'][0]['feature_file']); feature.write_bytes(b'changed')
    with pytest.raises(ValueError, match='observation changed'):
        handoff.build(cert, short, memory, seq6proof, [], cache_paths)
