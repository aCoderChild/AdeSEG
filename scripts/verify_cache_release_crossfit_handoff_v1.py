#!/usr/bin/env python3
"""Reconstruct a mixed eager/cache-release 143-read handoff, without labels.

Seq6 uses its original certified eager-gate observations, while its separate
cache-release full-sequence audit establishes limited compatibility. Four
other sequences use their independently audited eager captures and seven use
their own independently reconstructed cache-release V2 captures. This does
not authorize training or claim full-sequence eager equality for new videos.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_cache_release_full_v2 import audit as audit_cache_release_v2
from scripts.audit_crossfit_warm_video_v2 import audit as audit_eager_v2
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import (DATA, OUT, FRESH, PLAN_SHA, archived_frame_metadata,
    phase_artifact, sequence_inputs, validate_plan)
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_cache_release_full_study_v1 import verify_seq6_proof
from scripts.run_crossfit_warm_video_study_v2 import checked_audits, certified_seq6, source_hashes
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_bounded_loader_gate_v2 import verify_gate as verify_short_gate
from scripts.verify_mps_memory_diagnostic_v3 import verify_gate as verify_memory_gate
from scripts.warm_crossfit_video_contract_v1 import read_bound, require


EAGER = frozenset(('seq10', 'seq20', 'seq22', 'seq23'))
CACHE_RELEASE = frozenset(('seq8', 'seq11', 'seq16', 'seq17', 'seq18', 'seq19', 'seq21'))
PROTOCOL = 'protocols/CACHE_RELEASE_CROSSFIT_HANDOFF_V1.md'


def sources():
    names = set(local_import_closure(ROOT, ('scripts/verify_cache_release_crossfit_handoff_v1.py',)))
    names.add(PROTOCOL)
    return {name: digest(ROOT / name) for name in sorted(names)}


def verify_release_audit(path, sequence, certificate_path, short_gate_path, memory_gate_path, seq6_path,
                         certificate_sha, short_gate_sha, memory_gate_sha, seq6_sha):
    """Independently reconstruct one V2 audit and freeze producer bindings."""
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve(strict=True)) and path.name == 'audit.json' and
            path.parent.parent.name == sequence and sequence in CACHE_RELEASE, 'Invalid cache-release audit')
    root = path.parent.parent.parent; sha = phase_artifact(root, f'{sequence}_audit', path)
    saved = read_bound(path, sha)
    manifest_path = root / 'run_manifest.json'; manifest_sha = digest(manifest_path)
    manifest = read_bound(manifest_path, manifest_sha)
    require(saved.get('passed') is True and saved.get('sequence') == sequence and
            saved.get('label_guard', {}).get('open_attempts') == 0 and
            saved.get('certificate_sha256') == certificate_sha and saved.get('short_gate_sha256') == short_gate_sha and
            saved.get('memory_gate_sha256') == memory_gate_sha and saved.get('seq6_proof_sha256') == seq6_sha and
            manifest.get('sequence') == sequence and manifest.get('certificate_path') == str(certificate_path) and
            manifest.get('short_gate_path') == str(short_gate_path) and
            manifest.get('memory_gate_path') == str(memory_gate_path) and
            manifest.get('memory_gate_sha256') == memory_gate_sha and
            manifest.get('seq6_proof_path') == str(seq6_path) and manifest.get('seq6_proof_sha256') == seq6_sha,
            'Cache-release manifest, diagnostic or compatibility proof differs')
    rebuilt = audit_cache_release_v2(root, sequence, certificate_path, short_gate_path)
    require({key: value for key, value in saved.items() if key != 'label_guard'} == rebuilt and
            digest(path) == sha and digest(manifest_path) == manifest_sha,
            'Cache-release audit failed independent reconstruction')
    return root, saved, sha, manifest_sha


def verify_eager_audit(root, sequence, certificate_path):
    """Independently reconstruct an already phase-bound eager V2 audit."""
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(OUT.resolve(strict=True)) and sequence in EAGER, 'Invalid eager audit root')
    path = root / sequence / 'audit/audit.json'
    sha = phase_artifact(root, f'{sequence}_audit', path)
    saved = read_bound(path, sha)
    require(saved.get('passed') is True and saved.get('sequence') == sequence and
            saved.get('label_guard', {}).get('open_attempts') == 0, 'Eager audit identity differs')
    rebuilt = audit_eager_v2(root, sequence, certificate_path)
    require({key: value for key, value in saved.items() if key != 'label_guard'} == rebuilt and
            digest(path) == sha, 'Eager audit failed independent reconstruction')
    return sha


def cache_bindings(root, sequence, saved):
    """Bind both producer receipts, both sidecars and four cache traces."""
    result = {}
    for mode in ('reference', 'capture'):
        folder = root / sequence / mode
        receipt_path = folder / 'receipt.json'
        receipt_sha = phase_artifact(root, f'{sequence}_{mode}', receipt_path)
        receipt = read_bound(receipt_path, receipt_sha)
        side_path = folder / 'bounded_receipt.json'; side_sha = digest(side_path)
        side = read_bound(side_path, side_sha)
        require(receipt_sha == saved[f'{mode}_sha256'] and side_sha == saved[f'{mode}_bounded_sha256'] and
                side.get('passed') is True and side.get('original_receipt_sha256') == receipt_sha,
                'Cache-release receipt or sidecar differs')
        result[mode + '_receipt_path'] = str(receipt_path)
        result[mode + '_receipt_sha256'] = receipt_sha
        result[mode + '_sidecar_path'] = str(side_path)
        result[mode + '_sidecar_sha256'] = side_sha
        for kind in ('setup', 'frame'):
            trace_path = root / sequence / f'{mode}_{kind}_cache.jsonl'
            trace_sha = digest(trace_path)
            require(trace_sha == receipt[f'cache_{kind}_sha256'], 'Cache-release trace differs')
            result[f'{mode}_{kind}_trace_path'] = str(trace_path)
            result[f'{mode}_{kind}_trace_sha256'] = trace_sha
    return result


def make_observation(plan, sequence, item, metadata, audit_path, audit_sha, binding, *, kind,
                     seq6_path=None, seq6_sha=None, memory_path=None, memory_sha=None):
    ordinal = item['frame']; source = plan['source_frames'][sequence][ordinal]
    for name in ('feature', 'snapshot'):
        require(digest(item[name + '_file']) == item[name + '_sha256'], 'Handoff observation changed')
    row = {'sequence': sequence, 'frame': ordinal, 'view': plan['prompts'][sequence]['view'],
           'roi_xyxy': plan['prompts'][sequence]['prompt_provenance']['roi_xyxy'],
           'source_frame_index': source['source_frame_index'], 'image_sha256': source['image_sha256'],
           'label_metadata_path': metadata['path'], 'label_metadata_sha256': metadata['sha256'],
           'label_metadata_locator': ordinal, 'reference_kind': kind,
           'reference_receipt_path': binding['reference_receipt_path'],
           'reference_receipt_sha256': binding['reference_receipt_sha256'],
           'capture_receipt_path': binding['capture_receipt_path'],
           'capture_receipt_sha256': binding['capture_receipt_sha256'],
           'sequence_audit_path': str(audit_path), 'sequence_audit_sha256': audit_sha,
           'feature_file': item['feature_file'], 'feature_sha256': item['feature_sha256'],
           'snapshot_file': item['snapshot_file'], 'snapshot_sha256': item['snapshot_sha256']}
    if kind == 'matched_cache_release_reference':
        require(sequence in CACHE_RELEASE and seq6_path is not None and memory_path is not None,
                'Missing cache-release compatibility evidence')
        row.update({key: value for key, value in binding.items() if 'sidecar_' in key or '_trace_' in key})
        row.update(seq6_proof_path=str(seq6_path), seq6_proof_sha256=seq6_sha,
                   memory_gate_path=str(memory_path), memory_gate_sha256=memory_sha)
    return row


def build(certificate_path, short_gate_path, memory_gate_path, seq6_path, eager_audits, cache_audits):
    plan = validate_plan()
    require(set(FRESH) == {'seq6'} | EAGER | CACHE_RELEASE and not (EAGER & CACHE_RELEASE),
            'Declared sequence partition differs')
    certificate_path = Path(certificate_path).resolve(strict=True)
    short_gate_path = Path(short_gate_path).resolve(strict=True)
    memory_gate_path = Path(memory_gate_path).resolve(strict=True)
    seq6_path = Path(seq6_path).resolve(strict=True)
    certificate, cert_sha = verify_certificate(certificate_path)
    short_gate, short_sha = verify_short_gate(short_gate_path)
    memory_gate, memory_sha = verify_memory_gate(memory_gate_path)
    require(short_gate.get('certificate_sha256') == cert_sha and memory_gate.get('certificate_sha256') == cert_sha,
            'Compatibility diagnostics use a different warm certificate')
    seq6, seq6_sha = verify_seq6_proof(seq6_path, cert_sha, short_sha)
    require(seq6.get('memory_gate_sha256') == memory_sha, 'Seq6 uses a different memory diagnostic')
    certified_seq6(certificate, plan)
    eager = checked_audits(eager_audits, plan, source_hashes(), certificate_path)
    require(set(eager) == EAGER, 'Eager audit coverage differs')
    for sequence, root in eager.items():
        verify_eager_audit(root, sequence, certificate_path)
    cache_paths = [Path(path).resolve(strict=True) for path in cache_audits]
    require(len(cache_paths) == len(CACHE_RELEASE) and len(set(cache_paths)) == len(cache_paths),
            'Cache-release audit count or uniqueness differs')
    cache = {}
    for path in cache_paths:
        sequence = path.parent.parent.name
        require(sequence in CACHE_RELEASE and sequence not in cache, 'Unexpected or duplicate cache-release audit')
        cache[sequence] = verify_release_audit(path, sequence, certificate_path, short_gate_path, memory_gate_path, seq6_path,
                                               cert_sha, short_sha, memory_sha, seq6_sha)
    require(set(cache) == CACHE_RELEASE, 'Cache-release sequence coverage differs')
    rows = []; audits = {}; producers = {}
    for sequence in FRESH:
        sequence_inputs(plan, sequence)
        if sequence == 'seq6':
            capture_path = Path(certificate['seq6_gate_capture']['receipt_path']).resolve(strict=True)
            capture_sha = certificate['seq6_gate_capture']['receipt_sha256']
            reference_path = Path(certificate['seq6_historical_reference']['reference_path']).resolve(strict=True)
            reference_sha = certificate['seq6_historical_reference']['reference_sha256']
            audit_path = Path(certificate['warm_gate_path']).resolve(strict=True)
            audit_sha = certificate['warm_gate_sha256']; kind = 'historical_warm_reference'
            binding = {'reference_receipt_path': str(reference_path), 'reference_receipt_sha256': reference_sha,
                       'capture_receipt_path': str(capture_path), 'capture_receipt_sha256': capture_sha}
        elif sequence in EAGER:
            root = eager[sequence]; capture_path = root / sequence / 'capture/receipt.json'
            reference_path = root / sequence / 'reference/receipt.json'
            capture_sha = phase_artifact(root, f'{sequence}_capture', capture_path)
            reference_sha = phase_artifact(root, f'{sequence}_reference', reference_path)
            audit_path = root / sequence / 'audit/audit.json'; audit_sha = phase_artifact(root, f'{sequence}_audit', audit_path)
            kind = 'matched_warm_reference'
            binding = {'reference_receipt_path': str(reference_path), 'reference_receipt_sha256': reference_sha,
                       'capture_receipt_path': str(capture_path), 'capture_receipt_sha256': capture_sha}
        else:
            root, saved, audit_sha, manifest_sha = cache[sequence]
            audit_path = root / sequence / 'audit/audit.json'
            binding = cache_bindings(root, sequence, saved)
            capture_path = Path(binding['capture_receipt_path']); capture_sha = binding['capture_receipt_sha256']
            kind = 'matched_cache_release_reference'
            producers[sequence] = {'manifest_path': str(root / 'run_manifest.json'), 'manifest_sha256': manifest_sha,
                                   'audit_path': str(audit_path), 'audit_sha256': audit_sha, **binding}
        capture = read_bound(capture_path, capture_sha)
        metadata = archived_frame_metadata(plan, sequence)
        audits[sequence] = {'path': str(audit_path), 'sha256': audit_sha}
        for item in capture['observations']:
            rows.append(make_observation(plan, sequence, item, metadata, audit_path, audit_sha, binding, kind=kind,
                seq6_path=seq6_path, seq6_sha=seq6_sha, memory_path=memory_gate_path, memory_sha=memory_sha))
    expected = {(item['sequence'], item['frame']) for item in plan['fresh_reads']}
    require(len(rows) == len(expected) == 143 and len({(row['sequence'], row['frame']) for row in rows}) == 143 and
            {(row['sequence'], row['frame']) for row in rows} == expected, 'Mixed 143-read grid differs')
    return {'passed': True, 'training_authorized': False, 'format': 'cache_release_crossfit_handoff_v1',
        'scope': 'Mixed eager/cache-release evidence only; requires a separate reviewed training preflight.',
        'index_plan_sha256': PLAN_SHA, 'existing_manifest_sha256': plan['existing_manifest_sha256'],
        'existing_feature_pipeline_sha256': plan['feature_pipeline_sha256'],
        'warm_capture_certificate_path': str(certificate_path), 'warm_capture_certificate_sha256': cert_sha,
        'short_gate_path': str(short_gate_path), 'short_gate_sha256': short_sha,
        'memory_gate_path': str(memory_gate_path), 'memory_gate_sha256': memory_sha,
        'seq6_proof_path': str(seq6_path), 'seq6_proof_sha256': seq6_sha,
        'eager_sequences': sorted(EAGER), 'cache_release_sequences': sorted(CACHE_RELEASE),
        'audits': audits, 'cache_release_producers': producers, 'observations': rows, 'source_sha256': sources()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('certificate', 'short-gate', 'memory-gate', 'seq6-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--eager-audit', type=Path, action='append', default=[])
    parser.add_argument('--cache-audit', type=Path, action='append', default=[])
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(), 'Invalid handoff output')
    guard = install_mask_open_guard(DATA); frozen = sources()
    result = build(args.certificate, args.short_gate, args.memory_gate, args.seq6_audit, args.eager_audit, args.cache_audit)
    require(guard['open_attempts'] == 0 and frozen == result['source_sha256'] == sources(),
            'Target label opened or source changed')
    result['label_guard'] = dict(guard)
    out.mkdir(parents=True, exist_ok=False); path = out / 'handoff.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'handoff_sha256': digest(path),
        'training_authorized': False, 'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'training_authorized': False, 'output': str(out)}))


if __name__ == '__main__':
    main()
