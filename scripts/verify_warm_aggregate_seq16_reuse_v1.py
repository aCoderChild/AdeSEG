#!/usr/bin/env python3
"""Bind a completed warm aggregate to the independently verified seq16 reuse."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, EXISTING, PLAN_SHA, phase_artifact, require, validate_plan
from scripts.crossfit_training_contract_v1 import validate_warm_index_structure
from scripts.memory_token_io import digest
from scripts.run_crossfit_warm_video_v2 import WARM_SHA
from scripts.run_crossfit_warm_video_study_v2 import source_hashes
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_seq16_reference_reuse_v1 import verify as verify_reuse


def read_bound(path, expected):
    require(digest(path) == expected, 'Handoff artifact changed before reading')
    value = json.loads(Path(path).read_text())
    require(digest(path) == expected, 'Handoff artifact changed during reading')
    return value


def verify(aggregate_dir, reuse_verification):
    folder = Path(aggregate_dir).resolve(strict=True); path = folder/'captures.json'
    reuse_path = Path(reuse_verification).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)) and reuse_path.is_relative_to(BASE.resolve(strict=True)),
            'Handoff input outside Drive experiments')
    aggregate_sha = digest(path); aggregate = read_bound(path, aggregate_sha)
    completion_sha = digest(folder/'completion.json'); completion = read_bound(folder/'completion.json', completion_sha)
    require(completion.get('passed') is True and completion.get('captures_sha256') == aggregate_sha and
            completion.get('label_guard', {}).get('open_attempts') == 0 and aggregate.get('passed') is True and
            aggregate.get('plan_sha256') == aggregate.get('index_plan_sha256') == PLAN_SHA and
            aggregate.get('warm_preflight_sha256') == WARM_SHA and aggregate.get('sources') == source_hashes(),
            'Warm aggregate completion differs')
    certificate, certificate_sha = verify_certificate(aggregate['certificate_path'])
    require(certificate_sha == aggregate['certificate_sha256'] and
            certificate['warm_gate_sha256'] == aggregate['warm_gate_sha256'], 'Warm aggregate certificate differs')
    index = validate_plan(); manifest_path = EXISTING/'manifest.json'; manifest_sha = digest(manifest_path)
    old = read_bound(manifest_path, manifest_sha)
    require(manifest_sha == aggregate['existing_manifest_sha256'] == index['existing_manifest_sha256'],
            'Historical manifest differs')
    validate_warm_index_structure(index, old, aggregate, index_sha256=PLAN_SHA,
        warm_preflight_sha256=WARM_SHA, warm_gate_path=aggregate['warm_gate_path'],
        warm_gate_sha256=aggregate['warm_gate_sha256'], certificate_path=aggregate['certificate_path'],
        certificate_sha256=certificate_sha)
    reuse_sha = digest(reuse_path); saved = read_bound(reuse_path, reuse_sha)
    require(saved.get('passed') is True and saved.get('label_guard', {}).get('open_attempts') == 0,
            'Seq16 reuse verification differs')
    rebuilt = verify_reuse(saved['batch'])
    require({k: v for k, v in saved.items() if k != 'label_guard'} == rebuilt and digest(reuse_path) == reuse_sha,
            'Seq16 reuse verification cannot be reconstructed')
    batch = Path(saved['batch']).resolve(strict=True); audit = batch/'seq16/audit/audit.json'
    reference = batch/'seq16/reference/receipt.json'; capture = batch/'seq16/capture/receipt.json'
    audit_sha = phase_artifact(batch, 'seq16_audit', audit)
    reference_sha = phase_artifact(batch, 'seq16_reference', reference)
    capture_sha = phase_artifact(batch, 'seq16_capture', capture)
    require(audit_sha == saved['capture_audit_sha256'] and reference_sha == saved['reference_receipt_sha256'] and
            saved['certificate_sha256'] == certificate_sha and
            aggregate['audits']['seq16'] == {'path': str(audit), 'sha256': audit_sha},
            'Seq16 aggregate audit or certificate differs')
    capture_record = read_bound(capture, capture_sha)
    rows = [row for row in aggregate['observations'] if row['sequence'] == 'seq16']
    targets = [row['frame'] for row in index['fresh_reads'] if row['sequence'] == 'seq16']
    require([row['frame'] for row in rows] == targets and
            [row['frame'] for row in capture_record['observations']] == targets, 'Seq16 aggregate grid differs')
    for row, observed in zip(rows, capture_record['observations'], strict=True):
        require(row['reference_kind'] == 'matched_warm_reference' and
                row['reference_receipt_path'] == str(reference) and row['reference_receipt_sha256'] == reference_sha and
                row['capture_receipt_path'] == str(capture) and row['capture_receipt_sha256'] == capture_sha and
                row['sequence_audit_path'] == str(audit) and row['sequence_audit_sha256'] == audit_sha and
                all(row[k] == observed[k] for k in ('feature_file', 'feature_sha256', 'snapshot_file', 'snapshot_sha256')),
                'Seq16 aggregate observation lineage differs')
    require(digest(path) == aggregate_sha and digest(folder/'completion.json') == completion_sha and
            digest(reuse_path) == reuse_sha, 'Handoff artifact changed during verification')
    return {'passed': True, 'scope': 'Label-free seq16 reuse binding for completed exploratory warm aggregate.',
        'aggregate_path': str(path), 'aggregate_sha256': aggregate_sha, 'completion_sha256': completion_sha,
        'reuse_verification_path': str(reuse_path), 'reuse_verification_sha256': reuse_sha,
        'seq16_capture_audit_sha256': audit_sha, 'seq16_reference_sha256': reference_sha,
        'seq16_capture_sha256': capture_sha, 'certificate_sha256': certificate_sha,
        'verifier_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate-dir', type=Path, required=True)
    parser.add_argument('--reuse-verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid handoff output')
    guard = install_mask_open_guard(DATA); result = verify(args.aggregate_dir, args.reuse_verification)
    require(guard['open_attempts'] == 0, 'Annotation opened during handoff verification')
    result['label_guard'] = dict(guard); out.mkdir(parents=True)
    (out/'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(out)}))


if __name__ == '__main__':
    main()
