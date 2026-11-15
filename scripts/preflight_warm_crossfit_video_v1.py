#!/usr/bin/env python3
"""Label-free certification of out-of-fold warm full-video prerequisites.

Does not run inference, open labels, select by scores or establish efficacy.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import DATA, EXISTING, OUT, phase_artifact
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_crossfit_warm_video_v2 import warm_plan
from scripts.score_warm_crossfit_v1 import validate_all
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import load_preflight
from scripts.warm_crossfit_video_contract_v1 import (NO_INITIALIZATION, PROPOSED_ARM,
    fresh_sequence_inputs, heldout_fold, load_heldout_checkpoint, old_sequence_inputs, read_bound, require)

ENTRYPOINTS = ('scripts/preflight_warm_crossfit_video_v1.py',
    'scripts/warm_crossfit_video_contract_v1.py', 'modeling/warm_crossfit_video_runtime_v1.py',
    'scripts/run_warm_crossfit_video_v1.py', 'scripts/audit_warm_crossfit_video_smoke_v1.py',
    'scripts/run_warm_crossfit_video_study_v1.py', 'scripts/score_warm_crossfit_video_v1.py')
PROTOCOL = 'protocols/EXPLORATORY_WARM_CROSSFIT_VIDEO_V1.md'


def sources():
    names = set(local_import_closure(ROOT, ENTRYPOINTS)); names.add(PROTOCOL)
    return {name: digest(ROOT / name) for name in sorted(names)}


def certify(batch, preflight_path):
    batch = Path(batch).resolve(strict=True); preflight_path = Path(preflight_path).resolve(strict=True)
    require(batch.is_relative_to(OUT.resolve(strict=True)), 'Cross-fit batch outside experiment root')
    plan, sha = load_preflight(preflight_path)
    rows = [json.loads(line) for line in (batch / 'phases.jsonl').read_text().splitlines() if line.strip()]
    names = [x['phase'] for x in rows]
    require('verify' in names, 'Fixed-read all-fold verification is required')
    completed = tuple(name for name in ('verify', 'score', 'audit') if name in names)
    index, samples, records, proof = validate_all(batch, preflight_path, plan, sha, completed=completed)
    verified_path = batch / 'verify/verification.json'
    verified_sha = phase_artifact(batch, 'verify', verified_path)
    verified = read_bound(verified_path, verified_sha)
    require(verified.get('passed') is True and verified.get('label_guard', {}).get('open_attempts') == 0 and
            all(verified.get(k) == v for k, v in proof.items()), 'Fixed-read all-fold proof differs')
    aggregate_path = Path(plan['aggregate_path']).resolve(strict=True) / 'captures.json'
    aggregate = read_bound(aggregate_path, plan['aggregate_sha256'])
    certificate, certificate_sha = verify_certificate(aggregate['certificate_path'])
    require(certificate_sha == aggregate['certificate_sha256'] == plan['certificate_sha256'],
            'Warm capture certificate differs')
    warm = warm_plan(); manifest = read_bound(EXISTING / 'manifest.json', plan['old_manifest_sha256'])
    refs = {}; checkpoints = {}
    for i in range(1, 24):
        sequence = f'seq{i}'
        if sequence in NO_INITIALIZATION:
            continue
        fold = heldout_fold(sequence)
        if str(fold) not in checkpoints:
            _, checkpoints[str(fold)] = load_heldout_checkpoint(batch, preflight_path, plan, sha, sequence)
        info = (old_sequence_inputs(sequence, manifest, plan['old_manifest_sha256'])
                if sequence in manifest['sequences'] else
                fresh_sequence_inputs(sequence, aggregate, certificate, index, warm))
        if sequence in manifest['sequences']:
            require(info['reference'].get('identity') == warm['old_identity'],
                    'Historical warm source/runtime identity differs from certified captures')
        refs[sequence] = {'fold': fold, 'prompt_frame': info['start'], 'frames': len(info['frames']),
            'roi_xyxy': info['roi_xyxy'], 'image_sha256': info['image_sha256'],
            'image_sizes': [list(size) for size in info['image_sizes']],
            'reference_path': info['reference_path'], 'reference_sha256': info['reference_sha256']}
    require(len(refs) == 21 and len(checkpoints) == 5, 'Video patient/fold coverage differs')
    return {'passed': True, 'scope': 'Warm full-video prerequisites only; no label reads or efficacy.',
        'arm': PROPOSED_ARM, 'no_initialization': sorted(NO_INITIALIZATION),
        'crossfit_root': str(batch), 'preflight_path': str(preflight_path), 'preflight_sha256': sha,
        'fixed_read_verification_sha256': verified_sha, 'fixed_read_proof': proof,
        'warm_capture_certificate_path': aggregate['certificate_path'],
        'warm_capture_certificate_sha256': certificate_sha, 'aggregate_path': str(aggregate_path),
        'aggregate_sha256': plan['aggregate_sha256'], 'old_manifest_sha256': plan['old_manifest_sha256'],
        'checkpoints': checkpoints, 'references': refs, 'source_sha256': sources()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crossfit-root', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); output = a.output.resolve()
    require(output.is_relative_to(OUT.resolve(strict=True)) and not output.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA)
    result = certify(a.crossfit_root, a.preflight)
    require(guard['open_attempts'] == 0, 'Annotation open attempted in video certification')
    result['label_guard'] = dict(guard)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'certificate.json'; path.write_text(json.dumps(result, indent=2) + '\n')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'certificate_sha256': digest(path),
        'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(output)}))


if __name__ == '__main__': main()
