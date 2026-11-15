#!/usr/bin/env python3
"""Freeze a separate, four-control study against certified warm captures."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact
from scripts.local_import_closure_v1 import local_import_closure
from scripts.mechanism_crossfit_control_core_v1 import ARMS, make_arms
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import load_preflight, new_output, require, samples_for

SOURCES = ('scripts/preflight_mechanism_crossfit_v1.py',
           'scripts/mechanism_crossfit_control_core_v1.py',
           'scripts/run_mechanism_crossfit_fold_v1.py',
           'scripts/score_mechanism_crossfit_v1.py',
           'scripts/run_mechanism_crossfit_study_v1.py',
           'scripts/load_aligned_rgb_control_v1.py',
           'modeling/multiscale_mechanism_controls_v1.py',
           'modeling/rgb_contrast_control_v1.py',
           'protocols/MULTISCALE_MECHANISM_CONTROLS_V1.md',
           'scripts/local_import_closure_v1.py')


def sources():
    paths = set(SOURCES) | set(local_import_closure(ROOT, SOURCES))
    return {name: digest(ROOT / name) for name in sorted(paths)}


def reference_study(root, warm_path, warm, warm_sha):
    """Require the separately audited six-arm study before control training."""
    from scripts.score_warm_crossfit_v1 import validate_all
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(BASE.resolve(strict=True)), 'Reference study outside Drive')
    index, samples, records, proof = validate_all(root, warm_path, warm, warm_sha,
                                                  completed=('verify', 'score', 'audit'))
    verification_path = root/'verify/verification.json'
    verification_sha = phase_artifact(root, 'verify', verification_path)
    score_path = root/'score/results.json'; score_sha = phase_artifact(root, 'score', score_path)
    audit_path = root/'audit/audit.json'; audit_sha = phase_artifact(root, 'audit', audit_path)
    score = json.loads(score_path.read_text()); audit = json.loads(audit_path.read_text())
    require(digest(score_path) == score_sha and digest(audit_path) == audit_sha and score.get('passed') is True and
            score.get('verification_sha256') == verification_sha and
            score.get('report_sha256') == digest(root/'score/report.md') and
            score.get('prelabel_guard', {}).get('open_attempts') == 0 and
            audit.get('passed') is True and audit.get('score_sha256') == score_sha and audit.get('error_count') == 0 and
            audit.get('prelabel_guard', {}).get('open_attempts') == 0,
            'Reference study score or audit differs')
    require(len(samples) == 357, 'Reference read coverage differs')
    return records, {'root': str(root), 'manifest_sha256': proof['manifest_sha256'], 'verification_sha256': verification_sha,
                     'score_sha256': score_sha, 'audit_sha256': audit_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-preflight', type=Path, required=True)
    parser.add_argument('--reference-batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = new_output(args.output)
    guard = install_mask_open_guard(DATA)
    warm, warm_sha = load_preflight(args.warm_preflight)
    index, rows = samples_for(warm)
    _, reference = reference_study(args.reference_batch, args.warm_preflight.resolve(), warm, warm_sha)
    expected = warm['initial_state_sha256']['foreground_background_asinh']
    models, _, _ = make_arms(expected, 'cpu')
    require(len(models) == len(ARMS) and len(rows) == 357 and guard['open_attempts'] == 0,
            'Control preflight coverage differs')
    source_hashes = sources()
    result = {'passed': True, 'scope': 'Exploratory four-arm fixed-read study, no labels or model inference.',
        'arms': list(ARMS), 'warm_preflight_path': str(args.warm_preflight.resolve()),
        'warm_preflight_sha256': warm_sha, 'aggregate_sha256': warm['aggregate_sha256'],
        'index_plan_sha256': warm['index_plan_sha256'], 'samples': len(rows),
        'folds': warm['folds'], 'initial_state_sha256': expected, 'reference_study': reference,
        'optimizer': warm['optimizer'], 'loss': warm['loss'],
        'rgb_policy': 'decoded sRGB [0,1], same hash-bound frame and viewport, area-resized contrast only',
        'source_sha256': source_hashes, 'label_guard': dict(guard)}
    require(source_hashes == sources() and guard['open_attempts'] == 0, 'Preflight inputs changed')
    out.mkdir(parents=True); path = out / 'plan.json'; path.write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'plan_sha256': digest(path),
        'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'arms': list(ARMS), 'samples': len(rows), 'output': str(out)}))


if __name__ == '__main__':
    main()
