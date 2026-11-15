#!/usr/bin/env python3
"""Recompute CMSA crop and restored full-frame metrics independently on CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_additional_author_comparators_v2 import compare
from scripts.audit_boundary_video_factorial_v1 import label, metrics, summarize
from scripts.crossfit_observer_capture_shared_v1 import sequence_inputs, validate_plan
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import new_output
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_cmsa_seq23_viewport_v1 import DATA, PRIOR, read_bound, require, validate_inputs


def audit(verification, scores):
    verification = Path(verification).resolve(strict=True); scores = Path(scores).resolve(strict=True)
    verification_sha = digest(verification); proof = read_bound(verification, verification_sha)
    score_path = scores/'results.json'; score_sha = digest(score_path); primary = read_bound(score_path, score_sha)
    guard = install_phase_mask_guard(DATA)
    current = validate_inputs(proof['prediction_audit_path'])
    require(proof.get('passed') is True and primary.get('passed') is True and
            proof.get('label_guard', {}).get('open_attempts') == 0 and
            primary.get('verification_sha256') == verification_sha and
            primary.get('scorer_sha256') == proof.get('scorer_sha256') and
            primary.get('prediction_audit_sha256') == proof['prediction_audit_sha256'] and
            primary.get('report_sha256') == digest(scores/'report.md') and
            all(proof.get(k) == v for k, v in current.items()) and guard['open_attempts'] == 0,
            'Frozen inputs or primary score differ')
    guard['active'] = False
    plan = validate_plan(); frames, _, start, _, _, _, _, _ = sequence_inputs(plan, 'seq23')
    metadata = read_bound(proof['frame_metadata']['path'], proof['frame_metadata']['sha256'])
    prior = read_bound(PRIOR, proof['prior_sha256']); prior_rows = prior['rows']['seq23']
    require(len(frames) == len(metadata) == len(primary['rows']) == 56, 'Audit frame coverage differs')
    rows, errors = [], []
    for i, frame in enumerate(frames):
        path = Path(proof['run'])/'predictions'/'seq23'/f'{i:06d}.png'
        pred_sha, label_sha, image_sha = proof['prediction_sha256'][i], metadata[i]['label_sha256'], proof['images'][i]
        require(digest(path) == pred_sha and digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha,
                'Audit input changed')
        with Image.open(path) as image: crop = np.asarray(image.convert('L')) >= 128
        truth = label(frame.mask_path)
        require(digest(path) == pred_sha and digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha,
                'Audit input changed during decode')
        require(crop.shape == (720, 834) and truth.shape == (720, 1280), 'Audit geometry differs')
        full = np.zeros((720, 1280), dtype=bool); full[:, 446:] = crop
        full_metric = metrics(full, truth); viewport_metric = metrics(crop, truth[:, 446:])
        require(bool(full_metric['present']) == bool(prior_rows[i]['A']['present']), 'Prior label presence differs')
        row = {'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
               'initialization_frame': i == start, 'full_frame': full_metric, 'viewport': viewport_metric}
        compare(errors, [i], row, primary['rows'][i]); rows.append(row)
    for scope in ('full_frame', 'viewport'):
        compare(errors, [scope, 'post'], summarize(rows, scope, True), primary['summaries'][scope])
        compare(errors, [scope, 'all'], summarize(rows, scope, False), primary['all_frame_summaries'][scope])
    compare(errors, ['prior_project_lean_postprompt'], prior['summaries']['seq23']['A']['post_prompt'],
            primary['prior_project_lean_postprompt'])
    return {'passed': not errors, 'scope': 'Independent CPU metrics; shared structural prediction verification.',
            'verification_sha256': verification_sha, 'score_sha256': score_sha,
            'auditor_sha256': digest(Path(__file__)), 'checked_frames': len(rows),
            'prelabel_guard': dict(guard), 'error_count': len(errors), 'errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('verification', 'scores', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); out = new_output(a.output); result = audit(a.verification, a.scores); out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']: raise ValueError(f"Independent audit found {result['error_count']} discrepancies")
    print(json.dumps({'passed': True, 'frames': result['checked_frames']}))


if __name__ == '__main__': main()
