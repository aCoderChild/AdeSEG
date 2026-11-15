#!/usr/bin/env python3
"""Recompute Lean metrics independently; share structural frozen-run checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_additional_author_comparators_v2 import compare
from scripts.audit_boundary_video_factorial_v1 import label, metrics, summarize
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_study_v2 import CHECKPOINT, DETECTOR, EXTENDED, REMAINING
from scripts.run_author_lean_automatic_v1 import DATA, SEQUENCES, inputs, new_output
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_author_lean_automatic_v1 import read_bound, require, validate_frozen


def audit(folder):
    folder = Path(folder).resolve(strict=True); guard = install_phase_mask_guard(DATA)
    proof_path = folder/'verification/verification.json'; score_path = folder/'scores/results.json'
    proof_sha = digest(proof_path); score_sha = digest(score_path)
    proof = read_bound(proof_path, proof_sha); primary = read_bound(score_path, score_sha)
    current = validate_frozen(folder, ('verify', 'score'))
    require(guard['open_attempts'] == 0 and proof.get('passed') is True and primary.get('passed') is True and
            primary.get('verification_sha256') == proof_sha and
            primary.get('scorer_sha256') == proof.get('scorer_sha256') and
            primary.get('report_sha256') == digest(folder/'scores/report.md'), 'Score provenance differs')
    for key in ('manifest_sha256', 'gate_sha256', 'records'):
        require(current[key] == proof[key], 'Frozen inputs changed after verification')
    guard['active'] = False; errors = []; checked = 0
    patient_means, present_dice, empty_count, empty_alarms = [], [], 0, 0
    for seq in SEQUENCES:
        bound = proof['records'][seq]; meta = bound['frame_metadata']
        metadata = read_bound(meta['path'], meta['sha256'])
        frames, start, _, _ = inputs(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        receipt = read_bound(folder/'full'/seq/'receipt.json', bound['receipt_sha256'])
        require(len(frames) == len(metadata) == len(primary['rows'][seq]), 'Audit frame coverage differs')
        rows = []
        for i, frame in enumerate(frames):
            pred_path = folder/'full'/seq/f'{i:06d}.png'; pred_sha = receipt['records'][i]['mask_sha256']
            label_sha = metadata[i]['label_sha256']; image_sha = bound['images'][i]
            require(digest(frame.image_path) == image_sha and digest(frame.mask_path) == label_sha and
                    digest(pred_path) == pred_sha, 'Audit input changed')
            truth = label(frame.mask_path)
            with Image.open(pred_path) as image: prediction = np.asarray(image.convert('L')) > 0
            require(digest(frame.image_path) == image_sha and digest(frame.mask_path) == label_sha and
                    digest(pred_path) == pred_sha, 'Audit input changed during decode')
            row = {'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                   'initialization_frame': i == start, 'lean_author_mps': metrics(prediction, truth)}
            compare(errors, [seq, i], row, primary['rows'][seq][i]); rows.append(row); checked += 1
        compare(errors, [seq, 'post'], summarize(rows, 'lean_author_mps', True), primary['summaries'][seq])
        compare(errors, [seq, 'all'], summarize(rows, 'lean_author_mps', False), primary['all_frame_summaries'][seq])
        present = [r['lean_author_mps']['dice'] for r in rows if r['post_prompt'] and r['lean_author_mps']['present']]
        empty = [r['lean_author_mps'] for r in rows if r['post_prompt'] and not r['lean_author_mps']['present']]
        if present:
            patient_means.append(float(np.mean(present))); present_dice.extend(present)
        empty_count += len(empty); empty_alarms += sum(x['false_alarm'] for x in empty)
        pre = {'frames': start, 'labeled_present': sum(r['lean_author_mps']['present'] for r in rows[:start]),
               'present_misses': sum(r['lean_author_mps']['present'] and r['lean_author_mps']['dice'] == 0 for r in rows[:start])}
        compare(errors, [seq, 'preprompt'], pre, primary['preprompt'][seq])
    overall = {'patients_with_present_frames': len(patient_means), 'present_frames': len(present_dice),
               'patient_macro_present_dice': float(np.mean(patient_means)) if patient_means else None,
               'frame_weighted_present_dice': float(np.mean(present_dice)) if present_dice else None,
               'labeled_empty_frames': empty_count, 'labeled_empty_false_alarms': empty_alarms}
    compare(errors, ['overall'], overall, primary['overall'])
    return {'passed': not errors, 'scope': 'Independent CPU metrics; shared structural verifier.',
            'verification_sha256': proof_sha, 'score_sha256': score_sha, 'auditor_sha256': digest(Path(__file__)),
            'checked_frames': checked, 'prelabel_guard': dict(guard), 'error_count': len(errors), 'errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    out = new_output(a.run/'independent_audit'); result = audit(a.run); out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']: raise ValueError(f"Independent metrics found {result['error_count']} discrepancies")
    print(json.dumps({'passed': True, 'checked_frames': result['checked_frames']}))


if __name__ == '__main__': main()
