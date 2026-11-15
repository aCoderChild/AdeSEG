#!/usr/bin/env python3
"""Score independently frozen CMSA seq23 crop predictions after label-free verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT, restore_mask
from scripts.crossfit_observer_capture_shared_v1 import archived_frame_metadata, sequence_inputs, validate_plan
from scripts.memory_token_io import digest
from scripts.local_import_closure_v1 import local_import_closure
from scripts.run_additional_author_comparator_v2 import EXPERIMENT_ROOT, new_output
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_boundary_video_factorial_v1 import metric, summary
from scripts.source_group_label_guard_v1 import install_mask_open_guard

DATA = ROOT/'data/polypgen_sequence.jsonl'
BASE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1')
PRIOR = BASE/'boundary_video_factorial_v1_7c3zaram/scores/results.json'
PRIOR_AUDIT = BASE/'boundary_video_factorial_v1_7c3zaram_independent_audit_attempt01/audit.json'
AUDITOR = ROOT/'scripts/audit_cmsa_seq23_viewport_predictions_v1.py'
SOURCES = ('scripts/score_cmsa_seq23_viewport_v1.py', 'scripts/audit_cmsa_seq23_viewport_scores_v1.py',
           'scripts/audit_cmsa_seq23_viewport_predictions_v1.py', 'scripts/run_cmsa_seq23_viewport_v1.py',
           'scripts/verify_cmsa_seq23_viewport_gate_v1.py', 'scripts/run_cmsa_polypgen.py',
           'scripts/stage_cmsa_seq23_viewport_v1.py', 'scripts/crossfit_observer_capture_shared_v1.py',
           'scripts/score_boundary_video_factorial_v1.py', 'scripts/score_source_group_boundary_v1.py',
           'scripts/audit_boundary_video_factorial_v1.py', 'scripts/audit_additional_author_comparators_v2.py',
           'scripts/score_additional_author_comparators_v2.py', 'scripts/source_group_label_guard_v1.py',
           'scripts/source_group_training_label_guard_v1.py', 'scripts/memory_token_io.py',
           'datasets/io.py', 'datasets/video.py', 'modeling/seq23_viewport.py')


def require(value, message):
    if not value: raise ValueError(message)


def read_bound(path, expected):
    path = Path(path); require(digest(path) == expected, 'File changed before read')
    raw = path.read_bytes(); import hashlib
    require(hashlib.sha256(raw).hexdigest() == expected and digest(path) == expected, 'File changed during read')
    return json.loads(raw)


def source_hashes():
    # Static closure covers transitive project imports; separately pinned CMSA
    # author sources and runtime identity cover dynamic upstream imports.
    names = set(SOURCES) | {'scripts/local_import_closure_v1.py'} | set(local_import_closure(ROOT, SOURCES))
    return {name: digest(ROOT/name) for name in sorted(names)}


def reconstruct_prediction_audit(stage, checkpoint, run):
    """Keep irreversible audit hooks confined to a fresh child process."""
    code = ('import json,sys; from pathlib import Path; '
            'from scripts.audit_cmsa_seq23_viewport_predictions_v1 import audit; '
            'print(json.dumps(audit(*(Path(x) for x in sys.argv[1:]))))')
    proc = subprocess.run([sys.executable, '-c', code, str(stage), str(checkpoint), str(run)],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def verify_prior(prior, prior_audit, prior_sha, baseline, sequence='seq23'):
    run = PRIOR.parent.parent
    require(prior.get('passed') is True and prior_audit.get('passed') is True and
            prior_audit.get('error_count') == 0 and prior_audit.get('score_sha256') == prior_sha and
            prior_audit.get('audit_source_sha256') == digest(ROOT/'scripts/audit_boundary_video_factorial_v1.py') and
            prior_audit.get('run') == str(run) and
            prior_audit.get('run_map_sha256') == prior.get('run_map_sha256') == digest(run/'run_map.json') and
            prior_audit.get('run_manifest_sha256') == prior.get('launch_manifest_sha256') == digest(run/'run_manifest.json') and
            prior.get('run_receipt_sha256', {}).get(sequence, {}).get('A') == digest(baseline),
            'Prior independently audited automatic Lean baseline differs')


def validate_inputs(prediction_audit):
    """Reconstruct the no-label prediction audit and bind prior frame identities."""
    proof_path = Path(prediction_audit).resolve(strict=True); proof_sha = digest(proof_path)
    require(proof_path.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)), 'Prediction audit outside experiment root')
    proof = read_bound(proof_path, proof_sha)
    require(proof.get('passed') is True and proof.get('verifier_sha256') == digest(AUDITOR) and
            proof.get('label_guard', {}).get('open_attempts') == 0, 'Prediction audit differs')
    current = reconstruct_prediction_audit(proof['stage'], proof['checkpoint'], proof['run'])
    require(current == proof, 'Prediction audit cannot be reconstructed')
    plan = validate_plan(); frames, source_rows, start, _, _, baseline, _, prompt = sequence_inputs(plan, 'seq23')
    require(len(frames) == len(source_rows) == 56, 'Prior automatic frame coverage differs')
    prior_sha = digest(PRIOR); audit_sha = digest(PRIOR_AUDIT)
    prior = read_bound(PRIOR, prior_sha); prior_audit = read_bound(PRIOR_AUDIT, audit_sha)
    verify_prior(prior, prior_audit, prior_sha, baseline)
    rows = prior['rows']['seq23']; require(len(rows) == len(frames), 'Prior frame count differs')
    for i, (frame, row) in enumerate(zip(frames, rows, strict=True)):
        require(row['frame'] == i and row['source_frame_index'] == frame.frame_index and
                row['post_prompt'] == (i > start) and row['initialization_frame'] == (i == start),
                'Prior post-prompt frame identity differs')
    meta = archived_frame_metadata(plan, 'seq23')
    return {'prediction_audit_path': str(proof_path), 'prediction_audit_sha256': proof_sha,
            'run': proof['run'], 'prediction_sha256': proof['prediction_sha256'],
            'stage': proof['stage'], 'checkpoint': proof['checkpoint'], 'prompt_frame': start,
            'prompt': prompt['prompt_provenance'], 'baseline_sha256': digest(baseline),
            'prior_sha256': prior_sha, 'prior_audit_sha256': audit_sha,
            'frame_metadata': meta, 'images': [r['image_sha256'] for r in source_rows],
            'scorer_sha256': digest(Path(__file__)), 'source_sha256': source_hashes()}


def scored_rows(bound):
    plan = validate_plan(); frames, _, start, _, _, _, _, _ = sequence_inputs(plan, 'seq23')
    metadata = read_bound(bound['frame_metadata']['path'], bound['frame_metadata']['sha256'])
    prior = read_bound(PRIOR, bound['prior_sha256']); prior_rows = prior['rows']['seq23']
    x0, y0, x1, y1 = VIEWPORT; rows = []
    require(len(metadata) == len(frames) == len(bound['prediction_sha256']), 'Scoring coverage differs')
    for i, frame in enumerate(frames):
        path = Path(bound['run'])/'predictions'/'seq23'/f'{i:06d}.png'
        pred_sha, label_sha, image_sha = bound['prediction_sha256'][i], metadata[i]['label_sha256'], bound['images'][i]
        require(digest(path) == pred_sha and digest(frame.mask_path) == label_sha and
                digest(frame.image_path) == image_sha, 'Prediction, label or input changed')
        with Image.open(path) as image: crop_prediction = np.asarray(image.convert('L')) >= 128
        truth = load_label_mask(frame.mask_path, [1]) == 1
        require(digest(path) == pred_sha and digest(frame.mask_path) == label_sha and
                digest(frame.image_path) == image_sha, 'Input changed during decode')
        require(truth.shape == (720, 1280) and crop_prediction.shape == (720, 834), 'Mask geometry differs')
        full = metric(restore_mask(crop_prediction), truth)
        cropped = metric(crop_prediction, truth[y0:y1, x0:x1])
        require(bool(full['present']) == bool(prior_rows[i]['A']['present']), 'Prior label presence differs')
        rows.append({'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                     'initialization_frame': i == start, 'full_frame': full, 'viewport': cropped})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--mode', choices=('verify', 'score'), required=True)
    p.add_argument('--prediction-audit', type=Path, required=True); p.add_argument('--verification', type=Path)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); out = new_output(a.output)
    if a.mode == 'verify':
        guard = install_mask_open_guard(DATA); result = validate_inputs(a.prediction_audit)
        require(guard['open_attempts'] == 0, 'Annotation open before verification')
        result.update(passed=True, label_guard=dict(guard)); out.mkdir(parents=True)
        (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'passed': True, 'frames': 56})); return
    require(a.verification is not None, 'Separate no-label verification is required')
    guard = install_phase_mask_guard(DATA); verification = a.verification.resolve(strict=True)
    verification_sha = digest(verification); proof = read_bound(verification, verification_sha)
    require(proof.get('passed') is True and proof.get('label_guard', {}).get('open_attempts') == 0 and
            proof.get('scorer_sha256') == digest(Path(__file__)) and
            proof.get('prediction_audit_path') == str(a.prediction_audit.resolve(strict=True)), 'Verification differs')
    current = validate_inputs(a.prediction_audit)
    require(all(proof.get(k) == v for k, v in current.items()) and guard['open_attempts'] == 0,
            'Verified inputs changed before annotation access')
    out.mkdir(parents=True); guard['active'] = False
    rows = scored_rows(proof)
    summaries = {scope: summary(rows, scope) for scope in ('full_frame', 'viewport')}
    all_frames = {scope: summary(rows, scope, post_only=False) for scope in ('full_frame', 'viewport')}
    prior_lean = read_bound(PRIOR, proof['prior_sha256'])['summaries']['seq23']['A']['post_prompt']
    report = ['# CMSA seq23 viewport comparison', '',
              'Released column-wise preprocessing; unprompted clip model, not online equivalent. '
              'Seq23 crop was selected retrospectively; these are exploratory data.', '',
              '| Evaluation | Present frames | Mean present Dice | Labeled-empty alarms |', '|---|---:|---:|---:|']
    for scope, s in summaries.items():
        value = s['mean_present_dice']; value = 'N/A' if value is None else f'{value:.4f}'
        report.append(f"| {scope} | {s['present_frames']} | {value} | {s['false_alarms']}/{s['empty_frames']} |")
    report += ['', 'Full-frame masks restore zeros outside the main viewport and use the original full label. '
               'Viewport metrics crop both prediction and label; their presence denominators may differ. '
               'Predictions use the existing project saved-probability threshold of 128/255. '
               'Post-prompt summaries use the prior automatic initialization frame and exclude it. '
               'The author loop can use future images in its initial clip; labeled-empty does not prove absence.', '',
               f"Matched prior project Lean present Dice: {prior_lean['mean_present_dice']} on {prior_lean['present_frames']} "
               f"present frames; labeled-empty alarms {prior_lean['false_alarms']}/{prior_lean['empty_frames']}. "
               'Lean uses an automatic prompt and is a different runtime; CMSA is unprompted.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    result = {'passed': True, 'scope': 'Previously inspected seq23, not independent evaluation.',
              'verification_sha256': verification_sha, 'prediction_audit_sha256': digest(a.prediction_audit),
              'scorer_sha256': digest(Path(__file__)), 'report_sha256': digest(out/'report.md'),
              'prelabel_guard': dict(guard), 'rows': rows, 'summaries': summaries, 'all_frame_summaries': all_frames,
              'prior_project_lean_postprompt': prior_lean}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'frames': len(rows)}))


if __name__ == '__main__': main()
