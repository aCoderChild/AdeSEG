#!/usr/bin/env python3
"""Independently rescore the frozen exploratory video factorial on CPU.

All 12 complete runs, all saved PNG hashes, launch/gate receipts and the
primary score must exist and validate before any target mask is opened. The
independent arithmetic uses Pillow/NumPy/SciPy, not the primary scorer or
mask loader. This audit cannot make previously inspected videos independent.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output

SEQS = ('seq11', 'seq20', 'seq23')
ARMS = ('A', 'B', 'C', 'D')
FIELDS = ('present', 'dice', 'iou', 'precision', 'recall', 'boundary_f_5px', 'predicted_pixels', 'false_alarm')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def same(x, y):
    if x is None or y is None:
        return x is y
    if isinstance(x, bool) or isinstance(y, bool):
        return type(x) is type(y) and x == y
    return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)


def boundary_target(gt):
    inner = gt & ~ndimage.binary_erosion(gt, structure=np.ones((3,3), dtype=bool), border_value=0)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)).astype(bool)
    near = ndimage.binary_dilation(inner, structure=ellipse)
    return inner, near, int(inner.sum()), ellipse


def boundary(pred, gt, cached_target=None):
    p = pred & ~ndimage.binary_erosion(pred, structure=np.ones((3,3), dtype=bool), border_value=0)
    t, near_t, tn, ellipse = boundary_target(gt) if cached_target is None else cached_target
    pn = int(p.sum())
    if not pn or not tn:
        return 1.0 if not pn and not tn else 0.0
    precision = int((p & near_t).sum()) / pn
    recall = int((t & ndimage.binary_dilation(p, structure=ellipse)).sum()) / tn
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def metrics(pred, gt, cached_target=None):
    require(pred.shape == gt.shape, 'Prediction/label shapes differ')
    pp, gp, tp = int(pred.sum()), int(gt.sum()), int((pred & gt).sum())
    return {'present': bool(gp), 'dice': 2*tp/(pp+gp) if pp+gp else 1.,
            'iou': tp/(pp+gp-tp) if pp+gp-tp else 1.,
            'precision': tp/pp if pp else None, 'recall': tp/gp if gp else None,
            'boundary_f_5px': boundary(pred, gt, cached_target), 'predicted_pixels': pp,
            'false_alarm': bool(pp) if not gp else None}


def mean(items):
    values = [x for x in items if x is not None]
    return math.fsum(values)/len(values) if values else None


def summarize(rows, arm, post):
    chosen = [r for r in rows if not post or r['post_prompt']]
    present = [r[arm] for r in chosen if r[arm]['present']]
    empty = [r[arm] for r in chosen if not r[arm]['present']]
    return {'frames': len(chosen), 'present_frames': len(present),
            'mean_present_dice': mean([x['dice'] for x in present]),
            'mean_present_iou': mean([x['iou'] for x in present]),
            'mean_present_precision_when_defined': mean([x['precision'] for x in present]),
            'mean_present_recall': mean([x['recall'] for x in present]),
            'mean_present_boundary_f_5px': mean([x['boundary_f_5px'] for x in present]),
            'zero_overlap_present_frames': sum(x['dice'] == 0 for x in present),
            'empty_frames': len(empty), 'false_alarms': sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels': mean([x['predicted_pixels'] for x in empty])}


def paired(rows, arm, base):
    delta = [r[arm]['dice'] - r[base]['dice'] for r in rows if r['post_prompt'] and r[arm]['present']]
    return {'present_frames': len(delta), 'mean_present_dice_difference': mean(delta),
            'improved': sum(x > 0 for x in delta), 'worsened': sum(x < 0 for x in delta),
            'unchanged': sum(x == 0 for x in delta)}


def label(path):
    with Image.open(path) as im:
        raw = np.asarray(im)
        gray = np.asarray(im.convert('L'))
        if raw.ndim == 3 or (raw.max() > 1 and im.format == 'JPEG'):
            require(float(((gray > 64) & (gray < 192)).mean()) <= .05, 'Ambiguous grayscale label')
            return gray > 127
        require(set(np.unique(raw)).issubset({0,1}), 'Unexpected indexed label')
        return raw == 1


def compare(errors, prefix, actual, expected):
    for key, value in actual.items():
        if key not in expected or not same(value, expected[key]):
            errors.append([*prefix, key, value, expected.get(key)])


def precheck(run):
    """Validate finished receipts and all masks BEFORE reading a label."""
    run = Path(run).resolve(strict=True)
    launch, mapping = read(run/'run_manifest.json'), read(run/'run_map.json')
    score_path, report = run/'scores/results.json', run/'scores/report.md'
    score = read(score_path)
    require(score.get('passed') is True and score.get('scorer_sha256') == sha(ROOT/'scripts/score_boundary_video_factorial_v1.py') and
            score.get('report_sha256') == sha(report) and score.get('run_map_sha256') == sha(run/'run_map.json'),
            'Primary scorer is incomplete or changed')
    phase_path = run/'phases.jsonl'
    phases = [json.loads(line) for line in phase_path.read_text().splitlines() if line.strip()]
    expected_phases = [f'{seq}_{arm}' for seq in launch['sequences'] for arm in ARMS] + ['score']
    require([x.get('phase') for x in phases] == expected_phases and
            all(x.get('passed') is True and x.get('log_sha256') == sha(run/(x['phase']+'.log')) for x in phases),
            'Full launcher phases or logs are incomplete')
    require(launch.get('run_map') == mapping and set(mapping) == set(SEQS) and
            launch.get('scorer_sha256') == sha(ROOT/'scripts/score_boundary_video_factorial_v1.py') and
            launch.get('python_hash_seed') == '0' and launch.get('mps_fallback') == '0' and launch.get('mps_fast_math') == '0' and
            launch.get('runner_sha256') == sha(ROOT/'scripts/run_boundary_refiner_priority_v1.py') and
            launch.get('launcher_sha256') == sha(ROOT/'scripts/run_boundary_video_factorial_v1.py'),
            'Launch or map binding differs')
    gate, gate_audit = Path(launch['gate_path']), Path(launch['independent_gate_audit_path'])
    require(sha(gate) == launch['gate_sha256'] and sha(gate_audit) == launch['independent_gate_audit_sha256'],
            'Gate or independent gate audit changed')
    ga = read(gate_audit)
    require(ga.get('passed') is True and ga.get('gate_sha256') == sha(gate) and
            ga.get('audit_source_sha256') == sha(ROOT/'scripts/audit_boundary_video_short_v1.py'),
            'Independent short-gate audit differs')
    manifest_path = ROOT/'data/polypgen_sequence.jsonl'
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    frames = {seq: sorted((r for r in rows if r['video_id'] == seq), key=lambda r: r['frame_index']) for seq in SEQS}
    # Source preflight is a fixed, previously audited input, not a target mask.
    base = run.parent.parent
    pre = base/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03'
    pre_path = pre/'manifest.json'; predata = read(pre_path); done = read(pre/'completion.json')
    require(done.get('passed') and done.get('manifest_sha256') == sha(pre_path), 'Preflight changed')
    meta = {}
    for seq in SEQS:
        path = Path(predata['sequences'][seq]['source'])/seq/'frames.json'
        require(sha(path) == predata['sequences'][seq]['frames_metadata_sha256'], 'Frame metadata changed')
        meta[seq] = read(path)
        require(len(meta[seq]) == len(frames[seq]) and
                [f['frame_index'] for f in frames[seq]] == list(range(len(frames[seq]))), 'Manifest coverage differs')
    require(set(score.get('rows', {})) == set(SEQS), 'Score coverage differs')
    prior_path = base/'sam2long_automatic_priority_v1/four_system_report_attempt01/results.json'
    prior = read(prior_path)
    require(prior.get('passed') is True and score.get('prior_four_system_report_sha256') == sha(prior_path)
            and prior.get('report_sha256') == sha(prior_path.parent/'report.md')
            and prior.get('reporter_sha256') == sha(ROOT/'scripts/report_sam2long_automatic_priority.py'),
            'Previously scored comparator provenance differs')
    seen, receipts, runtime_identity = set(), {}, None
    mask_count = 0
    for seq in SEQS:
        expected_images = []
        for i, item in enumerate(frames[seq]):
            image_path = ROOT/'data'/item['image']
            image_sha = sha(image_path)
            require(image_sha == meta[seq][i]['image_sha256'], 'Original input image changed')
            expected_images.append({'frame': i, 'sha256': image_sha})
        old = run.parent/f'lean_mps_baseline_{seq}_full_attempt01'
        old_receipt = read(old/'receipt.json')
        require(prior['source_hashes'][seq]['lean_receipt_sha256'] == sha(old/'receipt.json')
                and old_receipt.get('passed') is True and old_receipt.get('full_video') is True
                and len(old_receipt.get('records',[])) == len(frames[seq]), 'Prior Lean receipt differs')
        for i, record in enumerate(old_receipt['records']):
            require(record['frame'] == i and record['image_sha256'] == expected_images[i]['sha256']
                    and record['mask_sha256'] == sha(old/f'{i:06d}.png'), 'Prior Lean input or mask differs')
        require(set(mapping[seq]) == set(ARMS), 'Arm coverage differs')
        require(len(score['rows'][seq]) == len(frames[seq]), 'Scored frame count differs')
        receipts[seq] = {}
        for arm in ARMS:
            folder = Path(mapping[seq][arm]).resolve(strict=True)
            require(folder not in seen and folder.is_relative_to(run), 'Run path duplicate or outside run')
            seen.add(folder)
            receipt_path = folder/'completion.json'; rec = read(receipt_path)
            require(rec.get('passed') is True and rec.get('full_video') is True and rec.get('limit') == 0 and
                    rec.get('sequence') == seq and rec.get('arm') == arm and
                    rec.get('runner_source_sha256') == sha(ROOT/'scripts/run_boundary_refiner_priority_v1.py') and
                    rec.get('engineering_gate_sha256') == sha(gate) and
                    rec.get('label_guard',{}).get('open_attempts') == 0 and
                    rec.get('input_images') == expected_images and
                    score['run_receipt_sha256'][seq][arm] == sha(receipt_path), 'Run receipt differs')
            start = rec['prompt_frame']; expected = list(range(len(frames[seq])))
            require([r['frame'] for r in rec['masks']] == expected and
                    [r['frame'] for r in rec['post_prompt_state']] == list(range(start,len(expected))), 'Output coverage differs')
            for key, name in (('runtime_verification_sha256','verification/runtime_verification.json'),
                              ('memory_events_sha256','memory_events.json'),('positions_sha256','positions.json'),
                              ('selections_sha256','selections.json'),('telemetry_sha256','telemetry.json')):
                require(rec[key] == sha(folder/name), f'{name} changed')
            runtime = read(folder/'verification/runtime_verification.json')
            require(runtime.get('passed') and runtime.get('candidate_checkpoint_sha256') == rec['candidate_checkpoint_sha256']
                    and runtime.get('device') == 'mps float32' and runtime.get('fallback') == 'disabled'
                    and runtime.get('fast_math') == 'disabled' and runtime.get('memory_budget') == 16384
                    and runtime.get('insurance_reserve') == 3072 and runtime.get('baseline_replays') == 2,
                    'Runtime/candidate binding differs')
            require(runtime.get('source_sha256') and all(sha(ROOT/name) == value for name,value in runtime['source_sha256'].items()),
                    'Frozen runtime sources differ')
            study = Path(runtime['study']).resolve(strict=True)
            require(runtime['candidate_checkpoint_sha256'] == ga.get('candidate_checkpoint_sha256')
                    and runtime['candidate_checkpoint_sha256'] == sha(study/'runs/region_boundary/checkpoint.pt')
                    and runtime['control_checkpoint_sha256'] == sha(study/'runs/mask_only_boundary/checkpoint.pt')
                    and runtime['same_objective_no_contrast_sha256'] == sha(study/'runs/no_contrast_boundary/checkpoint.pt')
                    and runtime['manifest_sha256'] == sha(study/'manifest/manifest.json')
                    and runtime['scores_sha256'] == sha(study/'scores/scores.json'),
                    'Actual candidate/control/manifest or study evidence changed')
            runtime_identity = runtime if runtime_identity is None else runtime_identity
            require(runtime == runtime_identity, 'Runtime differs between arms or sequences')
            for name in ('model','refiner'):
                require(sha(folder/f'verification/{name}_tensor_fingerprints.json') == runtime[name+'_tensor_fingerprints_sha256'],
                        'Fingerprint record differs')
            for r in rec['masks']:
                path = folder/f"{r['frame']:06d}.png"
                require(sha(path) == r['sha256'], 'Frozen prediction changed')
                mask_count += 1
            receipts[seq][arm] = rec
        a, b = receipts[seq]['A'], receipts[seq]['B']
        for i, row in enumerate(score['rows'][seq]):
            require(row.get('frame') == i and row.get('source_frame_index') == frames[seq][i]['frame_index']
                    and row.get('post_prompt') is (i > a['prompt_frame'])
                    and row.get('initialization_frame') is (i == a['prompt_frame']),
                    'Scored source-frame mapping differs')
        for arm in ('B','C','D'):
            r = receipts[seq][arm]
            for key in ('prompt_frame','prompt_sha256','prompt_mask_sha256','view','checkpoint_sha256',
                        'detector_sha256','candidate_checkpoint_sha256','input_images'):
                require(r[key] == a[key], 'Matched study input differs')
            for key in ('mask_sha256','pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(r['post_prompt_state'][0][key] == a['post_prompt_state'][0][key], 'Initial conditioning state differs')
        first = {arm: receipts[seq][arm]['post_prompt_state'][1] for arm in ARMS}
        require(first['A']['output_logits_sha256'] == first['C']['output_logits_sha256']
                and first['B']['output_logits_sha256'] == first['D']['output_logits_sha256']
                and first['C']['maskmem_features_sha256'] == first['D']['maskmem_features_sha256'],
                'First-frame output/memory routing differs')
        for arm in ('B','C','D'):
            require(all(first[arm][k] == first['A'][k] for k in
                        ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256')),
                    'First native decoder state differs')
        for x,y in zip(b['post_prompt_state'],a['post_prompt_state'],strict=True):
            for key in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(x[key] == y[key], 'B changed baseline compact state')
        for name in ('memory_events.json','positions.json','selections.json'):
            require(read(Path(mapping[seq]['A'])/name) == read(Path(mapping[seq]['B'])/name), 'B changed baseline trajectory')
    return run, mapping, frames, meta, receipts, score, mask_count


def audit(run):
    run, mapping, frames, metadata, receipts, saved, mask_count = precheck(run)
    # Reaching this point requires all twelve complete runs and the primary score.
    errors, recomputed, checked_labels = [], {}, 0
    for seq in SEQS:
        rows = []
        start = receipts[seq]['A']['prompt_frame']
        old = run.parent/'..'/'contrast_refinement_v1'/f'lean_mps_baseline_{seq}_full_attempt01'
        old = old.resolve(strict=True)
        for i, item in enumerate(frames[seq]):
            mask_path = ROOT/'data'/item['mask']
            require(sha(mask_path) == metadata[seq][i]['label_sha256'], 'Label hash differs')
            gt = label(mask_path); checked_labels += 1
            target_boundary = boundary_target(gt)
            row = {'post_prompt': i > start}
            for arm in (*ARMS,'prior_lean'):
                folder = old if arm == 'prior_lean' else Path(mapping[seq][arm])
                with Image.open(folder/f'{i:06d}.png') as im:
                    pred = np.asarray(im.convert('L')) > 0
                row[arm] = metrics(pred,gt,target_boundary)
                compare(errors,[seq,i,arm],row[arm],saved['rows'][seq][i][arm])
            rows.append(row)
        recomputed[seq] = {}
        for arm in (*ARMS,'prior_lean'):
            recomputed[seq][arm] = {}
            for label_name, post in (('post_prompt',True),('all',False)):
                value = summarize(rows,arm,post)
                compare(errors,[seq,arm,label_name],value,saved['summaries'][seq][arm][label_name])
                recomputed[seq][arm][label_name] = value
        for name,arm,base in (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A')):
            compare(errors,[seq,name],paired(rows,arm,base),saved['paired_descriptive'][seq][name])
        pre = {'frames':start,'present_misses':sum(r['A']['present'] for r in rows[:start])}
        compare(errors,[seq,'preprompt'],pre,saved['preprompt'][seq])
        for arm in ARMS:
            raw = [x for x in read(Path(mapping[seq][arm])/'telemetry.json') if not x['conditioning'] and x['frame']>start]
            stats={'frames':len(raw), 'native_absence_frames':sum(not x['native_object_present'] for x in raw),
                   'native_exact_suppressed_frames':sum(x['native_exact_suppressed'] for x in raw),
                   'native_absence_refiner_positive_frames':sum(not x['native_object_present'] and x['refined']['positive_pixels']>0 for x in raw),
                   'native_absence_output_positive_frames':sum(not x['native_object_present'] and x['output']['positive_pixels']>0 for x in raw),
                   'native_absence_memory_positive_frames':sum(not x['native_object_present'] and x['memory_high_res']['positive_pixels']>0 for x in raw)}
            compare(errors,[seq,arm,'telemetry'],stats,saved['suppression_telemetry'][seq][arm])
    return {'passed':not errors, 'scope':'CPU artifact and metric audit only; no independent-patient or efficacy claim.',
            'run':str(run),'score_sha256':sha(run/'scores/results.json'), 'run_manifest_sha256':sha(run/'run_manifest.json'),
            'run_map_sha256':sha(run/'run_map.json'), 'audit_source_sha256':sha(Path(__file__)),
            'label_files_checked':checked_labels,'prediction_files_checked':mask_count,
            'error_count':len(errors),'errors':errors,'summaries':recomputed,
            'uncertainty_note':'Three previously inspected videos with unverified patient independence cannot support patient-level confidence intervals or false-alarm noninferiority.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=audit(args.run)
    out=validate_output(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'errors':result['error_count'],'output':str(out)}))
    if not result['passed']: raise SystemExit(1)

if __name__ == '__main__': main()
