#!/usr/bin/env python3
"""Verify and score the exploratory boundary-refiner A/B/C/D video study.

All twelve full-video predictions are validated before any label mask is
opened. The development sequences were inspected previously; this is not a
patient-independent or preregistered efficacy evaluation.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.score_source_group_boundary_v1 import boundary_five_pixel

SEQUENCES = ('seq11', 'seq20', 'seq23')
ARMS = ('A', 'B', 'C', 'D')
RUNNER = ROOT / 'scripts/run_boundary_refiner_priority_v1.py'
MATCH_KEYS = ('prompt_frame', 'view', 'prompt_sha256', 'prompt_mask_sha256',
              'checkpoint_sha256', 'detector_sha256', 'candidate_checkpoint_sha256',
              'runner_source_sha256', 'engineering_gate_sha256', 'input_images')


def read(path):
    return json.loads(Path(path).read_text())


def mapped_runs(path, experiment_root):
    """Require exactly twelve distinct, in-Drive directories, no aliases."""
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate run map key')
            result[key] = value
        return result
    data = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicate_keys)
    if set(data) != set(SEQUENCES) or any(set(data[s]) != set(ARMS) for s in SEQUENCES):
        raise ValueError('Run map must contain exactly three sequences and four arms')
    root = experiment_root.resolve(strict=True)
    runs = {s: {a: Path(data[s][a]).resolve(strict=True) for a in ARMS} for s in SEQUENCES}
    paths = [runs[s][a] for s in SEQUENCES for a in ARMS]
    if len(set(paths)) != 12 or any(not p.is_dir() or not p.is_relative_to(root) for p in paths):
        raise ValueError('Run folders must be distinct and inside experiment root')
    return runs


def _check_events(folder, receipt):
    for name, key in (('memory_events.json', 'memory_events_sha256'),
                      ('positions.json', 'positions_sha256'),
                      ('selections.json', 'selections_sha256'),
                      ('telemetry.json', 'telemetry_sha256'),
                      ('verification/runtime_verification.json', 'runtime_verification_sha256')):
        if digest(folder / name) != receipt[key]:
            raise ValueError(f'Frozen {name} changed')
    runtime = read(folder / 'verification/runtime_verification.json')
    if (not runtime.get('passed') or runtime.get('candidate_checkpoint_sha256') != receipt.get('candidate_checkpoint_sha256')
            or runtime.get('device') != 'mps float32' or runtime.get('fallback') != 'disabled'
            or runtime.get('fast_math') != 'disabled' or runtime.get('memory_budget') != 16384
            or runtime.get('insurance_reserve') != 3072 or runtime.get('baseline_replays') != 2):
        raise ValueError('Runtime verification failed')
    sources = runtime.get('source_sha256', {})
    if not sources or any(digest(ROOT / name) != sha for name, sha in sources.items()):
        raise ValueError('Frozen runtime source changed')
    for name in ('model', 'refiner'):
        key = name + '_tensor_fingerprints_sha256'
        if digest(folder / f'verification/{name}_tensor_fingerprints.json') != runtime.get(key):
            raise ValueError('Live tensor fingerprints changed')
    return runtime


def check_conditioning(current, baseline):
    if current.get('mask_sha256') is None or current.get('mask_sha256') != baseline.get('mask_sha256'):
        raise ValueError('Conditioning output changed')
    for key in ('pred_masks_sha256', 'maskmem_features_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
        if current.get(key) is None or current.get(key) != baseline.get(key):
            raise ValueError('Conditioning native compact state changed')


def check_telemetry(folder, receipt, frame_count):
    """Validate all label-independent diagnostics before scoring begins."""
    start = receipt['prompt_frame']
    items = read(folder / 'telemetry.json')
    if not isinstance(items, list) or any(not isinstance(x, dict) or
            type(x.get('conditioning')) is not bool or type(x.get('frame')) is not int for x in items):
        raise ValueError('Invalid telemetry records')
    later = [x for x in items if not x['conditioning'] and x['frame'] > start]
    if [x['frame'] for x in later] != list(range(start + 1, frame_count)):
        raise ValueError('Per-frame native suppression telemetry is incomplete')
    for x in later:
        if (type(x.get('native_object_present')) is not bool or type(x.get('native_exact_suppressed')) is not bool
                or type(x.get('object_score_logit')) not in (int, float) or not math.isfinite(x['object_score_logit'])
                or x['native_object_present'] != (x['object_score_logit'] > 0)):
            raise ValueError('Invalid native suppression telemetry')
        for key in ('native', 'refined', 'output', 'memory_high_res'):
            s = x.get(key, {})
            if (type(s.get('numel')) is not int or s['numel'] <= 0 or type(s.get('positive_pixels')) is not int
                    or not 0 <= s['positive_pixels'] <= s['numel']
                    or any(type(s.get(k)) not in (int, float) or not math.isfinite(s[k]) for k in ('min', 'max'))
                    or s['min'] > s['max'] or not isinstance(s.get('sha256'), str) or len(s['sha256']) != 64):
                raise ValueError('Invalid mask-logit telemetry')
    wall = receipt.get('prediction_wall_seconds')
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError('Per-run measured wall time is missing or invalid')
    stats = {'frames': len(later),
             'native_absence_frames': sum(not x['native_object_present'] for x in later),
             'native_exact_suppressed_frames': sum(x['native_exact_suppressed'] for x in later),
             'native_absence_refiner_positive_frames': sum(not x['native_object_present'] and x['refined']['positive_pixels'] > 0 for x in later),
             'native_absence_output_positive_frames': sum(not x['native_object_present'] and x['output']['positive_pixels'] > 0 for x in later),
             'native_absence_memory_positive_frames': sum(not x['native_object_present'] and x['memory_high_res']['positive_pixels'] > 0 for x in later),
             'records': later}
    timing = {'prediction_wall_seconds': wall,
              'scope': 'Staging, propagation, saving and telemetry; not isolated model throughput.'}
    return stats, timing


def check_frozen_runs(runs, videos, lean_base, runner=RUNNER):
    """Validate coverage, inputs, routing invariants, and masks without labels."""
    runner_hash = digest(runner)
    receipts, baselines, telemetry, timing = {}, {}, {}, {}
    candidate_hash = None
    gate_hash = None
    common_runtime = None
    for seq in SEQUENCES:
        frames = videos[seq]
        baseline_folder = lean_base / f'lean_mps_baseline_{seq}_full_attempt01'
        old = read(baseline_folder / 'receipt.json')
        if (not old.get('passed') or not old.get('full_video') or old.get('sequence') != seq
                or len(old.get('records', [])) != len(frames)):
            raise ValueError('Archived comparator is incomplete')
        from scripts.run_boundary_video_factorial_v1 import DETECTOR_SHA256
        if old.get('identity', {}).get('detector_sha256') != DETECTOR_SHA256:
            raise ValueError('Archived detector checkpoint differs')
        for i, frame in enumerate(frames):
            row = old['records'][i]
            if (row['frame'] != i or row['source_frame_index'] != frame.frame_index
                    or row['image_sha256'] != digest(frame.image_path)
                    or row['mask_sha256'] != digest(baseline_folder / f'{i:06d}.png')):
                raise ValueError('Archived comparator input or output changed')
        baselines[seq] = (baseline_folder, old)
        receipts[seq] = {}
        telemetry[seq], timing[seq] = {}, {}
        for arm in ARMS:
            folder = runs[seq][arm]
            r = read(folder / 'completion.json')
            if (not r.get('passed') or not r.get('full_video') or r.get('limit') != 0
                    or r.get('sequence') != seq or r.get('arm') != arm
                    or r.get('runner_source_sha256') != runner_hash
                    or r.get('prompt_frame') != old['prompt_frame']
                    or r.get('prompt_sha256') != old['prompt']['prompt_metadata_sha256']
                    or r.get('prompt_mask_sha256') != old['prompt']['prompt_mask_sha256']
                    or r.get('checkpoint_sha256') != old['identity']['checkpoint_sha256']
                    or r.get('detector_sha256') != old['identity']['detector_sha256']
                    or not r.get('engineering_gate_sha256')
                    or r.get('label_guard', {}).get('open_attempts') != 0):
                raise ValueError('Run identity, scope, or comparator prompt differs')
            if candidate_hash is None:
                candidate_hash = r['candidate_checkpoint_sha256']
            if r.get('candidate_checkpoint_sha256') != candidate_hash:
                raise ValueError('Candidate checkpoint differs across runs')
            if gate_hash is None:
                gate_hash = r['engineering_gate_sha256']
            if r['engineering_gate_sha256'] != gate_hash:
                raise ValueError('Engineering gate differs across runs')
            expected_images = [{'frame': i, 'sha256': digest(f.image_path)} for i, f in enumerate(frames)]
            if r.get('input_images') != expected_images:
                raise ValueError('Input image hashes or frame order differ')
            if [m['frame'] for m in r.get('masks', [])] != list(range(len(frames))):
                raise ValueError('Full-frame output coverage differs')
            if [x['frame'] for x in r.get('post_prompt_state', [])] != list(range(r['prompt_frame'], len(frames))):
                raise ValueError('Post-prompt state coverage differs')
            expected_view = ({'kind': 'fixed_seq23_main_viewport', 'xyxy': list(VIEWPORT)}
                             if seq == 'seq23' else {'kind': 'full_source_frame'})
            if r.get('view') != expected_view:
                raise ValueError('Sequence viewport differs')
            runtime = _check_events(folder, r)
            telemetry[seq][arm], timing[seq][arm] = check_telemetry(folder, r, len(frames))
            if common_runtime is None:
                common_runtime = runtime
            elif runtime != common_runtime:
                raise ValueError('Live model, refiner or environment identity differs')
            for row in r['masks']:
                i = row['frame']
                mask = folder / f'{i:06d}.png'
                if digest(mask) != row['sha256']:
                    raise ValueError('Frozen mask hash differs')
                with Image.open(frames[i].image_path) as original, Image.open(mask) as image:
                    if image.format != 'PNG' or image.size != original.size:
                        raise ValueError('Frozen mask geometry or format differs')
                    binary = np.asarray(image.convert('L'))
                    if not np.isin(binary, (0, 255)).all():
                        raise ValueError('Frozen mask is not binary')
                    if i < r['prompt_frame'] and binary.any():
                        raise ValueError('Pre-prompt mask is not empty')
            receipts[seq][arm] = r
        a = receipts[seq]['A']
        for arm in 'BCD':
            r = receipts[seq][arm]
            if any(r.get(key) != a.get(key) for key in MATCH_KEYS):
                raise ValueError('Factorial prompt, runtime or input differs')
            check_conditioning(r['post_prompt_state'][0], a['post_prompt_state'][0])
            if len(r['post_prompt_state']) > 1:
                for key in ('pred_masks_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
                    if (r['post_prompt_state'][1].get(key) is None or
                            r['post_prompt_state'][1].get(key) != a['post_prompt_state'][1].get(key)):
                        raise ValueError('First post-prompt native decoder state differs')
        b = receipts[seq]['B']
        for current, original in zip(b['post_prompt_state'][1:], a['post_prompt_state'][1:], strict=True):
            for key in ('pred_masks_sha256', 'maskmem_features_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256'):
                if current.get(key) != original.get(key) or current.get(key) is None:
                    raise ValueError('Output-only arm changed native state')
        for file in ('memory_events.json', 'positions.json', 'selections.json'):
            if read(runs[seq]['A'] / file) != read(runs[seq]['B'] / file):
                raise ValueError('Output-only arm changed memory routing')
    return receipts, baselines, telemetry, timing


def metric(pred, truth):
    if pred.shape != truth.shape:
        raise ValueError('Prediction/label geometry differs')
    pp, gt = int(pred.sum()), int(truth.sum())
    tp = int((pred & truth).sum())
    return {'present': bool(gt), 'dice': (2 * tp / (pp + gt)) if pp + gt else 1.,
            'iou': tp / (pp + gt - tp) if pp + gt - tp else 1.,
            'precision': tp / pp if pp else None, 'recall': tp / gt if gt else None,
            'boundary_f_5px': boundary_five_pixel(pred, truth),
            'predicted_pixels': pp, 'false_alarm': bool(pp) if not gt else None}


def summary(rows, arm, post_only=True):
    chosen = [r for r in rows if not post_only or r['post_prompt']]
    present = [r[arm] for r in chosen if r[arm]['present']]
    empty = [r[arm] for r in chosen if not r[arm]['present']]
    def mean(key, items):
        values = [v[key] for v in items if v[key] is not None]
        return float(np.mean(values)) if values else None
    return {'frames': len(chosen), 'present_frames': len(present),
            'mean_present_dice': mean('dice', present),
            'mean_present_iou': mean('iou', present),
            'mean_present_precision_when_defined': mean('precision', present),
            'mean_present_recall': mean('recall', present),
            'mean_present_boundary_f_5px': mean('boundary_f_5px', present),
            'zero_overlap_present_frames': sum(v['dice'] == 0 for v in present),
            'empty_frames': len(empty), 'false_alarms': sum(v['false_alarm'] for v in empty),
            'mean_empty_predicted_pixels': mean('predicted_pixels', empty)}


def paired_summary(rows, arm, reference):
    """Paired post-prompt differences, descriptive only on these videos."""
    chosen = [r for r in rows if r['post_prompt'] and r[arm]['present']]
    dice = [r[arm]['dice'] - r[reference]['dice'] for r in chosen]
    return {'present_frames': len(chosen), 'mean_present_dice_difference': float(np.mean(dice)) if dice else None,
            'improved': sum(v > 0 for v in dice), 'worsened': sum(v < 0 for v in dice),
            'unchanged': sum(v == 0 for v in dice)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment-root', 'run-map', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve(strict=True)
    runs = mapped_runs(args.run_map, root)
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    lean = root / 'contrast_refinement_v1'
    # Do not move any label access above this complete twelve-run check.
    receipts, baselines, telemetry, timing = check_frozen_runs(runs, videos, lean)
    from scripts.run_boundary_video_factorial_v1 import (AUDIT, BASELINE, CHECKPOINT, PRE, REF, STUDY, verify_gate_audit)
    from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate
    launch_path = args.run_map.parent / 'run_manifest.json'
    launch = read(launch_path)
    gate_path = Path(launch.get('gate_path', ''))
    gate_audit = Path(launch.get('independent_gate_audit_path', ''))
    if (launch.get('run_map') != read(args.run_map) or set(launch.get('sequences', [])) != set(SEQUENCES)
            or launch.get('arms') != list(ARMS) or launch.get('runner_sha256') != digest(RUNNER)
            or launch.get('mps_fallback') != '0' or launch.get('mps_fast_math') != '0'
            or launch.get('python_hash_seed') != '0'
            or launch.get('scorer_sha256') != digest(Path(__file__))
            or launch.get('launcher_sha256') != digest(ROOT / 'scripts/run_boundary_video_factorial_v1.py')
            or launch.get('gate_sha256') != digest(gate_path)
            or launch.get('independent_gate_audit_sha256') != digest(gate_audit)
            or any(receipts[s][a].get('engineering_gate_sha256') != digest(gate_path) for s in SEQUENCES for a in ARMS)):
        raise ValueError('Full-video launch, source, map or engineering gate differs')
    verify_full_gate(gate_path, study=STUDY, audit=AUDIT, checkpoint=CHECKPOINT,
                     split_preflight=PRE, baseline_reference=BASELINE, seq4_capture=REF)
    verify_gate_audit(gate_path, gate_audit)
    pre_root = root / 'lean_channel_attention_training_pilot_v1/split_preflight_attempt03'
    pre_path = pre_root / 'manifest.json'
    pre_done = read(pre_root / 'completion.json')
    if not pre_done.get('passed') or pre_done.get('manifest_sha256') != digest(pre_path):
        raise ValueError('Source preflight manifest changed')
    pre = read(pre_path)
    # Bind every label metadata file before opening any label, not one at a time.
    label_metadata = {}
    for seq in SEQUENCES:
        source = Path(pre['sequences'][seq]['source']) / seq / 'frames.json'
        if digest(source) != pre['sequences'][seq]['frames_metadata_sha256']:
            raise ValueError('Label metadata changed')
        label_metadata[seq] = read(source)
        if len(label_metadata[seq]) != len(videos[seq]):
            raise ValueError('Label metadata coverage differs')
    prior_path = root / 'sam2long_automatic_priority_v1/four_system_report_attempt01/results.json'
    prior = read(prior_path)
    if (not prior.get('passed') or prior.get('reporter_sha256') != digest(ROOT / 'scripts/report_sam2long_automatic_priority.py')
            or prior.get('report_sha256') != digest(prior_path.parent / 'report.md')):
        raise ValueError('Archived comparator report differs')
    rows_all, summaries, source_hashes, preprompt, differences, paired = {}, {}, {}, {}, {}, {}
    for seq in SEQUENCES:
        frames = videos[seq]
        folder, old = baselines[seq]
        if prior['source_hashes'][seq]['lean_receipt_sha256'] != digest(folder / 'receipt.json'):
            raise ValueError('Archived comparator receipt differs from report')
        meta = label_metadata[seq]
        start = old['prompt_frame']
        rows = []
        changed_pixels, changed_frames = 0, 0
        for i, frame in enumerate(frames):
            if digest(frame.mask_path) != meta[i]['label_sha256']:
                raise ValueError('Label hash changed')
            truth = load_label_mask(frame.mask_path, [1]) == 1
            with Image.open(folder / f'{i:06d}.png') as image:
                legacy = np.asarray(image.convert('L')) > 0
            out = {'frame': i, 'source_frame_index': frame.frame_index,
                   'post_prompt': i > start, 'initialization_frame': i == start,
                   'prior_lean': metric(legacy, truth)}
            oldmetric = prior['rows'][seq][i]['lean']
            for key in ('present', 'dice', 'false_alarm', 'predicted_pixels'):
                if out['prior_lean'][key] != oldmetric[key]:
                    raise ValueError('Archived Lean score disagrees')
            for arm in ARMS:
                with Image.open(runs[seq][arm] / f'{i:06d}.png') as image:
                    pred = np.asarray(image.convert('L')) > 0
                out[arm] = metric(pred, truth)
                if arm == 'A':
                    count = int(np.count_nonzero(pred != legacy))
                    changed_pixels += count
                    changed_frames += bool(count)
            rows.append(out)
        rows_all[seq] = rows
        summaries[seq] = {arm: {'post_prompt': summary(rows, arm), 'all': summary(rows, arm, False)}
                          for arm in (*ARMS, 'prior_lean')}
        paired[seq] = {name: paired_summary(rows, arm, base) for name, arm, base in
                       (('B_minus_A', 'B', 'A'), ('C_minus_A', 'C', 'A'),
                        ('D_minus_B', 'D', 'B'), ('D_minus_A', 'D', 'A'))}
        differences[seq] = {'frames_with_different_pixels': changed_frames,
                            'total_different_pixels': changed_pixels}
        preprompt[seq] = {'frames': start, 'present_misses': sum(r['A']['present'] for r in rows if r['frame'] < start)}
        source_hashes[seq] = {arm: digest(runs[seq][arm] / 'completion.json') for arm in ARMS}
    outdir = validate_output(args.output)
    outdir.mkdir(parents=True, exist_ok=False)
    lines = ['# Exploratory full-video boundary refiner factorial', '',
             'Same archived automatic prompt, project Lean token-window system and seq23 viewport; post-prompt frames exclude the initialization frame. Previously inspected development videos; no independent validation or causal claim about SAM2Long.', '',
             '| Sequence | Present frames | Dice A / B / C / D | Labeled-empty alarms A / B / C / D |',
             '|---|---:|---:|---:|']
    for seq in SEQUENCES:
        s = summaries[seq]
        dice = ' / '.join(f"{s[a]['post_prompt']['mean_present_dice']:.4f}" for a in ARMS)
        fa = ' / '.join(f"{s[a]['post_prompt']['false_alarms']}/{s[a]['post_prompt']['empty_frames']}" for a in ARMS)
        lines.append(f"| {seq} | {s['A']['post_prompt']['present_frames']} | {dice} | {fa} |")
    lines.extend(['', 'Previously audited comparator scores on the same post-prompt frame indices:', '',
                  '| Sequence | Native SAM2 | Project Lean | SAMURAI | SAM2Long |',
                  '|---|---:|---:|---:|---:|'])
    for seq in SEQUENCES:
        values = prior['summaries'][seq]
        scores = ' | '.join(f"{values[k]['mean_present_dice']:.4f}" for k in ('native', 'lean', 'samurai', 'sam2long'))
        lines.append(f'| {seq} | {scores} |')
    lines.extend(['', 'Previously audited comparator labeled-empty false alarms:', '',
                  '| Sequence | Native SAM2 | Project Lean | SAMURAI | SAM2Long |',
                  '|---|---:|---:|---:|---:|'])
    for seq in SEQUENCES:
        values = prior['summaries'][seq]
        alarms = ' | '.join(f"{values[k]['false_alarms']}/{values[k]['empty_frames']}" for k in ('native', 'lean', 'samurai', 'sam2long'))
        lines.append(f'| {seq} | {alarms} |')
    lines.extend(['', 'Native no-object decisions with positive refined output or memory:', '',
                  '| Sequence | Arm | Native no-object frames | Output positive | Memory positive |',
                  '|---|---|---:|---:|---:|'])
    for seq in SEQUENCES:
        for arm in ARMS:
            t = telemetry[seq][arm]
            lines.append(f"| {seq} | {arm} | {t['native_absence_frames']} | {t['native_absence_output_positive_frames']} | {t['native_absence_memory_positive_frames']} |")
    lines.extend(['', 'A: native output and memory; B: refined output with native memory; C: native output with refined memory; D: refined output and memory. B vs A preserves the memory trajectory; C and D may change subsequent trajectories. Conditioning is protected.',
                  '', 'This is an explicitly requested exploratory override of the original candidate-selection screen. The boundary-trained region checkpoint was selected after development outcomes were inspected; no minimum clinically meaningful effect or noninferiority margin was prespecified.',
                  '', 'The models and prompts are matched within this study. The project Lean baseline is a modified token-window variant, not unmodified author Lean. Historical YOLO scanned until the first detection and no new redetection is introduced. The seq23 crop is retrospective.',
                  '', 'Full-run subprocesses set PYTHONHASHSEED=0, as observed in the short-gate invocation; the short-gate saved receipts did not independently record this variable.',
                  '', 'Comparator scores above use the same initialization and viewport; SAM2Long is noncausal and chooses its pathway after seeing the full video. CMSA-Net is separately documented in cmsa_net_polypgen_v1/automatic_priority_audit_attempt01/report.md: it uses its own checkpoint and preprocessing, and sees the seq23 inset.',
                  '', 'No-object telemetry reports the model’s existing object-score decision and whether the output or memory logits contain positive pixels; it does not add a new inference threshold. Low-resolution output and high-resolution memory counts are diagnostic and are not full-size mask areas.',
                  '', 'Boundary F uses the previously defined five-native-pixel elliptical tolerance, not a physical distance. Labeled-empty frames are not necessarily confirmed clinically absent. All three videos were previously inspected; patient independence and detector training exposure remain unverified. Full rows, pre-prompt misses, all-frame metrics and baseline pixel differences are in results.json.'])
    report = outdir / 'report.md'
    report.write_text('\n'.join(lines) + '\n')
    result = {'passed': True, 'scope': 'Exploratory A/B/C/D full-video study on previously inspected development videos.',
              'summaries': summaries, 'paired_descriptive': paired, 'rows': rows_all, 'preprompt': preprompt,
              'suppression_telemetry': telemetry, 'timing': timing,
              'prior_lean_pixel_differences': differences, 'run_receipt_sha256': source_hashes,
              'run_map_sha256': digest(args.run_map), 'launch_manifest_sha256': digest(launch_path),
              'engineering_gate_sha256': digest(gate_path), 'independent_gate_audit_sha256': digest(gate_audit),
              'prior_four_system_report_sha256': digest(prior_path),
              'scorer_sha256': digest(Path(__file__)), 'report_sha256': digest(report)}
    (outdir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(outdir), 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
