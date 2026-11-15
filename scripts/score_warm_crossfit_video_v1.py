#!/usr/bin/env python3
"""Freeze, score and independently recompute warm out-of-fold video metrics."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT
from scripts.audit_warm_crossfit_video_smoke_v1 import audit as audit_smoke
from scripts.channel_value_fixed_read_v1 import install_phase_mask_guard
from scripts.crossfit_observer_capture_shared_v1 import DATA, EXISTING, OUT, archived_frame_metadata, phase_artifact, validate_plan
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_video_v1 import sources
from scripts.project_archived_label_hashes_v1 import project_label_hashes
from scripts.score_boundary_video_factorial_v1 import metric
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_video_contract_v1 import read_bound, require

ARMS = ('A', 'B', 'C', 'D')


def telemetry_summary(items, start, end):
    require([x.get('frame') for x in items] == list(range(start + 1, end)), 'Telemetry coverage differs')
    for item in items:
        score = item.get('object_score_logit')
        require(item.get('conditioning') is False and type(item.get('native_object_present')) is bool and
                type(item.get('native_exact_suppressed')) is bool and isinstance(score, (int, float)) and
                math.isfinite(score) and item['native_object_present'] == (score > 0), 'Object telemetry differs')
        for key in ('native', 'refined', 'output', 'memory_high_res'):
            part = item[key]
            require(all(isinstance(part.get(k), (int, float)) and math.isfinite(part[k]) for k in ('min', 'max')) and
                    part['min'] <= part['max'] and
                    type(part.get('positive_pixels')) is int and type(part.get('numel')) is int and
                    part['numel'] > 0 and 0 <= part['positive_pixels'] <= part['numel'] and
                    len(part.get('sha256', '')) == 64 and
                    all(c in '0123456789abcdef' for c in part['sha256']),
                    'Mask telemetry differs')
    absent = [x for x in items if not x['native_object_present']]
    return {'post_prompt_frames': len(items), 'native_absent_score_frames': len(absent),
        'native_exact_suppressed_frames': sum(x['native_exact_suppressed'] for x in items),
        'refined_positive_with_native_absent_score': sum(x['refined']['positive_pixels'] > 0 for x in absent),
        'output_positive_with_native_absent_score': sum(x['output']['positive_pixels'] > 0 for x in absent)}


def validate_frozen(root, *, completed=()):
    from scripts.run_warm_crossfit_video_study_v1 import phases
    root = Path(root).resolve(strict=True); require(root.is_relative_to(OUT.resolve(strict=True)), 'Batch outside Drive')
    manifest_file = root / 'run_manifest.json'; manifest_sha = digest(manifest_file)
    manifest = read_bound(manifest_file, manifest_sha)
    tasks = phases(root, Path(manifest['crossfit_root']), Path(manifest['preflight_path']))
    require(manifest.get('source_sha256') == sources() and manifest.get('phases') == [x[0] for x in tasks],
            'Video study source or plan differs')
    rows = [json.loads(x) for x in (root / 'phases.jsonl').read_text().splitlines() if x.strip()]
    expected = [name for name, _, _ in tasks if name not in ('verify', 'score', 'audit')] + list(completed)
    require([x.get('phase') for x in rows] == expected, 'All video predictions must freeze before scoring')
    phase_sha = {}
    for row, (name, _, artifact) in zip(rows, tasks, strict=False):
        phase_sha[name] = phase_artifact(root, name, artifact)
        require(phase_sha[name] == row.get('artifact_sha256'), 'Video phase changed')
    certfile = root / 'certificate/certificate.json'; cert_sha = phase_sha['certificate']
    cert = read_bound(certfile, cert_sha)
    require(cert.get('passed') is True and cert.get('source_sha256') == sources() and
            cert.get('label_guard', {}).get('open_attempts') == 0, 'Video certificate differs')
    smoke_file = root / 'smoke_audit/audit.json'; saved_smoke = read_bound(smoke_file, phase_sha['smoke_audit'])
    rebuilt = audit_smoke(root, certfile, cert_sha)
    require({k: v for k, v in saved_smoke.items() if k != 'label_guard'} == rebuilt and
            saved_smoke.get('label_guard', {}).get('open_attempts') == 0, 'Video smoke audit differs')
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}; refs = {}; runs = {}
    for seq, spec in cert['references'].items():
        frames = videos[seq]; require(len(frames) == spec['frames'] == len(spec['image_sha256']), 'Frame coverage differs')
        require(all(digest(f.image_path) == sha for f, sha in zip(frames, spec['image_sha256'], strict=True)),
                'Video source image changed')
        ref = read_bound(spec['reference_path'], spec['reference_sha256']); start = spec['prompt_frame']
        require([r['frame'] for r in ref['rows']] == list(range(start, len(frames))), 'Reference frame coverage differs')
        for row in ref['rows']:
            path = Path(spec['reference_path']).parent / f"{row['frame']:06d}.png"
            require(digest(path) == row['mask_sha256'], 'Reference video mask changed')
        refs[seq] = {'path': spec['reference_path'], 'sha256': spec['reference_sha256']}; runs[seq] = {}
        for arm in ('B', 'C', 'D'):
            folder = root / 'full' / seq / arm; receipt_file = folder / 'receipt.json'
            sha = phase_sha[f'full_{seq}_{arm}']; rec = read_bound(receipt_file, sha)
            require(rec.get('passed') is True and rec.get('sequence') == seq and rec.get('arm') == arm and
                rec.get('limit') == 0 and rec.get('certificate_sha256') == cert_sha and
                rec.get('smoke_audit_sha256') == phase_sha['smoke_audit'] and
                rec.get('reference_sha256') == spec['reference_sha256'] and rec.get('prompt_frame') == start and
                rec.get('roi_xyxy') == spec['roi_xyxy'] and rec.get('label_guard', {}).get('open_attempts') == 0 and
                isinstance(rec.get('wall_seconds'), (int, float)) and math.isfinite(rec['wall_seconds']) and
                rec['wall_seconds'] > 0 and isinstance(rec.get('timing_scope'), str) and
                [x['frame'] for x in rec['rows']] == list(range(start, len(frames))) and
                [x['frame'] for x in rec['masks']] == list(range(len(frames))), 'Full video receipt differs')
            runtime = read_bound(folder / 'verification/runtime_verification.json', rec['runtime_verification_sha256'])
            require(runtime.get('passed') is True and runtime.get('sequence') == seq and runtime.get('device') == 'mps' and
                runtime.get('certificate_sha256') == cert_sha and
                runtime.get('fold_checkpoint') == cert['checkpoints'][str(spec['fold'])], 'Full runtime differs')
            for item, frame in zip(rec['masks'], frames, strict=True):
                file = folder / f"{item['frame']:06d}.png"; require(digest(file) == item['sha256'], 'Full mask changed')
                with Image.open(frame.image_path) as image, Image.open(file) as pred:
                    values = np.asarray(pred.convert('L'))
                    require(pred.format == 'PNG' and pred.size == image.size and np.isin(values, (0, 255)).all() and
                            (item['frame'] >= start or not values.any()), 'Full mask geometry differs')
                    if seq == 'seq23': require(not values[:, :VIEWPORT[0]].any(), 'Seq23 prediction outside ROI')
                require(digest(file) == item['sha256'], 'Full mask changed during decoding')
            initial = folder / f'{start:06d}.png'
            initial_reference = Path(spec['reference_path']).parent / f'{start:06d}.png'
            with Image.open(initial) as a, Image.open(initial_reference) as b:
                require(np.array_equal(np.asarray(a.convert('L')), np.asarray(b.convert('L'))),
                        'Full conditioning output differs from warm A')
            traces = {name: read_bound(folder / f'{name}.json', rec['trace_sha256'][name])
                      for name in ('positions', 'selection', 'anchor_events', 'token_window_events', 'telemetry')}
            telemetry_summary(traces['telemetry'], start, len(frames))
            if arm == 'B':
                require([x['frame'] for x in rec['native_masks']] == list(range(start, len(frames))),
                        'B native mask coverage differs')
                for item in rec['native_masks']:
                    file = folder / 'native' / f"{item['frame']:06d}.png"
                    other = Path(spec['reference_path']).parent / f"{item['frame']:06d}.png"
                    refrow = ref['rows'][item['frame'] - start]
                    require(digest(file) == item['sha256'] and digest(other) == refrow['mask_sha256'],
                            'B native or reference mask changed')
                    with Image.open(file) as a, Image.open(other) as b:
                        require(np.array_equal(np.asarray(a.convert('L')), np.asarray(b.convert('L'))),
                                'B native mask differs from warm A')
                    require(digest(file) == item['sha256'] and digest(other) == refrow['mask_sha256'],
                            'B native or reference mask changed during decoding')
                require(all(traces[k] == ref[k] for k in ('positions', 'selection', 'anchor_events', 'token_window_events')),
                        'B native trace differs from warm A')
            runs[seq][arm] = {'path': str(receipt_file), 'sha256': sha}
    require(len(refs) == 21 and digest(manifest_file) == manifest_sha, 'Video patient coverage differs')
    return {'launch_sha256': manifest_sha, 'certificate_sha256': cert_sha,
            'smoke_audit_sha256': phase_sha['smoke_audit'], 'references': refs, 'runs': runs}


def summarise(rows, arm, post):
    values = [r[arm] for r in rows if not post or r['post_prompt']]
    present = [x for x in values if x['present']]; empty = [x for x in values if not x['present']]
    def mean(key, items):
        values = [x[key] for x in items if x[key] is not None]
        return math.fsum(values) / len(values) if values else None
    return {'frames': len(values), 'present_frames': len(present),
        'mean_present_dice': mean('dice', present), 'mean_present_iou': mean('iou', present),
        'mean_present_precision_when_defined': mean('precision', present),
        'mean_present_recall': mean('recall', present),
        'mean_present_boundary_f_5px': mean('boundary_f_5px', present),
        'zero_overlap_present_frames': sum(x['dice'] == 0 for x in present),
        'empty_frames': len(empty), 'false_alarms': sum(x['false_alarm'] for x in empty),
        'mean_empty_predicted_pixels': mean('predicted_pixels', empty)}


def compute(root, proof, *, independent=False):
    if independent:
        from scripts.audit_boundary_video_factorial_v1 import label, metrics as calculate
    else:
        label = lambda path: load_label_mask(path, [1]) == 1
        calculate = metric
    cert = read_bound(root / 'certificate/certificate.json', proof['certificate_sha256'])
    manifest = read_bound(EXISTING / 'manifest.json', cert['old_manifest_sha256'])
    oldplan = read_bound(manifest['preflight_path'], manifest['preflight_sha256']); index = validate_plan()
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}; rows_all = {}; summaries = {}
    preprompt = {}; timing = {}; telemetry = {}
    for seq, spec in cert['references'].items():
        frames = videos[seq]; start = spec['prompt_frame']
        if seq in manifest['sequences']:
            old = oldplan['sequences'][seq]
            binding = {'path': str(Path(old['source']) / seq / 'frames.json'), 'sha256': old['archived_frames_sha256']}
        else:
            binding = archived_frame_metadata(index, seq)
        labels = project_label_hashes(binding, len(frames), list(range(len(frames))))
        reference = read_bound(spec['reference_path'], spec['reference_sha256']); refrows = {x['frame']: x for x in reference['rows']}
        receipts = {arm: read_bound(proof['runs'][seq][arm]['path'], proof['runs'][seq][arm]['sha256']) for arm in ('B','C','D')}
        hashes = {arm: {x['frame']: x['sha256'] for x in receipts[arm]['masks']} for arm in receipts}
        rows = []
        for i, frame in enumerate(frames):
            require(digest(frame.image_path) == spec['image_sha256'][i] and digest(frame.mask_path) == labels[i],
                    'Video image or evaluation label changed')
            truth = label(frame.mask_path); require(digest(frame.mask_path) == labels[i], 'Label changed during decode')
            row = {'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                   'initialization_frame': i == start}
            for arm in ARMS:
                if arm == 'A' and i < start:
                    pred = np.zeros_like(truth, dtype=bool)
                else:
                    folder = Path(spec['reference_path']).parent if arm == 'A' else root / 'full' / seq / arm
                    file = folder / f'{i:06d}.png'; expected = refrows[i]['mask_sha256'] if arm == 'A' else hashes[arm][i]
                    require(digest(file) == expected, 'Video prediction changed before score')
                    with Image.open(file) as image: pred = np.asarray(image.convert('L')) > 0
                    require(digest(file) == expected and pred.shape == truth.shape, 'Video prediction geometry changed')
                row[arm] = calculate(pred, truth)
                if seq == 'seq23':
                    x0, y0, x1, y1 = VIEWPORT
                    row.setdefault('viewport', {})[arm] = calculate(pred[y0:y1,x0:x1], truth[y0:y1,x0:x1])
            rows.append(row)
        rows_all[seq] = rows
        summaries[seq] = {arm: {'post_prompt': summarise(rows, arm, True), 'all': summarise(rows, arm, False),
            'initialization_frame': summarise([r for r in rows if r['initialization_frame']], arm, False)} for arm in ARMS}
        preprompt[seq] = {'frames': start, 'present_misses': sum(row['A']['present'] for row in rows[:start])}
        timing[seq] = {arm: {'wall_seconds': receipts[arm]['wall_seconds'],
                            'scope': receipts[arm]['timing_scope']} for arm in ('B', 'C', 'D')}
        telemetry[seq] = {arm: telemetry_summary(read_bound(root / 'full' / seq / arm / 'telemetry.json',
            receipts[arm]['trace_sha256']['telemetry']), start, len(frames)) for arm in ('B', 'C', 'D')}
    overall = {}
    for arm in ARMS:
        patients = [summaries[s][arm]['post_prompt']['mean_present_dice'] for s in summaries]
        patients = [x for x in patients if x is not None]
        flat = [row for rows in rows_all.values() for row in rows]
        overall[arm] = {'patient_macro_present_dice': math.fsum(patients) / len(patients) if patients else None,
                        'patients_with_present_frames': len(patients), 'pooled_post_prompt': summarise(flat, arm, True)}
    paired = {}
    for arm in ('B', 'C', 'D'):
        paired[arm] = {}
        for seq, rows in rows_all.items():
            delta = [row[arm]['dice'] - row['A']['dice'] for row in rows
                     if row['post_prompt'] and row['A']['present']]
            paired[arm][seq] = {'present_frames': len(delta),
                'mean_present_dice_difference': math.fsum(delta) / len(delta) if delta else None,
                'improved': sum(x > 0 for x in delta), 'worsened': sum(x < 0 for x in delta),
                'unchanged': sum(x == 0 for x in delta)}
    viewport_rows = [row['viewport'] | {'post_prompt': row['post_prompt']} for row in rows_all.get('seq23', [])]
    viewport = {arm: {'post_prompt': summarise(viewport_rows, arm, True),
                      'all': summarise(viewport_rows, arm, False)} for arm in ARMS}
    result = {'rows': rows_all, 'summaries': summaries, 'overall': overall,
            'paired_vs_A': paired, 'seq23_viewport': viewport, 'preprompt': preprompt, 'timing': timing,
            'telemetry': telemetry,
            'no_initialization': ['seq1','seq7'], 'no_initialization_dice': None}
    result['descriptive_analysis'] = descriptive_analysis(result)
    return result


def descriptive_analysis(result):
    analysis = {}
    for arm in ('B', 'C', 'D'):
        differences = [item['mean_present_dice_difference'] for item in result['paired_vs_A'][arm].values()
                       if item['mean_present_dice_difference'] is not None]
        baseline = result['overall']['A']['pooled_post_prompt']
        candidate = result['overall'][arm]['pooled_post_prompt']
        analysis[arm] = {'matched_patients': len(differences),
            'mean_patient_dice_difference': math.fsum(differences) / len(differences) if differences else None,
            'patients_improved': sum(x > 0 for x in differences),
            'patients_worsened': sum(x < 0 for x in differences),
            'patients_unchanged': sum(x == 0 for x in differences),
            'baseline_empty_false_alarms': baseline['false_alarms'],
            'candidate_empty_false_alarms': candidate['false_alarms'],
            'labeled_empty_frames': candidate['empty_frames']}
    return analysis


def render_report(result):
    def f(value):
        return 'N/A' if value is None else f'{value:.4f}'
    lines = ['# Exploratory warm out-of-fold video results', '',
        'Previously inspected patients; scores are not an independent external test. '
        'A is the matched project Lean token-window runtime with a 16,384 spatial-token budget, '
        '3,072-token insurance reserve and 0.5 multiscale anchor; it is not the unmodified author Lean baseline. '
        'B changes output only; C writes refined masks to memory; D does both.', '',
        '## Post-prompt present-frame Dice and empty-frame alarms', '',
        '| Sequence | A Dice | B Dice | C Dice | D Dice | A/B/C/D empty alarms | Pre-prompt present misses |',
        '|---|---:|---:|---:|---:|---|---:|']
    for number in range(1, 24):
        seq = f'seq{number}'
        if seq in result['no_initialization']:
            lines.append(f'| {seq} | N/A | N/A | N/A | N/A | N/A | N/A |')
            continue
        if seq not in result['summaries']:
            continue
        entries = [result['summaries'][seq][arm]['post_prompt'] for arm in ARMS]
        dice = ' | '.join(f(x['mean_present_dice']) for x in entries)
        alarms = ' / '.join(f"{x['false_alarms']}/{x['empty_frames']}" for x in entries)
        lines.append(f"| {seq} | {dice} | {alarms} | {result['preprompt'][seq]['present_misses']} |")
    lines += ['', '## Patient and frame denominators', '',
        '| Route | Patient-macro Dice | Patients with present frames | Pooled Dice | Present frames | Empty alarms |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        item = result['overall'][arm]; pooled = item['pooled_post_prompt']
        lines.append(f"| {arm} | {f(item['patient_macro_present_dice'])} | {item['patients_with_present_frames']} | "
                     f"{f(pooled['mean_present_dice'])} | {pooled['present_frames']} | "
                     f"{pooled['false_alarms']}/{pooled['empty_frames']} |")
    lines += ['', '## Descriptive paired differences against A', '',
        '| Route | Mean patient Dice difference | Matched patients | Improved / worsened / unchanged | Empty alarms A → route |',
        '|---|---:|---:|---|---|']
    for arm in ('B', 'C', 'D'):
        item = result['descriptive_analysis'][arm]
        lines.append(f"| {arm} | {f(item['mean_patient_dice_difference'])} | {item['matched_patients']} | "
            f"{item['patients_improved']} / {item['patients_worsened']} / {item['patients_unchanged']} | "
            f"{item['baseline_empty_false_alarms']} → {item['candidate_empty_false_alarms']} "
            f"of {item['labeled_empty_frames']} |")
    lines += ['', 'These paired differences describe this exploratory set; they do not establish statistical significance or causation.', '',
        'Seq1 and seq7 have no automatic initialization and are reported as missing, not assigned a Dice score. '
        'Seq23 primary scores use the full canvas; its separately recorded viewport scores exclude the discarded inset. '
        'Seq6 uses a legacy reference without a recorded label-open guard; a later phase-bound warm capture '
        'verifies conditioning and named trajectory, not unrecorded compact state. '
        'Per-sequence boundary F, precision/recall, zero-overlap counts, false-positive pixel area, paired differences, '
        'initialization-frame metrics and timing are in results.json. Wall time includes staging, prompt processing, '
        'synchronization and file writes; it is not inference-only FPS. Direct-mask-init comparator results are contextual.', '']
    return '\n'.join(lines)


def main():
    from scripts.score_warm_crossfit_v1 import compare
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('verify','score','audit'), required=True)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(strict=True); out = a.output.resolve()
    require(out.is_relative_to(root) and not out.exists(), 'Invalid score output')
    guard = install_mask_open_guard(DATA) if a.mode == 'verify' else install_phase_mask_guard(DATA)
    completed = () if a.mode == 'verify' else ('verify',) if a.mode == 'score' else ('verify','score')
    proof = validate_frozen(root, completed=completed); require(guard['open_attempts'] == 0, 'Early target open')
    if a.mode == 'verify':
        proof.update(passed=True, label_guard=dict(guard), source_sha256=digest(Path(__file__)))
        out.mkdir(parents=True); (out/'verification.json').write_text(json.dumps(proof, indent=2)+'\n'); return
    vpath = root/'verify/verification.json'; vsha = phase_artifact(root, 'verify', vpath); verified = read_bound(vpath,vsha)
    require(verified.get('passed') is True and verified.get('source_sha256') == digest(Path(__file__)) and
            verified.get('label_guard', {}).get('open_attempts') == 0 and
            all(verified.get(k) == v for k,v in proof.items()), 'Frozen verification differs')
    primary = None; score_sha = None
    if a.mode == 'audit':
        path = root/'score/results.json'; score_sha = phase_artifact(root,'score',path)
        primary = read_bound(path,score_sha)
        require(primary.get('passed') is True and primary.get('verification_sha256') == vsha and
                primary.get('report_sha256') == digest(root/'score/report.md'),
                'Frozen primary score or verification binding differs')
    guard['active'] = False; result = compute(root, proof, independent=a.mode == 'audit'); out.mkdir(parents=True)
    if a.mode == 'score':
        report = out/'report.md'; report.write_text(render_report(result))
        (out/'results.json').write_text(json.dumps({'passed':True,'verification_sha256':vsha,
            'report_sha256': digest(report), 'result':result},indent=2)+'\n')
    else:
        errors=[]; compare(result,primary['result'],[],errors)
        require(not errors, f'Independent metric recomputation differs: {errors[:3]}')
        (out/'audit.json').write_text(json.dumps({'passed':True,'score_sha256':score_sha,
            'report_sha256': primary['report_sha256'],
            'scope':'Independent metric recomputation on same data; not an independent test cohort.'},indent=2)+'\n')


if __name__ == '__main__': main()
