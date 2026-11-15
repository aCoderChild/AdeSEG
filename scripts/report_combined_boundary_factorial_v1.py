#!/usr/bin/env python3
"""Reaggregate two independently audited video studies without opening masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

ARMS = ('A', 'B', 'C', 'D')
EXPECTED = {'seq8', 'seq11', 'seq16', 'seq17', 'seq18', 'seq19', 'seq20', 'seq21', 'seq22', 'seq23'}
ROOT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1')
STUDIES = ('boundary_video_factorial_v1_7c3zaram', 'extended_boundary_factorial_v1_12ahgcx7')
IDENTITY_KEYS = ('checkpoint_sha256', 'detector_sha256', 'candidate_checkpoint_sha256',
                 'control_checkpoint_sha256', 'engineering_gate_sha256', 'runtime_verification_sha256')
PRIORITY = {'seq11', 'seq20', 'seq23'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_study(root, name):
    score_path = root / name / 'scores/results.json'
    audit_path = root / f'{name}_independent_audit_attempt01/audit.json'
    score, audit = (json.loads(p.read_text()) for p in (score_path, audit_path))
    if not score.get('passed') or not audit.get('passed') or audit.get('error_count') != 0:
        raise ValueError(f'{name}: study or independent audit did not pass')
    if audit.get('score_sha256') != digest(score_path):
        raise ValueError(f'{name}: independent audit references a different score artifact')
    if set(score['summaries']) != set(audit['summaries']) or set(score['rows']) != set(score['summaries']):
        raise ValueError(f'{name}: sequence sets differ')
    return score, audit, {'score_path': str(score_path), 'score_sha256': digest(score_path),
                          'audit_path': str(audit_path), 'audit_sha256': digest(audit_path)}


def reaggregate(studies, *, expected=EXPECTED):
    per_video = {}
    for score, audit, _ in studies:
        for seq, rows in score['rows'].items():
            if seq in per_video:
                raise ValueError(f'duplicate sequence {seq}')
            if len({r['frame'] for r in rows}) != len(rows):
                raise ValueError(f'{seq}: duplicate frame')
            post = [r for r in rows if r['post_prompt']]
            out = {}
            for arm in ARMS:
                present = [r[arm] for r in post if r[arm]['present']]
                empty = [r[arm] for r in post if not r[arm]['present']]
                if not present:
                    raise ValueError(f'{seq}/{arm}: no present frames')
                actual = {'frames': len(post), 'present_frames': len(present),
                          'mean_present_dice': sum(r['dice'] for r in present) / len(present),
                          'zero_overlap_present_frames': sum(r['dice'] == 0 for r in present),
                          'empty_frames': len(empty),
                          'false_alarms': sum(bool(r['false_alarm']) for r in empty),
                          'mean_empty_predicted_pixels': (sum(r['predicted_pixels'] for r in empty) / len(empty) if empty else None),
                          'mean_present_recall': sum(r['recall'] for r in present) / len(present),
                          'mean_present_boundary_f_5px': sum(r['boundary_f_5px'] for r in present) / len(present)}
                for r in post:
                    if r[arm]['present'] != r['A']['present']:
                        raise ValueError(f'{seq}: presence differs among arms')
                for src in (score['summaries'][seq][arm]['post_prompt'], audit['summaries'][seq][arm]['post_prompt']):
                    for key, value in actual.items():
                        ref = src[key]
                        if value is None or ref is None:
                            if value is not None or ref is not None:
                                raise ValueError(f'{seq}/{arm}: {key} null mismatch')
                        elif not math.isclose(value, ref, rel_tol=1e-10, abs_tol=1e-10):
                            raise ValueError(f'{seq}/{arm}: {key} differs from audited result')
                out[arm] = actual
            per_video[seq] = out
    if set(per_video) != set(expected):
        raise ValueError('Unexpected sequence set')
    aggregates = {}
    for arm in ARMS:
        data = [entry[arm] for entry in per_video.values()]
        n = sum(x['present_frames'] for x in data)
        e = sum(x['empty_frames'] for x in data)
        aggregates[arm] = {'patients': len(data), 'present_frames': n,
                           'equal_patient_mean_dice': sum(x['mean_present_dice'] for x in data)/len(data),
                           'pooled_present_frame_mean_dice': sum(x['mean_present_dice'] * x['present_frames'] for x in data)/n,
                           'zero_overlap_present_frames': sum(x['zero_overlap_present_frames'] for x in data),
                           'empty_frames': e, 'false_alarms': sum(x['false_alarms'] for x in data),
                           'mean_empty_predicted_pixels': sum((x['mean_empty_predicted_pixels'] or 0)*x['empty_frames'] for x in data)/e if e else None,
                           'mean_per_patient_empty_alarm_rate': sum(x['false_alarms']/x['empty_frames'] for x in data if x['empty_frames']) / sum(x['empty_frames'] > 0 for x in data)}
    return per_video, aggregates


def check_cross_study_identity(root, names, studies):
    shared, completions = None, []
    for name, (score, _, _) in zip(names, studies, strict=True):
        for seq in score['rows']:
            for arm in ARMS:
                path = root/name/seq/arm/'completion.json'
                c = json.loads(path.read_text())
                if not c.get('passed') or c.get('sequence') != seq or c.get('arm') != arm or not c.get('full_video') or c.get('limit') != 0:
                    raise ValueError(f'{seq}/{arm}: invalid full-video completion')
                identity = {k: c.get(k) for k in IDENTITY_KEYS}
                if any(not isinstance(v, str) or len(v) != 64 for v in identity.values()):
                    raise ValueError(f'{seq}/{arm}: missing identity')
                if shared is not None and identity != shared:
                    raise ValueError(f'{seq}/{arm}: cross-study model identity differs')
                shared = identity
                completions.append({'path': str(path), 'sha256': digest(path)})
    return shared, completions


def paired_analysis(per_video, *, replicates=20000, seed=20261115):
    """Post-hoc equal-patient bootstrap, independently seeded per comparison."""
    if set(per_video) != EXPECTED or replicates < 100:
        raise ValueError('Expected all ten patients and sufficient replicates')
    seqs = sorted(per_video, key=lambda s: int(s[3:]))
    output = {}
    for arm in ('B', 'D'):
        diffs = [per_video[s][arm]['mean_present_dice'] - per_video[s]['A']['mean_present_dice'] for s in seqs]
        rng = random.Random(seed)
        means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(replicates))
        def percentile(q):
            index = (len(means) - 1) * q
            lo, hi = math.floor(index), math.ceil(index)
            return means[lo] + (means[hi] - means[lo]) * (index - lo)
        subgroup = {}
        for name, members in (('priority_three', PRIORITY), ('additional_seven', EXPECTED - PRIORITY)):
            vals = [per_video[s][arm]['mean_present_dice'] - per_video[s]['A']['mean_present_dice'] for s in seqs if s in members]
            subgroup[name] = {'sequences': sorted(members, key=lambda s: int(s[3:])), 'mean_delta': sum(vals) / len(vals)}
        output[arm] = {'mean_delta': sum(diffs) / len(diffs), 'percentile_95_ci': [percentile(.025), percentile(.975)],
                       'improved_patients': sum(x > 0 for x in diffs), 'worsened_patients': sum(x < 0 for x in diffs),
                       'replicates': replicates, 'seed': seed, 'subgroups': subgroup}
    return output


def render(per_video, aggregates, paired, inputs, identity):
    lines = ['# Combined exploratory boundary-refiner video comparison', '',
        'Reaggregated from the two independently audited score records. No source images or label masks were opened for this report.', '',
        'The user states each sequence is a different patient; patient identifiers were not independently verified. All ten videos or their baseline outcomes were previously inspected, the checkpoint was selected after development, and detector/backbone exposure and near-duplicate/source relationships remain unresolved. This is not a blinded or independent final evaluation.', '',
        'A: original output and memory; B: refined output only; C: refined memory input only; D: refined output and memory input. The initial conditioning is protected; the refiner does not directly overwrite the native object score, pointer or compact mask, but their later values can change when C/D change memory. Historical YOLO scanned until first detection; these runs replay the saved automatic prompt and do not redetect. Sequence 23 uses a retrospectively chosen inset-excluded crop. Labeled-empty does not prove clinical absence.', '',
        '| Sequence | Present | Dice A | Dice B | Dice C | Dice D | Empty alarms A / B / C / D | Zero-overlap A / B / C / D | Mean empty pixels A / B / C / D |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seq in sorted(per_video, key=lambda s: int(s[3:])):
        x = per_video[seq]; p = x['A']['present_frames']; e = x['A']['empty_frames']
        dice = ' | '.join(f"{x[a]['mean_present_dice']:.4f}" for a in ARMS)
        alarms = ' / '.join(f"{x[a]['false_alarms']}/{e}" for a in ARMS)
        zero = ' / '.join(f"{x[a]['zero_overlap_present_frames']}/{p}" for a in ARMS)
        area = ' / '.join(f"{x[a]['mean_empty_predicted_pixels']:.0f}" if e else 'NA' for a in ARMS)
        lines.append(f'| {seq} | {p} | {dice} | {alarms} | {zero} | {area} |')
    lines += ['', '| Arm | Equal-patient Dice | Frame-weighted Dice | Zero-overlap | Empty alarms | Mean empty predicted pixels | Mean patient empty alarm rate |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, x in aggregates.items():
        lines.append(f"| {arm} | {x['equal_patient_mean_dice']:.4f} | {x['pooled_present_frame_mean_dice']:.4f} | {x['zero_overlap_present_frames']}/{x['present_frames']} | {x['false_alarms']}/{x['empty_frames']} | {x['mean_empty_predicted_pixels']:.1f} | {x['mean_per_patient_empty_alarm_rate']:.4f} |")
    b_up, d_up = paired['B']['improved_patients'], paired['D']['improved_patients']
    lines += ['', f'Compared with A, B improved mean present Dice in {b_up}/{len(per_video)} patients and D improved it in {d_up}/{len(per_video)}. D worsened seq8 and seq17 notably and increased zero-overlap and the mean per-patient empty alarm rate overall, despite its higher mean Dice. These are descriptive observations, not a clinical safety result.', '',
        '| Post-hoc comparison | Mean patient Dice change | 95% patient-bootstrap interval | Earlier three patients | Additional seven patients |',
        '|---|---:|---:|---:|---:|']
    for arm in ('B', 'D'):
        x = paired[arm]; low, high = x['percentile_95_ci']; groups = x['subgroups']
        lines.append(f"| {arm} − A | {x['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | {groups['priority_three']['mean_delta']:+.4f} | {groups['additional_seven']['mean_delta']:+.4f} |")
    lines += ['', 'Intervals are an exploratory 20,000-replicate percentile bootstrap resampling the 10 user-reported patients, with fixed seed 20261115. This assumes patients are independent; identity was user-reported, and shared sources, near-duplicates and other dependencies are unresolved. The checkpoint and videos were chosen after prior inspection, subgroups are retrospective, and n=10 is small. These intervals describe this selected set under the resampling assumption, not a prespecified population inference, clinical effect or significance test.', '',
        'These summaries describe the selected videos only. Frame-weighted averages give longer videos greater influence; equal-patient averages weight each user-reported patient once. No significance, noninferiority, safety or generalization claim follows from these reused development videos, one checkpoint and unverified exposure. Boundary F uses five native pixels, not physical distance.', '',
        'B and A share the same native memory trajectory; C and D may change future model inputs through the memory encoder. Refinement can cause both overextension and missed foreground: inspect the patient rows, false-positive area and zero-overlap counts, not mean Dice alone.', '',
        'Each input score artifact and its passing independent audit were SHA-256 bound and their frame-derived aggregate metrics checked against both recorded summaries before this report was produced. All 40 completion receipts were separately checked to share the same backbone, detector, candidate, control, original engineering gate and runtime-verification identities.', '',
        f"Shared candidate checkpoint SHA-256: `{identity['candidate_checkpoint_sha256']}`.", '']
    for x in inputs:
        lines.append(f"- Study score: `{x['score_path']}` SHA-256 `{x['score_sha256']}`; independent audit: `{x['audit_path']}` SHA-256 `{x['audit_sha256']}`.")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    studies = [checked_study(args.root, s) for s in STUDIES]
    per_video, aggregates = reaggregate(studies)
    identity, completions = check_cross_study_identity(args.root, STUDIES, studies)
    paired = paired_analysis(per_video)
    inputs = [s[2] for s in studies]
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / 'report.md'
    report.write_text(render(per_video, aggregates, paired, inputs, identity))
    receipt = {'passed': True, 'scope': 'Descriptive aggregation of independently audited frozen scores; no labels opened.',
               'source_path': str(Path(__file__).resolve()), 'source_sha256': digest(__file__),
               'inputs': inputs, 'cross_study_identity': identity, 'completions': completions,
               'per_video': per_video, 'aggregates': aggregates, 'post_hoc_paired_bootstrap': paired,
               'report_sha256': digest(report)}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
