#!/usr/bin/env python3
"""Combine three audited video studies without opening images or labels.

This is a retrospective post-hoc report, not a locked evaluation. Patient
distinctness is user-reported; seq6 and seq10 have prior development exposure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output
from scripts.report_combined_boundary_factorial_v1 import (
    ARMS, ROOT as EXPERIMENT_ROOT, checked_study, check_cross_study_identity,
    digest, reaggregate,
)

STUDIES = ('boundary_video_factorial_v1_7c3zaram',
           'extended_boundary_factorial_v1_12ahgcx7',
           'remaining_boundary_factorial_v1_a4_crb_n')
EXPECTED = {'seq6', 'seq8', 'seq10', 'seq11', 'seq16', 'seq17', 'seq18',
            'seq19', 'seq20', 'seq21', 'seq22', 'seq23'}
GROUPS = {'priority_three': {'seq11', 'seq20', 'seq23'},
          'additional_seven': {'seq8', 'seq16', 'seq17', 'seq18', 'seq19', 'seq21', 'seq22'},
          'prior_development_two': {'seq6', 'seq10'}}


def paired_analysis(per_video, *, replicates=20000, seed=20261115):
    """Post-hoc equal-patient percentile bootstrap, never frame resampling."""
    if set(per_video) != EXPECTED or replicates < 100:
        raise ValueError('Expected all twelve patients and sufficient replicates')
    seqs = sorted(per_video, key=lambda s: int(s[3:]))
    output = {}
    for arm in ('B', 'D'):
        diffs = [per_video[s][arm]['mean_present_dice'] - per_video[s]['A']['mean_present_dice'] for s in seqs]
        rng = random.Random(seed)
        means = sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(replicates))
        def quantile(q):
            x = (len(means)-1)*q
            lo, hi = math.floor(x), math.ceil(x)
            return means[lo] + (means[hi]-means[lo])*(x-lo)
        subgroup = {}
        for name, members in GROUPS.items():
            vals = [per_video[s][arm]['mean_present_dice']-per_video[s]['A']['mean_present_dice']
                    for s in seqs if s in members]
            subgroup[name] = {'sequences': sorted(members,key=lambda s:int(s[3:])),
                              'mean_delta': sum(vals)/len(vals)}
        output[arm] = {'mean_delta':sum(diffs)/len(diffs), 'percentile_95_ci':[quantile(.025),quantile(.975)],
                       'improved_patients':sum(x>0 for x in diffs), 'worsened_patients':sum(x<0 for x in diffs),
                       'replicates':replicates, 'seed':seed, 'subgroups':subgroup}
    return output


def render(per_video, aggregates, paired, inputs, identity):
    lines = ['# Twelve-patient exploratory boundary-refiner comparison', '',
        'Reaggregated from three independently audited score records; no source images or labels were opened for this report.', '',
        'The user states each sequence represents a different patient; an independent patient-ID table was not available. These videos or their baseline outcomes were previously inspected, and seq6 and seq10 were used in earlier model development and validation. The checkpoint was selected after development; detector/backbone exposure and near-duplicate or source relationships remain unresolved. This is not blinded or independent final evaluation.', '',
        'A: original output and memory; B: refined output only; C: refined memory input only; D: refined output and memory input. Initial conditioning is protected. C/D may change later native pointers and scores through feedback. The saved YOLO initialization is replayed; historical detection scanned until its first detection. No new redetection or inference gate was introduced. Seq23 uses the retrospective inset-excluded crop. Labeled-empty does not prove clinical absence.', '',
        '| Sequence | Present | Dice A | Dice B | Dice C | Dice D | Empty alarms A / B / C / D | Zero overlap A / B / C / D | Mean empty pixels A / B / C / D |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seq in sorted(per_video,key=lambda s:int(s[3:])):
        x=per_video[seq]; p=x['A']['present_frames']; e=x['A']['empty_frames']
        dice=' | '.join(f"{x[a]['mean_present_dice']:.4f}" for a in ARMS)
        alarms=' / '.join(f"{x[a]['false_alarms']}/{e}" for a in ARMS)
        zeros=' / '.join(f"{x[a]['zero_overlap_present_frames']}/{p}" for a in ARMS)
        area=' / '.join(f"{x[a]['mean_empty_predicted_pixels']:.0f}" if e else 'NA' for a in ARMS)
        lines.append(f'| {seq} | {p} | {dice} | {alarms} | {zeros} | {area} |')
    lines += ['', '| Arm | Equal-patient Dice | Frame-weighted Dice | Zero overlap | Empty alarms | Mean empty pixels | Mean patient empty-alarm rate |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, x in aggregates.items():
        lines.append(f"| {arm} | {x['equal_patient_mean_dice']:.4f} | {x['pooled_present_frame_mean_dice']:.4f} | {x['zero_overlap_present_frames']}/{x['present_frames']} | {x['false_alarms']}/{x['empty_frames']} | {x['mean_empty_predicted_pixels']:.1f} | {x['mean_per_patient_empty_alarm_rate']:.4f} |")
    lines += ['', '| Post-hoc comparison | Mean patient Dice change | 95% patient-bootstrap interval | Improved / worsened patients |',
              '|---|---:|---:|---:|']
    for arm in ('B','D'):
        v=paired[arm]; lo,hi=v['percentile_95_ci']
        lines.append(f"| {arm} − A | {v['mean_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {v['improved_patients']} / {v['worsened_patients']} |")
    lines += ['', 'The intervals are post-hoc 20,000-replicate percentile bootstraps resampling user-reported patients, with seed 20261115. They assume independent patients; shared source, detector and near-duplicate uncertainties remain. These selected, inspected videos, one checkpoint and small sample do not support a population, clinical, noninferiority or safety claim.', '',
        'Seq6 and seq10 are explicitly included as exploratory prior-development patients, not a new independent evaluation. Per-video, pooled-frame and equal-patient metrics serve different purposes; inspect zero-overlap and empty predicted area as well as Dice.', '',
        f"All 48 completion receipts share the same recorded backbone, detector, refiner, control, original gate and runtime identity. Candidate checkpoint SHA-256: `{identity['candidate_checkpoint_sha256']}`.", '']
    for x in inputs:
        lines.append(f"- Study score: `{x['score_path']}` SHA-256 `{x['score_sha256']}`; independent audit: `{x['audit_path']}` SHA-256 `{x['audit_sha256']}`.")
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=EXPERIMENT_ROOT)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    studies=[checked_study(a.root,s) for s in STUDIES]
    per_video,aggregates=reaggregate(studies,expected=EXPECTED)
    identity,completions=check_cross_study_identity(a.root,STUDIES,studies)
    paired=paired_analysis(per_video); inputs=[x[2] for x in studies]
    out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
    report=out/'report.md'; report.write_text(render(per_video,aggregates,paired,inputs,identity))
    receipt={'passed':True,'scope':'Post-hoc aggregation of audited scores; no source images or labels opened.',
             'source_sha256':digest(__file__),
             'aggregation_helper_sha256':digest(ROOT/'scripts/report_combined_boundary_factorial_v1.py'),
             'inputs':inputs,'cross_study_identity':identity,'completions':completions,
             'per_video':per_video,'aggregates':aggregates,'post_hoc_paired_bootstrap':paired,
             'report_sha256':digest(report)}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__': main()
