#!/usr/bin/env python3
"""Report the source-bound positive-teacher development study."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scores','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();score_path=a.scores/'scores.json'
    done=json.loads((a.scores/'completion.json').read_text());data=json.loads(score_path.read_text())
    if not done.get('passed') or done.get('scores_sha256')!=digest(score_path) or not data.get('passed') or data.get('plan_sha256')!=digest(a.plan):
        raise ValueError('Scoring or plan receipt changed')
    names={'baseline':'Captured baseline','region_robust_bce':'Region / robust BCE','region_robust_teacher':'Region / robust BCE + preservation',
           'global_robust_bce':'Global / robust BCE','global_robust_teacher':'Global / robust BCE + preservation',
           'mask_only_robust_bce':'Mask-only / robust BCE','mask_only_robust_teacher':'Mask-only / robust BCE + preservation'}
    if set(names)!=set(data['summary']):raise ValueError('Unexpected study arms')
    lines=['# Positive-teacher preservation development study','',
           'Single seed, 39 training reads and 10 previously inspected validation reads. All six predictions were frozen and independently replayed before validation labels were opened. CPU- and MPS-captured features have a composite provenance identity; patient independence is unknown.','',
           '| Method | Present frame Dice (7) | Equal-video Dice (2) | Empty false alarms (3) | Mean empty predicted pixels |',
           '|---|---:|---:|---:|---:|']
    for key,label in names.items():
        x=data['summary'][key]
        if x['present_frames']!=7 or x['empty_frames']!=3:raise ValueError('Unexpected denominators')
        lines.append(f"| {label} | {x['mean_present_frame_dice']:.4f} | {x['mean_per_video_present_dice']:.4f} | {x['empty_false_alarms']}/3 | {x['mean_empty_predicted_pixels']:,.0f} |")
    s=data['summary']
    if not (s['region_robust_bce']['mean_present_frame_dice'] < s['region_robust_teacher']['mean_present_frame_dice'] < s['baseline']['mean_present_frame_dice']
            and s['region_robust_teacher']['mean_present_frame_dice'] < s['mask_only_robust_teacher']['mean_present_frame_dice']
            and s['region_robust_teacher']['mean_empty_predicted_pixels'] > s['region_robust_bce']['mean_empty_predicted_pixels']
            and all(s[k]['empty_false_alarms']==3 for k in names)):
        raise ValueError('Narrative no longer matches the results')
    hard=[x for x in data['rows'] if x['sequence']=='seq4' and x['frame'] in (21,24) and x['split']=='train']
    if len(hard)!=2 or any(any(x['methods'][key]['dice']!=0 for key in names) for x in hard):
        raise ValueError('Hard-case narrative differs')
    lines += ['',
        'The training-only preservation term increased region-conditioned present-frame Dice compared with the same model under robust BCE alone, but it remained below the baseline and the mask-only control. It also increased mean empty predicted area relative to robust BCE alone; all methods retained 3/3 labeled-empty false alarms.',
        '', 'Neither label-selected seq4 training miss (frames 21 and 24) was recovered by any arm. The preserved teacher term is weighted by the training GT-positive mask and the original automatic probability, and divides by GT-positive area. It is absent from inference; it is not a new confidence gate or threshold.',
        '', 'This objective was proposed after inspecting the earlier validation errors. One seed, two previously inspected validation videos and three empty frames are insufficient for independent generalization or a claim of a novel attention mechanism. The full-video candidate has not been run.',
        '',f"Source scoring SHA-256: `{digest(score_path)}`; plan SHA-256: `{digest(a.plan)}`."]
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(report),'scores_sha256':digest(score_path)},indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
