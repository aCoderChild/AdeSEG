#!/usr/bin/env python3
"""Report the verified mixed-data fixed-read development study."""
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
    done=json.loads((a.scores/'completion.json').read_text());result=json.loads(score_path.read_text())
    if not done.get('passed') or done.get('scores_sha256')!=digest(score_path) or not result.get('passed') or result['plan_sha256']!=digest(a.plan):
        raise ValueError('Scoring receipt or plan changed')
    order=['baseline','region_bce','region_robust_bce','global_bce','global_robust_bce',
           'shuffled_bce','shuffled_robust_bce','mask_only_bce','mask_only_robust_bce']
    names={'baseline':'Captured baseline','region_bce':'Region / BCE','region_robust_bce':'Region / robust BCE',
           'global_bce':'Global / BCE','global_robust_bce':'Global / robust BCE',
           'shuffled_bce':'Fixed permutation / BCE','shuffled_robust_bce':'Fixed permutation / robust BCE',
           'mask_only_bce':'Mask-only / BCE','mask_only_robust_bce':'Mask-only / robust BCE'}
    if set(result['summary'])!=set(order):raise ValueError('Unexpected study coverage')
    lines=['# Mixed-data fixed-read development study','',
           'One seed; 39 training reads from sequences 2, 3, 4, 6 and 13; 10 previously inspected validation reads from sequences 9 and 10. All eight arms and predictions were frozen and replayed before validation labels were scored. CPU- and MPS-captured inputs have separate provenance; patient independence is unknown.','',
           '| Method | Present frame Dice (7) | Equal-video Dice (2 videos) | Empty false alarms (3) | Mean empty predicted pixels |',
           '|---|---:|---:|---:|---:|']
    for key in order:
        x=result['summary'][key]
        if x['present_frames']!=7 or x['empty_frames']!=3:raise ValueError('Unexpected validation denominators')
        lines.append(f"| {names[key]} | {x['mean_present_frame_dice']:.4f} | {x['mean_per_video_present_dice']:.4f} | {x['empty_false_alarms']}/3 | {x['mean_empty_predicted_pixels']:,.0f} |")
    if any(result['summary'][key]['mean_present_frame_dice'] >= result['summary']['baseline']['mean_present_frame_dice']
           or result['summary'][key]['mean_per_video_present_dice'] >= result['summary']['baseline']['mean_per_video_present_dice']
           or result['summary'][key]['empty_false_alarms'] != 3 for key in order[1:]):
        raise ValueError('Summary narrative no longer matches results')
    hard = [x for x in result['rows'] if x['sequence']=='seq4' and x['frame'] in (21,24) and x['split']=='train']
    if len(hard)!=2 or any(any(x['methods'][key]['dice']!=0 for key in order) for x in hard):
        raise ValueError('Hard-case narrative no longer matches results')
    if (result['summary']['region_robust_bce']['mean_empty_predicted_pixels'] >= result['summary']['baseline']['mean_empty_predicted_pixels']
            or result['summary']['region_robust_bce']['mean_present_frame_dice'] <= result['summary']['region_bce']['mean_present_frame_dice']):
        raise ValueError('Robust-loss narrative no longer matches results')
    lines += ['',
        'Every arm was below the captured baseline on present-frame and equal-video Dice; each retained three of three labeled-empty false alarms. Region/robust BCE reduced mean empty predicted area compared with baseline, but also reduced present Dice. Robust BCE improved the region arm relative to ordinary BCE on this set; this is not a generalization claim.',
        '', 'The fixed channel permutation is invertible and is a sensitivity comparison, not an information-destruction placebo. No additional inference gate, threshold, redetection or pruning rule was introduced.',
        '', 'Two label-selected seq4 training misses (21 and 24) had zero Dice in every arm; these are training diagnostics, not held-out outcomes. Frame 24 is suppressed to -1024 by native SAM2 presence handling.',
        '', 'No trained refiner in this study was run through full video. Validation was already inspected and reused; the small sample, one seed, label completeness and unknown patient/center or checkpoint overlap prevent confirmatory conclusions.',
        '',f"Source scoring SHA-256: `{digest(score_path)}`; plan SHA-256: `{digest(a.plan)}`."]
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    path=out/'report.md';path.write_text('\n'.join(lines)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(path),'scores_sha256':digest(score_path)},indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
