"""Offline descriptive report from independently audited native stage captures."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.decoder_training_mask import select_decoder_mask
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import prediction_nonempty
from scripts.memory_token_io import digest
from scripts.run_regional_refinement import load, write


def outcome(row):
    if not row['present']:
        return 'false_alarm' if row['absent_fp'] else 'correct_absence'
    if not prediction_nonempty(row):
        return 'missed_or_lost'
    return 'disjoint_prediction' if row['scores']['iou']==0 else 'overlapping_prediction'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_run',type=Path)
    args=parser.parse_args();base=args.native_run
    completed=json.loads((base/'completion.json').read_text())
    audit=json.loads((base/'independent_audit.json').read_text())
    setup=json.loads((base/'setup.json').read_text())
    if not completed.get('passed') or completed['scope']!='cohort' or not audit.get('passed'):
        raise RuntimeError('Completed, independently audited native cohort required')
    if audit['total_targets']!=2012 or audit['stage_cases']!=66 or set(audit['sequences'])!=set(completed['entries']):
        raise RuntimeError('Audit scope differs')
    if any(digest(ROOT/n)!=sha for n,sha in setup['sources'].items()):
        raise RuntimeError('Native source inventory changed')
    cases=json.loads((ROOT/'protocols/encoder_decoder_bottleneck_v1.json').read_text())['cases']
    selection={(r['sequence'],r['frame']):r['category'] for r in cases}
    counts=Counter();deficits=defaultdict(list);summaries={};snapshots=[]
    torch.set_num_threads(2)
    for name,path in completed['entries'].items():
        folder=Path(path);entry=audit['sequences'][name]
        if digest(folder/'completion.json')!=entry['completion_sha256'] or digest(folder/'receipt.json')!=entry['receipt_sha256']:
            raise RuntimeError('Audited evidence changed')
        receipt=json.loads((folder/'completion.json').read_text())
        if digest(folder/'frames.json')!=receipt['frames_sha256']:
            raise RuntimeError('Frame record changed')
        rows=json.loads((folder/'frames.json').read_text());lookup={r['frame']:r for r in rows}
        categories=Counter(outcome(r) for r in rows);counts.update(categories)
        present=[r for r in rows if r['present']]
        if not present:
            raise RuntimeError('Expected at least one positive target per evaluated sequence')
        for category in ('missed_or_lost','disjoint_prediction','overlapping_prediction'):
            deficits[category].append(sum(1-r['scores']['dice'] for r in present if outcome(r)==category)/len(present))
        summaries[name]={'counts':dict(categories),'present_dice':statistics.mean(r['scores']['dice'] for r in present),
            'present_targets':len(present),'empty_targets':len(rows)-len(present)}
        for artifact in receipt['artifacts']:
            values=load(Path(artifact['path']),artifact['sha256'])
            if fingerprint(values)['sha256']!=artifact['stage_fingerprint']:
                raise RuntimeError('Stage tensor layout/content differs')
            frame=artifact['frame'];row=lookup[frame];target=values['target'].numpy()
            native=F.interpolate(values['pred_masks'].float(),size=target.shape,mode='bilinear',align_corners=False)[0,0].numpy()>0
            scores={**calculate_scores(native,target),'boundary_f':float(db_eval_boundary(native,target))}
            if scores!=row['scores']:
                raise RuntimeError('Saved stage mask does not reproduce frame scores')
            raw=select_decoder_mask([(values['raw_masks'],values['predicted_ious'],values['mask_tokens'],values['object_score'])])
            raw=F.interpolate(raw,size=target.shape,mode='bilinear',align_corners=False)[0,0].numpy()>0
            current,conditioned=values['current_image'].float(),values['memory_conditioned'].float()
            snapshots.append({'sequence':name,'frame':frame,'selection_stratum':selection[(name,frame)],
                'official_outcome':outcome(row),'scores':scores,'raw_scores':calculate_scores(raw,target),
                'object_score':float(values['object_score'].item()),
                'current_feature_l2_mean':float(current.norm(dim=-1).mean()),
                'conditioned_feature_l2_mean':float(conditioned.norm(dim=-1).mean()),
                'feature_change_l2_mean':float((conditioned-current).norm(dim=-1).mean()),
                'feature_cosine_mean':float(F.cosine_similarity(current,conditioned,dim=-1).mean())})
    if len(snapshots)!=66 or sum(counts.values())!=2012:
        raise RuntimeError('Report coverage incomplete')
    contributions={k:statistics.mean(v) for k,v in deficits.items()}
    total=sum(contributions.values())
    groups={}
    for category in sorted({r['official_outcome'] for r in snapshots}):
        selected=[r for r in snapshots if r['official_outcome']==category]
        groups[category]={'cases':len(selected),**{key:statistics.median(r[key] for r in selected) for key in
            ('object_score','current_feature_l2_mean','conditioned_feature_l2_mean','feature_change_l2_mean','feature_cosine_mean')}}
    report={'exploratory':True,'inference_performed':False,'checkpoint_sha256':setup['checkpoint_sha256'],
        'counts':dict(counts),'sequence_summaries':summaries,'deficit_contributions':contributions,
        'total_equal_sequence_dice_deficit':total,'stage_cases':snapshots,'stage_group_medians':groups,
        'audit_sha256':digest(base/'independent_audit.json'),'reporter_sha256':digest(Path(__file__)),
        'limits':'Feature norms/cosines are descriptive, not semantic probes or causal effects. Overlapping masks include contour, extent and localisation errors. Fixed cases were selected using the previous medical checkpoint; official outcomes are reclassified, not reselected.'}
    write(base/'stage_analysis.json',report)
    lines=['# Official native stage audit — descriptive analysis','',
        'All 21 promptable sequences; 2,012 targets. No adaptation, inference rerun or claimed Dice gain.','',
        '| Outcome | Frames | Equal-sequence Dice deficit contribution | Share of total deficit |',
        '|---|---:|---:|---:|']
    for category in ('missed_or_lost','disjoint_prediction','overlapping_prediction','false_alarm','correct_absence'):
        value=contributions.get(category)
        amount='N/A' if value is None else f'{value:.6f}'
        share='N/A' if value is None or total==0 else f'{100*value/total:.2f}%'
        lines.append(f'| {category} | {counts[category]} | {amount} | {share} |')
    lines+=['','A deficit contribution is first averaged over each sequence\'s GT-present targets, then over sequences. Empty targets are reported separately, not inserted into present Dice.',
        '', '| Fixed-case outcome | Cases | Median feature cosine | Median feature-change L2 |', '|---|---:|---:|---:|']
    for category,group in groups.items():
        lines.append(f"| {category} | {group['cases']} | {group['feature_cosine_mean']:.6f} | {group['feature_change_l2_mean']:.6f} |")
    lines+=['',report['limits'],
        '', 'Per-case native/raw Dice, original-resolution Boundary F, object score and feature descriptors are in `stage_analysis.json`. A nonzero representation change does not establish that memory caused an error or that the encoder cannot localise the object. Ordinary adaptation is the next efficacy test.']
    (base/'stage_analysis.md').write_text('\n'.join(lines)+'\n')
    print('Audited descriptive report written:',base/'stage_analysis.md')


if __name__=='__main__':
    main()
