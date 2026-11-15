#!/usr/bin/env python3
"""Score repeated frozen validation outputs after prediction has completed."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.soft_occupancy_auc import soft_occupancy_auc
from modeling.lean_channel_training_loss import balanced_soft_bce
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('first','second','dataset','dataset-audit','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r1=read(a.first/'prediction_receipt.json');r2=read(a.second/'prediction_receipt.json');d=read(a.dataset/'dataset.json');da=read(a.dataset_audit/'verification.json');pre=read(a.preflight/'manifest.json')
    if not r1['passed'] or r1!=r2 or r1['runner_sha256']!=digest(ROOT/'scripts/predict_lean_channel_validation.py') or r1['dataset_sha256']!=digest(a.dataset/'dataset.json') or not da['passed'] or da['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_dataset.py') or da['dataset_sha256']!=r1['dataset_sha256']:
        raise ValueError('Predictions or dataset audit changed')
    if d['preflight_sha256']!=digest(a.preflight/'manifest.json') or d['groups']['validation']!=['seq9','seq10']:raise ValueError('Validation scope changed')
    for seq in d['groups']['validation']:
        meta=pre['sequences'][seq];folder=Path(meta['source'])/seq;rows=read(folder/'frames.json');first=meta['first_prompt']
        if r1['initial_image_sha256'][seq]!=rows[first]['image_sha256'] or r1['prompt_sha256'][seq]!=meta['prompt_mask_sha256'] or digest(folder/'prompt_refined.png')!=meta['prompt_mask_sha256']:
            raise ValueError('Validation RGB prompt source changed')
    samples={(x['sequence'],x['frame']):x for x in d['samples'] if x['group']=='validation'}
    if len(samples)!=10 or {(x['sequence'],x['frame']) for x in r1['events']}!=set(samples):raise ValueError('Validation coverage differs')
    methods={'prompt_conditioned','descriptor_independent','mismatched_prompt','unweighted','rgb'};frames=[]
    for event in r1['events']:
        seq=event['sequence'];idx=event['frame'];item=samples[(seq,idx)]
        if set(event['scores'])!=methods or digest(Path(item['target_path']))!=item['target_sha256'] or digest(Path(item['mask_path']))!=item['label_sha256']:
            raise ValueError('Labels or score methods changed')
        target=torch.load(item['target_path'],map_location='cpu',weights_only=True);present=bool(target.sum()>0);metrics={}
        for arm,info in event['scores'].items():
            f1=a.first/f'{seq}_{idx:06d}_{arm}.pt';f2=a.second/f'{seq}_{idx:06d}_{arm}.pt'
            if digest(f1)!=info['file_sha256'] or digest(f2)!=info['file_sha256']:raise ValueError('Repeated scores changed')
            data=torch.load(f1,map_location='cpu',weights_only=True);score=data['score']
            if score.shape!=(4096,) or sha(score)!=info['score_sha256'] or not torch.isfinite(score).all():raise ValueError('Invalid score tensor')
            auc=soft_occupancy_auc(score.numpy(),target.numpy())
            metrics[arm]={'weighted_auc':auc,'balanced_bce':float(balanced_soft_bce(score,target)), 'mean_score':float(score.mean())}
        frames.append({'sequence':seq,'frame':idx,'present':present,'soft_foreground_mass':float(target.sum()),'metrics':metrics})
    summary={}
    for seq in d['groups']['validation']:
        rows=[x for x in frames if x['sequence']==seq];summary[seq]={}
        for arm in sorted(methods):
            aucs=[x['metrics'][arm]['weighted_auc'] for x in rows if x['metrics'][arm]['weighted_auc'] is not None]
            summary[seq][arm]={'mean_weighted_auc':float(np.mean(aucs)) if aucs else None,'auc_frames':len(aucs),
              'empty_mean_score':float(np.mean([x['metrics'][arm]['mean_score'] for x in rows if not x['present']])) if any(not x['present'] for x in rows) else None}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'scope':'Small sequence-separated feature-score validation. Soft occupancy weighted AUC is a discrimination measure, not video Dice or false-alarm rate. Patient identity unknown.',
      'frames':frames,'summary':summary,'dataset_sha256':digest(a.dataset/'dataset.json'),'predictions_sha256':digest(a.first/'prediction_receipt.json'),
      'metric_sha256':digest(ROOT/'evaluation/soft_occupancy_auc.py'),'audit_sha256':digest(Path(__file__))}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Channel-attention feasibility: held-out feature-score validation','','Two validation videos, five preselected frames each. AUC is soft-occupancy-weighted pairwise ranking; it is not mask Dice. No validation labels were opened when generating scores.','',
      '| Sequence | Arm | Mean weighted AUC | Scorable frames | Mean score on empty frames |','| --- | --- | ---: | ---: | ---: |']
    for seq,arms in summary.items():
        for arm,v in arms.items():
            auc='—' if v['mean_weighted_auc'] is None else f"{v['mean_weighted_auc']:.6f}";empty='—' if v['empty_mean_score'] is None else f"{v['empty_mean_score']:.6f}"
            lines.append(f"| {seq} | {arm} | {auc} | {v['auc_frames']} | {empty} |")
    lines+=['','No thresholded segmentation was generated, so false-alarm rates and full-video Dice cannot be inferred. The data are previously inspected and patient independence is unverified. Training is a small fixed-schedule feasibility pilot, not evidence of generalization.','']
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print(str(out/'report.md'))
if __name__=='__main__':main()
