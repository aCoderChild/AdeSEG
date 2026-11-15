#!/usr/bin/env python3
"""Independently recompute validation score metrics from saved tensors."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def independent_auc(scores,labels):
    s=scores.double();y=labels.double();p=y.sum();n=(1-y).sum()
    if p==0 or n==0:return None
    order=torch.argsort(s,stable=True);s=s[order];y=y[order]
    _,inverse=torch.unique_consecutive(s,return_inverse=True)
    groups=int(inverse[-1])+1;pos=torch.zeros(groups,dtype=torch.float64);neg=torch.zeros_like(pos)
    pos.scatter_add_(0,inverse,y);neg.scatter_add_(0,inverse,1-y)
    return float((pos*(torch.cumsum(neg,0)-neg+neg*.5)).sum()/(p*n))
def independent_bce(scores,labels):
    z=scores.numpy().astype(np.float64)*8;y=labels.numpy().astype(np.float64);n=1-y;terms=[]
    if y.sum()>0:terms.append(float((np.logaddexp(0,-z)*y).sum()/y.sum()))
    if n.sum()>0:terms.append(float((np.logaddexp(0,z)*n).sum()/n.sum()))
    return sum(terms)/len(terms)

def main():
    p=argparse.ArgumentParser()
    for k in ('first','second','dataset','audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();done=read(a.audit/'completion.json');result=read(a.audit/'results.json');r1=read(a.first/'prediction_receipt.json');r2=read(a.second/'prediction_receipt.json');d=read(a.dataset/'dataset.json')
    if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or result['audit_sha256']!=digest(ROOT/'scripts/audit_lean_channel_validation.py') or result['metric_sha256']!=digest(ROOT/'evaluation/soft_occupancy_auc.py') or not r1['passed'] or r1!=r2 or result['predictions_sha256']!=digest(a.first/'prediction_receipt.json') or result['dataset_sha256']!=digest(a.dataset/'dataset.json'):
        raise ValueError('Validation receipts changed')
    samples={(x['sequence'],x['frame']):x for x in d['samples'] if x['group']=='validation'};events={(x['sequence'],x['frame']):x for x in r1['events']}
    if len(samples)!=10 or set(samples)!=set(events):raise ValueError('Unexpected held-out sample coverage')
    checks=0
    for row in result['frames']:
        key=(row['sequence'],row['frame']);sample=samples[key];target_path=Path(sample['target_path'])
        if digest(target_path)!=sample['target_sha256'] or digest(Path(sample['mask_path']))!=sample['label_sha256']:raise ValueError('Target changed')
        target=torch.load(target_path,map_location='cpu',weights_only=True)
        for arm,metric in row['metrics'].items():
            info=events[key]['scores'][arm];name=f'{key[0]}_{key[1]:06d}_{arm}.pt';p1=a.first/name;p2=a.second/name
            if digest(p1)!=info['file_sha256'] or digest(p2)!=info['file_sha256']:raise ValueError('Prediction changed')
            score=torch.load(p1,map_location='cpu',weights_only=True)['score'];auc=independent_auc(score,target);bce=independent_bce(score,target)
            if (auc is None)!=(metric['weighted_auc'] is None) or (auc is not None and abs(auc-metric['weighted_auc'])>1e-11) or abs(bce-metric['balanced_bce'])>1e-5 or abs(float(score.mean())-metric['mean_score'])>1e-7:
                raise ValueError('Independent metric arithmetic differs')
            checks+=1
    for seq in d['groups']['validation']:
        rows=[x for x in result['frames'] if x['sequence']==seq]
        for arm,summary in result['summary'][seq].items():
            vals=[x['metrics'][arm]['weighted_auc'] for x in rows if x['metrics'][arm]['weighted_auc'] is not None]
            if len(vals)!=summary['auc_frames'] or (vals and abs(sum(vals)/len(vals)-summary['mean_weighted_auc'])>1e-12):raise ValueError('Sequence summary differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    receipt={'passed':True,'score_checks':checks,'validation_frames':len(samples),'results_sha256':digest(a.audit/'results.json'),
      'predictions_sha256':digest(a.first/'prediction_receipt.json'),'verifier_sha256':digest(Path(__file__)),
      'scope':'Independent arithmetic and artifact validation; not evidence of patient-level generalization or video Dice.'}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
