#!/usr/bin/env python3
"""Independently reaggregate audited factorial score JSONs; no labels or model inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import validate_output

ARMS=('A','B','C','D')
EXPECTED={'seq6','seq8','seq10','seq11','seq16','seq17','seq18','seq19','seq20','seq21','seq22','seq23'}
IDS=('checkpoint_sha256','detector_sha256','candidate_checkpoint_sha256','control_checkpoint_sha256',
     'engineering_gate_sha256','runtime_verification_sha256')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def close(a,b):
    return a is None and b is None or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10))


def bootstrap(values, replicates=20000, seed=20261115):
    require(len(values)==12 and replicates>=100,'bootstrap sample mismatch')
    rng=random.Random(seed);out=[]
    for _ in range(replicates):out.append(sum(values[rng.randrange(len(values))] for _ in values)/len(values))
    out.sort();interval=[]
    for p in (.025,.975):
        x=(len(out)-1)*p;lo=math.floor(x);hi=math.ceil(x)
        interval.append(out[lo]+(out[hi]-out[lo])*(x-lo))
    return sum(values)/len(values),interval,sum(x>0 for x in values),sum(x<0 for x in values)


def audit(folder):
    folder=Path(folder).resolve(strict=True); receipt=read(folder/'receipt.json')
    require(receipt.get('passed') is True and receipt.get('report_sha256')==sha(folder/'report.md'),'report/receipt changed')
    require(receipt.get('source_sha256')==sha(ROOT/'scripts/report_combined_12_boundary_factorial_v1.py') and
            receipt.get('aggregation_helper_sha256')==sha(ROOT/'scripts/report_combined_boundary_factorial_v1.py'), 'report source changed')
    inputs=receipt['inputs'];require(len(inputs)==3,'input studies missing')
    per={}; completions=[]
    for item in inputs:
        score_path=Path(item['score_path']);audit_path=Path(item['audit_path'])
        require(sha(score_path)==item['score_sha256'] and sha(audit_path)==item['audit_sha256'],'input score/audit changed')
        score,aud=read(score_path),read(audit_path)
        require(score.get('passed') is True and aud.get('passed') is True and aud.get('error_count')==0 and
                aud.get('score_sha256')==sha(score_path),'underlying independent audit did not pass')
        require(set(score['rows'])==set(score['summaries'])==set(aud['summaries']),'input sequence mismatch')
        study=score_path.parent.parent
        for seq,rows in score['rows'].items():
            require(seq not in per and len({x['frame'] for x in rows})==len(rows),'duplicated sequence/frame')
            post=[row for row in rows if row['post_prompt']];per[seq]={}
            for arm in ARMS:
                require(all(r[arm]['present']==r['A']['present'] for r in post),'arm presence mismatch')
                pos=[r[arm] for r in post if r[arm]['present']];empty=[r[arm] for r in post if not r[arm]['present']]
                require(bool(pos),'no present frames')
                value={'frames':len(post),'present_frames':len(pos),'mean_present_dice':sum(r['dice'] for r in pos)/len(pos),
                    'zero_overlap_present_frames':sum(r['dice']==0 for r in pos),'empty_frames':len(empty),
                    'false_alarms':sum(bool(r['false_alarm']) for r in empty),
                    'mean_empty_predicted_pixels':sum(r['predicted_pixels'] for r in empty)/len(empty) if empty else None,
                    'mean_present_recall':sum(r['recall'] for r in pos)/len(pos),
                    'mean_present_boundary_f_5px':sum(r['boundary_f_5px'] for r in pos)/len(pos)}
                for original in (score['summaries'][seq][arm]['post_prompt'],aud['summaries'][seq][arm]['post_prompt'],receipt['per_video'][seq][arm]):
                    require(all(k in original and close(v,original[k]) for k,v in value.items()),f'frame aggregate differs {seq}/{arm}')
                per[seq][arm]=value
                path=study/seq/arm/'completion.json';comp=read(path)
                require(comp.get('passed') is True and comp.get('sequence')==seq and comp.get('arm')==arm and
                        comp.get('full_video') is True and comp.get('limit')==0 and
                        all(comp.get(k)==receipt['cross_study_identity'].get(k) for k in IDS),'completion identity mismatch')
                completions.append({'path':str(path),'sha256':sha(path)})
    require(set(per)==EXPECTED and len(completions)==48,'cohort/completion count mismatch')
    require(sorted(completions,key=lambda x:x['path'])==sorted(receipt['completions'],key=lambda x:x['path']),'completion hashes differ')
    total={}
    for arm in ARMS:
        values=[x[arm] for x in per.values()]; n=sum(x['present_frames'] for x in values); e=sum(x['empty_frames'] for x in values)
        total[arm]={'patients':len(values),'present_frames':n,'equal_patient_mean_dice':sum(x['mean_present_dice'] for x in values)/len(values),
          'pooled_present_frame_mean_dice':sum(x['mean_present_dice']*x['present_frames'] for x in values)/n,
          'zero_overlap_present_frames':sum(x['zero_overlap_present_frames'] for x in values),
          'empty_frames':e,'false_alarms':sum(x['false_alarms'] for x in values),
          'mean_empty_predicted_pixels':sum((x['mean_empty_predicted_pixels'] or 0)*x['empty_frames'] for x in values)/e,
          'mean_per_patient_empty_alarm_rate':sum(x['false_alarms']/x['empty_frames'] for x in values if x['empty_frames'])/sum(x['empty_frames']>0 for x in values)}
        require(all(k in receipt['aggregates'][arm] and close(v,receipt['aggregates'][arm][k]) for k,v in total[arm].items()),'pooled aggregate differs')
    comparison={}
    for arm in ('B','D'):
        seqs=sorted(per,key=lambda s:int(s[3:]));diffs=[per[s][arm]['mean_present_dice']-per[s]['A']['mean_present_dice'] for s in seqs]
        mean,ci,wins,losses=bootstrap(diffs); old=receipt['post_hoc_paired_bootstrap'][arm]
        require(close(mean,old['mean_delta']) and all(close(x,y) for x,y in zip(ci,old['percentile_95_ci'],strict=True)) and
                wins==old['improved_patients'] and losses==old['worsened_patients'] and
                old['seed']==20261115 and old['replicates']==20000, 'patient bootstrap differs')
        comparison[arm]={'mean_delta':mean,'percentile_95_ci':ci,'improved_patients':wins,'worsened_patients':losses}
    return {'passed':True,'scope':'Independent aggregation and source/receipt audit; existing score records only; no label/mask/model reads.',
            'report_sha256':sha(folder/'report.md'),'receipt_sha256':sha(folder/'receipt.json'),
            'audit_source_sha256':sha(Path(__file__)),'patients_checked':len(per),'completion_receipts_checked':len(completions),
            'aggregates':total,'post_hoc_paired_bootstrap':comparison,
            'limitations':['Patient distinctness is user-reported, not independently verified.',
             'Previously inspected and selected data; bootstrap is descriptive, not a prospective population inference.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.report); out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'patients_checked':result['patients_checked'],'completions_checked':result['completion_receipts_checked']}))

if __name__=='__main__':main()
