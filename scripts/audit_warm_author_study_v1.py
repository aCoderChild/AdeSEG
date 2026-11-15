#!/usr/bin/env python3
"""Recompute warm author-study metrics on CPU using a separate implementation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_additional_author_comparators_v2 import compare
from scripts.audit_boundary_video_factorial_v1 import label,metrics,summarize,paired
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import new_output
from scripts.run_warm_author_comparator_v1 import DATA,METHODS,SEQUENCES,frozen_preflight,input_data
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_author_lean_automatic_v1 import read_bound,require
from scripts.score_warm_author_study_v1 import validate_frozen,phase_bound_sha


def audit(folder):
    folder=Path(folder).resolve(strict=True);guard=install_phase_mask_guard(DATA)
    verify=folder/'verification/verification.json';verify_sha=phase_bound_sha(folder,'verify');proof=read_bound(verify,verify_sha)
    score_path=folder/'scores/results.json';score_sha=phase_bound_sha(folder,'score');primary=read_bound(score_path,score_sha)
    current=validate_frozen(folder,('verify','score'))
    require(proof.get('passed') is True and proof.get('label_guard',{}).get('open_attempts')==0 and
            primary.get('passed') is True and primary.get('verification_sha256')==verify_sha and
            primary.get('scorer_sha256')==proof.get('scorer_sha256') and
            primary.get('report_sha256')==digest(folder/'scores/report.md') and
            all(proof.get(k)==v for k,v in current.items()) and guard['open_attempts']==0,
            'Warm frozen proof or scoring differs')
    guard['active']=False;plan=frozen_preflight();errors=[];checked=0
    compare(errors,['feature_controls'],proof['feature_controls'],primary['feature_controls'])
    patient={m:[] for m in METHODS};all_dice={m:[] for m in METHODS};empty={m:0 for m in METHODS};alarms={m:0 for m in METHODS}
    for seq in SEQUENCES:
        frames,start,_,_=input_data(seq,plan);bound=proof['records'][seq]
        meta=read_bound(bound['frame_metadata']['path'],bound['frame_metadata']['sha256'])
        receipts={m:read_bound(folder/'full'/seq/m/'receipt.json',bound['receipts'][m]) for m in METHODS}
        require(len(frames)==len(meta)==len(primary['rows'][seq]),'Warm audit coverage differs');rows=[]
        for i,frame in enumerate(frames):
            label_sha=meta[i]['label_sha256'];image_sha=bound['images'][i]
            require(digest(frame.mask_path)==label_sha and digest(frame.image_path)==image_sha,'Warm audit input changed')
            truth=label(frame.mask_path);require(digest(frame.mask_path)==label_sha,'Warm label changed during audit')
            row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,'initialization_frame':i==start}
            for method in METHODS:
                path=folder/'full'/seq/method/f'{i:06d}.png';sha=receipts[method]['records'][i]['mask_sha256']
                require(digest(path)==sha,'Warm prediction changed')
                with Image.open(path) as image:mask=np.asarray(image.convert('L'))>0
                require(digest(path)==sha,'Warm prediction changed during audit');row[method]=metrics(mask,truth)
                if seq=='seq23':
                    x0,y0,x1,y1=VIEWPORT;row[method+'_viewport']=metrics(mask[y0:y1,x0:x1],truth[y0:y1,x0:x1])
            require(digest(frame.image_path)==image_sha,'Warm source changed during audit')
            compare(errors,[seq,i],row,primary['rows'][seq][i]);rows.append(row);checked+=1
        for method in METHODS:
            compare(errors,[seq,method,'post'],summarize(rows,method,True),primary['summaries'][seq][method])
            compare(errors,[seq,method,'all'],summarize(rows,method,False),primary['all_frame_summaries'][seq][method])
            if method!='native':compare(errors,[seq,method,'paired'],paired(rows,method,'native'),primary['paired_descriptive'][seq][method])
            if seq=='seq23':compare(errors,[seq,method,'viewport'],summarize(rows,method+'_viewport',True),
                                    primary['seq23_viewport_summaries'][method])
            compare(errors,[seq,method,'timing'],{'seconds':receipts[method]['prediction_wall_seconds'],
                    'scope':receipts[method]['timing_scope']},primary['timing'][seq][method])
            present=[r[method]['dice'] for r in rows if r['post_prompt'] and r[method]['present']]
            absent=[r[method] for r in rows if r['post_prompt'] and not r[method]['present']]
            if present:patient[method].append(float(np.mean(present)));all_dice[method].extend(present)
            empty[method]+=len(absent);alarms[method]+=sum(x['false_alarm'] for x in absent)
        pre={'frames':start,'labeled_present':sum(r['native']['present'] for r in rows[:start]),
             'present_misses':{m:sum(r[m]['present'] and r[m]['dice']==0 for r in rows[:start]) for m in METHODS}}
        compare(errors,[seq,'preprompt'],pre,primary['preprompt'][seq])
    for method in METHODS:
        overall={'patients_with_present_frames':len(patient[method]),'present_frames':len(all_dice[method]),
                 'patient_macro_present_dice':float(np.mean(patient[method])) if patient[method] else None,
                 'frame_weighted_present_dice':float(np.mean(all_dice[method])) if all_dice[method] else None,
                 'labeled_empty_frames':empty[method],'labeled_empty_false_alarms':alarms[method]}
        compare(errors,[method,'overall'],overall,primary['overall'][method])
    return {'passed':not errors,'scope':'Independent CPU metric calculations; shared structural verifier.',
            'verification_sha256':verify_sha,'score_sha256':score_sha,'auditor_sha256':digest(Path(__file__)),
            'checked_frames':checked,'checked_predictions':checked*len(METHODS),'prelabel_guard':dict(guard),
            'error_count':len(errors),'errors':errors}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=new_output(a.run/'independent_audit');result=audit(a.run);out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError(f"Independent audit found {result['error_count']} discrepancies")
    print(json.dumps({'passed':True,'frames':result['checked_frames']}))


if __name__=='__main__':main()
