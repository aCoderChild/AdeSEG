#!/usr/bin/env python3
"""Independently rescore 20 archived CMSA sequences on matched target frames."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.audit_additional_author_comparators_v2 import compare
from scripts.audit_boundary_video_factorial_v1 import label, metrics, summarize
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import new_output
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_archived_cmsa_automatic_cohort_v2 import (DATA, PRIOR, PRIORITY, SEQUENCES,
    read_bound, require, validate_inputs)


def audit(verification,scores):
    verification=Path(verification).resolve(strict=True);scores=Path(scores).resolve(strict=True)
    guard=install_phase_mask_guard(DATA)
    verify_sha=digest(verification);proof=read_bound(verification,verify_sha)
    score_path=scores/'results.json';score_sha=digest(score_path);primary=read_bound(score_path,score_sha)
    current=validate_inputs(proof['batch'],proof['reconciliation_path'])
    require(proof.get('passed') is True and proof.get('label_guard',{}).get('open_attempts')==0 and
            primary.get('passed') is True and primary.get('verification_sha256')==verify_sha and
            primary.get('scorer_sha256')==proof.get('scorer_sha256') and
            primary.get('report_sha256')==digest(scores/'report.md') and
            all(proof.get(k)==v for k,v in current.items()) and guard['open_attempts']==0,
            'Frozen proof or primary scores differ')
    guard['active']=False;batch=Path(proof['batch'])
    benchmark=read_bound(batch/'scores/results.json',proof['batch_hashes']['score'])
    prior=read_bound(PRIOR,proof['prior_sha256']);videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')}
    errors=[];checked=0;patient_means=[];present_dice=[];empty_count=0;false_alarms=0
    for seq in SEQUENCES:
        info=proof['records'][seq];frames=videos[seq];start=info['prompt_frame']
        meta=read_bound(info['frame_metadata']['path'],info['frame_metadata']['sha256'])
        reference=prior['rows'][seq] if seq in PRIORITY else benchmark['rows'][seq]
        require(len(frames)==len(meta)==len(reference)==len(primary['rows'][seq]),'Audit coverage differs')
        rows=[]
        for i,(frame,ref,pred) in enumerate(zip(frames,reference,info['predictions'],strict=True)):
            path=Path(pred['path']);label_sha=meta[i]['label_sha256'];image_sha=info['images'][i]
            require(digest(frame.mask_path)==label_sha and digest(frame.image_path)==image_sha and digest(path)==pred['sha256'],
                    'Audit input changed')
            truth=label(frame.mask_path)
            with Image.open(path) as image:mask=np.asarray(image.convert('L'))>=128
            require(digest(frame.mask_path)==label_sha and digest(frame.image_path)==image_sha and digest(path)==pred['sha256'],
                    'Audit input changed during decode')
            result=metrics(mask,truth)
            require(result['present']==ref[info['reference_arm']]['present'],'Reference label presence differs')
            row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,
                 'initialization_frame':i==start,'cmsa':result}
            compare(errors,[seq,i],row,primary['rows'][seq][i]);rows.append(row);checked+=1
        value=summarize(rows,'cmsa',True);compare(errors,[seq,'summary'],value,primary['summaries'][seq])
        present=[r['cmsa']['dice'] for r in rows if r['post_prompt'] and r['cmsa']['present']]
        empty=[r['cmsa'] for r in rows if r['post_prompt'] and not r['cmsa']['present']]
        if present:patient_means.append(float(np.mean(present)));present_dice.extend(present)
        empty_count+=len(empty);false_alarms+=sum(v['false_alarm'] for v in empty)
    overall={'patients_with_present_frames':len(patient_means),'present_frames':len(present_dice),
             'patient_macro_present_dice':float(np.mean(patient_means)) if patient_means else None,
             'frame_weighted_present_dice':float(np.mean(present_dice)) if present_dice else None,
             'empty_frames':empty_count,'false_alarms':false_alarms}
    compare(errors,['overall'],overall,primary['overall'])
    return {'passed':not errors,'scope':'Independent CPU metrics; shared structural verifier.',
            'verification_sha256':verify_sha,'score_sha256':score_sha,'auditor_sha256':digest(Path(__file__)),
            'checked_frames':checked,'prelabel_guard':dict(guard),'error_count':len(errors),'errors':errors}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('verification','scores','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=new_output(a.output);result=audit(a.verification,a.scores);out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError(f"Independent metrics found {result['error_count']} discrepancies")
    print(json.dumps({'passed':True,'frames':result['checked_frames']}))


if __name__=='__main__':main()
