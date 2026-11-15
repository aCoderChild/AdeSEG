#!/usr/bin/env python3
"""Score and describe audited development captures, after trajectories freeze.

This reads labels only for the ten prespecified training frames of seq6 or
seq13 after full reference/capture and prompt audits. It does not alter model
inputs, use labels in a prompt, or claim subject independence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.preflight_expanded_train_capture_v1 import TARGETS


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference','capture','verification','prompt-audit','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--sequence',choices=tuple(TARGETS),required=True);a=p.parse_args()
    plan=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Capture plan changed')
    ref=read(a.reference/'receipt.json');cap=read(a.capture/'receipt.json');verify=read(a.verification/'verification.json');prompt=read(a.prompt_audit/'verification.json')
    if (not verify['passed'] or verify['sequence']!=a.sequence or verify['targets']!=TARGETS[a.sequence]
            or verify['auditor_sha256']!=digest(ROOT/'scripts/audit_expanded_train_capture_v1.py')
            or verify['reference_receipt_sha256']!=digest(a.reference/'receipt.json')
            or verify['capture_receipt_sha256']!=digest(a.capture/'receipt.json')
            or not prompt['passed'] or prompt['sequence']!=a.sequence
            or prompt['runner_sha256']!=digest(ROOT/'scripts/audit_expanded_initial_prompt_v1.py')
            or prompt['reference_receipt_sha256']!=digest(a.reference/'receipt.json')
            or prompt['capture_receipt_sha256']!=digest(a.capture/'receipt.json')):
        raise ValueError('Full capture or prompt audit differs')
    if ref['identity']!=cap['identity'] or ref['rows']!=cap['rows']:
        raise ValueError('Reference/capture trajectory identity differs')
    obs={x['frame']:x for x in cap['observations']}
    if sorted(obs)!=TARGETS[a.sequence]:raise ValueError('Frozen target coverage differs')
    spec=plan['sequences'][a.sequence];folder=Path(spec['source'])/a.sequence
    if digest(folder/'frames.json')!=spec['source_frames_sha256']:raise ValueError('Source metadata changed')
    archive=read(folder/'frames.json');frames=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev')[a.sequence]
    # Labels are first opened below, after the complete two trajectories,
    # observation verification and prompt audit have been frozen.
    result_rows=[];samples=[]
    for i in TARGETS[a.sequence]:
        frame=frames[i];entry=obs[i]
        if digest(frame.image_path)!=spec['image_sha256'][i] or digest(frame.mask_path)!=archive[i]['label_sha256']:
            raise ValueError('Image or label changed')
        feature=Path(entry['feature_file']);snapshot=Path(entry['snapshot_file']);mask=a.capture/f'{i:06d}.png'
        if digest(feature)!=entry['feature_sha256'] or digest(snapshot)!=entry['snapshot_sha256']:
            raise ValueError('Captured sample changed')
        frozen=next(x for x in cap['rows'] if x['frame']==i)
        if digest(mask)!=frozen['mask_sha256']:raise ValueError('Frozen output mask changed')
        truth=load_label_mask(frame.mask_path,[1])==1;pred=np.asarray(Image.open(mask).convert('L'))>0
        if truth.shape!=pred.shape:raise ValueError('Label geometry differs')
        snap=torch.load(snapshot,map_location='cpu',weights_only=True)
        score=float(snap['decoder_outputs'][6][0,0]);low=snap['decoder_outputs'][3]
        den=int(pred.sum()+truth.sum());intersection=int((pred&truth).sum())
        row={'sequence':a.sequence,'frame':i,'present':bool(truth.any()),'target_pixels':int(truth.sum()),
             'prediction_pixels':int(pred.sum()),'overlap_pixels':intersection,'dice':2*intersection/den if den else 1.,
             'false_alarm':bool(pred.any()) if not truth.any() else None,'object_score':score,
             'suppressed_to_minus1024':bool(torch.all(low==-1024))}
        result_rows.append(row)
        samples.append({'split':'train','sequence':a.sequence,'frame':i,'image':str(frame.image_path),
                        'image_sha256':digest(frame.image_path),'label':str(frame.mask_path),'label_sha256':digest(frame.mask_path),
                        'feature':str(feature),'feature_sha256':digest(feature),'snapshot':str(snapshot),'snapshot_sha256':digest(snapshot),
                        'capture_device':'mps'})
    present=[x for x in result_rows if x['present']];empty=[x for x in result_rows if not x['present']]
    identity_sha=hashlib.sha256(json.dumps(cap['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result={'passed':True,'scope':'Retrospective training-only sample audit; not held-out or patient-independent evaluation.',
            'sequence':a.sequence,'samples':samples,'rows':result_rows,'mps_runtime_identity_sha256':identity_sha,
            'cpu_training_feature_pipeline_sha256':plan['cpu_feature_pipeline_sha256'],
            'device_shift_note':'Existing seq2/3/4 features were captured on CPU; these features were captured on MPS. Prior seq4 CPU/MPS checks cover only limited selected features.',
            'prompt_audit_sha256':digest(a.prompt_audit/'verification.json'),'capture_verification_sha256':digest(a.verification/'verification.json'),
            'capture_receipt_sha256':digest(a.capture/'receipt.json'),'preflight_sha256':digest(a.preflight/'manifest.json'),
            'runner_sha256':digest(Path(__file__)),'summary':{'present':len(present),'empty':len(empty),
            'mean_present_dice':float(np.mean([r['dice'] for r in present])) if present else None,
            'empty_false_alarms':sum(r['false_alarm'] for r in empty),
            'zero_overlap_present':sum(r['overlap_pixels']==0 for r in present),
            'suppressed_present':sum(r['suppressed_to_minus1024'] for r in present)}}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'samples.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'samples_sha256':digest(out/'samples.json')},indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'summary':result['summary'],'rows':result_rows},indent=2),flush=True)


if __name__=='__main__':main()
