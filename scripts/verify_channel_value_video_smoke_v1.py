#!/usr/bin/env python3
"""Audit minimal MPS A/B/C/D smoke and a separate cropped-seq23 A smoke."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_value_window_runtime_v1 import BASE, runtime_sources, require
from scripts.crossfit_observer_capture_shared_v1 import DATA,phase_artifact,sequence_inputs,validate_plan
from scripts.memory_token_io import digest
from scripts.run_channel_value_video_v1 import compare_archived
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def read(path):return json.loads(Path(path).read_text())


def inspect(root,seq,arm,certificate_sha,plan):
    folder=root/'smoke'/seq/arm;file=folder/'receipt.json'
    expected=phase_artifact(root,f'smoke_{seq}_{arm}',file);rec=read(file)
    require(digest(file)==expected and rec.get('passed') is True and rec.get('sequence')==seq and
            rec.get('arm')==arm and rec.get('limit')==4 and rec.get('full_video') is False and
            rec.get('certificate_sha256')==certificate_sha and rec.get('smoke_audit_sha256') is None and
            rec.get('label_guard',{}).get('open_attempts')==0 and
            rec.get('runner_sha256')==digest(ROOT/'scripts/run_channel_value_video_v1.py'),
            'Smoke receipt identity differs')
    frames,records,start,_,_,baseline,old,prompt=sequence_inputs(plan,seq)
    require(rec.get('prompt')==prompt and rec.get('prompt_frame')==start and rec.get('input_images')==records and
            rec.get('baseline_completion_path')==str(baseline) and rec.get('baseline_completion_sha256')==digest(baseline),
            'Smoke source, prompt or baseline differs')
    end=start+5;require([x['frame'] for x in rec['rows']]==list(range(start,end)) and
                         [x['frame'] for x in rec['masks']]==list(range(end)),'Smoke frame coverage differs')
    for row in rec['masks']:
        i=row['frame'];path=folder/f'{i:06d}.png';require(digest(path)==row['sha256'],'Smoke mask changed')
        with Image.open(frames[i].image_path) as source,Image.open(path) as image:
            data=np.asarray(image.convert('L'))
            require(image.format=='PNG' and image.size==source.size and np.isin(data,(0,255)).all() and
                    (i>=start or not data.any()),'Smoke mask geometry or encoding differs')
        require(digest(path)==row['sha256'],'Smoke mask changed while decoding')
    trace={}
    for name in ('memory_events','positions','selections','telemetry'):
        file=folder/(name+'.json');sha=rec['trace_sha256'][name]
        require(digest(file)==sha,'Smoke trace changed');trace[name]=read(file)
        require(digest(file)==sha,'Smoke trace changed during read')
    require([x['frame'] for x in trace['telemetry']]==list(range(start,end)),'Smoke telemetry coverage differs')
    compare_archived(rec['rows'],{k:trace[k] for k in ('memory_events','positions','selections')},baseline,old,arm,True)
    runtime=folder/'verification/runtime_verification.json'
    require(digest(runtime)==rec['runtime_verification_sha256'],'Smoke runtime changed')
    r=read(runtime);require(digest(runtime)==rec['runtime_verification_sha256'] and r.get('passed') is True and
        r.get('certificate_sha256')==certificate_sha and r.get('runtime_sources')==runtime_sources() and
        r.get('device')=='mps' and r.get('dtype')=='float32' and r.get('memory_budget')==16384 and
        r.get('insurance_reserve')==3072,'Smoke runtime identity differs')
    for name in ('model','refiner'):
        require(digest(folder/'verification'/(name+'_tensor_fingerprints.json'))==r[name+'_tensor_fingerprints_sha256'],
                'Smoke model fingerprint changed')
    return {'receipt_sha256':expected,'rows':rec['rows'],'telemetry':trace['telemetry'],
            'runtime_sha256':rec['runtime_verification_sha256']}


def audit(root,certificate_sha):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve()),'Batch outside experiment root')
    plan=validate_plan();items={(s,a):inspect(root,s,a,certificate_sha,plan)
                                for s,a in [('seq20','A'),('seq20','B'),('seq20','C'),('seq20','D'),('seq23','A')]}
    # At the first post-prompt frame every arm starts with the same state.
    a,b,c,d=(items[('seq20',name)]['rows'][1] for name in ('A','B','C','D'))
    require(a['output_logits_sha256']==c['output_logits_sha256'] and
            b['output_logits_sha256']==d['output_logits_sha256'] and
            c['maskmem_features_sha256']==d['maskmem_features_sha256'],
            'First-step factorial routes differ')
    return {'passed':True,'scope':'Minimal MPS-only routing smoke; no repeated/cross-device equivalence or efficacy claim.',
            'certificate_sha256':certificate_sha,'runs':{f'{s}_{a}':{'receipt_sha256':v['receipt_sha256'],
                 'runtime_sha256':v['runtime_sha256']} for (s,a),v in items.items()},
            'runner_sha256':digest(ROOT/'scripts/run_channel_value_video_v1.py'),
            'auditor_sha256':digest(Path(__file__))}


def verify_smoke_audit(path,root,certificate_sha,*,reconstruct=True):
    root=Path(root).resolve(strict=True);path=Path(path).resolve(strict=True)
    expected=phase_artifact(root,'smoke_audit',path);saved=read(path)
    require(digest(path)==expected and saved.get('passed') is True and saved.get('certificate_sha256')==certificate_sha and
            saved.get('auditor_sha256')==digest(Path(__file__)) and
            saved.get('runner_sha256')==digest(ROOT/'scripts/run_channel_value_video_v1.py') and
            saved.get('label_guard',{}).get('open_attempts')==0,'Smoke audit or inputs changed')
    if reconstruct:
        require({k:v for k,v in saved.items() if k!='label_guard'}==audit(root,certificate_sha),
                'Reconstructed smoke audit differs')
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-root',type=Path,required=True);p.add_argument('--certificate-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.batch_root.resolve(strict=True);out=a.output.resolve()
    require(out.is_relative_to(root) and not out.exists(),'Invalid smoke audit output')
    guard=install_mask_open_guard(DATA);result=audit(root,a.certificate_sha256)
    require(guard['open_attempts']==0,'Label open attempted during smoke audit')
    result['label_guard']=dict(guard);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out)}))


if __name__=='__main__':main()
