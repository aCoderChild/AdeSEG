#!/usr/bin/env python3
"""Audit short offline SAM2Long CPU/MPS replays without opening labels.

The author selects a path after each complete clip; these short-clip outputs
are not a prefix or performance estimate of a full-video run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_sam2long_automatic_priority import CONFIG,MANIFEST,PATHWAY_CONFIG,dependency_hashes


def read(path):return json.loads(path.read_text())


def audit(cpu,mps_a,mps_b,samurai):
    paths=[Path(cpu),Path(mps_a),Path(mps_b)];runs=[read(path/'receipt.json') for path in paths]
    common=('sequence','prompt_frame','prompt','end_frame','state_sha256')
    for index,(path,run) in enumerate(zip(paths,runs,strict=True)):
        ident=run['identity']
        if not run.get('passed') or not run.get('noncausal') or run.get('full_video') or not run.get('deterministic') or ident['device']!=('cpu' if index==0 else 'mps'):
            raise ValueError('Unexpected offline engineering receipt')
        if (ident['runner_sha256']!=digest(ROOT/'scripts/run_sam2long_automatic_priority.py') or ident['config_sha256']!=digest(CONFIG)
            or ident['manifest_sha256']!=digest(MANIFEST) or ident['pathways']!=PATHWAY_CONFIG or ident['dependencies']!=dependency_hashes()
            or (index and (ident['fallback']!='0' or ident['fast_math']!='0'))):
            raise ValueError('Author runtime configuration differs')
        if len(run['records'])!=run['end_frame']+1 or [x['frame'] for x in run['records']]!=list(range(run['end_frame']+1)) or run['end_frame']-run['prompt_frame']<3:
            raise ValueError('Clip coverage differs')
        pa=run.get('pathway_audit',{})
        if (pa.get('audited_post_prompt_frames')!=run['end_frame']-run['prompt_frame']
            or pa.get('pathways_after_initial_step')!=3
            or pa.get('checked_tensor_fields')!=['pred_masks','maskmem_features','obj_ptr','object_score_logits','ious']):
            raise ValueError('Author pathway state was not audited')
        if any(digest(path/f"{x['frame']:06d}.png")!=x['mask_sha256'] for x in run['records']):raise ValueError('Frozen output changed')
    if any(any(r[k]!=runs[0][k] for k in common) for r in runs[1:]):raise ValueError('Clip, prompt, or model state differs')
    base={k:v for k,v in runs[0]['identity'].items() if k!='device'}
    if any({k:v for k,v in r['identity'].items() if k!='device'}!=base for r in runs[1:]):raise ValueError('Cross-device asset/environment identity differs')
    sam=read(samurai/'receipt.json')
    if not sam.get('passed') or not sam.get('full_video') or sam['sequence']!=runs[0]['sequence'] or sam['prompt_frame']!=runs[0]['prompt_frame'] or sam['prompt']!=runs[0]['prompt'] or sam['checkpoint_sha256']!=base['checkpoint_sha256']:
        raise ValueError('SAMURAI archived automatic mask differs')
    rows=[]
    for item in runs[0]['records']:
        frame=item['frame']
        if any(run['records'][frame]['image_sha256']!=item['image_sha256'] for run in runs[1:]):raise ValueError('Source images differ')
        masks=[]
        for path in paths:
            with Image.open(path/f'{frame:06d}.png') as im:masks.append(np.asarray(im.convert('L'))>0)
        if len({x.shape for x in masks})!=1:raise ValueError('Output geometry differs')
        union=int((masks[0]|masks[1]).sum());inter=int((masks[0]&masks[1]).sum())
        rows.append({'frame':frame,'cpu_mps_pixels':int((masks[0]!=masks[1]).sum()),
                     'mps_repeat_pixels':int((masks[1]!=masks[2]).sum()),'cpu_mps_iou':inter/union if union else 1.})
    return {'passed':True,'scope':'Offline short-clip engineering comparison, no efficacy or full-video prefix claim.',
      'sequence':runs[0]['sequence'],'prompt':runs[0]['prompt'],'identity':runs[1]['identity'],'state_sha256':runs[0]['state_sha256'],
      'verifier_sha256':digest(Path(__file__)),'samurai_receipt_sha256':digest(samurai/'receipt.json'),
      'source_receipt_sha256':[digest(path/'receipt.json') for path in paths],'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cpu','mps-a','mps-b','samurai','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=audit(a.cpu,a.mps_a,a.mps_b,a.samurai)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':result['sequence'],'frames':len(result['rows']),
      'cpu_mps_pixels':sum(x['cpu_mps_pixels'] for x in result['rows']),
      'mps_repeat_pixels':sum(x['mps_repeat_pixels'] for x in result['rows'])}))


if __name__=='__main__':main()
