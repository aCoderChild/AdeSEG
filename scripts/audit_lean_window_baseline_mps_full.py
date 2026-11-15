#!/usr/bin/env python3
"""Audit a frozen full MPS Lean baseline against the archived CPU run.

No labels are opened. Measured differences are recorded, not used as a
post-hoc statistical equivalence claim.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_window_baseline_mps import verify_gate


def read(path):return json.loads(path.read_text())


def audit(mps,cpu,samurai,gate_path):
    r=read(mps/'receipt.json');sam=read(samurai/'receipt.json')
    if not r.get('passed') or not r.get('full_video') or r['identity']['device']!='mps':raise ValueError('MPS baseline incomplete')
    if r['gate_sha256']!=digest(gate_path):raise ValueError('Engineering gate changed')
    gate=verify_gate(gate_path,r['sequence'],r['prompt'],r['identity'])
    if gate['state_sha256']!=r['state_sha256']:raise ValueError('Live model state differs from engineering check')
    if not sam.get('passed') or not sam.get('full_video') or sam['sequence']!=r['sequence'] or sam['prompt']!=r['prompt'] or sam['prompt_frame']!=r['prompt_frame']:
        raise ValueError('SAMURAI input prompt differs')
    done=read(cpu/'completion.json')
    if not done.get('passed'):raise ValueError('CPU archive incomplete')
    if r['sequence']=='seq23':
        if digest(cpu/'frames.json')!=done['frames_sha256']:raise ValueError('CPU frame receipt differs')
        hashes=[x['mask_sha256'] for x in read(cpu/'frames.json')]
    else:
        if digest(cpu/'records.json')!=done['records_sha256']:raise ValueError('CPU records receipt differs')
        hashes=read(cpu/'records.json')['mask_sha256']
    if len(r['records'])!=len(hashes) or r['end_frame']!=len(hashes)-1:raise ValueError('Full video coverage differs')
    rows=[]
    for i,(record,expected) in enumerate(zip(r['records'],hashes,strict=True)):
        if record['frame']!=i or digest(mps/f'{i:06d}.png')!=record['mask_sha256'] or digest(cpu/f'{i:06d}.png')!=expected:
            raise ValueError('Frozen mask hash or frame differs')
        with Image.open(mps/f'{i:06d}.png') as img:a=np.asarray(img.convert('L'))>0
        with Image.open(cpu/f'{i:06d}.png') as img:b=np.asarray(img.convert('L'))>0
        if a.shape!=b.shape:raise ValueError('Mask geometries differ')
        union=int((a|b).sum());inter=int((a&b).sum())
        rows.append({'frame':i,'different_pixels':int((a!=b).sum()),'iou':inter/union if union else 1.})
    writes=[e for e in r['token_window_events'] if e['stage']=='after_write']
    plans=[e for e in r['token_window_events'] if e['stage']=='admission_plan']
    if len(writes)!=r['end_frame']-r['prompt_frame'] or len(plans)!=len(writes) or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):
        raise ValueError('Memory accounting differs')
    return {'passed':True,'scope':'Frozen CPU/MPS automatic-mask project Lean baseline audit, no labels or efficacy inference.',
      'sequence':r['sequence'],'prompt_frame':r['prompt_frame'],'frames':len(rows),'rows':rows,
      'cpu_mps_differing_pixels':sum(x['different_pixels'] for x in rows),
      'minimum_frame_iou':min(x['iou'] for x in rows),'differing_frames':sum(x['different_pixels']>0 for x in rows),
      'memory_writes':len(writes),'evictions':sum(bool(x['evict_frames']) for x in plans),
      'max_total_tokens':max(x['total_tokens'] for x in writes),
      'mps_receipt_sha256':digest(mps/'receipt.json'),'cpu_completion_sha256':digest(cpu/'completion.json'),
      'samurai_receipt_sha256':digest(samurai/'receipt.json'),'gate_sha256':digest(gate_path),'auditor_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('mps','cpu','samurai','gate','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=audit(a.mps,a.cpu,a.samurai,a.gate)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
