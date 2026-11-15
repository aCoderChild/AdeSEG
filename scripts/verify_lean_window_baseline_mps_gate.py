#!/usr/bin/env python3
"""Independently verify two short frozen-prompt MPS Lean baseline runs.

Checks exact SAMURAI prompt, source frames, fork/runtime identity, token budget,
and records CPU/MPS and MPS-repeat mask differences without choosing a
post-hoc numerical acceptance threshold. This does not evaluate quality.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(path):return json.loads(path.read_text())


def verify(a,b,cpu_reference,samurai):
    paths=[Path(a),Path(b)];runs=[read(x/'receipt.json') for x in paths]
    if any(not r.get('passed') or r.get('full_video') for r in runs):raise ValueError('Unexpected short engineering run')
    for key in ('sequence','prompt_frame','prompt','end_frame','identity','state_sha256'):
        if runs[0][key]!=runs[1][key]:raise ValueError(f'Engineering identities differ: {key}')
    one,two=runs;sequence=one['sequence'];identity=one['identity']
    if identity['runner_sha256']!=digest(ROOT/'scripts/run_lean_window_baseline_mps.py') or identity['device']!='mps' or identity['fallback']!='0' or identity['fast_math']!='0':
        raise ValueError('Runner or MPS settings differ')
    sam=read(samurai/'receipt.json')
    if not sam.get('passed') or not sam.get('full_video') or sam['sequence']!=sequence or sam['prompt_frame']!=one['prompt_frame'] or sam['prompt']!=one['prompt'] or sam['checkpoint_sha256']!=identity['checkpoint_sha256']:
        raise ValueError('SAMURAI prompt, frame or checkpoint does not match')
    done=read(cpu_reference/'completion.json')
    if not done.get('passed'):raise ValueError('Archived CPU baseline incomplete')
    if sequence=='seq23':
        if digest(cpu_reference/'frames.json')!=done['frames_sha256']:raise ValueError('CPU baseline frame receipt differs')
        old=read(cpu_reference/'frames.json');hashes=[x['mask_sha256'] for x in old]
    else:
        if digest(cpu_reference/'records.json')!=done['records_sha256']:raise ValueError('CPU baseline record receipt differs')
        old=read(cpu_reference/'records.json');hashes=old['mask_sha256']
    if len(hashes)<=one['end_frame']:raise ValueError('CPU reference too short')
    for run,path in zip(runs,paths,strict=True):
        rows=run['records']
        if len(rows)!=run['end_frame']+1 or [r['frame'] for r in rows]!=list(range(run['end_frame']+1)):
            raise ValueError('Engineering frame coverage differs')
        if any(digest(path/f"{r['frame']:06d}.png")!=r['mask_sha256'] for r in rows):raise ValueError('Engineering mask hash differs')
        writes=[e for e in run['token_window_events'] if e['stage']=='after_write']
        plans=[e for e in run['token_window_events'] if e['stage']=='admission_plan']
        if len(writes)!=run['end_frame']-run['prompt_frame'] or len(plans)!=len(writes) or not any(e['evict_frames'] for e in plans) or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):
            raise ValueError('Memory coverage or budget differs')
    if [r['image_sha256'] for r in one['records']] != [r['image_sha256'] for r in two['records']]:raise ValueError('Images differ between replays')
    rows=[]
    for i in range(one['end_frame']+1):
        cpu_path=cpu_reference/f'{i:06d}.png'
        if digest(cpu_path)!=hashes[i]:raise ValueError('Archived CPU prediction changed')
        masks=[]
        for path in (cpu_reference,*paths):
            with Image.open(path/f'{i:06d}.png') as image:masks.append(np.asarray(image.convert('L'))>0)
        if len({x.shape for x in masks})!=1:raise ValueError('Output geometry differs')
        union=int((masks[0]|masks[1]).sum());inter=int((masks[0]&masks[1]).sum())
        rows.append({'frame':i,'cpu_mps_pixels':int((masks[0]!=masks[1]).sum()),
                     'mps_repeat_pixels':int((masks[1]!=masks[2]).sum()),'cpu_mps_iou':inter/union if union else 1.})
    return {'passed':True,'scope':'Short frozen-automatic-mask engineering replay only; no efficacy or guarantee for other videos.',
      'sequence':sequence,'prompt':one['prompt'],'identity':identity,'state_sha256':one['state_sha256'],
      'source_receipt_sha256':[digest(path/'receipt.json') for path in paths],
      'samurai_receipt_sha256':digest(samurai/'receipt.json'),'cpu_completion_sha256':digest(cpu_reference/'completion.json'),
      'window_events_equal':one['token_window_events']==two['token_window_events'],
      'selection_equal':one['selection']==two['selection'],
      'rows':rows,'verifier_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('mps-a','mps-b','cpu-reference','samurai','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=verify(a.mps_a,a.mps_b,a.cpu_reference,a.samurai)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':result['sequence'],'frames':len(result['rows']),
      'cpu_mps_pixels':sum(x['cpu_mps_pixels'] for x in result['rows']),
      'mps_repeat_pixels':sum(x['mps_repeat_pixels'] for x in result['rows'])}))


if __name__=='__main__':main()
