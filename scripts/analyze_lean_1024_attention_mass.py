#!/usr/bin/env python3
"""Summarize captured author usage mass for retained-token experiments."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','replay','replay-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');r=read(a.replay/'completion.json');v=read(a.replay_verification/'verification.json')
if not all(x['passed'] for x in (cap,r,v)) or r['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or v['first_sha256']!=digest(a.replay/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_retention_1024.py'):raise ValueError('Inputs changed')
snaps={x['frame']:x for x in cap['snapshots']};results=[]
for e in r['events']:
 f=e['frame'];sp=a.capture/'snapshots'/f'{f:06d}.pt';src=a.capture/'source_snapshots'/f'{f:06d}.pt'
 if digest(sp)!=snaps[f]['read_sha256'] or digest(src)!=snaps[f]['sources_sha256']:raise ValueError('Snapshot changed')
 data=torch.load(sp,map_location='cpu',weights_only=True);s=torch.load(src,map_location='cpu',weights_only=True);attention=data['attention_outputs']
 if not isinstance(attention,tuple) or len(attention)!=2 or not torch.is_tensor(attention[1]):raise ValueError('Author usage unavailable')
 u=attention[1].double();ptr=int(data['attention_inputs']['num_obj_ptr_tokens']);total=u.sum()
 if u.ndim!=1 or len(u)!=data['attention_inputs']['memory'].shape[0] or not torch.isfinite(u).all() or (u<0).any() or total<=0:raise ValueError('Usage invalid')
 eligible=sorted(j for source in s['sources'] for j in range(source['start'],source['start']+len(source['token_ids'])))
 if len(eligible)!=5120 or len(set(eligible))!=5120 or min(eligible)<4096 or max(eligible)>=len(u)-ptr:raise ValueError('Eligible source geometry changed')
 partition={'conditioning':float(u[:4096].sum()/total),'eligible_historical':float(u[eligible].sum()/total),'pointer':float(u[-ptr:].sum()/total) if ptr else 0.}
 partition['other_spatial']=1.-sum(partition.values());removed={m:float(u[ids].sum()/total) for m,ids in e['deleted'].items()}
 results.append({'frame':f,'usage_sum':float(total),'partition':partition,'removed_attention_mass':removed})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'sequence':cap['sequence'],'scope':'Last-layer attention mass on the frozen full read before intervention; no target labels and no assertion of causal importance.','frames':results,'capture_sha256':digest(a.capture/'priority_capture_receipt.json'),'replay_sha256':digest(a.replay/'completion.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n');print(cap['sequence'], 'conditioning',round(float(np.mean([x['partition']['conditioning'] for x in results])),6),'eligible',round(float(np.mean([x['partition']['eligible_historical'] for x in results])),6),'prompt-selected removal',round(float(np.mean([x['removed_attention_mass']['prompt_conditioned'] for x in results])),6))
