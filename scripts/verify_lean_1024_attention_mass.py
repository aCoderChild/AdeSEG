#!/usr/bin/env python3
"""Recompute captured attention-mass summaries using independent NumPy sums."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','replay','analysis','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();c=read(a.capture/'priority_capture_receipt.json');r=read(a.replay/'completion.json');done=read(a.analysis/'completion.json');d=read(a.analysis/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.analysis/'results.json') or d['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or d['replay_sha256']!=digest(a.replay/'completion.json') or d['runner_sha256']!=digest(ROOT/'scripts/analyze_lean_1024_attention_mass.py'):raise ValueError('Analysis provenance changed')
shots={e['frame']:e for e in c['snapshots']}
for row,e in zip(d['frames'],r['events'],strict=True):
 f=row['frame'];path=a.capture/'snapshots'/f'{f:06d}.pt';source_path=a.capture/'source_snapshots'/f'{f:06d}.pt'
 if f!=e['frame'] or digest(path)!=shots[f]['read_sha256'] or digest(source_path)!=shots[f]['sources_sha256']:raise ValueError('Snapshot changed')
 t=torch.load(path,map_location='cpu',weights_only=True);s=torch.load(source_path,map_location='cpu',weights_only=True);u=t['attention_outputs'][1].numpy().astype(np.float64);ptr=int(t['attention_inputs']['num_obj_ptr_tokens']);total=u.sum();eligible=np.zeros(len(u),dtype=bool)
 for v in s['sources']:eligible[v['start']:v['start']+len(v['token_ids'])]=True
 cond=np.zeros(len(u),dtype=bool);cond[:4096]=True;pointer=np.zeros(len(u),dtype=bool)
 if ptr:pointer[-ptr:]=True
 other=~(cond|pointer|eligible)
 for name,select in [('conditioning',cond),('eligible_historical',eligible),('pointer',pointer),('other_spatial',other)]:
  if abs(float(u[select].sum()/total)-row['partition'][name])>1e-12:raise ValueError('Partition differs')
 for name,ids in e['deleted'].items():
  if abs(float(u[ids].sum()/total)-row['removed_attention_mass'][name])>1e-12:raise ValueError('Removed mass differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':c['sequence'],'analysis_sha256':digest(a.analysis/'results.json'),'frames':len(d['frames']),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print(c['sequence'],'attention mass verified')
