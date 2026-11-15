#!/usr/bin/env python3
"""Freeze historical usage scores on verified whole-source deletion groups."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_historical_usage_score import percentile_ranks
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('usage','usage-verification','capture','replay','replay-verification','loso','loso-audit','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());u=read(a.usage/'historical_usage_receipt.json');v=read(a.usage_verification/'verification.json');c=read(a.capture/'source_capture_receipt.json');r=read(a.replay/'completion.json');rv=read(a.replay_verification/'verification.json');l=read(a.loso/'predictions.json');done=read(a.loso_audit/'completion.json');seq=u['sequence'];repeat=a.loso.parent/f'{seq}_attempt02'
if not u['passed'] or not v['passed'] or v['capture_sha256']!=digest(a.usage/'historical_usage_receipt.json') or v['source_reference_sha256']!=digest(a.capture/'source_capture_receipt.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_historical_usage.py') or not c['passed'] or not r['passed'] or r['sequence']!=seq or r['capture_sha256']!=digest(a.capture/'source_capture_receipt.json') or not rv['passed'] or rv['first_sha256']!=digest(a.replay/'completion.json') or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_deletion.py') or not l['passed'] or l['holdout']!=seq or seq in l['training_outcome_sha256'] or l!=read(repeat/'predictions.json') or digest(a.loso/'research_gate.pt')!=l['checkpoint_sha256'] or not done['passed']:raise ValueError('Source, usage, replay or LOSO verification changed')
events={e['source_frame']:e for e in u['events']};preds={x['frame']:x for x in l['predictions']};rows=[]
for e in r['events']:
 if 'groups' not in e:continue
 frame=e['frame'];path=a.capture/'source_snapshots'/f'{frame:06d}.pt'
 if digest(path)!=c['source_snapshots'][path.name]:raise ValueError('Source snapshot changed')
 snap=torch.load(path,map_location='cpu',weights_only=True);values={}
 for source in snap['sources']:
  i=source['frame']
  if i>=frame-1:continue
  if i not in events:raise ValueError('Historical source usage missing')
  e0=events[i];fp=a.usage/'historical_usage'/e0['file']
  if digest(fp)!=e0['sha256']:raise ValueError('Usage changed')
  item=torch.load(fp,map_location='cpu',weights_only=True);ids=source['token_ids'].long()
  if item['source_frame']!=i or item['read_frame']>=frame or not torch.equal(item['keep_idx'],ids):raise ValueError('Usage identity or causality differs')
  ranks=percentile_ranks(item['usage'].numpy());start=source['start']
  for j,token in enumerate(ids.tolist()):
   if start+j in values:raise ValueError('Source spans overlap')
   values[start+j]=float(ranks[token])
 x=preds[frame];scores={'utility_trained':x['scores'],**x['controls']};scores['historical_usage']=[float(np.mean([values[j] for j in group])) for group in e['groups']]
 if any(len(vals)!=5 or not np.isfinite(vals).all() for vals in scores.values()):raise ValueError('Invalid scores')
 rows.append({'frame':frame,'scores':scores})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'sequence':seq,'events':rows,'usage_sha256':digest(a.usage/'historical_usage_receipt.json'),'usage_verification_sha256':digest(a.usage_verification/'verification.json'),'source_sha256':digest(a.capture/'source_capture_receipt.json'),'source_replay_sha256':digest(a.replay/'completion.json'),'loso_predictions_sha256':digest(a.loso/'predictions.json'),'loso_audit_completion_sha256':digest(a.loso_audit/'completion.json'),'percentile_sha256':digest(ROOT/'modeling/lean_historical_usage_score.py'),'protocol_sha256':digest(ROOT/'protocols/lean_historical_usage_group_control_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Label-free historical-source scores; no target outcomes opened.'};(out/'scores.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'reads':len(rows)}))
